# Troubleshooting and FAQ

Entries marked **(observed)** were hit while running this repository on CPU/gloo;
the rest are standard distributed-training failure modes mapped onto this code.

## Start-up

**`ValueError: tp_size=X does not divide world_size=Y`** — `resolve_dp_size`
(`src/parallelism/mesh.py`). Pick `tp` among the divisors it lists.

**`ValueError: model.n_kv_heads=2 not divisible by tp_size=4`** — TP splits heads and
the FFN evenly (`TrainingConfig.validate`). Use `tp <= n_kv_heads` and a divisor of
`n_heads`, `n_kv_heads` and `ffn_hidden_size`. (Before the audit this surfaced as an
opaque DTensor sharding error.)

**`RuntimeError: backend='nccl' requested but torch.cuda.is_available() is False`** —
`src/utils/env.py`. Pass `--backend gloo` (CPU) or fix the CUDA install. The YAMLs
other than `test_tiny.yaml` default to `nccl`.

**`global_batch_size identity violated`** — set `global_batch_size: 0` to derive it,
or make it `micro_batch_size · grad_accum_steps · dp_size`.

**`RuntimeError: ... batch config disagrees across ranks`** — ranks loaded different
configs (typically a stale file on one node).

**Warning `DTensor random operators may not have complete support on cpu device
mesh` (observed)** — emitted by torch when TP runs on CPU; harmless for this model
(weights are created before sharding on every rank with the same seed).

**`HYBRID_SHARD` raised `Expected device_mesh to have ndim=2 but got 1` (observed,
fixed)** — the wrapper passed the 1-D DP mesh. It now builds a 2-D
`(replicate, shard)` mesh; choose the shard-group size with
`parallel.hybrid_shard_size`.

**Port in use / `EADDRINUSE`** — use `torchrun --standalone` (picks a free port) or
`--master_port`. The test helper uses fixed ports `29600+world_size`; do not run two
pytest sessions concurrently.

## Hangs and timeouts

A collective that some rank never enters blocks the others until the timeout (30 min
by default, `init_distributed(timeout_seconds=1800)`). Typical causes in this code
base:

1. **Different step counts per DP rank** at an epoch boundary. Fixed in
   `ShardedSampler` (equal-length shards); if you replace the dataset/sampler keep
   that invariant, and keep `drop_last=True`.
2. **A rank died.** The survivors block in the next collective; `torchrun`'s agent
   notices the dead worker and SIGTERMs the rest. In the measured fault test
   (BENCHMARKS.md) detection took a few seconds because the agent, not NCCL/gloo
   timeouts, ended the job.
3. **Wrong group in a collective** (world vs `dp_group` vs `tp_group`).
4. **Checkpoint save**: every rank must call `save_checkpoint` (it contains two
   barriers). Calling it under `if rank == 0:` deadlocks.
5. **Eval**: every rank must run the same number of eval batches.

Diagnostics: `TORCH_DISTRIBUTED_DEBUG=DETAIL` (works with gloo and NCCL),
`NCCL_DEBUG=INFO`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `py-spy dump --pid <rank pid>`
on each rank to see who is in which collective.

## Gloo-specific

* Gloo does not support all dtypes/ops that NCCL does (the shipped CPU config pins
  fp32). Do not enable bf16 reductions on gloo.
* Gloo transfers go through TCP on loopback; many ranks on few cores oversubscribe
  the CPU (collectives and compute compete for the same cores). Keep
  `world_size × OMP_NUM_THREADS <= physical cores`; see BENCHMARKS.md for the policy
  used and its effect.
* Hide a GPU on a mixed box: `CUDA_VISIBLE_DEVICES=""`, otherwise FSDP/`DeviceMesh`
  may bind to the GPU.

## NCCL-specific (not exercised here)

`NCCL_SOCKET_IFNAME`, `NCCL_IB_*` for multi-node; TP groups should be intra-node
(contiguous ranks) — verify with the layout table printed at start-up. On a single
GPU `LOCAL_RANK >= device_count` is rejected up front.

## Memory

**CPU OOM-kill with several ranks (observed risk)** — each rank is a full Python +
torch process (~250-300 MB RSS before any model) and FSDP builds the *full* model on
every rank before sharding, so peak RSS at start-up is `P·4 B` per rank regardless
of `dp`. Estimate `ranks × (300 MB + 4P + activations)` before launching; this is why
the 125M config was not run in this repository's benchmarks.

**CUDA OOM** — first lever order: `activation_checkpointing`, smaller
`micro_batch_size` with more `grad_accum_steps`, larger `tp_size`, shorter `seq_len`.
Remember the logits tensor (`B·S·V`) is replicated on every TP rank (MODEL.md) and
that under FSDP `no_sync` accumulates unsharded gradients (TRAINING_LOOP.md).

## Numerics

**`GradNotFiniteError`** — raised on all ranks together before the optimizer step.
Lower LR / lengthen warmup, keep clipping on, prefer bf16 to fp16.

**Loss differs between layouts** — it should not, beyond float reassociation
(~1e-4 relative in the equivalence tests). If it does, run
`pytest tests/integration/test_parallel_equivalence.py` first; an unequal init across
DP ranks, a wrong-group reduction or a data-sharding bug are the usual suspects.

**Grad norm differs between FSDP-only and FSDP×TP** — was a real bug (see
AUDIT_FINDINGS.md); guarded by the equivalence test.

## Checkpoints and resume

**`ValueError: sharded checkpoint ... was saved with (world, tp, dp)=... but the
current topology is ...`** — shards are per-rank. Resume on the original topology.
There is currently no resharding path; `full=True` exports exist at the API level but
are not wired into the trainer.

**`CheckpointValidationError` with `missing _SUCCESS marker`** — the save did not
complete (crash mid-save). Resume from an earlier step: pass the run directory as
`--resume-from` and the newest valid `step_*` is selected.

**`UnpicklingError: Weights only load failed ... numpy._core.multiarray._reconstruct`
(observed, fixed)** — torch ≥ 2.6 defaults `torch.load(weights_only=True)`, which
rejects the pickled NumPy RNG state. `load_checkpoint` now passes
`weights_only=False` for the RNG file only (trusted, self-written).

**Resume is slow for long runs** — data position is restored by replaying batches
(`_fast_forward_data`), O(steps).

## Profiling

`communication_fraction` is 0 on CPU unless you are on this repository's fixed
version (it used a profiler attribute that no longer exists on torch 2.6, the version used here; see
OBSERVABILITY.md).

## FAQ

**Why is `125m.yaml` 152M parameters?** SwiGLU has three matrices and the vocabulary
is 50,304 (MODEL.md).

**Why does a one-process run skip FSDP?** FSDP over one rank shards nothing; pure TP
and single-process runs use the bare module (ARCHITECTURE.md §4.10).

**Can I use the hand-written `ColumnParallelLinear`/`RowParallelLinear`?** They are a
reference implementation verified against dense layers (forward and backward, 2 ranks)
but the model uses the DTensor plan.

**Is the mixed-precision path tested?** The policy objects and dtype aliases are;
the GPU-only test that checks fp32 gradient reduction under FSDP is a skipped stub.
Bf16 training was not run for this documentation (CPU only).
