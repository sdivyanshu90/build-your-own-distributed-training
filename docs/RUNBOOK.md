# Operations Runbook

Practical commands for launching, resuming, diagnosing, profiling, and tuning a
run. Assumes the repo is the working directory and dependencies from
`requirements.txt` are installed.

---

## 1. Launching

### Single-node, multi-GPU

`torchrun` spawns one process per GPU; `--tp-size` sets the tensor-parallel degree
and `dp_size` is inferred as `world_size / tp_size`.

```bash
# 8 GPUs, TP=2  ->  DP=4
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/125m.yaml --tp-size 2 --run-id gpt125m_run1
```

```bash
# 8 GPUs, pure FSDP (TP=1, DP=8)
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/125m.yaml --tp-size 1
```

### Multi-node

Place TP **inside** a node (NVLink) and DP **across** nodes. Run the same command
on every node with a shared rendezvous endpoint:

```bash
# 2 nodes x 8 GPUs, TP=8 intra-node, DP=2 across nodes
torchrun \
    --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_id=llama7b --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py --config config/7b.yaml --tp-size 8 --run-id llama7b_run1
```

For multi-node, prefer `sharding_strategy: HYBRID_SHARD` (already set in
`config/7b.yaml`): shard within a node, replicate across nodes — this keeps the
expensive inter-node traffic to a periodic all-reduce instead of every-layer
reduce-scatter.

### Local CPU functional run (no GPU)

```bash
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=4 train.py \
    --config config/test_tiny.yaml --tp-size 2 --backend gloo --max-steps 50
```

> On a box with a single GPU, the CPU/Gloo path also needs `CUDA_VISIBLE_DEVICES=""`
> so FSDP does not bind to the lone GPU. Pure-FSDP and pure-TP run on CPU; full
> 2D FSDP+TP requires real multi-GPU NCCL on torch 2.3 (see ARCHITECTURE §6).

---

## 2. Resuming from a checkpoint

Checkpoints are written to `{checkpoint_dir}/{run_id}/step_{N}/`. Resume with the
**same** topology used to save (sharded checkpoints are per-rank):

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/125m.yaml --tp-size 2 --run-id gpt125m_run1 \
    --resume-from checkpoints/gpt125m_run1/step_10000
```

* **Automatic latest-valid recovery (fault tolerance):** point `--resume-from` at
  the run directory and the loop will pick the newest *valid* checkpoint, skipping
  any half-written one. Programmatically:
  `find_latest_valid_checkpoint("checkpoints/gpt125m_run1")`.
* **Resume on a different topology / for inference:** save a `FULL_STATE_DICT`
  export (`save_checkpoint(..., full=True)`) and load with `full=True`. The sharded
  fast-path requires the original `world_size`.
* **What is restored:** model params, optimizer moments, LR scheduler, global step,
  per-rank RNG, and the data position (the loop fast-forwards the window stream so
  step `N` sees the same data as a continuous run).

Verify a checkpoint before trusting it:

```python
from src.checkpointing.recovery import validate_checkpoint
print(validate_checkpoint("checkpoints/gpt125m_run1/step_10000", deep=True))
```

---

## 3. Diagnosing common failures

### NCCL timeout / hang
**Symptoms:** the job stops making progress; eventually a `Watchdog caught
collective operation timeout` or it blocks forever.
**Most common cause:** ranks disagree on *which collective to run* — a wrong
process group, a per-rank-divergent control-flow branch, or a mismatched
`world_size`. Check:
* All ranks launched with identical config and the same `WORLD_SIZE`.
* Every `all_reduce`/`all_gather`/`reduce_scatter` names the correct group
  (`ctx.dp_group` vs `ctx.tp_group` vs world) — this is the #1 2D-parallel bug.
* The data loader hands every DP rank the **same number of steps** (`drop_last=True`
  guarantees this); an unequal step count desyncs collectives at epoch end.
* Set `NCCL_DEBUG=INFO` and `TORCH_NCCL_BLOCKING_WAIT=1` to surface the stuck op
  and its participants; the default collective timeout is 30 min (`init_distributed`).

### OOM (CUDA out of memory)
In order of leverage:
1. Turn on `activation_checkpointing: true` (biggest activation-memory win).
2. Reduce `micro_batch_size` and raise `grad_accum_steps` to keep the global batch.
3. Increase `tp_size` (shards each layer's params **and** its activations).
4. Lower `seq_len` if the task allows.
5. As a last resort, `cpu_offload: true` (heavy throughput cost).
Read the per-step `peak_mem_mb` in the logs to see where you stand.

### Rank desynchronization
**Symptom:** DP ranks report different losses for what should be identical state, or
grad norms diverge. Check seeding (`base_seed + dp_rank`, constant within a TP
group), that dropout is 0 for pretraining, and that no rank took a different code
path (e.g. a config that differs per rank — the trainer's startup all-reduce check
catches a batch-config skew and aborts loudly).

### Non-finite loss / gradient
The step runs a **collective finite check** before the optimizer step: if any rank
has a NaN/Inf gradient, *all* ranks raise `GradNotFiniteError` together (no hang).
Usual causes: LR too high (lower it or extend warmup), missing grad clipping
(`max_grad_norm > 0`), fp16 overflow (use bf16 — the default), or a corrupt batch.

### Checkpoint corruption
`validate_checkpoint(path, deep=True)` reports the exact problem: missing
`_SUCCESS` marker (incomplete save), a missing/zero-byte/unloadable shard
(truncation), or a model-config mismatch (`d_model` etc. differ from the current
model). `find_latest_valid_checkpoint` skips corrupt checkpoints automatically.

---

## 4. Reading the profiler output

The trainer profiles a small window (`warmup_profile_steps` then `profile_steps`)
and writes a trace under `traces/{run_id}/`.

* Open the Chrome trace JSON in `chrome://tracing` or Perfetto. Look at whether the
  FSDP `all_gather` of block *i+1* overlaps the compute of block *i* — gaps mean
  prefetch is not hiding communication.
* `communication_fraction(prof)` (used by `bench_communication.py`) prints the
  share of CUDA time in collectives. A high fraction (> ~0.3) means the run is
  communication-bound: check `forward_prefetch`/`BACKWARD_PRE`, whether the TP
  group accidentally spans a slow inter-node link, and whether `reduce_dtype=fp32`
  doubling DP traffic is worth it at your `dp_size`.
* The text summary (`text_summary(prof)`) ranks ops by self CUDA time — the top
  entries tell you whether you are matmul-bound (good) or collective/overhead-bound.

---

## 5. Tuning the key knobs

| Knob | Where | Effect & guidance |
|---|---|---|
| `tp_size` | `--tp-size` / `parallel.tp_size` | Shards each layer across GPUs. Keep ≤ GPUs-per-node (NVLink). Raises per-step `all_reduce` traffic; use the smallest TP that makes the model fit. |
| `dp_size` | inferred / `parallel.dp_size` | FSDP degree. More DP = larger global batch and more param-shard savings, but fp32 grad reduction traffic grows. |
| `grad_accum_steps` | `grad_accum_steps` | Increases effective batch without extra memory; the first `K-1` micro-steps skip the DP reduce-scatter via `no_sync`. |
| `sharding_strategy` | `parallel.sharding_strategy` | `FULL_SHARD` (max memory saving) vs `HYBRID_SHARD` (multi-node: shard intra-node, replicate across) vs `SHARD_GRAD_OP` (keep params, shard grad+opt — more memory, less comm). |
| `activation_checkpointing` | `parallel.activation_checkpointing` | ~30% extra compute for a large activation-memory cut. Turn on first under memory pressure. |
| `sequence_parallel` | `parallel.sequence_parallel` | Shards norm activations by `tp_size`; extra resharding per block. Worth it at long `seq_len`. |
| `backward_prefetch` / `forward_prefetch` | `parallel.*` | Overlap next-layer all-gather with current compute. Leave on; turning off can ~halve throughput. |
| `limit_all_gathers` | `parallel.limit_all_gathers` | Caps concurrent all-gathers to bound peak memory in deep models. Leave on. |
| `lr` / `warmup_steps` / `min_lr` | `optimizer`/`scheduler` | Peak LR scales roughly with global batch; warmup avoids early-Adam instability; cosine anneals to `min_lr` at `max_steps`. |
| `param/reduce/buffer dtype` | `parallel.*` | Keep `reduce_dtype=float32` at scale (unbiased grad sum). bf16 needs no loss scaling. |

### A tuning recipe for a new cluster
1. Start `tp_size=1` (pure FSDP) at the largest `micro_batch_size` that fits with
   `activation_checkpointing` on; record tokens/sec and MFU.
2. If the model does not fit even at `micro_batch_size=1`, raise `tp_size` to the
   GPUs-per-node count.
3. Sweep `micro_batch_size` up until MFU plateaus; use `grad_accum_steps` to reach
   the target global batch.
4. Run `bench_communication.py` with prefetch on/off to confirm comm is hidden.
5. Gate CI with `bench_throughput.py --assert-min-mfu 0.40` so regressions fail.
