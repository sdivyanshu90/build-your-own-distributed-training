# Design decisions (ADR-style)

The original ten decisions (FSDP over DDP, per-block wrapping, `use_orig_params`,
fp32 reduce, DTensor TP + reference linears, replicated embedding/head, `-1` head
reshape, sharded vs full checkpoints, atomic writes + marker, skip FSDP at
`dp == 1`) are recorded with alternatives and rationale in
[ARCHITECTURE.md §4](ARCHITECTURE.md). This file adds the decisions made or changed
by the audit, plus the ones the benchmark design depends on. Format: **context →
decision → consequences**.

## ADR-11 Same model-init seed on every rank, per-rank seed afterwards
*Context.* `seed_everything(seed, dp_rank)` ran before `build_model`, so DP ranks
created different weights. With FULL_SHARD, FSDP does not broadcast module state (`sync_module_states` is
not set), so each rank's shard comes from that rank's own init: a valid but
dp-size-dependent initialisation (inferred from FSDP semantics, not separately
measured); with `NO_SHARD`
or `HYBRID_SHARD` replicas silently diverged from step 0 and the sharded runs were not
reproducible across DP degrees. *Decision.* Seed with `seed` for construction, then
re-seed with `seed + dp_rank`. *Consequences.* Initialisation is independent of
layout (the equivalence tests rely on it); the cost is none. `NO_SHARD` replica
identity is covered by `test_no_shard_replicas_start_identical`.

## ADR-12 Equal-length data shards (truncate the remainder)
*Context.* Strided partition of `n` indices gives shard lengths `ceil/floor(n/dp)`,
which with `drop_last` batches can differ in batch count. *Alternatives.* Pad by
repeating indices (what `DistributedSampler` does by default: no sample lost, a few
duplicated); keep the remainder and rely on loader length checks. *Decision.* Truncate
to a multiple of `dp`: duplicates are worse than dropping `< dp` samples per epoch
for training data, and the dropped set changes every epoch under shuffling.
*Consequences.* Unshuffled runs always skip the same last `n % dp` samples; documented
in DATA_PIPELINE.md.

## ADR-13 Explicit 2D clip for FSDP×TP
*Context.* `FSDP.clip_grad_norm_` reduces over the FSDP group only. *Decision.*
`_fsdp_tp_clip` sums squared norms with weight 1 for TP-sharded and `1/tp` for
TP-replicated parameters, reduces over the FSDP (shard) group — skipped for
`NO_SHARD` — and over the TP group. TP membership is decided by parameter name
(`is_tp_sharded_param`), which couples it to the plan in `apply_tensor_parallelism`.
*Alternatives.* Walk DTensor placements (not pursued: FSDP flattens parameters, and the name
rule was simpler to verify); clip per-group (wrong). *Consequences.* If the TP plan gains a new
sharded module, update `_TP_SHARDED_MODULES`; the equivalence test fails loudly if
the two drift apart.

## ADR-14 HYBRID_SHARD builds its own 2-D mesh
*Context.* FSDP wants a 2-D mesh for HYBRID_SHARD; the trainer's DP axis is 1-D.
*Decision.* Reshape the `(dp, tp)` rank grid to `(replicate, shard, tp)` and pass the
`(replicate, shard)` sub-mesh; shard-group size from `parallel.hybrid_shard_size`
(default: per-node DP ranks). *Consequences.* One extra `DeviceMesh` (a few extra
process groups) is created; DP ranks adjacent in the original mesh share a shard
group.

## ADR-15 Checkpoint compatibility is enforced, not assumed
*Decision.* `load_checkpoint` compares `(world, tp, dp)` from `meta.json` with the live
topology and raises; `validate_checkpoint` also requires `optim.pt`, `rng.pt` and
`scheduler.pt`; `save_checkpoint` deletes a stale `_SUCCESS` before overwriting.
*Rationale.* "Silently load whatever `rank_N` exists" turns a topology change into
wrong weights or an opaque shape error. *Consequences.* Resharding remains
unsupported (a documented limitation rather than a trap).

## ADR-16 `weights_only=False` for the RNG file only
torch ≥ 2.6 defaults to `weights_only=True`, which cannot unpickle NumPy RNG state.
Model/optimizer/scheduler files load under the safe default; the RNG file is
self-written inside the run's checkpoint directory, and is loaded with
`weights_only=False`. Do not load checkpoints from untrusted sources.

## ADR-17 Benchmarks are CPU/gloo with a per-rank thread policy
*Context.* The only hardware is an 8-core laptop CPU with ~6 GB RAM shared with other
jobs. *Decision.* Measure with `gloo`, report two thread policies (1 thread/rank to
isolate parallel overhead, and a ~6-thread total budget), never exceed 4 ranks,
bound every run with `timeout` and a free-RAM guard, and label every table as CPU.
*Rejected.* Extrapolating GPU numbers (unmeasurable here); using the 2 GB MX330
(single device, not useful for multi-GPU behaviour). *Consequences.* Absolute
tokens/s are irrelevant for GPU planning; what transfers is the *structure*:
collective counts, memory partitioning, equivalence, checkpoint/recovery behaviour.

## ADR-18 Keep the hand-written TP linears as a tested reference
They are not on the training path (DTensor is), but they are the only place where
every forward/backward collective is explicit. Their backward collectives are now
tested on 2 ranks (`test_handrolled_tp_backward_matches_reference`), which the
single-rank unit tests could not do.

## ADR-19 CI is CPU-only and fast
`.github/workflows/ci.yml` installs CPU torch, runs `ruff`, `mypy` and the whole
pytest suite on CPU/gloo for Python 3.10 and 3.12. Benchmarks are deliberately not in
CI (timing on shared runners is noise).
