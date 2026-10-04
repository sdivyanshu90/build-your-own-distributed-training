# Observability

Files: `src/observability/{logging,metrics,profiler}.py`.

## Structured logging (`RankLogger`)

One JSON object per line on stdout:
`{"ts", "level", "rank", "run_id", "event", ...fields}`. `info` and `metric` are
**rank-0 gated**; `warning` and `error` are emitted by every rank (so a failure on
rank 3 is visible). Values that are not JSON-native are `str()`-ed; dataclasses are
expanded. Events emitted by the trainer: `train_start`, `step` (METRIC),
`eval` (METRIC), `checkpoint`, `checkpoint_on_interrupt`, `resumed`, `interrupted`
(WARNING), `train_done`.

The `step` metric record carries: `loss`, `ppl` (`exp(min(loss, 20))`), `grad_norm`
(pre-clip), `lr`, `tok_per_s`, `mfu`, `step_time_s`, `data_wait_s`, `peak_mem_mb`.
Parse it with any JSON-lines tool, e.g. `grep '"event": "step"' log | jq .loss`.

The mesh layout table (`format_mesh_layout`) is printed once by rank 0 at start-up,
mapping `(dp_rank, tp_rank) -> global rank`.

## Metrics (`metrics.py`)

* `compute_throughput(tokens, seconds)`; global tokens = local tokens × `dp_size`
  (TP ranks process the same tokens).
* `estimate_flops_per_token(ModelConfig)` = `6·N + 12·L·d·max_seq_len`.
* `compute_mfu(tokens/s, flops/token, num_gpus, gpu_type)` = achieved FLOP/s ÷
  (`PEAK_TFLOPS_BF16[gpu_type]·1e12·num_gpus`); keys: `a100`, `a100-80gb`, `h100`,
  `h100-sxm`, `v100`, `cpu` (peak 1 TFLOP/s placeholder). Unknown key returns
  `None`. Peak numbers are dense BF16 tensor-core peaks; `v100` has no BF16 and is
  listed with its FP16 number.
* `peak_memory_bytes`: `torch.cuda.max_memory_allocated`, **0 on CPU**. The CPU
  benchmarks in this repo therefore measure RSS and resident tensor bytes
  externally (`benchmarks/_common.py`).
* `aggregate_loss`: SUM-all-reduce over the DP group then divide by its size.

## Profiler (`profiler.py`)

`build_profiler(output_dir, warmup_steps, profile_steps)` returns a
`torch.profiler.profile` with schedule `wait = warmup-1, warmup = 1, active =
profile_steps, repeat = 1`, CPU (+CUDA when available) activities, shapes and memory
recording, and a TensorBoard trace handler. The trainer enables it on **rank 0 only**
when `profile_steps > 0` and writes to `traces/{run_id}/`.

`communication_fraction(prof)` classifies events whose name contains one of
`nccl, all_reduce, allreduce, all_gather, allgather, reduce_scatter,
reducescatter, broadcast, c10d` as communication and divides their self time by the
total self time. Notes:

* With CUDA it uses device self time; **on CPU/gloo it uses host self time**, i.e.
  the time blocked inside collective ops (the dataclass fields are still named
  `*_cuda_us`).
* **Bug fixed during the audit:** the code read `self_cuda_time_total` via
  `getattr(..., 0.0)`. That attribute no longer exists on torch 2.6 (it is
  `self_device_time_total`), so the fraction silently read 0 for every run on a
  recent torch, GPU included. `_time_key` now supports both names;
  `tests/unit/test_profiler.py` guards it.
* `text_summary` sorts by device time when CUDA is available, otherwise by
  `self_cpu_time_total`.

## What to look at when a run is slow

1. `data_wait_s` vs `step_time_s` — data stall.
2. `tok_per_s` per rank across ranks (compare logs of ranks if you enable per-rank
   metrics) — stragglers.
3. A Chrome/Perfetto trace of the profiled window: FSDP all-gathers should overlap
   compute with `forward_prefetch`/`BACKWARD_PRE`.
4. `grad_norm` spikes and `GradNotFiniteError` — see TROUBLESHOOTING.md.
