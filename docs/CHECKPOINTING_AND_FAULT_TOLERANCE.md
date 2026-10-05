# Checkpointing and fault tolerance

Files: `src/checkpointing/checkpoint.py` (save/load), `recovery.py` (validate/find),
trainer integration in `src/training/trainer.py`.

## On-disk layout

```
{checkpoint_dir}/{run_id}/step_{N}/
  meta.json            step, world_size, tp_size, dp_size, format, full TrainingConfig
  scheduler.pt         LambdaLR state (rank 0 writes)
  rank_{r}/model.pt    this rank's shard (FSDP sharded state dict, or plain state dict)
  rank_{r}/optim.pt    this rank's optimizer state (see caveat below)
  rank_{r}/rng.pt      Python / NumPy / torch (/ CUDA) RNG states
  _SUCCESS             contains the step; written last
```

`full=True` instead writes `model_full.pt` / `optim_full.pt` from rank 0 (API only;
the trainer does not use it).

## Save protocol (`save_checkpoint`)

1. Rank 0 removes any stale `_SUCCESS` in the target dir; barrier.
2. Every rank writes its shard files via `torch.save(tmp)` + `os.replace`
   (atomic rename; **no fsync of tensor files**).
3. Rank 0 writes `scheduler.pt`, then `meta.json` (`fsync` + `os.replace`).
4. Barrier (all shards are in place), rank 0 writes `_SUCCESS` (`fsync`ed), barrier.

Properties: a *process* crash at any point leaves either no marker (checkpoint
invisible to recovery) or a complete set; a stale marker cannot vouch for files being
overwritten (fixed). Not provided: power-loss durability of tensor files, directory
fsync, or a single atomic directory rename. Every rank must call it (two barriers).

## Validation and recovery (`recovery.py`)

`validate_checkpoint(path, expected_config=None, deep=False)` reports *all* problems:
not a directory; missing `_SUCCESS`; missing/unreadable `meta.json`; for sharded
format and each rank `0..world_size-1`: missing or zero-byte `model.pt`, missing
`optim.pt`/`rng.pt`; missing `scheduler.pt`; for `full` format the full model file;
with `deep=True` every model shard is actually `torch.load`ed (detects truncation);
and a comparison of the model-config keys (`vocab_size, d_model, n_layers, n_heads,
n_kv_heads, ffn_hidden_size, max_seq_len`) against the current config.
`find_latest_valid_checkpoint(run_dir)` scans `step_*` newest-first and returns the
first valid one. `Trainer` validates (`deep=False`, a size check, for speed) before
loading and `load_checkpoint` additionally refuses a different `(world, tp, dp)`.

## What is restored on resume

Model and optimizer shards, scheduler state, global step, per-rank RNG, and the data
position (by replaying `step·K` batches). Resume is **same-topology only**.

**Optimizer-state caveat.** `fsdp_utils.get_optimizer_state_dict` uses FSDP's
re-keyed optimizer state dict only when CUDA is available; otherwise it stores the
rank's raw `optimizer.state_dict()` marked `__fsdp_per_rank__`. All CPU
measurements here use the raw per-rank path; the CUDA path was not run (see
BENCHMARKS.md for the single-GPU status).

## Measured behaviour (CPU/gloo, `small` preset, 4.0M params)

| world | dp | params | steps | resumed at | save (s) | build+load (s) | ckpt size (MB, all ranks) | max abs dloss | bit-identical steps |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3997696 | 30 | 15 | 0.7468 | 2.6121 | 48.06 | 0.0 | 15/15 |

See BENCHMARKS.md for the interpretation and caveats.

## Fault injection

Unit/integration: `tests/fault/*` (corrupt/missing/truncated shards, config mismatch,
crash mid-save with recovery to the previous checkpoint).
End-to-end with real processes: `benchmarks/bench_fault.py` launches
`torchrun ... train.py` (2 ranks, test_tiny, checkpoint every 10 steps), SIGKILLs one
worker after step 25, times how long `torchrun` takes to tear the job down, relaunches
with `--resume-from <run dir>` and times the relaunch to the first post-resume
training step.

| world | save every | step at kill | committed ckpts | resumed from | steps lost | detect (s) | relaunch->first step (s) |
|---|---|---|---|---|---|---|---|
| 2 | 10 | 25 | step_10,step_20 | 20 | 5 | 5.61 | 62.28 |

Not covered: node loss during a *save* with ranks blocked in the barrier (they wait
for the collective timeout, 30 min by default, unless the launcher kills them first
as `torchrun` does), elastic re-rendezvous (`--max-restarts` would restart workers
but the trainer only resumes if `--resume-from` is given), and storage faults other
than truncation/missing files.
