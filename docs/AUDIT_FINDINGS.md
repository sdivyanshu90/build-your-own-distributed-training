# Audit findings

Environment for every finding: torch 2.6.0 (CPU build), Python 3.10, gloo, WSL2, run on
2026-10-04/05. Line numbers refer to the code **before** the fix (`main` at the start of
the audit). "Test" is the regression test that fails before / passes after, where one
could be written. Severity: **H** wrong results or crash in a documented feature, **M**
wrong behaviour under specific configurations, **L** robustness / documentation.

The complete suite was not re-run to completion after the fixes (see TESTING.md for the exact status).

Baseline on the unmodified repository with torch 2.6.0 (inside the declared
`torch>=2.3,<2.7` range): `5 failed, 73 passed, 1 skipped` (79 tests). The five
failures were all the same bug (#1).

| # | Sev | Where (before) | Finding | Fix | Test |
|---|---|---|---|---|---|
| 1 | H | `src/checkpointing/checkpoint.py:222` | `torch.load(rng.pt)` fails under torch 2.6's `weights_only=True` default (NumPy RNG pickle): **every resume crashed** on torch 2.6 (5 test failures). | `weights_only=False` for the self-written RNG file only | existing resume/recovery/serialization tests |
| 2 | H | `src/training/grad_utils.py:95-97` | Under FSDP×TP the clip delegated to `FSDP.clip_grad_norm_`, which reduces over the FSDP group only. Measured grad norm 1.088 vs 1.120 reference at tp=2,dp=2; TP ranks could clip differently. | `_fsdp_tp_clip` + `is_tp_sharded_param` | `test_2d_fsdp_x_tp_matches_reference` |
| 3 | H | `src/parallelism/fsdp_utils.py:160` | `HYBRID_SHARD` (used by `config/7b.yaml`) raised `ValueError: Expected device_mesh to have ndim=2 but got 1` for every `dp_size > 1`. | 2-D `(replicate, shard)` sub-mesh, `parallel.hybrid_shard_size` | `test_hybrid_shard_2x2_matches_reference` |
| 4 | H | `src/parallelism/tensor_parallel.py:489-497` | `sequence_parallel: true` (set in `config/7b.yaml`) crashed (`PrepareModuleInput` got one layout for the 3-argument attention) and the residual stream was never scattered/gathered. | 3-entry layouts, embedding-output scatter, pre-final-norm gather | `test_sequence_parallel_tp2_matches_reference` |
| 5 | H | `src/training/loop.py:154` | 2D (FSDP×TP) with `grad_accum_steps > 1` crashed in FSDP1 `_writeback_orig_params` ("Attempted to access the data pointer on an invalid python storage") from the second micro-batch: `no_sync` leaves unsharded DTensor grads. Reproduced in the scaling benchmark and in a test. | do not use `no_sync` when `tp_size > 1` (reduce every micro-step; same math, `K×` reduce-scatter traffic) | `test_2d_with_grad_accumulation_matches_reference` (fails before the fix; post-fix run **not completed in time, unverified**) |
| 6 | M | `src/training/trainer.py:99-104` | Per-`dp_rank` seed applied before model construction → DP replicas (`NO_SHARD`/hybrid) initialised with different weights; init depends on `dp_size`. | seed with `seed` for construction, per-rank seed after | `test_no_shard_replicas_start_identical`, equivalence tests |
| 7 | M | `src/data/dataloader.py:96-99` | Strided sharding gave unequal shard lengths (±1) → possibly unequal batch counts per DP rank at epoch ends → collective desync/hang. | truncate to a multiple of `dp` | `test_sizes_equal_across_ranks` (replaces the old "differ by at most one" test, which encoded the bug) |
| 8 | M | `src/checkpointing/checkpoint.py:198-216` | Loading a sharded checkpoint saved on a different `(world,tp,dp)` silently used whatever `rank_N` existed (wrong shards / opaque shape errors). | explicit topology check against `meta.json` | `test_load_rejects_topology_mismatch` |
| 9 | M | `src/checkpointing/checkpoint.py:120-158` | Re-saving an existing `step_N` left the old `_SUCCESS` valid while files were being overwritten. | remove marker (rank 0) + barrier before writing | `test_resave_clears_stale_success_marker` |
| 10 | M | `src/observability/profiler.py:107,129` | `self_cuda_time_total` does not exist on torch 2.6; `communication_fraction` silently returned 0 (comm/compute breakdown wrong on GPU as well). | support `self_device_time_total`; host time on CPU | `tests/unit/test_profiler.py` |
| 11 | M | `src/training/trainer.py:144`, docs/RUNBOOK §2 | RUNBOOK promised `--resume-from <run dir>` picks the newest valid checkpoint; the code validated and loaded that exact path. | `_resolve_resume_path` | covered by the fault benchmark (`bench_fault.py` resumes from a run dir) |
| 12 | M | `src/config.py:259-332` | No validation of `n_heads/n_kv_heads/ffn % tp`, GQA grouping, or `scheduler.max_steps` vs `max_steps` → opaque DTensor errors / schedule not ending at the last step. | `TrainingConfig.validate` | `tests/unit/test_config_validation.py` |
| 13 | L | `src/checkpointing/recovery.py:113-130` | Validation did not check `optim.pt`, `rng.pt`, `scheduler.pt` presence. | added | `test_validate_flags_missing_optimizer_shard` |
| 14 | L | `src/config.py:151`, `tensor_parallel.py:400` | `parallel.cpu_offload` and `apply_tensor_parallelism(loss_parallel=...)` were accepted and ignored (RUNBOOK recommends `cpu_offload`). | `cpu_offload` wired to FSDP `CPUOffload`; `loss_parallel=True` raises | – (offload is GPU-oriented; untested) |
| 15 | L | `src/training/grad_utils.py:140` | Pure-TP norm accumulator created on CPU regardless of the grads' device (would break the NCCL all-reduce path). | allocate on the params' device | pure-TP equivalence test (CPU only) |
| 16 | L | repo | `mypy src/` reported 7 errors although README claimed "mypy clean"; no CI workflow existed. | type fixes; `.github/workflows/ci.yml` | CI |
| 17 | L | `README.md`, `docs/ARCHITECTURE.md §6` | Docs claimed 2D FSDP×TP and FSDP resume were GPU-only (torch-2.3 bugs). On torch 2.6 CPU both work (except 2D accumulation, #5, now fixed). Also: RUNBOOK topology claims, `125m.yaml` is ~152M params, `7b.yaml` ships features that did not run (#3, #4). | docs rewritten | – |

## Observations not changed (documented limitations)

* No checkpoint resharding; FSDP optimizer state on CPU is a per-rank raw state dict
  (`__fsdp_per_rank__`); the CUDA re-keyed path was not run.
* Tensor files are not `fsync`ed; only `meta.json`/`_SUCCESS` are.
* Only `KeyboardInterrupt` writes an emergency checkpoint (not `SIGTERM`); no
  checkpoint at `max_steps` unless it is a multiple of `save_interval`.
* Resume replays data batches (O(step)); `LAMB` trust ratios are per local shard.
* `data.tokenizer_name` is unused; no tokenisation script produces `uint16` corpora.
* `VocabParallelEmbedding` and the hand-written TP linears are not on the training
  path. Logits are replicated across TP ranks.
* FSDP1 (`FullyShardedDataParallel`) is used; FSDP2 (`fully_shard`) is the supported
  composition with TP in newer PyTorch and would remove findings #5's workaround.
* The GPU-only mixed-precision test is a skipped stub.
