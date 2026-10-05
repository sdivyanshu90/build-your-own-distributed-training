# Testing guide

## Running

```bash
pip install -r requirements.txt          # or CPU torch + pytest ruff mypy
CUDA_VISIBLE_DEVICES="" pytest -q        # whole suite, CPU + gloo
ruff check src/ train.py tests/ benchmarks/
mypy src/
```

Multi-rank tests use `tests/_dist_utils.py::run_distributed(fn, world_size, *args)`:
`torch.multiprocessing.spawn` of gloo processes on `127.0.0.1:(29600 + world_size)`,
`CUDA_VISIBLE_DEVICES=""`, and child exceptions re-raised in the parent so a failure
on rank 1 fails the test rather than hanging it. Because of the fixed ports **do not
run two pytest sessions at once**. Tests are CPU-bound and slow on a loaded machine
(single-digit seconds to minutes each); FSDP/TP tests import torch in every spawned
process.

## Status of the suite (98 tests collected after the audit)

* **Baseline, unmodified repo, torch 2.6.0 CPU:** 79 tests, `5 failed, 73 passed, 1
  skipped` in 670 s (all 5 failures: the `weights_only` bug, finding #1; the skip is the
  GPU stub below).
* **After the fixes the complete suite was NOT run to completion**: the machine was
  shared and heavily loaded, and the one full attempt hit its 18-minute timeout after 24
  tests with no failure. Evidence for the post-fix state is per file, run separately:
  checkpoint serialization + fault + resume + dataloader files (18 passed); config
  validation (5 passed) and profiler (1 passed); equivalence file: FSDP dp=2, TP=2,
  FSDP x TP, NO_SHARD replicas and the name rule (passed before the last two
  fixes), sequence-parallel, HYBRID_SHARD and FSDP-with-accumulation (passed);
  **`test_2d_with_grad_accumulation_matches_reference` failed before the 2D
  `no_sync` fix and its post-fix run did not complete in time, so that fix is
  unverified by a passing test** (the 2D convergence benchmark that exercises the same
  path did not complete either).
* Files not re-run after the later changes (seeding order, sampler, `no_sync` change):
  `test_convergence_tiny`, `test_2d_parallel_forward`, `test_fsdp_wrapping`,
  `test_grad_accumulation` and the remaining unit files; they passed in the baseline.
  Re-run the suite before relying on it: `CUDA_VISIBLE_DEVICES="" pytest -q` (allow a
  quiet machine ~10 minutes).

## Layout and what each file proves

| File | Tests | Proves |
|---|---|---|
| `unit/test_column_row_linear.py` | 20 | `gradcheck` of the hand-written Column/Row linears incl. divisibility errors (1 rank) |
| `unit/test_tensor_parallel.py` | 5 | shapes, gather and chain equal `nn.Linear` (1 rank) |
| `unit/test_process_groups.py` | 7 | `resolve_dp_size` arithmetic/errors, 2×2, pure-FSDP, pure-TP, invalid meshes |
| `unit/test_fsdp_wrapping.py` | 3 | parameter-count invariant across shards, sharding+forward, MP policy dtypes |
| `unit/test_grad_accumulation.py` | 3 | accumulated grad = full-batch grad; `no_sync` does not change it; loss/K |
| `unit/test_grad_clipping.py` | 6 | pre-clip norm, clipping, no-op, finite check (NaN/Inf) |
| `unit/test_scheduler.py` | 4 | warm-up monotone, cosine formula, `min_lr` at end, resume reproduces LR |
| `unit/test_metrics.py` | 5 | throughput, FLOPs/token, MFU by hand, unknown GPU, loss aggregate |
| `unit/test_checkpoint_serialization.py` | 8 | round trip, optimizer state, truncation, latest-valid skips corrupt, config mismatch, **stale marker, topology mismatch, missing optimizer shard** |
| `unit/test_config_validation.py` | 5 | **TP divisibility, GQA group, scheduler horizon warning** |
| `unit/test_profiler.py` | 1 | **collectives are classified as communication on CPU** |
| `integration/test_2d_parallel_forward.py` | 3 | hand-rolled TP chain and TP model logits equal the dense reference (2 ranks); FSDP DP loss consistency |
| `integration/test_parallel_equivalence.py` | 10 | **loss + global grad-norm equal a single-process reference for FSDP, TP, FSDP×TP (4 ranks), sequence-parallel TP, HYBRID_SHARD 2×2, accumulation (FSDP K=3, 2D K=2); NO_SHARD replicas start identical; hand-rolled TP backward on 2 ranks; TP-name rule** |
| `integration/test_convergence_tiny.py` | 3 | single / FSDP / TP converge on the synthetic corpus (loss falls by > 1.0) |
| `integration/test_checkpoint_resume.py` | 2 | resume equals continuous run (single process; FSDP dp=2) |
| `integration/test_dataloader_sharding.py` | 4 | disjoint, no dup, equal length, reshuffle per epoch |
| `integration/test_mixed_precision.py` | 5 | autocast keeps fp32 master, policy dtypes; **1 skipped on CPU** (GPU stub, see below) |
| `fault/test_corrupt_checkpoint.py` | 3 | missing shard, truncated tensor, config mismatch detected |
| `fault/test_rank_failure.py` | 1 | crash mid-save leaves an unmarked checkpoint; recovery picks the previous one and resumes |
| `performance/bench_*.py` | – | not tests (no `test_` prefix); original GPU-oriented benchmark scripts |

Bold entries were added or rewritten during the audit.

## The skipped test

`integration/test_mixed_precision.py::test_fsdp_reduces_in_fp32_runtime` is marked
`skipif(not cuda)` and its body is `pytest.skip(...)` — a placeholder, so it skips
even on a GPU box. It documents an intent (gradient reduction in fp32 under FSDP
mixed precision) that no test checks.

## Notes on test design

* Tolerances: forward equality `atol 1e-4–1e-5`; loss `rel 1e-4`, grad norm `rel 1e-3`
  in the equivalence tests (float reassociation across shards). Resume comparisons
  use `1e-4`/`1e-3` in the tests, although the benchmark measured exactly 0.0
  difference on CPU (BENCHMARKS.md).
* The equivalence tests run **one** optimizer-step's forward/backward per layout; they
  verify gradients, not multi-step optimizer equivalence. Multi-step behaviour is
  covered by the convergence and resume tests.
* `shuffle=False` + equal shard lengths are what let the single-process reference see
  exactly the union of the DP ranks' micro-batches.
* A bug can hide in a test that encodes it: the old
  `test_sizes_differ_by_at_most_one` asserted the unequal-shard behaviour that caused
  the hang risk (finding #7).

## CI

`.github/workflows/ci.yml` (added in the audit; **there was none before**): Python 3.10
and 3.12, CPU torch `>=2.3,<2.7` from the PyTorch CPU index (resolves to the newest
2.6.x), `ruff`, `mypy src/`, then `pytest -q -x`. The workflow has not been executed
on GitHub from this environment; its commands were run locally in the venv.
Benchmarks are intentionally not part of CI.
