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

## Latest complete verification (2026-10-14)

Source `3f96b82778688f27f87f710592b1912103939921`, Python 3.12.15, torch 2.6.0+cu124. The correctness suite runs with CUDA hidden, Gloo, and `OMP_NUM_THREADS=MKL_NUM_THREADS=1` on six WSL logical CPUs.

| Check | Result | Evidence |
| --- | --- | --- |
| Complete pytest suite | 97 passed, 1 skipped; 98 collected, 83.99 s | [Log](../benchmarks/results/20261014/tests.log), [JUnit](../benchmarks/results/20261014/tests.xml) |
| Ruff | All source, train.py, tests, and benchmark scripts pass | [Log](../benchmarks/results/20261014/ruff.log) |
| Mypy | No issues in 33 source files | [Log](../benchmarks/results/20261014/mypy.log) |
| 2D accumulation regression | Passes in the complete suite; full throughput and 150-step convergence cases complete | [Report](BENCHMARKS.md) |
| GPU functional measurements | Single-GPU fp32/bf16 train steps, memory, reduced 151.9M shape, and exact resume complete | [Report](BENCHMARKS.md#single-gpu-cuda-throughput) |

The one skip is the existing GPU reduction-dtype stub, which skips even on a multi-GPU box. Single-GPU measurements do not verify multi-GPU bf16 collectives or GPU FSDP optimizer-state conversion. Missing WSL GPU kernel profiler events are reported as unavailable, not a 0% communication measurement. The separate user-executed Colab Tesla T4 supplement completes 15 cases, checks all 15 resumed losses in each of two policies, and captures 6,147 GPU kernel events. Its BF16 autocast uses software support without native BF16 acceleration. No complete cloud pytest run is claimed. [Colab validation](../benchmarks/results/20261014-colab-t4/validation.json).

The earlier partial post-fix verification is historical. Its missing CPU matrix points and full-suite status are superseded by this run; original timings remain in [the October 4/5 archive](archive/BENCHMARKS-20261004.md).

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
2.6.x), `ruff`, `mypy src/`, then `pytest -q -x`. The October 14 verification runs these checks locally; no new GitHub CI result is claimed for this benchmark.
Benchmarks are intentionally not part of CI.
