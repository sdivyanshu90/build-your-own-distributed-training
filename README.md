# build-your-own-distributed-training

A **from-scratch, production-grade distributed training loop** for a LLaMA-style
causal language model, built entirely on raw PyTorch primitives — **no DeepSpeed,
Megatron, Accelerate, or apex**. It composes two kinds of sharding into a 2D
`(DP × TP)` process mesh:

* **FSDP (ZeRO-3)** shards parameters, gradients, and optimizer state across the
  data-parallel axis, all-gathering each layer's params just in time.
* **Tensor Parallelism** shards individual weight matrices (column- and
  row-parallel) across the tensor-parallel axis so one layer's compute is split
  across GPUs.

Every design decision is documented with its rationale and the alternatives
rejected. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the deep dive and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for operations.

---

## Highlights

- **2D parallelism** via a named `(dp, tp)` `DeviceMesh`; degenerate `tp=1`
  (pure FSDP) and `dp=1` (pure TP) are handled without special-casing the rest of
  the code.
- **Hand-rolled tensor-parallel linears** (`ColumnParallelLinear`,
  `RowParallelLinear`) on four explicit `autograd.Function` collectives — the
  executable specification, verified with `torch.autograd.gradcheck` — *plus* the
  production DTensor `parallelize_module` path the trainer runs.
- **Correct gradient accumulation under FSDP**: `no_sync()` defers the DP
  reduce-scatter for all but the last micro-step; the TP all-reduce intentionally
  is **not** deferred (documented asymmetry).
- **Mixed precision** with `param_dtype=bf16`, `reduce_dtype=fp32` (unbiased
  gradient sum at scale), fp32 optimizer master — no `GradScaler` needed.
- **Atomic, sharded checkpointing** with a `_SUCCESS` marker and a validator that
  detects missing shards, truncation, and config mismatch; **bit-exact resume**
  (model + optimizer + scheduler + RNG + data position).
- **Observability**: structured JSON logs (rank-0 gated), tokens/sec, MFU, peak
  memory, grad norm, data-stall time, and a scoped `torch.profiler` with a
  comm-vs-compute breakdown.
- **Strong correctness tests**: TP forward is proven numerically identical to a
  single-GPU reference; pure-FSDP and pure-TP both converge on a learnable
  synthetic corpus; resume is proven bit-exact; fault recovery is tested.
- `ruff` and `mypy src/` clean; CI workflow in `.github/workflows/ci.yml`.

---

## Quickstart

```bash
pip install -r requirements.txt

# Single-node, 8 GPUs, TP=2 (DP=4):
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/125m.yaml --tp-size 2 --run-id my_run

# Local CPU functional run (no GPU needed):
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc_per_node=4 train.py \
    --config config/test_tiny.yaml --tp-size 2 --backend gloo --max-steps 50
```

Multi-node, resume, profiling, and tuning are covered in the
[RUNBOOK](docs/RUNBOOK.md).

---

## Repository layout

```
src/
  config.py                  # typed nested-dataclass config + YAML loader
  parallelism/               # mesh, process groups, tensor parallel, FSDP utils
  model/                     # transformer, attention, mlp, embeddings (RoPE, GQA, SwiGLU)
  training/                  # trainer, train/eval step, optimizer, scheduler, grad utils
  data/                      # synthetic + packed datasets, sharded sampler, tokenizer
  checkpointing/             # atomic save/load, validation & recovery
  observability/             # metrics (MFU), profiler, structured logging
  utils/                     # dtype policy, seeding, env validation
train.py                     # torchrun entry point
tests/{unit,integration,performance,fault}/
config/{base,125m,7b,test_tiny}.yaml
docs/                       # handbook (index: docs/README.md)
benchmarks/                 # CPU/Gloo and CUDA benchmark tools, Colab notebook, raw results
.github/workflows/ci.yml
```

---

## Testing

```bash
# Full suite (CPU + Gloo; multi-rank tests use torch.multiprocessing.spawn).
CUDA_VISIBLE_DEVICES="" pytest -q

# Lint + type-check.
ruff check src/ train.py tests/
mypy src/
```

Multi-rank tests spawn Gloo processes and **propagate child exceptions** to the
parent, so a failure on rank 1 surfaces as a test failure rather than a hang.

### What runs where

Verified on **2026-10-14**, Python 3.12.15 / torch 2.6.0+cu124, Ryzen 7 8845HS under WSL2:

- Full CPU/Gloo suite: **97 passed, 1 skipped in 83.99 seconds**, including 2D accumulation, sequence-parallel and HYBRID_SHARD equivalence, convergence, checkpoint/resume, and fault tests.
- Complete CPU matrix: scaling, memory, 2-/4-rank collectives, exact resume, SIGKILL recovery, and all four convergence layouts.
- Real single-GPU CUDA training on RTX 3050 Laptop 4 GiB and a separately executed Colab Tesla T4: fp32 and bf16 autocast, activation-checkpointing memory, reduced 151.9M model shape, and bit-exact resume in both precision policies. The T4 BF16 runs use software support without native BF16 acceleration; all 15 Colab cases complete.
- Still unmeasured: multi-GPU NCCL/FSDP/TP/2D, CUDA FSDP optimizer re-keying, multi-node, 7B, and the shipped full-batch 125M workload. The skipped test remains a multi-GPU mixed-precision placeholder. GPU profiler kernel timings are unavailable in WSL; the separate Colab trace captures 6,147 kernel events.

Details and evidence: [testing](docs/TESTING.md), [full benchmark](docs/BENCHMARKS.md).

The audit found and fixed 17 issues (several that broke documented features: resume on
torch 2.6, 2D gradient clipping, HYBRID_SHARD, sequence parallelism, 2D gradient
accumulation); see the findings table.

---

## Benchmarks (CPU/Gloo and single-GPU CUDA)

Measured **2026-10-14**, with laptop results in `benchmarks/results/20261014/` and the user-executed Colab run in `benchmarks/results/20261014-colab-t4/`. Training source and shipped configs are unchanged.

| Measurement | Latest result |
| --- | --- |
| CPU small model, 1 thread/rank, median of three DP observations | 1 rank 2,060.4 tok/s; FSDP dp=2 3,551.3; dp=4 6,247.0 |
| CPU budget control, four total threads requested | Observed 1×3 threads 3,987.3 tok/s; 2×2 5,066.7; 4×1 6,206.5 |
| CPU TP/2D scaling, first complete sweep | TP=2 1,983.5 tok/s; TP=4 1,777.6; DP2×TP2 3,170.3 |
| Mid-model resident state, fp32 | 424.1 MiB unsharded; FSDP dp=2 212.1/rank, dp=4 106.0/rank; TP=2 244.1/rank; 2D 122.1/rank |
| Checkpoint/resume | CPU 1/2 ranks and single-GPU fp32/bf16: 15/15 post-resume losses bit-identical in every case |
| SIGKILL recovery, two CPU ranks | Detection 0.47 s; relaunch→first metric 3.73 s; resumed from step 20 after reaching 25 |
| Laptop CUDA small/mid throughput | Small fp32 19,716.6 tok/s; mid bf16 autocast 9,368.7 tok/s |
| Laptop CUDA activation checkpointing, mid model, sequence 1024 | Peak allocated memory 731.3→497.8 MiB; throughput 10,713.7→9,201.4 tok/s |
| Laptop 151.9M architecture, reduced seq=128 / micro-batch=1 | 782.8 tok/s; peak allocated 2,611.4 MiB (bf16 autocast, fp32 persistent state) |
| Colab Tesla T4 throughput | Small fp32 18,270.0 tok/s; mid fp32 11,172.1 tok/s; mid bf16 autocast (non-native) 8,816.1 tok/s |
| Colab T4 activation checkpointing, mid model, sequence 1024 | Peak allocated memory 1,015.4→603.4 MiB; throughput 7,610.0→6,031.7 tok/s (non-native bf16 autocast) |
| Colab T4 reduced 151.9M architecture | 941.9 tok/s without checkpointing; 825.7 with checkpointing; same reduced seq/batch as laptop |
| CPU convergence | Single/FSDP/TP/2D all complete 150 synthetic-data steps; final loss ~1.59 |

CPU/Gloo timing does not predict multi-GPU performance. Four CPU ranks share three WSL-reported physical cores; DP points increase global batch. GPU results use one device, and the 151.9M measurement reduces the shipped workload to fit 4 GiB VRAM. The laptop profiler captures CUDA API activity without kernel timing data; that breakdown is unavailable. Colab captures GPU kernel timings. T4 software-supported BF16 autocast is not a native BF16 hardware benchmark.

[Full report and figures](docs/BENCHMARKS.md), [dated snapshot](docs/BENCHMARKS-20261014.md), [raw artifacts](benchmarks/results/20261014/README.md), [historical October 4/5 report](docs/archive/BENCHMARKS-20261004.md). Reproduce with `benchmarks/rebenchmark.py` and `benchmarks/supplement.py`.

The [Colab Tesla T4 report](docs/BENCHMARKS-COLAB-20261014.md) includes matching-workload comparisons, exact resume checks, and [verified raw artifacts](benchmarks/results/20261014-colab-t4/README.md). Both hosts have one GPU; these measurements do not establish multi-GPU scaling.

For another Colab run, upload [the prepared notebook](benchmarks/colab.ipynb), select a GPU runtime, and click **Runtime → Run all**. It now records native BF16 support separately from emulation and uses float32 automatically on T4. Notebook sharing exposes saved code/output; execution happens in your Colab session.

## Documentation

Index: [docs/README.md](docs/README.md). Handbook: overview, architecture, parallelism
(with measured collective counts and memory formulas), model, training loop, data
pipeline, checkpointing and fault tolerance, observability, configuration reference,
code walkthrough, benchmarks, testing, runbook, troubleshooting, design decisions,
audit findings, glossary.

---

## Constraints honored

- PyTorch ≥ 2.3 stable `torch.distributed.fsdp` + `torch.distributed.tensor` APIs.
- NCCL for GPU, Gloo only for CPU test fallback.
- No third-party training frameworks; everything is built from PyTorch primitives.
- Full type annotations; `ruff` + `mypy` clean; `pytest`-discoverable tests.
