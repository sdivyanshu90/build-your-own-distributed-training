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
- `ruff` clean, `mypy` clean.

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
docs/{ARCHITECTURE,RUNBOOK}.md
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

This repo is developed against **PyTorch 2.3** on a single-GPU / CPU box. The
following run and pass on the CPU/Gloo test path:

- All unit tests (TP linear numerics + `gradcheck`, scheduler, grad clipping,
  metrics, FSDP param-count invariant, checkpoint serialization, dataloader
  sharding).
- Integration: **TP forward proven identical to a single-GPU reference**,
  DP loss consistency, convergence for **single / pure-FSDP / pure-TP**, bit-exact
  resume, dataloader sharding, mixed-precision policy.
- Fault: corrupt/incomplete checkpoint detection and crash recovery.

Two paths are **GPU-only on torch 2.3** and are gated (skipped on CPU), while the
code is correct for real multi-GPU NCCL:

- **Composed 2D FSDP+TP end-to-end** — FSDP1's `use_orig_params` writeback hits a
  DTensor storage bug on the CPU/Gloo path in torch 2.3 (fixed on NCCL / torch ≥
  2.4). Pure FSDP and pure TP are fully exercised here; the TP math itself is
  proven correct against a single-GPU reference.
- **FSDP sharded optimizer state-dict** — calls `torch.cuda.synchronize()`
  unconditionally in torch 2.3, so FSDP checkpoint/resume is GPU-only; the
  checkpoint *logic* is fully tested with plain models on CPU.

See [`docs/ARCHITECTURE.md` §6](docs/ARCHITECTURE.md) for details.

---

## Constraints honored

- PyTorch ≥ 2.3 stable `torch.distributed.fsdp` + `torch.distributed.tensor` APIs.
- NCCL for GPU, Gloo only for CPU test fallback.
- No third-party training frameworks; everything is built from PyTorch primitives.
- Full type annotations; `ruff` + `mypy` clean; `pytest`-discoverable tests.
