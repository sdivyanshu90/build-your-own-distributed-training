# Overview

## Problem

Training a language model larger than one accelerator's memory, or faster than one
accelerator allows, needs the model state and the arithmetic to be split across
processes. The mainstream answers (DeepSpeed, Megatron-LM, Accelerate, NeMo) are
large and hide the collectives. This repository implements the core of such a
trainer — a LLaMA-style decoder trained with **FSDP along a data-parallel axis and
tensor parallelism along a second axis** — directly on `torch.distributed`,
`DeviceMesh`, `DTensor` and `FullyShardedDataParallel`, small enough to read in a
day and verified against single-process references.

## Goals

* A readable, typed, tested implementation of 2D (DP × TP) parallel LM training:
  mesh/process-group construction, FSDP wrapping policy, DTensor TP plan, gradient
  accumulation, global-norm clipping across both axes, mixed precision, scheduler,
  sharded atomic checkpoints with validation and recovery, deterministic resume.
* Every parallel layout must be **numerically equivalent** to the unparallelised
  model: `tests/integration/test_parallel_equivalence.py` checks loss and global
  gradient norm for FSDP, TP, FSDP×TP, sequence-parallel TP and HYBRID_SHARD
  against a local single-process reference.
* Everything runs on CPU with the `gloo` backend so that the logic is testable
  without GPUs (`CUDA_VISIBLE_DEVICES=""`), and the same code paths run on NCCL.
* Operational basics: structured logs, throughput/MFU, profiler hooks, a runbook.

## Non-goals

* Not a framework or a library with a stable API. There is no pipeline parallelism,
  expert parallelism/MoE, context parallelism, FP8, `torch.compile` integration,
  KV-cache/inference, or elastic membership change (restart + resume instead).
* Not performance-tuned for any GPU. **All benchmark numbers in this repository are
  CPU/gloo on a laptop-class machine** (BENCHMARKS.md). They validate that the code
  scales its *work* correctly and expose software overhead; they say nothing
  quantitative about GPU/NCCL throughput.
* Not a data-preparation pipeline: the repo ships a synthetic corpus and a reader for
  pre-tokenised `uint16` files, but no tokenisation script.
* Not a numerics research tool: loss-scale handling is bf16-only, there is no
  fp16 `GradScaler`.

## Scope of the claims in these docs

Every number is from a run described in BENCHMARKS.md (command, hardware, torch
version, date 2026-10-04) or from the code itself (parameter counts, formulas,
collective counts measured with the profiler). Statements about GPUs/NCCL are
qualitative and labelled as such. Statements about what the code does cite file
paths. Where something was not run (125M/7B training, NCCL, multi-node, the FSDP
sharded optimizer-state path on GPU), the docs say so.

## Repository map

```
train.py                  torchrun entry point (CLI overrides on a YAML config)
src/config.py             nested dataclass config + YAML loader + validation
src/parallelism/          mesh, process groups, TP (DTensor + hand-written), FSDP utils
src/model/                transformer, attention (RoPE, GQA), SwiGLU MLP, embeddings
src/training/             trainer, train/eval step, optimizer, scheduler, grad utils
src/data/                 synthetic + packed datasets, sharded sampler, tokenizers
src/checkpointing/        atomic sharded save/load, validation and recovery
src/observability/        JSON logging, metrics/MFU, profiler
src/utils/                dtype policy, seeding, launch environment
config/                   test_tiny, base, 125m, 7b YAML
tests/                    unit / integration / fault / performance(bench_*)
benchmarks/               reproducible CPU/gloo harness + raw results
docs/                     this handbook
.github/workflows/ci.yml  ruff + mypy + CPU/gloo pytest
```

Reading order: this file → ARCHITECTURE → PARALLELISM → MODEL → TRAINING_LOOP →
DATA_PIPELINE → CHECKPOINTING_AND_FAULT_TOLERANCE → OBSERVABILITY → CONFIGURATION →
BENCHMARKS → TESTING → RUNBOOK / TROUBLESHOOTING → DESIGN_DECISIONS → AUDIT_FINDINGS.
