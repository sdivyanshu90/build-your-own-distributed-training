# Documentation index

Start with [OVERVIEW](OVERVIEW.md); the rest can be read in any order, but the
sequence below builds up the picture.

| # | Document | Read it to learn |
|---|---|---|
| 1 | [OVERVIEW](OVERVIEW.md) | Problem, goals, non-goals, scope of claims, repository map |
| 2 | [ARCHITECTURE](ARCHITECTURE.md) | Mesh, communication schedule, memory lifecycle, original design log, determinism, limitations |
| 3 | [PARALLELISM](PARALLELISM.md) | DP vs FSDP vs TP vs 2D: what is sharded, which collectives run where (measured), volume and memory formulas checked against measurements |
| 4 | [MODEL](MODEL.md) | Transformer details, parameter counts for every config |
| 5 | [TRAINING_LOOP](TRAINING_LOOP.md) | Step lifecycle, accumulation, clipping, mixed precision, optimizer, scheduler, seeding |
| 6 | [DATA_PIPELINE](DATA_PIPELINE.md) | Datasets, tokenizers, sharding, epoch reseeding, resume replay |
| 7 | [CHECKPOINTING_AND_FAULT_TOLERANCE](CHECKPOINTING_AND_FAULT_TOLERANCE.md) | On-disk layout, atomicity, validation, recovery, measured costs and recovery time |
| 8 | [OBSERVABILITY](OBSERVABILITY.md) | Logs, metrics, MFU, profiler |
| 9 | [CONFIGURATION](CONFIGURATION.md) | Every config field and every shipped YAML |
| 10 | [CODE_WALKTHROUGH](CODE_WALKTHROUGH.md) | File-by-file tour |
| 11 | [BENCHMARKS](BENCHMARKS.md) | Methodology, hardware, results, interpretation, limitations, GPU expectations |
| 12 | [TESTING](TESTING.md) | Test layout, how to run, what each test proves, CI |
| 13 | [RUNBOOK](RUNBOOK.md) | Launching, resuming, diagnosing, tuning |
| 14 | [TROUBLESHOOTING](TROUBLESHOOTING.md) | Errors, hangs, OOM, gloo/NCCL notes, FAQ |
| 15 | [DESIGN_DECISIONS](DESIGN_DECISIONS.md) | ADRs added by the audit (+ pointer to the original log) |
| 16 | [AUDIT_FINDINGS](AUDIT_FINDINGS.md) | Bugs found, severity, fix, regression test |
| 17 | [GLOSSARY](GLOSSARY.md) | Terms |

Raw benchmark data: [`../benchmarks/results/`](../benchmarks/results/) (JSONL, plus a
generated `SUMMARY.md`).
