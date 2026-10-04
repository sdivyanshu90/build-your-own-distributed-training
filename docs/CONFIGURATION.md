# Configuration reference

All configuration lives in `src/config.py` as nested dataclasses and is loaded from
YAML by `TrainingConfig.from_yaml`. Unknown keys raise `TypeError` at load time
(`_from_dict`), and every omitted key falls back to the dataclass default shown
below. `train.py` applies a handful of CLI overrides on top (see the last section).

Validation happens in two places:

* **Load time** – unknown keys (typos) fail loudly.
* **Trainer start-up** – `TrainingConfig.validate(world_size, dp_size)` is called by
  `Trainer.__init__` once the process groups exist (`src/training/trainer.py`). It
  derives or checks the global batch, checks head divisibility, TP divisibility,
  the GQA group size and the sharding-strategy name, and **warns** if
  `scheduler.max_steps != max_steps`.

> Dtype fields are plain strings (`float32`/`fp32`, `float16`/`fp16`/`half`,
> `bfloat16`/`bf16`) resolved by `src/utils/dtype.py::resolve_dtype`.

## Top-level fields (`TrainingConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `seed` | int | 1234 | Base RNG seed. Model init uses `seed` on **every** rank; afterwards each rank is re-seeded with `seed + dp_rank` (see `Trainer.__init__`, `src/utils/seed.py`). Also seeds the synthetic dataset (`seed`, val: `seed+7`) and the sampler shuffle (`seed + epoch`). |
| `grad_accum_steps` | int | 1 | Micro-batches per optimizer step. Loss is divided by this value (`loop.py`). |
| `max_grad_norm` | float | 1.0 | Global L2 clip threshold. `<= 0` computes/logs the norm but never clips. |
| `max_steps` | int | 10000 | Optimizer steps to run (`Trainer.train` loop condition). Overridden by `--max-steps`, which also overwrites `scheduler.max_steps`. |
| `log_interval` | int | 10 | Emit a `METRIC step` record every N steps (rank 0). |
| `eval_interval` | int | 500 | Run `eval_step` every N steps; `0` disables. |
| `eval_steps` | int | 20 | Validation micro-batches per evaluation (per DP rank). |
| `save_interval` | int | 1000 | Write a sharded checkpoint every N steps; `0` disables. **No checkpoint is written at `max_steps` unless it is a multiple of `save_interval`.** |
| `warmup_profile_steps` | int | 5 | Profiler wait/warm-up steps before the active window. |
| `profile_steps` | int | 3 | Profiler active steps; `0` disables the profiler (rank 0 only). |
| `checkpoint_dir` | str | `checkpoints` | Root; checkpoints go to `{checkpoint_dir}/{run_id}/step_{N}/`. |
| `run_id` | str | `run` | Namespaces checkpoints, logs and `traces/{run_id}`. |
| `resume_from` | str \| null | null | A `step_N` checkpoint directory **or** a run directory (newest valid `step_*` is chosen, `Trainer`/`_resolve_resume_path`). |
| `backend` | str | `nccl` | `nccl` or `gloo`. `nccl` raises if CUDA is unavailable (`src/utils/env.py`). |

## `model:` (`ModelConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `vocab_size` | int | 32000 | Embedding rows / LM-head width. |
| `d_model` | int | 768 | Residual width; must be divisible by `n_heads`. |
| `n_layers` | int | 12 | Transformer blocks. |
| `n_heads` | int | 12 | Query heads. `head_dim = d_model / n_heads` must be even (RoPE). |
| `n_kv_heads` | int \| null | `n_heads` | KV heads (GQA). `n_heads % n_kv_heads == 0`; with TP, `n_kv_heads % tp_size == 0`. |
| `ffn_hidden_size` | int \| null | `256·ceil(int(8d/3)/256)` | SwiGLU inner width. With TP must be divisible by `tp_size`. |
| `max_seq_len` | int | 2048 | Size of the RoPE cache; `data.seq_len` must not exceed it. |
| `rope_theta` | float | 10000.0 | RoPE base. |
| `norm_eps` | float | 1e-5 | RMSNorm epsilon. |
| `tie_embeddings` | bool | true | LM head shares the embedding matrix. |
| `dropout` | float | 0.0 | Attention dropout probability (SDPA `dropout_p`, train mode only). There is no residual dropout in the code. |
| `attention_bias` / `mlp_bias` | bool | false | Bias on attention / MLP linears. |

## `parallel:` (`ParallelConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `tp_size` | int | 1 | Tensor-parallel degree; must divide `world_size`. |
| `dp_size` | int | -1 | Data-parallel degree; `-1` ⇒ `world_size // tp_size`. `dp * tp` must equal `world_size`. |
| `sharding_strategy` | str | `FULL_SHARD` | One of `FULL_SHARD`, `SHARD_GRAD_OP`, `HYBRID_SHARD`, `NO_SHARD`. Only used when `dp_size > 1` (with `dp_size == 1` FSDP is skipped entirely). |
| `hybrid_shard_size` | int | 0 | `HYBRID_SHARD` only: FSDP group size; `0` ⇒ `local_world_size // tp_size`. Must divide `dp_size`. |
| `activation_checkpointing` | bool | false | Non-reentrant checkpoint wrapper on every `TransformerBlock` (`fsdp_utils.apply_activation_checkpointing`). |
| `sequence_parallel` | bool | false | Adds `SequenceParallel` norms and `Shard(1)` layouts to the TP plan (see PARALLELISM.md for the status of this path). |
| `backward_prefetch` | str | `BACKWARD_PRE` | `BACKWARD_PRE` \| `BACKWARD_POST`. |
| `forward_prefetch` | bool | true | FSDP explicit forward prefetch. |
| `limit_all_gathers` | bool | true | FSDP rate limiter. |
| `cpu_offload` | bool | false | `CPUOffload(offload_params=True)`. Wired, not exercised by tests. |
| `param_dtype` | str | `bfloat16` | FSDP compute dtype for params, and the autocast dtype on CUDA. |
| `reduce_dtype` | str | `float32` | FSDP gradient reduce-scatter dtype. |
| `buffer_dtype` | str | `bfloat16` | FSDP buffer dtype. |

## `optimizer:` (`OptimizerConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `name` | str | `adamw` | `adamw` (torch) or `lamb` (in-repo `LAMB`; per-rank trust ratios, see TRAINING_LOOP.md). |
| `lr` | float | 3e-4 | Peak LR (scheduler multiplies it). |
| `weight_decay` | float | 0.1 | Applied to params with `dim() >= 2` only. |
| `betas` | [float, float] | (0.9, 0.95) | YAML list becomes a tuple. |
| `eps` | float | 1e-8 | Adam epsilon (LAMB default is overridden with this value as well). |
| `fused` | bool | true | Fused AdamW only when CUDA is available. |

## `scheduler:` (`SchedulerConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `warmup_steps` | int | 100 | Linear warmup `lr·(step+1)/warmup`. |
| `max_steps` | int | 10000 | Cosine horizon; must exceed `warmup_steps`. |
| `min_lr` | float | 3e-5 | Floor; must be `<= optimizer.lr`. |

## `data:` (`DataConfig`)

| Field | Type | Default | Effect |
|---|---|---|---|
| `dataset_path` | str | `synthetic` | `synthetic` or path to a flat `uint16` token file (`PackedTokenDataset`). |
| `tokenizer_name` | str | `gpt2` | **Not read by the trainer** (the packed file is assumed pre-tokenised); kept as metadata. |
| `seq_len` | int | 1024 | Tokens per sequence (inputs; labels are shifted by one). |
| `micro_batch_size` | int | 8 | Sequences per micro-batch per rank. |
| `global_batch_size` | int | 64 | Must equal `micro_batch_size · grad_accum_steps · dp_size`; `<= 0` ⇒ derived at start-up. |
| `num_workers` | int | 2 | DataLoader workers per rank. |
| `shuffle` | bool | true | Per-epoch reshuffle for the **train** loader (val never shuffles). |

## The shipped YAMLs

Parameter counts below come from instantiating `build_model` on the `meta` device
for each file (`benchmarks`-independent; reproducible with the snippet in
BENCHMARKS.md §Reproduction). "Analytic" is `ModelConfig.num_parameters()`, which
ignores norm weights.

| File | d_model / layers / heads (kv) / ffn / vocab / max_seq | Tied | Exact params | Analytic | Notes |
|---|---|---|---|---|---|
| `config/test_tiny.yaml` | 64 / 2 / 4 (2) / 128 / 256 / 64 | yes | 90,432 | 90,112 | fp32 everywhere, gloo, `global_batch_size: 0`, `grad_accum_steps: 2`, `seq_len 32`, mbs 4. CI/CPU. |
| `config/base.yaml` | 768 / 12 / 12 (12) / 2048 (auto) / 32000 / 2048 | yes | 109,529,856 | 109,510,656 | Defaults; bf16/fp32-reduce; seq 2048, mbs 8, 100k steps. |
| `config/125m.yaml` | 768 / 12 / 12 (12) / 3072 / 50304 / 1024 | yes | **151,898,880** | 151,879,680 | Named "125m" but, with a 3-matrix SwiGLU of width 3072 and a 50k vocab, it is **~152M** parameters. lr 6e-4, accum 4, mbs 16, 50k steps. |
| `config/7b.yaml` | 4096 / 32 / 32 (32) / 11008 / 32000 / 4096 | no | 6,738,415,616 | 6,738,149,376 | `tp_size: 8`, `HYBRID_SHARD`, activation checkpointing and `sequence_parallel: true`, accum 8, mbs 1, `tokenizer_name: meta-llama/Llama-2-7b-hf` (gated on Hugging Face; unused by the trainer). |

`7b.yaml` and `125m.yaml` default to `backend: nccl` (inherited default) and need
GPUs; `test_tiny.yaml` sets `backend: gloo`. None of the 125m/7b numbers were run
for this documentation (see BENCHMARKS.md for why).

## CLI overrides (`train.py`)

| Flag | Overrides |
|---|---|
| `--config PATH` (required) | YAML file |
| `--tp-size N` / `--dp-size N` | `parallel.tp_size` / `parallel.dp_size` |
| `--backend nccl\|gloo` | `backend` |
| `--run-id ID` | `run_id` |
| `--resume-from PATH` | `resume_from` |
| `--max-steps N` | `max_steps` **and** `scheduler.max_steps` |
| `--gpu-type KEY` | MFU peak-FLOPs table key (`a100` default; keys in `src/observability/metrics.py`). On CPU use `cpu` for a defined-but-meaningless value. |

There is no CLI flag for `log_interval`, `save_interval`, batch sizes or model
size; edit a YAML copy (the benchmark harness does exactly that, see
`benchmarks/bench_fault.py`).
