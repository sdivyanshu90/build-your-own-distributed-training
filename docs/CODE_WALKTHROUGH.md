# Code walkthrough

Every source file, in dependency order, with its key functions and what to know
before changing it. Line numbers are deliberately omitted (they rot); search for
the symbol.

## `train.py`

`parse_args` → `build_config` (YAML + overrides; `--max-steps` also sets
`scheduler.max_steps`) → `Trainer(config, gpu_type).train()` inside
`try/finally destroy_distributed()`. Run it with `torchrun`; running it directly
gives world size 1.

## `src/config.py`

Dataclasses `ModelConfig`, `ParallelConfig`, `OptimizerConfig`, `SchedulerConfig`,
`DataConfig`, `TrainingConfig`. Notable: `ModelConfig.__post_init__` fills
`n_kv_heads` and `ffn_hidden_size`; `num_parameters()` is the analytic count used by
the MFU formula; `TrainingConfig.validate` (see CONFIGURATION.md);
`from_yaml/_from_dict` recursion rejects unknown keys and converts YAML lists to
tuples for tuple-typed fields (string-annotation heuristic `_expects_tuple`).
`to_dict` (via `dataclasses.asdict`) is what checkpoints store.

## `src/utils/`

* `dtype.py` — `resolve_dtype`, `build_mixed_precision(ParallelConfig) →
  torch.distributed.fsdp.MixedPrecision(..., cast_forward_inputs=True)`,
  `autocast_dtype` (= `param_dtype`).
* `env.py` — `read_launch_env` (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`,
  `MASTER_*`; for `nccl` checks CUDA and `LOCAL_RANK < device_count`),
  `select_device` (`cuda:LOCAL_RANK` for nccl else CPU), `describe_versions`.
* `seed.py` — `seed_everything(base, dp_rank, deterministic)`,
  `get_rng_state/set_rng_state` (Python, NumPy, torch, CUDA).

## `src/parallelism/`

* `mesh.py` — `resolve_dp_size` (validates `tp | world` and `dp·tp = world`),
  `build_device_mesh` → `init_device_mesh(shape=(dp, tp), names=("dp","tp"))`,
  `get_parallel_dims` (coordinates from `mesh[axis].get_local_rank()`),
  `get_dp_group/get_tp_group`, `format_mesh_layout`. TP is the fastest-varying
  axis: ranks `0..tp-1` form the first TP group, so TP should map to one node.
* `process_groups.py` — `init_distributed` (idempotent), `build_process_context`
  → frozen `ProcessContext(env, device, mesh, dims, dp_group, tp_group)` with
  `is_rank0`, `is_dp_rank0`, `barrier`; `destroy_distributed`.
* `tensor_parallel.py` — four autograd Functions (`_CopyToTPRegion`: identity /
  all-reduce-grad; `_ReduceFromTPRegion`: all-reduce / identity;
  `_GatherFromTPRegion`: all-gather / take-chunk; `_ScatterToTPRegion`: take-chunk /
  all-gather) and the modules `ColumnParallelLinear` and `RowParallelLinear` built on
  them (reference implementation, `gradcheck`-tested, **not** used by the trainer);
  `apply_tensor_parallelism(model, tp_mesh, sequence_parallel=...)` — the
  production path: a `parallelize_module` plan of `ColwiseParallel` for
  `attention.wq/wk/wv`, `mlp.gate_proj/up_proj`, `RowwiseParallel` for
  `attention.wo`, `mlp.down_proj`, plus `SequenceParallel` norms and
  `PrepareModuleInput` when sequence parallelism is on; `is_tp_sharded_param(name)`
  used by the 2D clip.
* `fsdp_utils.py` — `wrap_model_with_fsdp` (auto-wrap each `TransformerBlock`,
  `use_orig_params=True`, `device_mesh=mesh["dp"]` — or the 2-D replicate×shard
  sub-mesh from `_hybrid_shard_mesh` for `HYBRID_SHARD`, `device_id=ctx.device`),
  `apply_activation_checkpointing` (non-reentrant wrapper on blocks),
  `count_unsharded_parameters`, and the state-dict helpers
  `get/load_model_state_dict`, `get/load_optimizer_state_dict`. The optimizer helpers
  fall back to a **per-rank raw `optimizer.state_dict()`** (marker key
  `__fsdp_per_rank__`) whenever `torch.cuda.is_available()` is false — the path every
  CPU test and benchmark takes. It is topology-locked and keyed by local shard
  layout; the re-keyed FSDP optimizer-state path is only taken on CUDA and is **not
  exercised here**.

## `src/model/`

See MODEL.md. `attention.py::repeat_kv`, `Attention`; `embeddings.py::
TokenEmbedding, precompute_rope_cache, apply_rotary_emb, VocabParallelEmbedding`
(unused); `mlp.py::SwiGLUMLP`; `transformer.py::RMSNorm, TransformerBlock,
Transformer, build_model`.

## `src/training/`

* `trainer.py` — `Trainer` (see TRAINING_LOOP.md), `_resolve_resume_path`,
  `_metric_fields`. Public pieces used by tests/benchmarks: `_next_window()`,
  `_fast_forward_data()`, `.model/.optimizer/.scheduler/.ctx`.
* `loop.py` — `train_step`, `eval_step`, `_maybe_no_sync`, `_move_batch`.
* `grad_utils.py` — `clip_grad_norm_` dispatcher, `_fsdp_tp_clip`,
  `_tp_aware_clip/_tp_aware_total_norm`, `grad_finite_check`, `GradNotFiniteError`.
* `optimizer.py` — `build_param_groups`, `LAMB`, `build_optimizer`.
* `scheduler.py` — `lr_lambda_factory`, `build_scheduler`.

## `src/data/`

See DATA_PIPELINE.md: `SyntheticTokenDataset`, `PackedTokenDataset`,
`ShardedSampler`, `build_dataloader` (`drop_last=True`, `pin_memory` only with CUDA),
`Tokenizer` protocol, `HFTokenizer`, `ByteTokenizer`, `build_tokenizer`.

## `src/checkpointing/`

See CHECKPOINTING_AND_FAULT_TOLERANCE.md: `save_checkpoint`, `load_checkpoint`,
`checkpoint_path`, `_atomic_torch_save`, `_atomic_write_text`;
`validate_checkpoint`, `find_latest_valid_checkpoint`, `require_valid_checkpoint`,
`CheckpointValidationResult`, `CheckpointValidationError`.

## `src/observability/`

See OBSERVABILITY.md: `RankLogger`, `StepMetrics` + metric helpers, profiler helpers.

## `tests/` and `benchmarks/`

See TESTING.md and BENCHMARKS.md. `tests/_dist_utils.py::run_distributed(fn, world,
*args)` spawns gloo processes with `mp.spawn` and re-raises child exceptions in the
parent.
