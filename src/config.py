"""Typed configuration schema for the distributed training system.

What this module does
---------------------
Defines the full hyperparameter surface of the system as nested, frozen-ish
``@dataclass`` objects (``ModelConfig``, ``ParallelConfig``, ``OptimizerConfig``,
``SchedulerConfig``, ``DataConfig`` and the top-level ``TrainingConfig``) plus a
YAML loader that materialises them. Every other module receives a typed config
object rather than reading globals or a free-form ``dict``; this keeps the data
flow explicit (Code Quality requirement: *no global state*) and lets ``mypy
--strict`` check that callers use real fields.

Why dataclasses over a dict / argparse namespace
-------------------------------------------------
A ``dict`` config silently accepts typos (``cfg["lerning_rate"]`` returns
``KeyError`` only at runtime, often on a different rank than the one that fired,
producing a confusing distributed hang). Dataclasses give us:
  * a single authoritative list of valid keys with defaults,
  * static type checking of every access,
  * trivial (de)serialisation for checkpoint metadata,
  * ``__eq__`` for free, which the checkpoint validator uses to detect a config
    mismatch between save and load time.

Key invariants
--------------
  * ``world_size == parallel.tp_size * parallel.dp_size`` is asserted at mesh
    construction time, not here, because ``world_size`` is a runtime property.
  * ``global_batch_size == micro_batch_size * grad_accum_steps * dp_size`` —
    enforced by :meth:`TrainingConfig.validate` once ``dp_size`` is known.
  * Dtype fields are stored as strings in YAML and resolved to ``torch.dtype``
    lazily (see :mod:`src.utils.dtype`) so the YAML stays framework-agnostic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class ModelConfig:
    """Architecture hyperparameters for a LLaMA-style causal LM.

    Attributes:
        vocab_size: Size of the token vocabulary. Must match the tokenizer.
        d_model: Residual-stream / hidden width. Must be divisible by
            ``n_heads`` and by the TP degree (each TP rank owns
            ``d_model / tp_size`` of the attention/MLP projections).
        n_layers: Number of stacked ``TransformerBlock`` modules.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads (grouped-query attention). Equal
            to ``n_heads`` for vanilla multi-head attention; smaller for GQA.
            Must be divisible by the TP degree.
        ffn_hidden_size: Inner width of the SwiGLU MLP. If ``None`` it is
            derived as ``int(8/3 * d_model)`` rounded to a multiple of 256, the
            LLaMA convention that keeps the SwiGLU parameter count comparable to
            a 4x GeLU MLP.
        max_seq_len: Maximum context length; bounds the RoPE cache.
        rope_theta: Base period for rotary position embeddings.
        norm_eps: Epsilon for RMSNorm numerical stability.
        tie_embeddings: If True the LM head reuses the input embedding matrix.
        dropout: Residual/attention dropout probability (0.0 for pretraining).
        attention_bias: Whether qkv/out projections carry bias (LLaMA: False).
        mlp_bias: Whether MLP projections carry bias (LLaMA: False).
    """

    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    ffn_hidden_size: int | None = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_hidden_size is None:
            # LLaMA's SwiGLU sizing: 8/3 * d_model rounded up to a multiple of 256.
            hidden = int(8 * self.d_model / 3)
            self.ffn_hidden_size = 256 * ((hidden + 255) // 256)

    @property
    def head_dim(self) -> int:
        """Per-head dimension. Invariant: ``d_model == n_heads * head_dim``."""
        return self.d_model // self.n_heads

    def num_parameters(self) -> int:
        """Approximate (analytical) parameter count, used by the MFU metric.

        Counts embeddings, per-layer attention + MLP, and the LM head. Norm and
        bias terms are negligible and folded into the estimate. Returns the
        *unsharded* total — the number a single-GPU model would hold.
        """
        assert self.n_kv_heads is not None and self.ffn_hidden_size is not None
        d = self.d_model
        # Attention: q (d*d) + k,v (d * n_kv_heads*head_dim each) + out (d*d).
        kv_dim = self.n_kv_heads * self.head_dim
        attn = d * d + 2 * d * kv_dim + d * d
        # SwiGLU MLP: gate + up (d -> ffn each) + down (ffn -> d).
        mlp = 3 * d * self.ffn_hidden_size
        per_layer = attn + mlp
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return embed + self.n_layers * per_layer + head


@dataclass
class ParallelConfig:
    """2D (DP x TP) parallelism and FSDP/mixed-precision policy.

    Attributes:
        tp_size: Tensor-parallel degree (GPUs that cooperatively hold one
            layer). 1 disables TP.
        dp_size: Data-parallel / FSDP degree. 1 disables sharding. If left at
            the sentinel ``-1`` it is inferred as ``world_size // tp_size``.
        sharding_strategy: ``"FULL_SHARD"`` (ZeRO-3) or ``"HYBRID_SHARD"``
            (shard within a node, replicate across nodes).
        activation_checkpointing: Recompute each block's activations in
            backward to trade compute for memory.
        sequence_parallel: Shard LayerNorm/RMSNorm activations along the
            sequence dim within the TP group (Megatron sequence parallelism).
        backward_prefetch: ``"BACKWARD_PRE"`` or ``"BACKWARD_POST"``; controls
            when FSDP all-gathers the next layer's params during backward.
        forward_prefetch: Prefetch the next layer's all-gather in forward.
        limit_all_gathers: Rate-limit concurrent all-gathers to bound peak
            memory (FSDP "rate limiter").
        cpu_offload: Offload sharded params/grads to CPU (last-resort memory
            relief; heavy throughput cost).
        param_dtype / reduce_dtype / buffer_dtype: Mixed-precision policy.
            See :mod:`src.utils.dtype` for why ``reduce_dtype`` is fp32.
    """

    tp_size: int = 1
    dp_size: int = -1
    sharding_strategy: str = "FULL_SHARD"
    activation_checkpointing: bool = False
    sequence_parallel: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    cpu_offload: bool = False
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    buffer_dtype: str = "bfloat16"


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters.

    Attributes:
        name: ``"adamw"`` or ``"lamb"``.
        lr: Peak learning rate (the scheduler ramps to this then decays).
        weight_decay: Decoupled weight decay (applied only to 2D+ weights, not
            norms/biases — see :func:`src.training.optimizer.build_param_groups`).
        betas: Adam moment decay rates.
        eps: Adam denominator epsilon.
        fused: Use the fused CUDA optimizer kernel when available.
    """

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool = True


@dataclass
class SchedulerConfig:
    """Cosine-decay-with-linear-warmup schedule.

    Attributes:
        warmup_steps: Linear ramp from 0 to ``OptimizerConfig.lr``.
        max_steps: Total optimizer steps; cosine reaches ``min_lr`` here.
        min_lr: Floor learning rate at the end of the cosine.
    """

    warmup_steps: int = 100
    max_steps: int = 10000
    min_lr: float = 3e-5


@dataclass
class DataConfig:
    """Data pipeline configuration.

    Attributes:
        dataset_path: Path/identifier of the corpus (or ``"synthetic"``).
        tokenizer_name: HuggingFace tokenizer id or local path.
        seq_len: Training sequence length.
        micro_batch_size: Per-GPU per-microstep batch size.
        global_batch_size: Tokens-equivalent global batch in *sequences*. Must
            equal ``micro_batch_size * grad_accum_steps * dp_size``.
        num_workers: Dataloader worker processes per rank.
        shuffle: Whether to shuffle shards each epoch.
    """

    dataset_path: str = "synthetic"
    tokenizer_name: str = "gpt2"
    seq_len: int = 1024
    micro_batch_size: int = 8
    global_batch_size: int = 64
    num_workers: int = 2
    shuffle: bool = True


@dataclass
class TrainingConfig:
    """Top-level config aggregating every sub-config plus loop controls.

    Attributes:
        model / parallel / optimizer / scheduler / data: Nested configs.
        seed: Base RNG seed; per-rank seed is ``seed + dp_rank`` so DP ranks see
            different data but TP ranks (same dp_rank) stay in lockstep.
        grad_accum_steps: Micro-batches accumulated before an optimizer step.
        max_grad_norm: Global gradient-norm clip threshold (<=0 disables).
        log_interval / eval_interval / save_interval: Step cadences.
        eval_steps: Number of eval micro-batches per evaluation.
        warmup_profile_steps / profile_steps: Profiler window.
        checkpoint_dir: Root directory for checkpoints.
        run_id: Unique run identifier; all artifacts are namespaced under it to
            prevent cross-run overwrites (self-review checklist item).
        resume_from: Path to a checkpoint directory to resume, or ``None``.
        backend: ``"nccl"`` for GPU, ``"gloo"`` for CPU fallback in tests.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    seed: int = 1234
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    max_steps: int = 10000
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 20
    save_interval: int = 1000
    warmup_profile_steps: int = 5
    profile_steps: int = 3
    checkpoint_dir: str = "checkpoints"
    run_id: str = "run"
    resume_from: str | None = None
    backend: str = "nccl"

    def validate(self, world_size: int, dp_size: int) -> None:
        """Cross-field consistency checks that need the runtime world size.

        Args:
            world_size: ``dist.get_world_size()`` at launch.
            dp_size: Resolved data-parallel degree (``world_size // tp_size``).

        Raises:
            ValueError: If the global batch identity is violated or a dtype /
                strategy string is unknown. The message includes the offending
                values so a failure is debuggable from the log line alone.
        """
        expected_global = self.data.micro_batch_size * self.grad_accum_steps * dp_size
        # global_batch_size <= 0 means "auto-derive" (used by topology-agnostic
        # test configs where dp_size is only known at runtime).
        if self.data.global_batch_size <= 0:
            self.data.global_batch_size = expected_global
        elif self.data.global_batch_size != expected_global:
            raise ValueError(
                "global_batch_size identity violated: "
                f"global_batch_size={self.data.global_batch_size} != "
                f"micro_batch_size({self.data.micro_batch_size}) * "
                f"grad_accum_steps({self.grad_accum_steps}) * dp_size({dp_size}) "
                f"= {expected_global}. Adjust grad_accum_steps or batch sizes."
            )
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError(
                f"d_model={self.model.d_model} not divisible by "
                f"n_heads={self.model.n_heads}."
            )
        valid_strategies = {"FULL_SHARD", "HYBRID_SHARD", "SHARD_GRAD_OP", "NO_SHARD"}
        if self.parallel.sharding_strategy not in valid_strategies:
            raise ValueError(
                f"sharding_strategy={self.parallel.sharding_strategy!r} not in "
                f"{sorted(valid_strategies)}."
            )

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        """Load a ``TrainingConfig`` from a YAML file with nested overrides.

        The YAML mirrors the dataclass structure; any omitted key falls back to
        the dataclass default. Unknown keys raise ``TypeError`` from the
        dataclass constructor so a typo fails loudly at load time on every rank.

        Args:
            path: Filesystem path to the YAML config.

        Returns:
            A fully-populated ``TrainingConfig``.

        Example:
            >>> cfg = TrainingConfig.from_yaml("config/test_tiny.yaml")
            >>> cfg.model.n_layers
            2
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        """Build a ``TrainingConfig`` from a plain (possibly nested) dict."""
        return _from_dict(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for checkpoint metadata / logging)."""
        return dataclasses.asdict(self)


def _from_dict(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively instantiate a (possibly nested) dataclass from a dict.

    Each field whose type is itself a dataclass is recursed into; tuples are
    coerced from YAML lists so ``betas: [0.9, 0.95]`` becomes a ``tuple``.
    """
    if not is_dataclass(cls):
        return raw
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    field_types = {f.name: f for f in fields(cls)}
    for key, value in raw.items():
        if key not in field_types:
            raise TypeError(
                f"Unknown config key {key!r} for {cls.__name__}. "
                f"Valid keys: {sorted(field_types)}."
            )
        f = field_types[key]
        ftype = f.type
        # Resolve nested dataclasses by their concrete default_factory type.
        nested_cls = _resolve_dataclass_type(ftype, f)
        if nested_cls is not None and isinstance(value, dict):
            kwargs[key] = _from_dict(nested_cls, value)
        elif isinstance(value, list) and _expects_tuple(type_hints[key]):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_dataclass_type(ftype: Any, f: Any) -> type | None:
    """Return the nested dataclass type for a field, or None if not a dataclass."""
    # When using ``from __future__ import annotations`` field types are strings.
    if isinstance(ftype, str):
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            candidate = f.default_factory()  # type: ignore[misc]
            if is_dataclass(candidate):
                return type(candidate)
        return None
    return ftype if is_dataclass(ftype) else None


def _expects_tuple(type_hint: Any) -> bool:
    """Heuristic: does this field's annotation mention ``tuple``?"""
    return isinstance(type_hint, str) and "tuple" in type_hint.lower()
