"""Mixed-precision dtype policy helpers.

What this module does
---------------------
Maps the string dtype names carried in YAML/config (``"bfloat16"`` etc.) to
``torch.dtype`` objects, and constructs the FSDP ``MixedPrecision`` policy from a
:class:`~src.config.ParallelConfig`.

Why the dtypes are split three ways
-----------------------------------
FSDP's ``MixedPrecision`` exposes three independent knobs and choosing them
wrongly silently corrupts training:

  * ``param_dtype=bfloat16`` — params are *all-gathered* and used in the
    forward/backward in bf16. This is the big memory and bandwidth win: the
    all-gather moves half the bytes of fp32.
  * ``reduce_dtype=float32`` — gradients are *reduce-scattered* in fp32. This is
    the subtle one. Gradient reduction sums ``dp_size`` partial gradients; doing
    that sum in bf16 accumulates rounding error proportional to ``dp_size`` and,
    at 64+ ranks, measurably degrades the gradient direction (the small bf16
    mantissa cannot represent the running sum of many small magnitudes). fp32
    reduction costs more bandwidth but keeps the optimizer's gradient estimate
    unbiased. **This is non-negotiable for large DP degrees.**
  * ``buffer_dtype=bfloat16`` — non-learned buffers (RoPE caches, masks). bf16
    is fine; they are not reduced.

The optimizer master weights remain fp32 regardless: FSDP keeps a fp32 copy of
the sharded ``FlatParameter`` and applies the update there, then re-casts to
``param_dtype`` for the next forward. That fp32 master copy is what makes bf16
training stable without loss scaling.
"""

from __future__ import annotations

import torch

from src.config import ParallelConfig

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype string to a ``torch.dtype``.

    Args:
        name: One of the keys in ``_DTYPE_MAP`` (case-insensitive).

    Returns:
        The corresponding ``torch.dtype``.

    Raises:
        ValueError: If ``name`` is not a recognised dtype, listing valid names.

    Example:
        >>> resolve_dtype("bf16")
        torch.bfloat16
    """
    key = name.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {name!r}. Valid options: {sorted(set(_DTYPE_MAP))}."
        )
    return _DTYPE_MAP[key]


def build_mixed_precision(cfg: ParallelConfig) -> torch.distributed.fsdp.MixedPrecision:
    """Construct the FSDP ``MixedPrecision`` policy from a ``ParallelConfig``.

    Args:
        cfg: Parallelism config carrying the three dtype strings.

    Returns:
        A ``MixedPrecision`` instance with ``param_dtype``, ``reduce_dtype`` and
        ``buffer_dtype`` resolved. ``cast_forward_inputs=True`` so module inputs
        are cast to ``param_dtype`` automatically.

    Performance note:
        Setting ``reduce_dtype`` equal to ``param_dtype`` (both bf16) roughly
        halves gradient-reduction bandwidth but, per the module docstring,
        biases the gradient at scale. Only do so for tiny DP degrees.
    """
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=resolve_dtype(cfg.param_dtype),
        reduce_dtype=resolve_dtype(cfg.reduce_dtype),
        buffer_dtype=resolve_dtype(cfg.buffer_dtype),
        cast_forward_inputs=True,
    )


def autocast_dtype(cfg: ParallelConfig) -> torch.dtype:
    """The dtype to use for ``torch.autocast`` in the forward pass.

    Mirrors ``param_dtype`` so activations are produced in the same precision the
    params are gathered in, avoiding spurious up/down casts at layer boundaries.
    """
    return resolve_dtype(cfg.param_dtype)
