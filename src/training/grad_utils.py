"""Gradient utilities: FSDP-aware clipping, global norm, and finiteness check.

What this module does
---------------------
Provides the two gradient-side operations a distributed step needs after
backward and before the optimizer step: (1) global gradient-norm clipping that is
correct under sharding, and (2) a cross-rank NaN/Inf check that halts *all* ranks
together rather than letting one rank hang the job.

Why ``FSDP.clip_grad_norm_`` and not ``torch.nn.utils.clip_grad_norm_``
-----------------------------------------------------------------------
The global gradient norm is ``sqrt(sum_i ||g_i||^2)`` over *all* parameters. Under
FSDP each rank holds only its shard of each gradient, so a naive per-rank
``torch.nn.utils.clip_grad_norm_`` computes the norm of a *fraction* of the
gradients and clips by the wrong factor — silently scaling updates differently on
every rank and corrupting training. ``FSDP.clip_grad_norm_`` instead computes each
rank's partial sum-of-squares, all_reduces it across the (DP, and with DTensor
params also TP) process groups, takes the global ``sqrt``, and applies one
consistent clip coefficient everywhere. It returns the true pre-clip global norm.

Why a collective finiteness check (and why ``ReduceOp.MIN``)
------------------------------------------------------------
A single NaN gradient on one rank must stop the run *with an error*, not a hang.
If we let it through, the optimizer poisons that rank's weights and the next
all-gather spreads NaNs everywhere — or, worse, the run limps on producing
garbage. We compute a local 0/1 "all finite" flag and all_reduce it with ``MIN``:
if *any* rank saw a non-finite gradient the reduced value is 0 on *every* rank,
so all ranks raise the same exception simultaneously (collective, no hang).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Runtime alias used only in annotations; kept as typing.Union so it is a valid
# value at module import on Python 3.10 regardless of operand types.
ParamsOrModule = Union[nn.Module, Iterable[torch.nn.Parameter]]  # noqa: UP007


class GradNotFiniteError(RuntimeError):
    """Raised (on all ranks) when any rank has a NaN/Inf gradient."""


def _as_param_list(target: ParamsOrModule) -> list[torch.nn.Parameter]:
    if isinstance(target, nn.Module):
        return list(target.parameters())
    return list(target)


def clip_grad_norm_(
    target: ParamsOrModule,
    max_norm: float,
    norm_type: float = 2.0,
    tp_group: dist.ProcessGroup | None = None,
) -> float:
    """Clip gradients by global norm, correctly under FSDP/TP sharding.

    Three code paths, chosen by what ``target`` is:
      * **FSDP module** — delegate to ``FSDP.clip_grad_norm_``, which reduces the
        partial norm across the DP (and DTensor TP) groups. The right call for
        the production FSDP / FSDP+TP runs.
      * **Module/params containing DTensors** (pure-TP, no FSDP) — compute a
        TP-aware global norm (see :func:`_tp_aware_total_norm`) and scale.
      * **Plain dense params** — ``torch.nn.utils.clip_grad_norm_``.

    Args:
        target: An ``FSDP`` module, a plain ``nn.Module``, or a parameter iterable.
        max_norm: Clip threshold. If ``<= 0`` only the norm is computed/returned.
        norm_type: The p-norm order. The DTensor path supports L2 only (the
            default); other orders raise.
        tp_group: TP process group, needed only for the DTensor path to reduce
            replicated-param contributions correctly.

    Returns:
        The **pre-clip** global gradient norm as a Python float, for logging.

    Side effects:
        Scales ``.grad`` in place by ``max_norm / (norm + eps)`` when the norm
        exceeds ``max_norm``.

    Example:
        >>> lin = torch.nn.Linear(4, 4)
        >>> lin(torch.randn(2, 4)).sum().backward()
        >>> pre = clip_grad_norm_(lin, max_norm=0.1)
        >>> pre >= 0.0
        True
    """
    if isinstance(target, FSDP):
        total = target.clip_grad_norm_(max_norm if max_norm > 0 else float("inf"), norm_type)
        return float(total)

    params = _as_param_list(target)
    has_dtensor = any(isinstance(p.grad, DTensor) for p in params if p.grad is not None)
    if has_dtensor:
        return _tp_aware_clip(params, max_norm, norm_type, tp_group)

    if max_norm <= 0:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        total = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type
        )
        return float(total)
    total = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)
    return float(total)


def _tp_aware_total_norm(
    params: list[torch.nn.Parameter], tp_group: dist.ProcessGroup | None
) -> torch.Tensor:
    """Compute the global L2 grad norm over a mix of DTensor and dense params.

    The global norm-squared is the sum of every parameter's *full*-tensor
    norm-squared. Two cases per param's gradient:
      * **Sharded** (a DTensor with a ``Shard`` placement): the per-rank local
        gradients are disjoint pieces, so the full norm² is the sum of local
        norm² across the TP group — contribute the local norm² and let the
        all_reduce sum them.
      * **Replicated** (a dense tensor, or a fully-``Replicate`` DTensor): every
        TP rank holds an identical gradient, so summing across the group would
        overcount by ``tp_size``; contribute ``local_norm² / tp_size`` so the
        post-all_reduce sum counts it exactly once.

    Args:
        params: The parameters to include.
        tp_group: The TP process group (``None`` => degree 1, no reduction).

    Returns:
        A scalar tensor: the global gradient L2 norm.
    """
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    local_sq = torch.zeros((), dtype=torch.float32)
    for p in params:
        g = p.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            local = g.to_local().float()
            replicate = all(pl.is_replicate() for pl in g.placements)
            sq = local.pow(2).sum()
            local_sq = local_sq + (sq / tp_size if replicate else sq)
        else:
            # Dense param: replicated across TP -> count once.
            local_sq = local_sq + g.float().pow(2).sum() / tp_size
    if tp_size > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=tp_group)
    return local_sq.sqrt()


def _tp_aware_clip(
    params: list[torch.nn.Parameter],
    max_norm: float,
    norm_type: float,
    tp_group: dist.ProcessGroup | None,
) -> float:
    """Clip a TP (DTensor) model's grads by global norm; return the pre-clip norm.

    Raises:
        ValueError: If ``norm_type != 2`` (only L2 is implemented for DTensors).
    """
    if norm_type != 2.0:
        raise ValueError(
            f"TP-aware clipping supports L2 only, got norm_type={norm_type}."
        )
    total_norm = _tp_aware_total_norm(params, tp_group)
    if max_norm > 0:
        coef = max_norm / (float(total_norm) + 1e-6)
        if coef < 1.0:
            for p in params:
                if p.grad is not None:
                    # Multiply by a Python float scalar (DTensor-safe, unlike a
                    # device tensor coefficient).
                    p.grad.mul_(coef)
    return float(total_norm)


def grad_finite_check(
    target: ParamsOrModule,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> bool:
    """Verify no rank has a non-finite gradient; raise on all ranks if any does.

    Args:
        target: Module or parameter iterable whose ``.grad`` tensors to check.
        group: Process group to reduce over (default: the world group). For 2D
            parallelism pass the world group so a NaN anywhere stops everyone.
        device: Device for the flag tensor (the rank's compute device).

    Returns:
        ``True`` if all gradients on all ranks are finite.

    Raises:
        GradNotFiniteError: On *every* rank if *any* rank had a NaN/Inf gradient.
            The message includes the local rank's verdict so the offending rank
            is identifiable in the logs.
    """
    params = _as_param_list(target)
    local_finite = 1
    for p in params:
        g = p.grad
        if g is None:
            continue
        # DTensor: check the local shard to avoid a collective inside isfinite.
        local = g.to_local() if isinstance(g, DTensor) else g
        if not torch.isfinite(local).all():
            local_finite = 0
            break
    flag = torch.tensor([local_finite], device=device, dtype=torch.int32)
    if dist.is_initialized():
        # MIN: result is 0 on all ranks if ANY rank reported 0 -> symmetric raise.
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    if int(flag.item()) == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        raise GradNotFiniteError(
            f"[rank {rank}] Non-finite gradient detected on at least one rank "
            f"(this rank local_finite={local_finite}). Aborting before the "
            f"optimizer step to avoid poisoning weights. Common causes: LR too "
            f"high, missing grad clipping, fp16 overflow (use bf16), or a bad "
            f"batch."
        )
    return True
