"""Cosine-decay-with-linear-warmup learning-rate schedule.

What this module does
---------------------
A self-contained LR schedule that ramps linearly from 0 to the peak LR over
``warmup_steps`` and then follows a half-cosine down to ``min_lr`` at
``max_steps``. Implemented as a closure over the step count plus a tiny
``LambdaLR`` so it composes with PyTorch's optimizer/checkpoint machinery and
restores exactly on resume.

Why warmup then cosine
----------------------
  * **Warmup** avoids the early-training instability where a large LR on a
    randomly-initialised model produces huge, noisy updates (especially harmful
    with Adam's small initial second-moment estimate). A linear ramp lets the
    optimizer statistics settle.
  * **Cosine decay** spends most of the budget at a high LR (fast progress) then
    smoothly anneals to a small LR (fine convergence). It outperforms step decay
    for LM pretraining and has no tuning knobs beyond ``min_lr``.

Determinism / resume invariant
------------------------------
The LR at step ``t`` is a pure function of ``t`` and the config — it carries no
hidden state. So resuming at step 500 and continuing yields the identical LR at
step 1000 as a continuous run (tested in ``test_scheduler.py``). The
``LambdaLR``'s ``last_epoch`` is what the checkpoint restores.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.config import SchedulerConfig


def lr_lambda_factory(
    peak_lr: float, warmup_steps: int, max_steps: int, min_lr: float
) -> Callable[[int], float]:
    """Build the multiplicative LR function used by ``LambdaLR``.

    The returned function maps a step index to a *multiplier* in ``[0, 1]``
    relative to ``peak_lr`` (``LambdaLR`` multiplies the optimizer's base LR by
    it). The optimizer's base LR must therefore equal ``peak_lr``.

    Args:
        peak_lr: The LR at the end of warmup (the optimizer's base LR).
        warmup_steps: Number of linear-warmup steps. ``0`` disables warmup.
        max_steps: Step at which the cosine reaches ``min_lr``.
        min_lr: Floor LR after decay.

    Returns:
        ``fn(step: int) -> float`` giving the LR multiplier.

    Raises:
        ValueError: If ``max_steps <= warmup_steps`` (no room to decay) or
            ``min_lr > peak_lr``.
    """
    if max_steps <= warmup_steps:
        raise ValueError(
            f"max_steps({max_steps}) must exceed warmup_steps({warmup_steps})."
        )
    if min_lr > peak_lr:
        raise ValueError(f"min_lr({min_lr}) must be <= peak_lr({peak_lr}).")
    min_ratio = min_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear ramp 0 -> 1. (step+1)/warmup so step 0 is not exactly 0.
            return float(step + 1) / float(max(1, warmup_steps))
        if step >= max_steps:
            return min_ratio
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Interpolate between 1.0 (peak) and min_ratio.
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, peak_lr: float
) -> LambdaLR:
    """Construct the ``LambdaLR`` cosine-with-warmup scheduler.

    Args:
        optimizer: The optimizer whose base LR equals ``peak_lr``.
        cfg: Scheduler config (warmup, max_steps, min_lr).
        peak_lr: Peak LR (== optimizer base LR).

    Returns:
        A ``LambdaLR`` whose ``.step()`` advances the schedule and whose
        ``.state_dict()`` round-trips through the checkpoint.

    Example:
        >>> opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        >>> sched = build_scheduler(opt, SchedulerConfig(warmup_steps=2,
        ...     max_steps=10, min_lr=0.0), peak_lr=1.0)
        >>> opt.param_groups[0]["lr"]
        0.5
    """
    lr_lambda = lr_lambda_factory(
        peak_lr=peak_lr,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        min_lr=cfg.min_lr,
    )
    return LambdaLR(optimizer, lr_lambda)
