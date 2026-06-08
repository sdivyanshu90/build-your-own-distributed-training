"""The forward/backward step: gradient accumulation, clipping, optimizer step.

What this module does
---------------------
Implements ``train_step`` (one optimizer step over a window of ``grad_accum_steps``
micro-batches) and ``eval_step``. This is where the subtle FSDP correctness rules
live.

Gradient accumulation under FSDP — the ``no_sync()`` rule
---------------------------------------------------------
FSDP reduce-scatters gradients across the DP group at the end of *each* backward
by default. When accumulating ``K`` micro-batches before one optimizer step, that
is ``K-1`` wasted reduce-scatters — we only need the averaged gradient once, after
the last micro-batch. ``model.no_sync()`` defers the reduce-scatter: gradients
accumulate locally in the unsharded grad buffer, and only the final
(non-``no_sync``) backward triggers the collective. So the rule is:

    enter ``no_sync()`` for micro-batches ``0..K-2``; run the last one normally.

Getting this wrong is a classic bug in both directions: wrap *all* K (including
the last) and the gradients are never reduced — DP ranks silently diverge; wrap
*none* and you pay K reduce-scatters and peak grad memory is unsharded the whole
time.

The TP asymmetry (self-review checklist)
----------------------------------------
``no_sync()`` only suppresses the **DP/FSDP** reduce-scatter. The **TP** all_reduce
inside every ``RowParallel`` layer (``wo``, ``down_proj``) fires on *every*
micro-batch regardless — it is part of the forward/backward math, not a gradient
sync, and deferring it would produce wrong activations. This asymmetry is
intentional and must not be "optimised away".

Loss scaling
------------
Each micro-batch loss is divided by ``grad_accum_steps`` so the summed gradient
equals the gradient of the full batch mean. We report the *un-scaled* mean loss
(sum of the divided losses) for logging.

Mixed precision
---------------
On CUDA the forward runs under ``torch.autocast(bf16)``; FSDP's ``MixedPrecision``
already holds params in bf16, and autocast additionally keeps numerically
sensitive ops (softmax, cross-entropy) in fp32. bf16 has the dynamic range of
fp32, so **no GradScaler is needed** (unlike fp16). On CPU we stay in fp32 for
deterministic cross-config comparisons.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.observability.metrics import (
    StepMetrics,
    aggregate_loss,
    compute_mfu,
    compute_throughput,
    estimate_flops_per_token,
    peak_memory_bytes,
)
from src.parallelism.process_groups import ProcessContext
from src.training.grad_utils import clip_grad_norm_, grad_finite_check
from src.utils.dtype import autocast_dtype

Batch = Mapping[str, torch.Tensor]


def _maybe_no_sync(model: nn.Module, enabled: bool) -> contextlib.AbstractContextManager:
    """Return ``model.no_sync()`` when accumulating, else a null context.

    Args:
        model: The (FSDP-wrapped) model. A plain module without ``no_sync`` falls
            back to the null context.
        enabled: True for every micro-batch except the last.
    """
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return contextlib.nullcontext()


def _move_batch(batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move ``input_ids``/``labels`` to ``device`` non-blocking; return the pair."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return input_ids, labels


def train_step(
    model: nn.Module,
    micro_batches: Sequence[Batch],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ctx: ProcessContext,
    config: TrainingConfig,
    *,
    gpu_type: str = "a100",
    data_wait_s: float = 0.0,
) -> StepMetrics:
    """Run one optimizer step over a window of micro-batches.

    Steps: zero grads -> for each micro-batch {(no_sync if not last) forward,
    scaled loss, backward} -> finite check -> clip -> optimizer step ->
    scheduler step -> assemble metrics.

    Args:
        model: The FSDP(+TP)-wrapped model.
        micro_batches: ``grad_accum_steps`` batches, each a mapping with
            ``input_ids`` and ``labels`` of shape ``(micro_bs, seq)``.
        optimizer: The optimizer.
        scheduler: An LR scheduler with ``.step()`` and ``.get_last_lr()``.
        ctx: Process context (device, groups, dims).
        config: The training config.
        gpu_type: GPU key for MFU (default ``"a100"``).
        data_wait_s: Seconds the caller spent fetching this window (stall time).

    Returns:
        A :class:`StepMetrics` with DP-averaged loss, global grad norm, LR,
        throughput, MFU and peak memory.

    Raises:
        GradNotFiniteError: If any rank produced a non-finite gradient.

    Performance note:
        The first ``K-1`` backwards run under ``no_sync()`` (no DP reduce-scatter);
        only the last triggers it. Removing ``no_sync()`` keeps the unsharded
        gradient resident across the whole window — up to ``K``x more grad memory
        and ``K``x the DP communication.
    """
    device = ctx.device
    accum = len(micro_batches)
    if accum == 0:
        raise ValueError("train_step received an empty micro_batch window.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    use_autocast = device.type == "cuda"
    autocast_dt = autocast_dtype(config.parallel)

    summed_mean_loss = torch.zeros((), device=device)
    local_tokens = 0

    for i, batch in enumerate(micro_batches):
        input_ids, labels = _move_batch(batch, device)
        local_tokens += input_ids.numel()
        is_last = i == accum - 1
        with _maybe_no_sync(model, enabled=not is_last):
            with torch.autocast(
                device_type=device.type, dtype=autocast_dt, enabled=use_autocast
            ):
                _, loss = model(input_ids, labels=labels)
            assert loss is not None, "model must return a loss when labels given"
            scaled = loss / accum
            scaled.backward()
        # Track the mean loss over the window (scaled is loss/accum already).
        summed_mean_loss = summed_mean_loss + scaled.detach()

    # Finite check over the WORLD group: a NaN anywhere halts everyone together.
    grad_finite_check(model, group=None, device=device)
    grad_norm = clip_grad_norm_(model, config.max_grad_norm, tp_group=ctx.tp_group)
    optimizer.step()
    scheduler.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_time = time.perf_counter() - t0

    # DP ranks process distinct data; TP ranks process identical data. Global
    # tokens therefore scale by dp_size, NOT world_size.
    global_tokens = local_tokens * ctx.dims.dp_size
    tokens_per_second = compute_throughput(global_tokens, step_time)
    flops_per_token = estimate_flops_per_token(config.model)
    mfu = compute_mfu(
        tokens_per_second, flops_per_token, ctx.world_size, gpu_type
    )
    loss_value = aggregate_loss(summed_mean_loss, ctx.dp_group)

    return StepMetrics(
        loss=loss_value,
        grad_norm=grad_norm,
        learning_rate=float(scheduler.get_last_lr()[0]),
        tokens_per_second=tokens_per_second,
        step_time_s=step_time,
        data_wait_s=data_wait_s,
        mfu=mfu,
        peak_memory_bytes=peak_memory_bytes(device),
    )


@torch.inference_mode()
def eval_step(
    model: nn.Module,
    micro_batches: Sequence[Batch],
    ctx: ProcessContext,
    config: TrainingConfig,
) -> float:
    """Compute the DP-averaged eval loss over a set of micro-batches.

    Uses ``torch.inference_mode`` (stronger than ``no_grad``: also disables
    view-tracking and version counters) so FSDP does not build any backward
    state. Returns the mean loss across the batches and DP ranks.

    Args:
        model: The model (FSDP-wrapped or plain).
        micro_batches: Evaluation batches.
        ctx: Process context.
        config: Training config.

    Returns:
        The DP-averaged evaluation loss (float). Returns ``0.0`` for an empty set.
    """
    if len(micro_batches) == 0:
        return 0.0
    model.eval()
    device = ctx.device
    use_autocast = device.type == "cuda"
    autocast_dt = autocast_dtype(config.parallel)
    total = torch.zeros((), device=device)
    for batch in micro_batches:
        input_ids, labels = _move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dt, enabled=use_autocast
        ):
            _, loss = model(input_ids, labels=labels)
        assert loss is not None
        total = total + loss.detach()
    total = total / len(micro_batches)
    return aggregate_loss(total, ctx.dp_group)
