"""Training metrics: loss aggregation, throughput, MFU, memory, perplexity.

What this module does
---------------------
Computes the numbers that tell you whether a run is healthy and efficient, and
packages a step's worth into a :class:`StepMetrics` dataclass. The two
non-trivial metrics are loss aggregation across DP ranks and Model FLOP
Utilization (MFU).

Loss aggregation
----------------
Each DP rank computes the loss on its own micro-batches. The *reported* loss is
the mean across the DP group (``all_reduce`` / ``dp_size``). We reduce over the
**DP group only**, not the world group: TP ranks within a group all compute the
*same* loss (they process the identical input — only weights are sharded), so
including them would just average a value with itself ``tp_size`` times. Reducing
over the wrong group here is a classic, silent 2D-parallel bug (the loss looks
fine but is ``tp_size``x under-counted in the average denominator).

Model FLOP Utilization (MFU)
----------------------------
MFU = (achieved model FLOPs/s) / (peak hardware FLOPs/s). The achieved model
FLOPs use the "6N" rule (2N for the forward matmuls, 4N for backward) plus the
attention score/value matmuls, per nanoGPT::

    flops_per_token = 6 * N + 12 * n_layers * d_model * seq_len

This counts the *useful* model FLOPs. Real hardware does more — activation
recomputation (checkpointing) and TP-replicated embedding/head compute — so
Hardware FLOPs Utilization (HFU) >= MFU. We report MFU because it is the
architecture-independent efficiency number; the docstring of
:func:`estimate_flops_per_token` explains the TP caveat the self-review checklist
calls out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from src.config import ModelConfig

# Peak bf16 TFLOPS (dense, no sparsity) for common accelerators.
PEAK_TFLOPS_BF16: dict[str, float] = {
    "a100": 312.0,
    "a100-80gb": 312.0,
    "h100": 989.0,
    "h100-sxm": 989.0,
    "v100": 125.0,
    "cpu": 1.0,  # placeholder so MFU is defined (meaningless) on CPU test runs
}


@dataclass
class StepMetrics:
    """Per-(optimizer)-step metrics returned by ``train_step``.

    Attributes:
        loss: DP-averaged cross-entropy loss for the step.
        grad_norm: Global pre-clip gradient norm.
        learning_rate: Current LR (first param group).
        tokens_per_second: Global throughput over the step.
        step_time_s: Wall-clock seconds for the step (all micro-batches).
        data_wait_s: Seconds spent waiting on the dataloader (stall time).
        mfu: Model FLOP Utilization in ``[0, 1]`` (``None`` if unknown GPU).
        peak_memory_bytes: ``torch.cuda.max_memory_allocated`` for the step.
    """

    loss: float
    grad_norm: float
    learning_rate: float
    tokens_per_second: float
    step_time_s: float
    data_wait_s: float
    mfu: float | None
    peak_memory_bytes: int

    def perplexity(self) -> float:
        """Exp of the loss, clamped to avoid overflow on early noisy steps."""
        return float(torch.exp(torch.tensor(min(self.loss, 20.0))))


def aggregate_loss(
    local_loss: torch.Tensor, dp_group: dist.ProcessGroup | None
) -> float:
    """Average a scalar loss across the data-parallel group.

    Args:
        local_loss: This rank's scalar loss tensor.
        dp_group: The DP process group (NOT the world or TP group).

    Returns:
        The DP-mean loss as a float.
    """
    val = local_loss.detach().clone()
    if dist.is_initialized() and dp_group is not None and dist.get_world_size(dp_group) > 1:
        dist.all_reduce(val, op=dist.ReduceOp.SUM, group=dp_group)
        val /= dist.get_world_size(dp_group)
    return float(val.item())


def estimate_flops_per_token(config: ModelConfig) -> float:
    """Estimate forward+backward FLOPs per token for MFU.

    Uses ``6N + 12 * n_layers * d_model * seq_len`` where ``N`` is the parameter
    count and the second term is the attention QK^T / AV matmuls (which the 6N
    term does not capture because they are not parameterised).

    Args:
        config: The model config (uses ``num_parameters``, ``n_layers``,
            ``d_model``, ``max_seq_len``).

    Returns:
        Estimated FLOPs per token (a float).

    TP caveat (self-review checklist):
        Under tensor parallelism the embedding and LM head are computed
        *redundantly* on every TP rank, and the TP all_reduce moves bytes that
        are not counted as FLOPs. This function returns the **model** FLOPs (the
        useful work, identical to the non-parallel baseline). Per-GPU hardware
        does strictly more, so measured HFU > MFU; the gap is the TP/recompute
        overhead. We report MFU for comparability across parallelism configs.
    """
    n = config.num_parameters()
    attn = 12 * config.n_layers * config.d_model * config.max_seq_len
    return float(6 * n + attn)


def compute_mfu(
    tokens_per_second: float,
    flops_per_token: float,
    num_gpus: int,
    gpu_type: str,
) -> float | None:
    """Compute Model FLOP Utilization.

    Args:
        tokens_per_second: Global achieved throughput.
        flops_per_token: From :func:`estimate_flops_per_token`.
        num_gpus: Total GPUs (world size).
        gpu_type: Key into :data:`PEAK_TFLOPS_BF16` (case-insensitive).

    Returns:
        MFU in ``[0, 1]``, or ``None`` if the GPU type is unknown (e.g. CPU runs).

    Example:
        >>> # 1.6e5 tok/s * 6e9 flops/tok over 8 A100s (peak 8 * 312e12)
        >>> round(compute_mfu(1.6e5, 6e9, 8, "a100"), 3)
        0.385
    """
    key = gpu_type.lower()
    if key not in PEAK_TFLOPS_BF16:
        return None
    peak = PEAK_TFLOPS_BF16[key] * 1e12 * num_gpus
    achieved = tokens_per_second * flops_per_token
    return achieved / peak


def compute_throughput(num_tokens: int, elapsed_seconds: float) -> float:
    """Tokens per second over an interval (guards divide-by-zero)."""
    if elapsed_seconds <= 0:
        return 0.0
    return num_tokens / elapsed_seconds


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (call at the start of each log window)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int:
    """Return ``max_memory_allocated`` for ``device`` (0 on CPU)."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0
