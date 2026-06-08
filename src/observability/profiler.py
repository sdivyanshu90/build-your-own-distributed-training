"""``torch.profiler`` integration: scoped tracing + comm/compute breakdown.

What this module does
---------------------
Wraps ``torch.profiler.profile`` so a run profiles only a *window* of steps (not
the whole job — a full-run trace is gigabytes and perturbs timing), exports a
Chrome trace + a text summary, and computes the fraction of step time spent in
communication vs. compute.

Why a scoped schedule
---------------------
The first few steps include lazy allocation, autotuning and cuDNN benchmark
warmup; their timings are not representative. We ``wait`` past the noise, ``warmup``
the profiler, then ``active``-record exactly ``profile_steps`` steps. This yields a
small, faithful trace centred on steady-state.

Reading the output
------------------
``trace.json`` opens in ``chrome://tracing`` / Perfetto and shows the
CPU/CUDA/NCCL timeline — you can see whether all-gathers overlap compute. The text
summary ranks ops by CUDA time. :func:`communication_fraction` distils the single
most important number: if comm is a large fraction of wall-clock, prefetch/overlap
is failing and throughput is comm-bound.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

# Substrings identifying NCCL/collective kernels in profiler op names.
_COMM_OP_HINTS = ("nccl", "all_reduce", "allreduce", "all_gather", "allgather",
                  "reduce_scatter", "reducescatter", "broadcast", "c10d")


@dataclass
class CommComputeBreakdown:
    """Communication-vs-compute timing split from a profiler trace.

    Attributes:
        total_cuda_us: Total self CUDA time across recorded ops (microseconds).
        comm_cuda_us: CUDA time attributed to collective/NCCL ops.
        compute_cuda_us: ``total - comm``.
        comm_fraction: ``comm / total`` in ``[0, 1]`` (0 if no CUDA time).
    """

    total_cuda_us: float
    comm_cuda_us: float
    compute_cuda_us: float
    comm_fraction: float


def build_profiler(
    output_dir: str,
    warmup_steps: int,
    profile_steps: int,
    *,
    record_memory: bool = True,
) -> profile:
    """Construct a scoped ``torch.profiler.profile``.

    Args:
        output_dir: Directory for the trace artifacts (created if absent).
        warmup_steps: Steps to skip before recording (``wait`` + ``warmup``).
        profile_steps: Number of steps to actively record.
        record_memory: Capture memory-allocation events (adds overhead).

    Returns:
        A ``profile`` object. Use it as a context manager and call ``.step()``
        once per training step so the schedule advances.

    Example:
        >>> prof = build_profiler("/tmp/prof", warmup_steps=2, profile_steps=3)
        >>> with prof:                       # doctest: +SKIP
        ...     for _ in range(6):
        ...         ...                      # one training step
        ...         prof.step()
    """
    os.makedirs(output_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    # wait long enough to clear startup noise, warmup 1, then record the window.
    wait = max(0, warmup_steps - 1)
    prof_schedule = schedule(wait=wait, warmup=1, active=profile_steps, repeat=1)
    return profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=record_memory,
        with_stack=False,
    )


def export_chrome_trace(prof: profile, path: str) -> None:
    """Export the captured trace to a Chrome/Perfetto JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prof.export_chrome_trace(path)


def text_summary(prof: profile, row_limit: int = 20) -> str:
    """Return a text table of the top ops by self CUDA (or CPU) time."""
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    return prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)


def communication_fraction(prof: profile) -> CommComputeBreakdown:
    """Compute the communication share of CUDA time from a finished profile.

    Args:
        prof: A ``profile`` that has recorded at least one active step.

    Returns:
        A :class:`CommComputeBreakdown`. On CPU-only runs (no CUDA events) all
        fields are 0 and ``comm_fraction`` is 0.

    Performance note:
        A high ``comm_fraction`` (say > 0.3) means collectives are not hidden
        behind compute — check ``forward_prefetch`` / ``BACKWARD_PRE`` and
        whether the TP group spans a slow (inter-node) link.
    """
    total = 0.0
    comm = 0.0
    for evt in prof.key_averages():
        cuda_us = float(getattr(evt, "self_cuda_time_total", 0.0))
        total += cuda_us
        name = evt.key.lower()
        if any(hint in name for hint in _COMM_OP_HINTS):
            comm += cuda_us
    compute = total - comm
    fraction = (comm / total) if total > 0 else 0.0
    return CommComputeBreakdown(
        total_cuda_us=total,
        comm_cuda_us=comm,
        compute_cuda_us=compute,
        comm_fraction=fraction,
    )
