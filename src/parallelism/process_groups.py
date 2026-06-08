"""Distributed bootstrap and the per-rank :class:`ProcessContext`.

What this module does
---------------------
Initialises the default process group from the torchrun environment, builds the
2D mesh, and packages everything a rank needs to know about its place in the job
into one immutable :class:`ProcessContext`. Passing this object explicitly (never
reading rank/world_size from globals) is how we satisfy the *no global state*
requirement: any function that needs to communicate takes a ``ProcessContext``
and asks it for the correct group.

Why a context object instead of module globals
-----------------------------------------------
The classic distributed-training bug is calling ``dist.all_reduce(x)`` with the
*default* (world) group when you meant the TP group, because the rank was read
from an ambient global. A context object forces the call site to name the group
(``ctx.tp_group`` vs ``ctx.dp_group``), which is exactly the distinction the
self-review checklist flags as "the most common and hardest-to-debug error in 2D
parallelism." It also makes the code testable: a unit test can construct a
fake context without a real launcher.

Invariants
----------
  * Exactly one default process group per process, created once here.
  * ``ctx.dims.world_size == ctx.dims.tp_size * ctx.dims.dp_size``.
  * ``ctx.is_rank0`` is true on exactly one global rank (logging gate).
  * ``ctx.is_dp_rank0`` is true on one rank per TP group; used when an artifact
    should be written once per TP group (e.g. TP-sharded checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from src.parallelism.mesh import (
    ParallelDims,
    build_device_mesh,
    format_mesh_layout,
    get_dp_group,
    get_parallel_dims,
    get_tp_group,
)
from src.utils.env import LaunchEnv, read_launch_env, select_device


@dataclass(frozen=True)
class ProcessContext:
    """Everything a rank needs to address the job's process groups.

    Attributes:
        env: The validated launch environment.
        device: This rank's compute device.
        mesh: The 2D ``(dp, tp)`` device mesh.
        dims: Resolved sizes + this rank's coordinates.
        dp_group: Data-parallel (FSDP) process group.
        tp_group: Tensor-parallel process group.
    """

    env: LaunchEnv
    device: torch.device
    mesh: DeviceMesh
    dims: ParallelDims
    dp_group: dist.ProcessGroup
    tp_group: dist.ProcessGroup

    @property
    def rank(self) -> int:
        return self.dims.global_rank

    @property
    def world_size(self) -> int:
        return self.dims.world_size

    @property
    def is_rank0(self) -> bool:
        """True on the single global rank-0 process (the logging gate)."""
        return self.dims.global_rank == 0

    @property
    def is_dp_rank0(self) -> bool:
        """True once per TP group (``dp_rank == 0``).

        Used for artifacts that are replicated across the DP axis but sharded
        across TP — e.g. each TP shard of a checkpoint is written by the
        ``dp_rank==0`` member of its TP group to avoid ``dp_size`` redundant
        writes of identical bytes.
        """
        return self.dims.dp_rank == 0

    def barrier(self) -> None:
        """Synchronise all ranks on the default (world) group."""
        if dist.is_initialized():
            dist.barrier()


def init_distributed(
    backend: str = "nccl",
    timeout_seconds: int = 1800,
) -> LaunchEnv:
    """Initialise the default process group from the torchrun environment.

    Idempotent: if the default group is already initialised (e.g. a test set it
    up) this returns the current environment without re-initialising.

    Args:
        backend: ``"nccl"`` (GPU) or ``"gloo"`` (CPU fallback for tests).
        timeout_seconds: Collective timeout. A generous default (30 min) avoids
            spurious timeouts during slow checkpoint loads, but is finite so a
            genuinely hung rank eventually raises instead of blocking forever.

    Returns:
        The :class:`LaunchEnv` for this process.

    Raises:
        RuntimeError: From :func:`read_launch_env` if the backend/CUDA state is
            inconsistent.
    """
    env = read_launch_env(backend=backend)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=env.world_size,
            rank=env.rank,
            timeout=timedelta(seconds=timeout_seconds),
        )
    return env


def build_process_context(
    tp_size: int,
    dp_size: int = -1,
    backend: str = "nccl",
) -> ProcessContext:
    """Bootstrap distributed and assemble the :class:`ProcessContext`.

    This is the single entry point the trainer calls. It initialises the PG,
    selects the device, builds the mesh, derives the rank coordinates, fetches
    the per-axis groups, and logs the mesh layout from rank 0.

    Args:
        tp_size: Tensor-parallel degree.
        dp_size: Data-parallel degree, or ``-1`` to infer.
        backend: Collective backend.

    Returns:
        A fully-populated :class:`ProcessContext`.

    Side effects:
        Initialises the default process group (if needed) and, on CUDA, sets the
        current device. Prints the mesh layout from rank 0.
    """
    env = init_distributed(backend=backend)
    device = select_device(env)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    mesh = build_device_mesh(tp_size=tp_size, dp_size=dp_size, device_type=device_type)
    dims = get_parallel_dims(mesh)
    ctx = ProcessContext(
        env=env,
        device=device,
        mesh=mesh,
        dims=dims,
        dp_group=get_dp_group(mesh),
        tp_group=get_tp_group(mesh),
    )
    if ctx.is_rank0:
        print(format_mesh_layout(mesh), flush=True)
    return ctx


def destroy_distributed() -> None:
    """Tear down the default process group if it exists.

    Always call this at process exit (including on exception paths) so NCCL
    communicators are released; a leaked communicator can wedge the next job on
    the same GPUs.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
