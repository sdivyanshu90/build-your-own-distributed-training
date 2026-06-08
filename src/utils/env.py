"""Environment & distributed-launch validation.

What this module does
---------------------
Reads the ``torchrun``-provided environment (``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR`` ...), validates that CUDA/NCCL are usable for the
requested backend, and returns a small immutable :class:`LaunchEnv` describing
the process's place in the job. Doing this once, loudly, at startup turns a
class of confusing mid-training hangs (mismatched world size, missing
``CUDA_VISIBLE_DEVICES``, NCCL version skew) into a clear pre-flight error.

Why validate eagerly
--------------------
The worst distributed failures are silent: rank 3 was launched with the wrong
``WORLD_SIZE`` and every collective now waits forever for a participant that
will never arrive. NCCL's default behaviour is to block, not error. We assert the
invariants we can check locally *before* the first collective so the failure has
a stack trace and a rank attached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LaunchEnv:
    """Immutable snapshot of the distributed launch environment.

    Attributes:
        rank: Global rank in ``[0, world_size)``.
        world_size: Total number of processes in the job.
        local_rank: Rank within this node; selects the GPU.
        local_world_size: Processes on this node (GPUs per node).
        master_addr / master_port: Rendezvous endpoint.
        backend: ``"nccl"`` or ``"gloo"``.
    """

    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    master_addr: str
    master_port: str
    backend: str


def read_launch_env(backend: str = "nccl") -> LaunchEnv:
    """Read and validate the torchrun environment for the calling process.

    Falls back to a single-process default when the torchrun variables are
    absent, so unit tests and ``python train.py`` (no ``torchrun``) still work.

    Args:
        backend: Requested collective backend. ``"nccl"`` requires CUDA.

    Returns:
        A populated :class:`LaunchEnv`.

    Raises:
        RuntimeError: If ``backend == "nccl"`` but CUDA is unavailable, or if
            ``LOCAL_RANK`` indexes a GPU that does not exist. The message names
            the rank and the observed device count for fast diagnosis.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[rank {rank}] backend='nccl' requested but torch.cuda."
                f"is_available() is False. Use backend='gloo' for CPU runs."
            )
        n_devices = torch.cuda.device_count()
        if local_rank >= n_devices:
            raise RuntimeError(
                f"[rank {rank}] LOCAL_RANK={local_rank} >= device_count="
                f"{n_devices}. Each local rank needs its own GPU; check "
                f"CUDA_VISIBLE_DEVICES and nproc_per_node."
            )

    return LaunchEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
    )


def select_device(env: LaunchEnv) -> torch.device:
    """Pick (and for CUDA, set) the device for this rank.

    Args:
        env: The launch environment.

    Returns:
        ``cuda:{local_rank}`` when the backend is NCCL, else ``cpu``.

    Side effects:
        Calls ``torch.cuda.set_device`` so subsequent allocations and NCCL
        communicators bind to the right GPU.
    """
    if env.backend == "nccl" and torch.cuda.is_available():
        device = torch.device(f"cuda:{env.local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def describe_versions() -> dict[str, str]:
    """Return a dict of relevant library versions for the run header log."""
    versions = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "n/a",
    }
    if torch.cuda.is_available():
        versions["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
        versions["gpu"] = torch.cuda.get_device_name(0)
    return versions
