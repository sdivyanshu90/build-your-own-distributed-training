"""Helpers for multi-process distributed tests on CPU+Gloo.

Spawns ``world_size`` processes with a Gloo backend (CUDA hidden so a box with a
single GPU still runs functional multi-rank tests), runs a top-level function on
each, and — critically — **propagates child exceptions to the parent** so a
failure on rank 1 surfaces as a test failure rather than a hang (self-review
checklist item). ``torch.multiprocessing.spawn(join=True)`` re-raises a child's
exception as ``ProcessRaisedException`` in the parent, which pytest reports.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp

# A spread of ports so concurrently-running tests do not collide on the rendezvous.
_BASE_PORT = 29600


def _entry(rank: int, world_size: int, port: int, fn: Callable[..., None], args: tuple) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def run_distributed(
    fn: Callable[..., None], world_size: int, *args: Any, port: int | None = None
) -> None:
    """Run ``fn(rank, world_size, *args)`` on ``world_size`` Gloo processes.

    Args:
        fn: A *top-level* (picklable) function; assertions inside it that fail
            propagate to the parent as a test failure.
        world_size: Number of processes to spawn.
        *args: Extra picklable positional args forwarded to ``fn``.
        port: Optional fixed rendezvous port (defaults to a per-world-size value).

    Raises:
        ProcessRaisedException: If any child raises (so the test fails, not hangs).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    chosen_port = port if port is not None else _BASE_PORT + world_size
    mp.spawn(  # type: ignore[no-untyped-call]
        _entry,
        args=(world_size, chosen_port, fn, args),
        nprocs=world_size,
        join=True,
    )
