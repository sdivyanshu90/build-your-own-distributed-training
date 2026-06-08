"""Atomic, sharded/full checkpoint save & load for FSDP(+TP) training.

What this module does
---------------------
Persists and restores the *complete* training state — model params, optimizer
moments, LR scheduler, global step, per-config metadata, and **per-rank RNG
state** — so a resumed run is bit-identical to one that never stopped.

Layout (namespaced by run_id and step to prevent overwrites)
------------------------------------------------------------
    {checkpoint_dir}/{run_id}/step_{step}/
        meta.json            # step, world/tp/dp sizes, config, format  (rank 0)
        scheduler.pt         # replicated scheduler state               (rank 0)
        rank_{r}/model.pt     # this rank's SHARDED model state
        rank_{r}/optim.pt     # this rank's SHARDED optimizer state
        rank_{r}/rng.pt       # this rank's RNG snapshot
        _SUCCESS             # written last, after a barrier             (rank 0)

Why sharded for training, full for export
-----------------------------------------
``SHARDED_STATE_DICT`` lets every rank write its own ``O(P/dp)`` shard in parallel
— fast, storage-cheap, the only sane option for frequent checkpoints at scale.
``FULL_STATE_DICT`` consolidates everything onto rank 0 (``O(P)`` there) — slow,
but produces a single portable file for inference/export and for resuming on a
*different* topology.

Why atomic writes + a _SUCCESS marker
-------------------------------------
A job killed mid-write must never leave a half-written checkpoint that a later
resume mistakes for valid. Every file is written to ``*.tmp`` then ``os.replace``'d
(atomic on POSIX). The ``_SUCCESS`` marker is written by rank 0 **after a barrier**
guarantees all ranks finished their shards, so its presence certifies the whole
checkpoint is complete. The recovery scanner trusts only marked checkpoints.

Invariant
---------
All ranks save/load the same set of keys over disjoint shards; the scheduler is
identical on every rank (saved once); RNG is per-rank (distinct streams).
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from src.config import TrainingConfig
from src.parallelism.fsdp_utils import (
    get_model_state_dict,
    get_optimizer_state_dict,
    load_model_state_dict,
    load_optimizer_state_dict,
)
from src.parallelism.process_groups import ProcessContext
from src.utils.seed import get_rng_state, set_rng_state

SUCCESS_MARKER = "_SUCCESS"
META_FILE = "meta.json"
SCHEDULER_FILE = "scheduler.pt"


def _atomic_torch_save(obj: Any, path: str) -> None:
    """Serialise ``obj`` to ``path`` atomically (write ``*.tmp`` then rename)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: str) -> None:
    """Write ``text`` to ``path`` atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint_path(config: TrainingConfig, step: int) -> str:
    """Return the directory for ``(run_id, step)`` (namespaced, collision-free)."""
    return os.path.join(config.checkpoint_dir, config.run_id, f"step_{step}")


def save_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: TrainingConfig,
    ctx: ProcessContext,
    *,
    full: bool = False,
) -> str:
    """Save a complete, atomic checkpoint for the current step.

    Args:
        model: The FSDP(+TP)-wrapped model.
        optimizer: The optimizer.
        scheduler: The LR scheduler (replicated; saved by rank 0).
        step: Global optimizer step (embedded in the path and meta).
        config: The training config (serialised into meta for validation).
        ctx: Process context (rank, groups, barrier).
        full: ``False`` => per-rank sharded (training); ``True`` => consolidated
            on rank 0 (export/portable).

    Returns:
        The checkpoint directory path.

    Side effects:
        Creates directories and files under :func:`checkpoint_path`; issues two
        barriers (one before, one after the ``_SUCCESS`` marker) so the marker
        certifies a complete checkpoint.

    Performance note:
        ``full=True`` all-gathers all params to rank 0 and is ``O(P)`` memory and
        time there — use it sparingly (final export), not every ``save_interval``.
    """
    ckpt_dir = checkpoint_path(config, step)
    rank_dir = os.path.join(ckpt_dir, f"rank_{ctx.rank}")
    os.makedirs(rank_dir, exist_ok=True)

    model_sd = get_model_state_dict(model, full=full)
    optim_sd = get_optimizer_state_dict(model, optimizer, full=full)

    if full:
        # Only rank 0 holds the consolidated dicts.
        if ctx.is_rank0:
            _atomic_torch_save(model_sd, os.path.join(ckpt_dir, "model_full.pt"))
            _atomic_torch_save(optim_sd, os.path.join(ckpt_dir, "optim_full.pt"))
    else:
        _atomic_torch_save(model_sd, os.path.join(rank_dir, "model.pt"))
        _atomic_torch_save(optim_sd, os.path.join(rank_dir, "optim.pt"))

    # RNG is always per-rank (distinct stochastic streams across DP ranks).
    _atomic_torch_save(get_rng_state(), os.path.join(rank_dir, "rng.pt"))

    if ctx.is_rank0:
        _atomic_torch_save(scheduler.state_dict(), os.path.join(ckpt_dir, SCHEDULER_FILE))
        meta = {
            "step": step,
            "world_size": ctx.world_size,
            "tp_size": ctx.dims.tp_size,
            "dp_size": ctx.dims.dp_size,
            "format": "full" if full else "sharded",
            "config": config.to_dict(),
        }
        _atomic_write_text(json.dumps(meta, indent=2), os.path.join(ckpt_dir, META_FILE))

    # Barrier: every rank has flushed its shard before the marker is written.
    ctx.barrier()
    if ctx.is_rank0:
        _atomic_write_text(str(step), os.path.join(ckpt_dir, SUCCESS_MARKER))
    # Barrier: no rank proceeds until the marker exists (so concurrent cleanup
    # cannot race a half-marked checkpoint).
    ctx.barrier()
    return ckpt_dir


def load_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    path: str,
    ctx: ProcessContext,
    *,
    full: bool = False,
    restore_rng: bool = True,
) -> int:
    """Restore model, optimizer, scheduler, RNG and step from a checkpoint.

    Args:
        model: The FSDP(+TP)-wrapped model (same architecture as at save time).
        optimizer: The optimizer to repopulate.
        scheduler: The scheduler to restore.
        path: A checkpoint directory (as returned by :func:`save_checkpoint`).
        ctx: Process context.
        full: Must match the save-time format.
        restore_rng: Restore this rank's RNG state (set False to intentionally
            reshuffle on resume).

    Returns:
        The global step stored in ``meta.json`` so the loop resumes correctly.

    Raises:
        FileNotFoundError: If the checkpoint is missing the ``_SUCCESS`` marker or
            this rank's shard.

    Side effects:
        Mutates model/optimizer/scheduler/RNG state in place.
    """
    if not os.path.exists(os.path.join(path, SUCCESS_MARKER)):
        raise FileNotFoundError(
            f"[rank {ctx.rank}] checkpoint at {path} has no {SUCCESS_MARKER} "
            f"marker — it is incomplete or corrupt; refusing to load."
        )
    rank_dir = os.path.join(path, f"rank_{ctx.rank}")

    if full:
        model_sd = torch.load(os.path.join(path, "model_full.pt"), map_location="cpu")
        load_model_state_dict(model, model_sd, full=True)
        optim_sd = torch.load(os.path.join(path, "optim_full.pt"), map_location="cpu")
        load_optimizer_state_dict(model, optimizer, optim_sd, full=True)
    else:
        model_file = os.path.join(rank_dir, "model.pt")
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"[rank {ctx.rank}] missing shard {model_file}; the checkpoint "
                f"was saved with a different world size. Use full=True to load "
                f"the consolidated export, or resume on the original topology."
            )
        model_sd = torch.load(model_file, map_location="cpu")
        load_model_state_dict(model, model_sd, full=False)
        optim_sd = torch.load(os.path.join(rank_dir, "optim.pt"), map_location="cpu")
        load_optimizer_state_dict(model, optimizer, optim_sd, full=False)

    sched_sd = torch.load(os.path.join(path, SCHEDULER_FILE), map_location="cpu")
    scheduler.load_state_dict(sched_sd)

    if restore_rng:
        rng = torch.load(os.path.join(rank_dir, "rng.pt"), map_location="cpu")
        set_rng_state(rng)

    with open(os.path.join(path, META_FILE)) as f:
        meta = json.load(f)
    return int(meta["step"])
