"""Checkpoint validation and latest-valid-checkpoint discovery.

What this module does
---------------------
Decides whether a checkpoint directory is safe to load, and finds the most recent
*valid* one to resume from. This is the difference between a fault-tolerant run
(restart -> resume from the last good checkpoint) and a run that crashes again on
a corrupt checkpoint it should have skipped.

What "valid" means
------------------
A checkpoint is valid iff: the ``_SUCCESS`` marker exists (so the save completed),
``meta.json`` parses, every expected per-rank shard directory is present (for
sharded format) or the consolidated file exists (full format), no shard file is
zero-bytes, and — when ``deep=True`` — every shard actually ``torch.load``s
(catches truncation). Optionally, the saved model config must match the current
config (catches "resume a ``d_model=256`` checkpoint into a ``d_model=512`` model"
before it explodes with a cryptic shape error mid-load).

Why scan rather than trust the newest directory
------------------------------------------------
The newest ``step_*`` directory may be the one the job was writing when it died —
incomplete, unmarked, or truncated. :func:`find_latest_valid_checkpoint` walks
steps newest-first and returns the first that passes validation, so recovery
never picks the half-written one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from src.checkpointing.checkpoint import META_FILE, SUCCESS_MARKER

MODEL_CONFIG_KEYS = (
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "ffn_hidden_size",
    "max_seq_len",
)


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails validation and a load is attempted anyway."""


@dataclass
class CheckpointValidationResult:
    """Outcome of validating one checkpoint directory.

    Attributes:
        path: The checkpoint directory examined.
        is_valid: Whether it is safe to load.
        step: The step from ``meta.json`` (``-1`` if unreadable).
        world_size: Expected world size from meta (``-1`` if unreadable).
        errors: Human-readable reasons it is invalid (empty iff ``is_valid``).
    """

    path: str
    is_valid: bool
    step: int = -1
    world_size: int = -1
    errors: list[str] = field(default_factory=list)


def validate_checkpoint(
    path: str,
    *,
    expected_config: dict[str, Any] | None = None,
    deep: bool = False,
) -> CheckpointValidationResult:
    """Validate a checkpoint directory.

    Args:
        path: The checkpoint directory.
        expected_config: If given, the current model config dict; mismatched
            architecture keys make the checkpoint invalid.
        deep: If True, actually ``torch.load`` each shard to detect truncation /
            corruption (slower but thorough). If False, only structural checks.

    Returns:
        A :class:`CheckpointValidationResult`. Never raises for an invalid
        checkpoint — inspect ``.is_valid`` / ``.errors``.
    """
    errors: list[str] = []
    if not os.path.isdir(path):
        return CheckpointValidationResult(path, False, errors=[f"not a directory: {path}"])
    if not os.path.exists(os.path.join(path, SUCCESS_MARKER)):
        errors.append(f"missing {SUCCESS_MARKER} marker (save incomplete)")

    meta_path = os.path.join(path, META_FILE)
    step = -1
    world_size = -1
    meta: dict[str, Any] = {}
    if not os.path.exists(meta_path):
        errors.append(f"missing {META_FILE}")
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            step = int(meta.get("step", -1))
            world_size = int(meta.get("world_size", -1))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"unreadable {META_FILE}: {exc}")

    fmt = meta.get("format", "sharded")
    if fmt == "sharded" and world_size > 0:
        for r in range(world_size):
            shard = os.path.join(path, f"rank_{r}", "model.pt")
            if not os.path.exists(shard):
                errors.append(f"missing shard for rank {r}: {shard}")
            elif os.path.getsize(shard) == 0:
                errors.append(f"zero-byte shard for rank {r}: {shard}")
            elif deep:
                _deep_check(shard, errors)
    elif fmt == "full":
        full = os.path.join(path, "model_full.pt")
        if not os.path.exists(full):
            errors.append("missing model_full.pt for full-format checkpoint")
        elif os.path.getsize(full) == 0:
            errors.append("zero-byte model_full.pt")
        elif deep:
            _deep_check(full, errors)

    if expected_config is not None and meta.get("config"):
        saved_model = meta["config"].get("model", {})
        for key in MODEL_CONFIG_KEYS:
            want = expected_config.get("model", {}).get(key)
            have = saved_model.get(key)
            if want is not None and have is not None and want != have:
                errors.append(
                    f"model config mismatch on {key!r}: checkpoint={have} "
                    f"current={want}"
                )

    return CheckpointValidationResult(
        path=path,
        is_valid=not errors,
        step=step,
        world_size=world_size,
        errors=errors,
    )


def _deep_check(file_path: str, errors: list[str]) -> None:
    """Attempt to ``torch.load`` a shard, recording corruption in ``errors``."""
    try:
        torch.load(file_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - we want to capture ANY load failure
        errors.append(f"corrupt/unloadable shard {file_path}: {exc!r}")


def find_latest_valid_checkpoint(
    checkpoint_dir: str,
    *,
    expected_config: dict[str, Any] | None = None,
    deep: bool = False,
) -> str | None:
    """Return the newest valid checkpoint path under ``checkpoint_dir``, or None.

    Args:
        checkpoint_dir: A run directory containing ``step_*`` subdirectories.
        expected_config: Forwarded to :func:`validate_checkpoint`.
        deep: Forwarded to :func:`validate_checkpoint`.

    Returns:
        The path of the most recent checkpoint that passes validation, or
        ``None`` if none do. Corrupt/incomplete newer checkpoints are skipped.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    steps: list[tuple[int, str]] = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step_"):
            try:
                steps.append((int(name[len("step_") :]), os.path.join(checkpoint_dir, name)))
            except ValueError:
                continue
    for _, path in sorted(steps, key=lambda t: t[0], reverse=True):
        result = validate_checkpoint(path, expected_config=expected_config, deep=deep)
        if result.is_valid:
            return path
    return None


def require_valid_checkpoint(
    path: str,
    *,
    expected_config: dict[str, Any] | None = None,
    deep: bool = True,
) -> CheckpointValidationResult:
    """Validate and raise :class:`CheckpointValidationError` if invalid.

    Use at load time when a checkpoint *must* be good (e.g. an explicit
    ``resume_from``): fails loudly with all reasons rather than half-loading.
    """
    result = validate_checkpoint(path, expected_config=expected_config, deep=deep)
    if not result.is_valid:
        raise CheckpointValidationError(
            f"Checkpoint {path} is invalid:\n  - " + "\n  - ".join(result.errors)
        )
    return result
