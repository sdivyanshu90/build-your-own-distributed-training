"""Fault injection: corrupt/incomplete checkpoints are detected, not loaded.

Verifies the validator catches the three failure modes that silently corrupt a
resume: a missing rank shard, truncated tensor data, and a model-config mismatch
between save and load. Uses a plain (single-process) checkpoint so it runs on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.checkpointing.checkpoint import save_checkpoint
from src.checkpointing.recovery import (
    CheckpointValidationError,
    require_valid_checkpoint,
    validate_checkpoint,
)
from src.config import ModelConfig, SchedulerConfig, TrainingConfig
from src.model.transformer import build_model
from src.parallelism.process_groups import build_process_context
from src.training.scheduler import build_scheduler


def _save(tmp: str):
    cfg = TrainingConfig()
    cfg.model = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    cfg.scheduler = SchedulerConfig(warmup_steps=2, max_steps=20, min_lr=1e-4)
    cfg.checkpoint_dir = tmp
    cfg.run_id = "fault"
    cfg.backend = "gloo"
    ctx = build_process_context(tp_size=1, backend="gloo")
    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, cfg.scheduler, 1e-3)
    path = save_checkpoint(model, opt, sched, step=5, config=cfg, ctx=ctx)
    return cfg, path


def test_missing_rank_shard_detected(single_process_pg: None, tmp_path) -> None:
    cfg, path = _save(str(tmp_path))
    # Fake a multi-rank checkpoint by editing meta to claim world_size=2 with no rank_1.
    import json

    meta_path = os.path.join(path, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["world_size"] = 2
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    result = validate_checkpoint(path)
    assert not result.is_valid
    assert any("rank 1" in e for e in result.errors)
    with pytest.raises(CheckpointValidationError):
        require_valid_checkpoint(path)


def test_truncated_tensor_detected(single_process_pg: None, tmp_path) -> None:
    cfg, path = _save(str(tmp_path))
    shard = os.path.join(path, "rank_0", "model.pt")
    # Truncate the file to half its length (simulates a write cut short by a
    # crash); this corrupts the zip archive so torch.load fails.
    size = os.path.getsize(shard)
    with open(shard, "r+b") as f:
        f.truncate(size // 2)
    result = validate_checkpoint(path, deep=True)
    assert not result.is_valid
    assert any("corrupt" in e or "unloadable" in e for e in result.errors)


def test_config_mismatch_detected(single_process_pg: None, tmp_path) -> None:
    cfg, path = _save(str(tmp_path))
    bad = cfg.to_dict()
    bad["model"]["d_model"] = 512  # saved with 32
    result = validate_checkpoint(path, expected_config=bad)
    assert not result.is_valid
    assert any("d_model" in e for e in result.errors)
