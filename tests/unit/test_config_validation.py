"""Config cross-field validation (no process group needed)."""

from __future__ import annotations

import pytest

from src.config import ModelConfig, TrainingConfig


def _cfg(tp: int, **model_kw) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_hidden_size=128, max_seq_len=32, **model_kw,
    )
    cfg.parallel.tp_size = tp
    cfg.scheduler.max_steps = cfg.max_steps
    cfg.data.global_batch_size = 0
    return cfg


def test_valid_tp_passes() -> None:
    _cfg(2).validate(world_size=2, dp_size=1)


def test_tp_must_divide_kv_heads() -> None:
    # n_kv_heads=2 cannot be split over 4 TP ranks (previously a cryptic DTensor error).
    with pytest.raises(ValueError, match="n_kv_heads"):
        _cfg(4).validate(world_size=4, dp_size=1)


def test_tp_must_divide_ffn() -> None:
    cfg = _cfg(2)
    cfg.model.ffn_hidden_size = 131
    with pytest.raises(ValueError, match="ffn_hidden_size"):
        cfg.validate(world_size=2, dp_size=1)


def test_gqa_group_must_divide() -> None:
    cfg = _cfg(1)
    cfg.model.n_kv_heads = 3
    with pytest.raises(ValueError, match="GQA"):
        cfg.validate(world_size=1, dp_size=1)


def test_scheduler_horizon_mismatch_warns() -> None:
    cfg = _cfg(1)
    cfg.scheduler.max_steps = cfg.max_steps + 5
    with pytest.warns(UserWarning, match="scheduler.max_steps"):
        cfg.validate(world_size=1, dp_size=1)
