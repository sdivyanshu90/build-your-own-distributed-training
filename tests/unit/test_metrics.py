"""Unit tests for metrics: MFU, throughput, flops estimate, loss aggregation."""

from __future__ import annotations

import math

import torch

from src.config import ModelConfig
from src.observability.metrics import (
    aggregate_loss,
    compute_mfu,
    compute_throughput,
    estimate_flops_per_token,
)


def test_throughput() -> None:
    assert math.isclose(compute_throughput(1000, 2.0), 500.0)
    assert compute_throughput(1000, 0.0) == 0.0  # guard divide-by-zero


def test_flops_per_token_matches_formula() -> None:
    cfg = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    n = cfg.num_parameters()
    expected = 6 * n + 12 * cfg.n_layers * cfg.d_model * cfg.max_seq_len
    assert estimate_flops_per_token(cfg) == float(expected)


def test_mfu_hand_computed() -> None:
    # 1e5 tok/s * 6e9 flops/tok = 6e14 achieved; 8 A100 peak = 8*312e12 = 2.496e15.
    mfu = compute_mfu(1e5, 6e9, num_gpus=8, gpu_type="a100")
    assert mfu is not None
    assert math.isclose(mfu, 6e14 / (8 * 312e12), rel_tol=1e-9)


def test_mfu_unknown_gpu_returns_none() -> None:
    assert compute_mfu(1e5, 6e9, num_gpus=8, gpu_type="tpu-v9") is None


def test_aggregate_loss_no_group_returns_local() -> None:
    # Without an initialised PG / group, aggregation returns the local value.
    loss = torch.tensor(3.5)
    assert math.isclose(aggregate_loss(loss, dp_group=None), 3.5, rel_tol=1e-6)
