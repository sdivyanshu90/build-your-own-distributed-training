"""Unit tests for gradient clipping and the cross-rank finiteness check."""

from __future__ import annotations

import math

import pytest
import torch

from src.training.grad_utils import GradNotFiniteError, clip_grad_norm_, grad_finite_check


def _param_with_grad(grad_values: list[float]) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros(len(grad_values)))
    p.grad = torch.tensor(grad_values)
    return p


def test_returns_correct_preclip_norm() -> None:
    # grad = [3, 4] -> L2 norm 5. max_norm large => no clipping, norm reported.
    p = _param_with_grad([3.0, 4.0])
    norm = clip_grad_norm_([p], max_norm=100.0)
    assert math.isclose(norm, 5.0, rel_tol=1e-6)


def test_clips_to_max_norm() -> None:
    # grad norm 2.0 clipped to 1.0 -> resulting norm exactly 1.0.
    p = _param_with_grad([2.0, 0.0])
    pre = clip_grad_norm_([p], max_norm=1.0)
    assert math.isclose(pre, 2.0, rel_tol=1e-6)
    assert p.grad is not None
    assert math.isclose(p.grad.norm().item(), 1.0, rel_tol=1e-6)


def test_noop_when_below_threshold() -> None:
    p = _param_with_grad([0.3, 0.4])  # norm 0.5 < 1.0
    before = p.grad.clone()  # type: ignore[union-attr]
    clip_grad_norm_([p], max_norm=1.0)
    assert torch.allclose(p.grad, before)  # type: ignore[arg-type]


def test_finite_check_passes_for_finite_grads() -> None:
    p = _param_with_grad([1.0, -2.0])
    assert grad_finite_check([p]) is True


def test_finite_check_raises_on_nan() -> None:
    p = _param_with_grad([float("nan"), 1.0])
    with pytest.raises(GradNotFiniteError, match="Non-finite gradient"):
        grad_finite_check([p])


def test_finite_check_raises_on_inf() -> None:
    p = _param_with_grad([float("inf"), 1.0])
    with pytest.raises(GradNotFiniteError):
        grad_finite_check([p])
