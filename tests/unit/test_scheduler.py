"""Unit tests for the cosine-with-warmup LR schedule."""

from __future__ import annotations

import math

import torch

from src.config import SchedulerConfig
from src.training.scheduler import build_scheduler, lr_lambda_factory


def _make(peak: float = 1.0, warmup: int = 10, max_steps: int = 100, min_lr: float = 0.1):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=peak)
    sched = build_scheduler(
        opt, SchedulerConfig(warmup_steps=warmup, max_steps=max_steps, min_lr=min_lr), peak
    )
    return opt, sched


def test_linear_warmup_is_monotonic() -> None:
    opt, sched = _make(peak=1.0, warmup=10, max_steps=100, min_lr=0.0)
    prev = -1.0
    for _ in range(10):
        lr = opt.param_groups[0]["lr"]
        assert lr > prev, "LR must strictly increase during warmup"
        prev = lr
        sched.step()


def test_cosine_decay_formula() -> None:
    peak, warmup, max_steps, min_lr = 2.0, 5, 25, 0.2
    fn = lr_lambda_factory(peak, warmup, max_steps, min_lr)
    step = 15  # midpoint-ish of the cosine
    progress = (step - warmup) / (max_steps - warmup)
    expected_mult = (min_lr / peak) + (1 - min_lr / peak) * 0.5 * (1 + math.cos(math.pi * progress))
    assert math.isclose(fn(step), expected_mult, rel_tol=1e-9)


def test_reaches_min_lr_at_end() -> None:
    opt, sched = _make(peak=1.0, warmup=5, max_steps=20, min_lr=0.05)
    for _ in range(20):
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.05, rel_tol=1e-6)


def test_resume_reproduces_lr() -> None:
    # Continuous run to step 15.
    opt_a, sched_a = _make(peak=1.0, warmup=5, max_steps=30, min_lr=0.1)
    for _ in range(15):
        sched_a.step()
    lr_continuous = opt_a.param_groups[0]["lr"]

    # Run to 8, snapshot, restore into a fresh scheduler, continue to 15.
    opt_b, sched_b = _make(peak=1.0, warmup=5, max_steps=30, min_lr=0.1)
    for _ in range(8):
        sched_b.step()
    state = sched_b.state_dict()
    opt_c, sched_c = _make(peak=1.0, warmup=5, max_steps=30, min_lr=0.1)
    sched_c.load_state_dict(state)
    for _ in range(7):
        sched_c.step()
    assert math.isclose(opt_c.param_groups[0]["lr"], lr_continuous, rel_tol=1e-9)
