"""Unit tests for gradient-accumulation correctness (the no_sync semantics).

These tests verify the *math* of accumulation without a real FSDP runtime: a
plain module accumulating K micro-batch gradients must equal the gradient of the
concatenated batch (scaled), and a mocked ``no_sync`` (deferred reduce-scatter)
must not change the accumulated result. The distributed equivalence under real
FSDP is covered by the integration suite.
"""

from __future__ import annotations

import contextlib

import torch

from src.training.loop import _maybe_no_sync


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_accumulated_grad_equals_full_batch_grad() -> None:
    torch.manual_seed(0)
    model = _Model()
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)

    # Full batch (mean loss) gradient.
    model.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    full_grad = model.lin.weight.grad.clone()

    # Accumulate over 4 micro-batches of size 2, each loss divided by accum.
    model.zero_grad()
    accum = 4
    micro = x.shape[0] // accum
    for i in range(accum):
        xb = x[i * micro : (i + 1) * micro]
        yb = y[i * micro : (i + 1) * micro]
        # mse over micro-batch == mean; dividing by accum gives the full-batch mean.
        (torch.nn.functional.mse_loss(model(xb), yb) / accum).backward()
    acc_grad = model.lin.weight.grad.clone()

    assert torch.allclose(full_grad, acc_grad, atol=1e-6)


def test_no_sync_does_not_change_accumulated_grad() -> None:
    """A mock 'no_sync' (just defers a hypothetical reduce) must not alter grads."""
    torch.manual_seed(1)

    class _NoSyncModel(_Model):
        @contextlib.contextmanager
        def no_sync(self):  # type: ignore[no-untyped-def]
            yield  # deferring the (here absent) gradient sync is a no-op on grads

    model = _NoSyncModel()
    x = torch.randn(6, 4)
    y = torch.randn(6, 1)
    accum = 3
    micro = 2

    def run(use_no_sync: bool) -> torch.Tensor:
        model.zero_grad()
        for i in range(accum):
            xb = x[i * micro : (i + 1) * micro]
            yb = y[i * micro : (i + 1) * micro]
            is_last = i == accum - 1
            ctx = _maybe_no_sync(model, enabled=use_no_sync and not is_last)
            with ctx:
                (torch.nn.functional.mse_loss(model(xb), yb) / accum).backward()
        return model.lin.weight.grad.clone()

    with_ns = run(True)
    without_ns = run(False)
    assert torch.allclose(with_ns, without_ns, atol=1e-7)


def test_loss_division_by_accum() -> None:
    # The reported per-window loss is the mean of (micro_loss / accum) == mean loss.
    losses = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(6.0)]
    accum = len(losses)
    summed = sum(loss_val / accum for loss_val in losses)
    assert torch.isclose(summed, torch.tensor(4.0))  # mean of [2,4,6]
