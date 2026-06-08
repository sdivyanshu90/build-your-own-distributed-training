"""Unit tests for the hand-rolled column/row parallel linears.

Single-process (world_size=1) tests: over a degenerate 1-rank TP group every
collective is an identity, so these validate the *layer algebra* and autograd
wiring. The cross-rank numerical-equivalence behaviour (sharded weights summing
to the reference) is covered by the multi-process integration tests; here we
verify shapes, exact equivalence to ``nn.Linear`` at tp_size=1, and that the
column->row composition reproduces a two-layer MLP.
"""

from __future__ import annotations

import torch

from src.parallelism.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_shapes(single_process_pg: None) -> None:
    layer = ColumnParallelLinear(8, 16, tp_group=None, bias=True)
    # tp_size==1 => owns all output columns.
    assert layer.weight.shape == (16, 8)
    assert layer.out_per_partition == 16


def test_row_parallel_shapes(single_process_pg: None) -> None:
    layer = RowParallelLinear(16, 8, tp_group=None, bias=True)
    assert layer.weight.shape == (8, 16)
    assert layer.in_per_partition == 16


def test_column_gather_matches_nn_linear(single_process_pg: None) -> None:
    torch.manual_seed(0)
    ref = torch.nn.Linear(8, 16)
    col = ColumnParallelLinear(8, 16, tp_group=None, gather_output=True, bias=True)
    col.weight.data.copy_(ref.weight.data)
    col.bias.data.copy_(ref.bias.data)
    x = torch.randn(4, 8)
    assert torch.allclose(ref(x), col(x), atol=1e-6)


def test_column_then_row_matches_two_linears(single_process_pg: None) -> None:
    torch.manual_seed(1)
    ref1 = torch.nn.Linear(8, 16)
    ref2 = torch.nn.Linear(16, 8)
    col = ColumnParallelLinear(8, 16, tp_group=None, gather_output=False, bias=True)
    row = RowParallelLinear(16, 8, tp_group=None, input_is_parallel=True, bias=True)
    col.weight.data.copy_(ref1.weight.data)
    col.bias.data.copy_(ref1.bias.data)
    row.weight.data.copy_(ref2.weight.data)
    row.bias.data.copy_(ref2.bias.data)
    x = torch.randn(4, 8)
    assert torch.allclose(ref2(ref1(x)), row(col(x)), atol=1e-5)


def test_column_backward_matches_nn_linear(single_process_pg: None) -> None:
    torch.manual_seed(2)
    ref = torch.nn.Linear(8, 16)
    col = ColumnParallelLinear(8, 16, tp_group=None, gather_output=True, bias=True)
    col.weight.data.copy_(ref.weight.data)
    col.bias.data.copy_(ref.bias.data)
    x1 = torch.randn(4, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ref(x1).sum().backward()
    col(x2).sum().backward()
    assert x1.grad is not None and x2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
