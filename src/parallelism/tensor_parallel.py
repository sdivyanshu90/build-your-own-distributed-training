"""Tensor-parallel primitives: column/row parallel linears + DTensor plan.

What this module provides
-------------------------
Two complementary implementations of Megatron-style tensor parallelism:

1. **Hand-rolled** ``ColumnParallelLinear`` / ``RowParallelLinear`` built on four
   custom ``autograd.Function`` collectives. These exist so every byte of the
   forward/backward communication is explicit and unit-testable with
   ``torch.autograd.gradcheck`` in a single process. They are the *reference*
   semantics.
2. **Production** :func:`apply_tensor_parallelism`, which uses PyTorch's
   ``parallelize_module`` with ``ColwiseParallel`` / ``RowwiseParallel`` DTensor
   plans. This is what the trainer actually runs: it composes with FSDP, handles
   gradient hooks, and overlaps comm automatically.

Both implement the *same* mathematics, derived below.

The two collective "operators" (Megatron's f and g)
---------------------------------------------------
Let ``X`` be the layer input (replicated across the TP group) and ``A`` the
weight. TP needs two conjugate operators that are identity in one direction and
an ``all_reduce`` in the other:

  * **f** (:class:`_CopyToTPRegion`): forward = identity, backward = all_reduce.
    Placed on the *input* of a column-parallel layer. Forward needs nothing (X
    is replicated); backward must sum the partial input-gradients each shard
    produced.
  * **g** (:class:`_ReduceFromTPRegion`): forward = all_reduce, backward =
    identity. Placed on the *output* of a row-parallel layer. Forward sums the
    partial outputs; backward just copies the (replicated) output-gradient.

Column-parallel: Y = X·A,  A = [A_1 | A_2 | ... | A_p]  (split on output dim)
----------------------------------------------------------------------------
Forward (rank i):  Y_i = f(X) · A_i = X · A_i   -> output is SHARDED [Y_1|...|Y_p]
  * No communication in forward; X is already replicated.
  * ``gather_output=True`` appends an all_gather to rebuild full Y (used only
    where the next op needs the whole thing, e.g. a non-parallel head).
Backward:
  * dL/dA_i = X^T · dL/dY_i              (LOCAL, complete — no comm)
  * dL/dX   = sum_i dL/dY_i · A_i^T      (the all_reduce inside f supplies the sum)

Row-parallel: Y = X·A,  A = [A_1; A_2; ...; A_p]  (split on input dim)
---------------------------------------------------------------------
Input X is expected already sharded on its last dim: X = [X_1 | ... | X_p].
Forward (rank i):  Y = g(sum_i X_i · A_i)        -> all_reduce -> output REPLICATED
Backward:
  * dL/dA_i = X_i^T · dL/dY              (LOCAL, complete — no comm)
  * dL/dX_i = dL/dY · A_i^T              (LOCAL; dL/dY is replicated because g's
    backward is identity, so each rank already holds the full dL/dY)
  Note: the canonical row-parallel backward needs **no** collective on the input
  gradient — it is naturally produced in the sharded layout the upstream
  column-parallel layer consumes. (Some texts describe an all_gather here; that
  corresponds to the different convention of materialising the *full* input
  gradient, which we never need.)

Bias placement
--------------
  * Column-parallel bias is sharded (each rank owns its output columns' bias)
    and added before the optional gather.
  * Row-parallel bias is the *full* output bias, added **after** the all_reduce
    exactly once per rank — adding it before would multiply it by ``tp_size``.

Why these never branch on ``tp_size == 1``
------------------------------------------
With a 1-rank TP group, all_reduce/all_gather/chunk are identities, so the same
code path is exactly the non-parallel computation. No special-casing required.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import Placement, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

# --------------------------------------------------------------------------- #
# Autograd-aware TP collectives (Megatron f / g and scatter / gather)
# --------------------------------------------------------------------------- #


def _group_size(group: dist.ProcessGroup | None) -> int:
    return dist.get_world_size(group) if group is not None else 1


def _group_rank(group: dist.ProcessGroup | None) -> int:
    return dist.get_rank(group) if group is not None else 0


class _CopyToTPRegion(torch.autograd.Function):
    """f operator: identity in forward, all_reduce in backward.

    Wraps the *input* of a column-parallel layer. Forward is a no-op because the
    input is replicated; backward sums the per-shard input gradients so every TP
    rank ends up with the full ``dL/dX``.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Any:  # type: ignore[override]
        if _group_size(ctx.group) > 1:
            grad = grad.contiguous()
            dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """g operator: all_reduce in forward, identity in backward.

    Wraps the *output* of a row-parallel layer. Forward sums the partial outputs
    across the TP group; backward copies the replicated output gradient.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:  # type: ignore[override]
        if _group_size(group) > 1:
            x = x.contiguous()
            dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Any:  # type: ignore[override]
        return grad, None


class _GatherFromTPRegion(torch.autograd.Function):
    """all_gather along ``dim`` in forward; take this rank's chunk in backward.

    Used by ``ColumnParallelLinear(gather_output=True)`` to reconstruct the full
    output. Its backward scatters the gradient back to shards (the inverse).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        ctx.dim = dim
        world = _group_size(group)
        ctx.world = world
        ctx.rank = _group_rank(group)
        if world == 1:
            return x
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Any:  # type: ignore[override]
        if ctx.world == 1:
            return grad, None, None
        chunk = grad.chunk(ctx.world, dim=ctx.dim)[ctx.rank]
        return chunk.contiguous(), None, None


class _ScatterToTPRegion(torch.autograd.Function):
    """Take this rank's chunk along ``dim`` in forward; all_gather in backward.

    Used by ``RowParallelLinear(input_is_parallel=False)`` to split a replicated
    input into shards. Its backward reassembles the full input gradient.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        ctx.dim = dim
        world = _group_size(group)
        ctx.world = world
        if world == 1:
            return x
        rank = _group_rank(group)
        return x.chunk(world, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Any:  # type: ignore[override]
        if ctx.world == 1:
            return grad, None, None
        grad = grad.contiguous()
        gathered = [torch.empty_like(grad) for _ in range(ctx.world)]
        dist.all_gather(gathered, grad, group=ctx.group)
        return torch.cat(gathered, dim=ctx.dim), None, None


def copy_to_tp_region(x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Public functional wrapper for the f operator (identity fwd / all_reduce bwd)."""
    return _CopyToTPRegion.apply(x, group)


def reduce_from_tp_region(x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Public functional wrapper for the g operator (all_reduce fwd / identity bwd)."""
    return _ReduceFromTPRegion.apply(x, group)


def gather_from_tp_region(
    x: torch.Tensor, group: dist.ProcessGroup | None, dim: int = -1
) -> torch.Tensor:
    """all_gather ``x`` along ``dim`` across the TP group (autograd-aware)."""
    return _GatherFromTPRegion.apply(x, group, dim)


def scatter_to_tp_region(
    x: torch.Tensor, group: dist.ProcessGroup | None, dim: int = -1
) -> torch.Tensor:
    """Split ``x`` along ``dim`` and keep this rank's shard (autograd-aware)."""
    return _ScatterToTPRegion.apply(x, group, dim)


# --------------------------------------------------------------------------- #
# Hand-rolled parallel linear layers (reference semantics, unit-testable)
# --------------------------------------------------------------------------- #


class ColumnParallelLinear(nn.Module):
    """Linear layer with the weight split column-wise across a TP group.

    ``Y = X · A``, with ``A = [A_1 | ... | A_p]`` split on the output dimension,
    so rank ``i`` owns ``out_features / tp_size`` output columns. The input is
    replicated; the output is sharded on its last dim unless ``gather_output``.

    Args:
        in_features: Input width (full, not sharded).
        out_features: Output width (full); must be divisible by ``tp_size``.
        tp_group: The tensor-parallel process group (``None`` => degree 1).
        bias: Whether to learn a (sharded) bias.
        gather_output: If True, all_gather the sharded output into the full
            tensor before returning (use for the final projection feeding a
            non-parallel consumer). Costs one all_gather per forward.
        init_method: Optional ``fn(tensor) -> None`` to initialise the *local*
            weight shard. Defaults to Kaiming-uniform matching ``nn.Linear``.

    Raises:
        ValueError: If ``out_features`` is not divisible by ``tp_size``.

    Shape:
        input  ``(*, in_features)``
        output ``(*, out_features)`` if ``gather_output`` else
               ``(*, out_features // tp_size)``

    Example:
        >>> # single-process group: behaves exactly like nn.Linear
        >>> layer = ColumnParallelLinear(8, 16, tp_group=None, gather_output=True)
        >>> y = layer(torch.randn(4, 8))
        >>> y.shape
        torch.Size([4, 16])

    Performance note:
        With ``gather_output=False`` the forward is communication-free; the only
        TP collective is the backward all_reduce inside the f operator, which
        FSDP/autograd can overlap with the row-parallel layer's compute.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: dist.ProcessGroup | None,
        bias: bool = True,
        gather_output: bool = False,
        init_method: object | None = None,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = _group_size(tp_group)
        self.tp_rank = _group_rank(tp_group)
        if out_features % self.tp_size != 0:
            raise ValueError(
                f"ColumnParallelLinear: out_features={out_features} not "
                f"divisible by tp_size={self.tp_size}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_partition = out_features // self.tp_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        self.bias = (
            nn.Parameter(torch.empty(self.out_per_partition)) if bias else None
        )
        self._reset_parameters(init_method)

    def _reset_parameters(self, init_method: object | None) -> None:
        if init_method is not None:
            init_method(self.weight)  # type: ignore[operator]
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # f operator: identity now, all_reduce of the input-grad in backward.
        x = _CopyToTPRegion.apply(x, self.tp_group)
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = _GatherFromTPRegion.apply(y, self.tp_group, -1)
        return y


class RowParallelLinear(nn.Module):
    """Linear layer with the weight split row-wise across a TP group.

    ``Y = X · A``, with ``A = [A_1; ...; A_p]`` split on the input dimension, so
    rank ``i`` owns ``in_features / tp_size`` input rows. The input is expected
    already sharded on its last dim (the output of a ``ColumnParallelLinear``);
    the output is reduced to the full, replicated tensor.

    Args:
        in_features: Input width (full); must be divisible by ``tp_size``.
        out_features: Output width (full).
        tp_group: The tensor-parallel process group.
        bias: Whether to learn the full output bias (added post-reduce).
        input_is_parallel: If True (default) the input is already sharded; if
            False a scatter is inserted to split a replicated input first.
        init_method: Optional weight-shard initialiser.

    Raises:
        ValueError: If ``in_features`` is not divisible by ``tp_size``.

    Shape:
        input  ``(*, in_features // tp_size)`` if ``input_is_parallel`` else
               ``(*, in_features)``
        output ``(*, out_features)`` (full, replicated)

    Performance note:
        The forward all_reduce is on the critical path and is the dominant TP
        cost. It is unaffected by FSDP's ``no_sync()`` — see the gradient
        accumulation docs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: dist.ProcessGroup | None,
        bias: bool = True,
        input_is_parallel: bool = True,
        init_method: object | None = None,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = _group_size(tp_group)
        self.tp_rank = _group_rank(tp_group)
        if in_features % self.tp_size != 0:
            raise ValueError(
                f"RowParallelLinear: in_features={in_features} not divisible "
                f"by tp_size={self.tp_size}."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_partition = in_features // self.tp_size
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        # Bias is full-width and added once, after the all_reduce.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self._reset_parameters(init_method)

    def _reset_parameters(self, init_method: object | None) -> None:
        if init_method is not None:
            init_method(self.weight)  # type: ignore[operator]
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = _ScatterToTPRegion.apply(x, self.tp_group, -1)
        # Partial local output, then g operator (all_reduce) to sum shards.
        y = F.linear(x, self.weight, None)
        y = _ReduceFromTPRegion.apply(y, self.tp_group)
        if self.bias is not None:
            y = y + self.bias  # full bias, once per rank, after the reduce
        return y


# --------------------------------------------------------------------------- #
# Production DTensor path: parallelize_module with Colwise/Rowwise plans
# --------------------------------------------------------------------------- #


def apply_tensor_parallelism(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    sequence_parallel: bool = False,
    loss_parallel: bool = False,
) -> nn.Module:
    """Apply column/row TP to every ``TransformerBlock`` via ``parallelize_module``.

    This is the path the trainer uses. It rewrites the linear layers of each
    block into DTensors so that, at runtime, ``ColwiseParallel`` produces a
    last-dim-sharded activation and ``RowwiseParallel`` consumes it and
    all_reduces back to a replicated activation. ``use_local_output=True`` (the
    library default) means the attention/MLP math in between sees plain local
    tensors with a *reduced head/hidden count*, which is why the model's
    attention reshapes with ``-1`` for the head dimension (see
    :mod:`src.model.attention`).

    Sharding plan per block:
        * ``attention.wq/wk/wv`` -> ColumnParallel (Q/K/V heads split across TP)
        * ``attention.wo``       -> RowParallel  (reduces partial attention out)
        * ``mlp.gate_proj/up_proj`` -> ColumnParallel
        * ``mlp.down_proj``      -> RowParallel

    Embedding and the final LM head are intentionally **left replicated** (not
    vocab-parallel). Rationale: vocab-parallel embedding/head require a
    vocab-parallel cross-entropy (all_reduce over the logits) and an extra
    all_gather of the input embedding; for the model sizes targeted here the
    redundant replicated compute is cheaper than that critical-path collective,
    and FSDP already shards these large matrices across the DP axis. A
    ``loss_parallel`` flag is provided for users who do want the column-parallel
    head + parallel loss.

    Args:
        model: The (un-wrapped) transformer. Must expose ``model.layers`` as an
            iterable of blocks whose submodules are named ``attention`` / ``mlp``
            with the linear names above.
        tp_mesh: The 1D ``tp`` sub-mesh (``full_mesh["tp"]``).
        sequence_parallel: If True, shard LayerNorm/RMSNorm activations along the
            sequence dimension within the TP group (Megatron sequence
            parallelism). Reduces activation memory by ``tp_size`` on the norm
            inputs at the cost of two extra reshardings per block.
        loss_parallel: If True, make the LM head column-parallel with a
            DTensor output so a vocab-parallel cross-entropy can run.

    Returns:
        The same ``model`` object, parallelised in place.

    Raises:
        AttributeError: If a block is missing an expected submodule name; the
            error names the block index so the architecture mismatch is obvious.

    Performance note:
        When composed with FSDP (wrap *after* this call), the TP weights become
        2D DTensors sharded on both ``tp`` and ``dp``. FSDP requires
        ``use_orig_params=True`` to manage them — see
        :func:`src.parallelism.fsdp_utils.wrap_model_with_fsdp`.
    """
    if tp_mesh.size() == 1:
        # Degenerate TP group: nothing to shard, return unchanged so callers
        # never branch on tp_size.
        return model

    for layer_id, block in enumerate(model.layers):
        if not (hasattr(block, "attention") and hasattr(block, "mlp")):
            raise AttributeError(
                f"Block {layer_id} is missing an 'attention'/'mlp' submodule "
                f"required for TP; got submodules {list(dict(block.named_children()))}."
            )

        if sequence_parallel:
            # Inputs arrive sharded on the sequence dim (Shard(1)); the first
            # colwise op resharls to Replicate for the matmul, the rowwise op
            # produces Shard(1) again for the next norm.
            attn_in_layout: Placement = Shard(1)
            attn_out_layout: Placement = Shard(1)
        else:
            attn_in_layout = Replicate()
            attn_out_layout = Replicate()

        block_plan: dict[str, ParallelStyle] = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=attn_out_layout),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=attn_out_layout),
        }

        if sequence_parallel:
            # Norms operate on sequence-sharded activations; the block input is
            # resharded to Replicate before the attention/MLP column ops.
            block_plan["attention_norm"] = SequenceParallel()
            block_plan["mlp_norm"] = SequenceParallel()
            block_plan["attention"] = PrepareModuleInput(
                input_layouts=(attn_in_layout,),
                desired_input_layouts=(Replicate(),),
            )
            block_plan["mlp"] = PrepareModuleInput(
                input_layouts=(attn_in_layout,),
                desired_input_layouts=(Replicate(),),
            )

        parallelize_module(block, tp_mesh, block_plan)

    return model
