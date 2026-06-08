"""FSDP wrapping policy, mixed precision, and (sharded/full) state-dict I/O.

What this module does
---------------------
Wraps a (possibly already TP-parallelised) model in FSDP with a per-``TransformerBlock``
auto-wrap policy, and provides the state-dict helpers used by checkpointing.

Why FSDP over DDP
-----------------
DDP replicates the *entire* model on every GPU: memory is ``O(P)`` per GPU for
params, plus ``O(P)`` for grads, plus ``O(2P)``-ish for Adam's moments — a 7B
model in fp32 master + bf16 params needs well over 100 GB before activations.
FSDP (ZeRO-3) shards params, grads, and optimizer state across the DP group, so
each GPU holds ``O(P / dp_size)``. It all-gathers a layer's full params *just in
time* for that layer's forward/backward, then frees them. Peak parameter memory
is therefore ``P/dp_size`` (the resident shard) ``+ P_layer`` (the single
largest layer transiently gathered) — which is why we wrap *per block*: the
transient term is one block, not the whole model.

Why per-block wrapping (``transformer_auto_wrap_policy``)
---------------------------------------------------------
If we wrapped the whole model as one FSDP unit, FSDP would all-gather *every*
parameter at once for the forward — defeating the memory saving (transient term
becomes ``P``). Wrapping each ``TransformerBlock`` as its own unit means only one
block's params are materialised at a time, and the all-gather of block ``i+1``
overlaps with the compute of block ``i`` (prefetch).

Why ``use_orig_params=True``
----------------------------
Two reasons: (1) it is **required** to compose with tensor parallelism — the TP
weights are DTensors, and only the orig-params path keeps them as addressable
parameters FSDP can further shard along DP; (2) it preserves the original
``nn.Parameter`` objects so the optimizer can build sensible per-parameter groups
(weight-decay vs no-decay) *after* wrapping.

Communication & memory per training step (FULL_SHARD)
-----------------------------------------------------
  * Forward, per block: all_gather params (O(P_block/dp_size) -> O(P_block)),
    compute, free the gathered params.
  * Backward, per block: all_gather params again (or reuse via BACKWARD_PRE
    prefetch), compute grads, reduce_scatter grads (O(P_block) -> O(P_block/dp)),
    free.
  * Optimizer step: update the resident shard only (O(P/dp_size)).

Invariants
----------
  * Every rank holds the *same set* of optimizer-state keys but *disjoint*
    parameter shards (ZeRO-3).
  * After wrapping, ``sum_ranks(local_param_numel)`` equals the original
    unsharded parameter count (no params lost or duplicated).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from src.config import ParallelConfig
from src.parallelism.process_groups import ProcessContext
from src.utils.dtype import build_mixed_precision

_SHARDING_STRATEGY = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
}

_BACKWARD_PREFETCH = {
    "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
    "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
}


def wrap_model_with_fsdp(
    model: nn.Module,
    ctx: ProcessContext,
    cfg: ParallelConfig,
    transformer_layer_cls: Iterable[type[nn.Module]],
) -> FSDP:
    """Wrap ``model`` in FSDP with a per-block auto-wrap policy.

    Args:
        model: The model to shard. If TP was applied, it must already be
            parallelised (TP first, then FSDP) so FSDP sees the DTensor weights.
        ctx: The process context; supplies the ``dp`` sub-mesh FSDP shards over
            and the compute device.
        cfg: Parallelism config (strategy, prefetch, mixed precision flags).
        transformer_layer_cls: The block class(es) to wrap individually, e.g.
            ``{TransformerBlock}``. Passed to ``transformer_auto_wrap_policy``.

    Returns:
        The FSDP-wrapped model (a ``FullyShardedDataParallel`` instance).

    Raises:
        ValueError: If ``cfg.sharding_strategy`` / ``cfg.backward_prefetch`` are
            unknown (message lists valid values).

    Side effects:
        Moves and shards parameters onto ``ctx.device``; allocates the flat
        sharded parameter storage.

    Performance note:
        ``limit_all_gathers=True`` caps the number of in-flight all-gathers so a
        deep model does not stack ``n_layers`` worth of transient full-param
        buffers and OOM. ``forward_prefetch`` + ``BACKWARD_PRE`` overlap the next
        block's all-gather with the current block's compute; turning them off
        roughly serialises comm and compute and can halve throughput.
    """
    if cfg.sharding_strategy not in _SHARDING_STRATEGY:
        raise ValueError(
            f"Unknown sharding_strategy={cfg.sharding_strategy!r}; "
            f"valid: {sorted(_SHARDING_STRATEGY)}."
        )
    if cfg.backward_prefetch not in _BACKWARD_PREFETCH:
        raise ValueError(
            f"Unknown backward_prefetch={cfg.backward_prefetch!r}; "
            f"valid: {sorted(_BACKWARD_PREFETCH)}."
        )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layer_cls),
    )

    mixed_precision: MixedPrecision = build_mixed_precision(cfg)

    # FSDP shards along the DP axis only. Passing the 1D dp sub-mesh is the
    # documented composition with TP: the resulting params are 2D DTensors
    # (Shard on tp, Shard on dp).
    dp_mesh = ctx.mesh["dp"]

    fsdp_kwargs: dict[str, Any] = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": _SHARDING_STRATEGY[cfg.sharding_strategy],
        "backward_prefetch": _BACKWARD_PREFETCH[cfg.backward_prefetch],
        "forward_prefetch": cfg.forward_prefetch,
        "limit_all_gathers": cfg.limit_all_gathers,
        "use_orig_params": True,
        "device_mesh": dp_mesh,
        # Pass device_id explicitly for BOTH cuda and cpu. Without it FSDP's
        # device-handle init defaults to cuda:current_device() when params are on
        # CPU, which breaks the Gloo/CPU test path on a box that happens to have
        # a GPU but is running a CPU job.
        "device_id": ctx.device,
    }

    wrapped = FSDP(model, **fsdp_kwargs)
    return wrapped


def apply_activation_checkpointing(
    model: nn.Module, transformer_layer_cls: Iterable[type[nn.Module]]
) -> None:
    """Wrap each transformer block in non-reentrant activation checkpointing.

    Recomputes a block's activations during backward instead of storing them,
    trading ~30% extra compute for a large activation-memory reduction (often the
    difference between fitting a long sequence or OOMing). ``NO_REENTRANT`` is the
    modern implementation: it composes with FSDP and supports nested autograd.

    Apply this **before** FSDP wrapping so the checkpoint wrapper sits inside the
    FSDP unit and the recompute uses the just-gathered params.

    Args:
        model: The (TP-applied, not-yet-FSDP) model.
        transformer_layer_cls: Block class(es) to checkpoint.

    Side effects:
        Replaces matching submodules with ``CheckpointWrapper`` instances in place.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing as _apply,
    )

    wrap_classes = set(transformer_layer_cls)
    wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    _apply(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda submodule: type(submodule) in wrap_classes,
    )


def count_unsharded_parameters(model: FSDP, ctx: ProcessContext) -> int:
    """Sum local parameter numel across all ranks (== original unsharded count).

    Used by tests to assert no parameter was lost or duplicated by sharding.

    Args:
        model: The FSDP-wrapped model.
        ctx: Process context (for the world group all_reduce).

    Returns:
        The total parameter count summed over every rank's local shard.
    """
    local = sum(p.numel() for p in model.parameters())
    t = torch.tensor([local], device=ctx.device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def get_model_state_dict(model: nn.Module, *, full: bool) -> dict[str, torch.Tensor]:
    """Extract the model state dict in sharded or full form.

    Args:
        model: The FSDP-wrapped model.
        full: If True, consolidate to a single full state dict offloaded to CPU
            on rank 0 only (``FULL_STATE_DICT``) — portable but slow, for final
            export. If False, each rank returns its own shard
            (``SHARDED_STATE_DICT``) — fast and storage-efficient, for training
            checkpoints.

    Returns:
        The state dict (empty on non-zero ranks when ``full=True`` with
        ``rank0_only``).

    Performance note:
        ``FULL_STATE_DICT`` all-gathers every parameter to rank 0 and is ``O(P)``
        memory on that rank; never use it for frequent training checkpoints at
        scale — use ``SHARDED_STATE_DICT``, which writes ``O(P/dp_size)`` per
        rank in parallel.
    """
    if not isinstance(model, FSDP):
        # Pure-TP (no FSDP) path: each rank holds its own state directly.
        return model.state_dict()
    if full:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    cfg_s = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg_s):
        return model.state_dict()


def load_model_state_dict(
    model: FSDP, state_dict: dict[str, torch.Tensor], *, full: bool
) -> None:
    """Load a sharded or full state dict into an FSDP model.

    Args:
        model: The FSDP-wrapped model (same architecture as at save time).
        state_dict: The dict produced by :func:`get_model_state_dict`.
        full: Must match how the dict was saved.

    Side effects:
        Copies parameters into the model's sharded storage in place.
    """
    if not isinstance(model, FSDP):
        model.load_state_dict(state_dict)
        return
    if full:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state_dict)
        return
    cfg_s = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg_s):
        model.load_state_dict(state_dict)


def get_optimizer_state_dict(
    model: FSDP, optimizer: torch.optim.Optimizer, *, full: bool
) -> dict[str, Any]:
    """Extract the FSDP-decomposed optimizer state dict.

    FSDP stores optimizer state (Adam moments) sharded just like the params. The
    ``optim_state_dict`` call re-keys the flat sharded state back to the original
    parameter names so it is portable and resharded correctly on load.

    Args:
        model: The FSDP-wrapped model.
        optimizer: The optimizer built on ``model.parameters()``.
        full: Sharded (training) vs full (export) decomposition, mirroring
            :func:`get_model_state_dict`.

    Returns:
        The optimizer state dict. On a CPU-only job (no CUDA) it is a per-rank
        fallback (see below), tagged with ``__fsdp_per_rank__``.

    Note (torch 2.3 CPU path):
        FSDP's ``optim_state_dict`` calls ``torch.cuda.synchronize()``
        unconditionally in torch 2.3, which raises on a CPU job. When CUDA is
        unavailable we fall back to each rank saving its *local* optimizer state
        (``optimizer.state_dict()``). This is valid for resuming on the **same
        topology** (each rank reloads its own shard) but is not cross-topology
        portable; the portable re-keyed path is used whenever CUDA is present.
    """
    if not isinstance(model, FSDP):
        return optimizer.state_dict()
    if not torch.cuda.is_available():
        return {"__fsdp_per_rank__": True, "state": optimizer.state_dict()}
    if full:
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, state_cfg, optim_cfg
        ):
            return FSDP.optim_state_dict(model, optimizer)
    state_cfg_s = ShardedStateDictConfig(offload_to_cpu=True)
    optim_cfg_s = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model, StateDictType.SHARDED_STATE_DICT, state_cfg_s, optim_cfg_s
    ):
        return FSDP.optim_state_dict(model, optimizer)


def load_optimizer_state_dict(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    optim_state_dict: dict[str, Any],
    *,
    full: bool,
) -> None:
    """Load an FSDP optimizer state dict, resharding to the current layout.

    ``optim_state_dict_to_load`` maps the saved (possibly differently-sharded)
    optimizer state onto the current model's sharding so a run can resume even if
    the DP degree changed between save and load.

    Args:
        model: The FSDP-wrapped model.
        optimizer: The optimizer to populate.
        optim_state_dict: The dict from :func:`get_optimizer_state_dict`.
        full: Must match the save-time decomposition.

    Side effects:
        Calls ``optimizer.load_state_dict`` with the resharded state.
    """
    if isinstance(optim_state_dict, dict) and optim_state_dict.get("__fsdp_per_rank__"):
        # Per-rank fallback saved on a CPU job (see get_optimizer_state_dict).
        optimizer.load_state_dict(optim_state_dict["state"])
        return
    if not isinstance(model, FSDP):
        optimizer.load_state_dict(optim_state_dict)
        return
    if full:
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ctx_mgr = FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, state_cfg, optim_cfg
        )
    else:
        state_cfg_s = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg_s = ShardedOptimStateDictConfig(offload_to_cpu=True)
        ctx_mgr = FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, state_cfg_s, optim_cfg_s
        )
    with ctx_mgr:
        loaded = FSDP.optim_state_dict_to_load(model, optimizer, optim_state_dict)
        optimizer.load_state_dict(loaded)
