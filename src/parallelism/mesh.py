"""2D ``DeviceMesh`` construction and rank mapping for (DP x TP) parallelism.

What this module does
---------------------
Builds the ``(dp, tp)`` process mesh that the whole system is organised around,
and provides the rank-mapping helpers everything else uses to address the right
process group. This is the foundation: get the mesh wrong and every collective
in the system talks to the wrong peers.

The mesh layout (and why row-major matters)
-------------------------------------------
We build a 2D mesh of shape ``(dp_size, tp_size)`` with the named axes
``("dp", "tp")``. ``init_device_mesh`` lays ranks out **row-major**, so the *last*
axis (``tp``) varies fastest::

    global_rank = dp_rank * tp_size + tp_rank

Concretely with ``dp_size=2, tp_size=4`` on 8 GPUs:

    (dp=0): ranks [0 1 2 3]     <- one TP group, ideally one node / NVLink island
    (dp=1): ranks [4 5 6 7]     <- another TP group

This contiguity is deliberate. TP fires an ``all_reduce`` *every* micro-step on
the critical path, so its 4 ranks must sit on the fastest interconnect (NVLink,
intra-node). DP/FSDP communicates less frequently and tolerates inter-node
links. Mapping ``tp`` to the fast-varying (contiguous) axis places each TP group
inside a node by default.

Why always build a full 2D mesh, even for 1D cases
--------------------------------------------------
The spec demands the rest of the code never branch on "is TP enabled". We satisfy
that by *always* constructing a ``(dp_size, tp_size)`` mesh, allowing a degenerate
size-1 dimension:

  * ``tp_size == 1`` -> pure FSDP. The ``tp`` sub-mesh has one rank; the TP
    ``all_reduce`` becomes a no-op over a 1-process group.
  * ``dp_size == 1`` -> pure TP. The ``dp`` sub-mesh has one rank; FSDP shards
    across 1 process, i.e. no real sharding, which is correct.

So ``mesh["dp"]`` and ``mesh["tp"]`` are *always* valid sub-meshes and callers
never special-case the degenerate dimension.

Communication patterns
-----------------------
This module triggers **no** collectives itself; it only constructs the groups.
The groups it returns carry these costs once used:
  * ``tp`` group: ``all_reduce`` / ``all_gather`` / ``reduce_scatter`` per layer,
    O(activation_bytes) each, on the critical path.
  * ``dp`` group: FSDP ``all_gather`` (params) + ``reduce_scatter`` (grads) per
    FSDP unit, O(param_bytes / dp_size) each, overlapped with compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


@dataclass(frozen=True)
class ParallelDims:
    """Resolved parallel dimensions and this rank's coordinates within them.

    Attributes:
        tp_size: Tensor-parallel degree.
        dp_size: Data-parallel (FSDP) degree.
        world_size: ``tp_size * dp_size``.
        dp_rank: This rank's index along the ``dp`` axis (which FSDP shard).
        tp_rank: This rank's index along the ``tp`` axis (which weight shard).
        global_rank: This rank's global rank.
    """

    tp_size: int
    dp_size: int
    world_size: int
    dp_rank: int
    tp_rank: int
    global_rank: int

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1


def resolve_dp_size(world_size: int, tp_size: int, dp_size: int) -> int:
    """Infer ``dp_size`` from ``world_size`` and ``tp_size`` when unset.

    Args:
        world_size: Total processes (``dist.get_world_size()``).
        tp_size: Requested tensor-parallel degree.
        dp_size: Requested DP degree, or ``-1`` to infer as
            ``world_size // tp_size``.

    Returns:
        The resolved DP degree.

    Raises:
        ValueError: If ``tp_size`` does not divide ``world_size``, or the
            resolved/explicit dims do not multiply to ``world_size``. The error
            spells out the arithmetic so the misconfiguration is obvious.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    if world_size % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} does not divide world_size={world_size}. "
            f"A TP group must fit evenly; choose tp_size in "
            f"{[t for t in range(1, world_size + 1) if world_size % t == 0]}."
        )
    inferred = world_size // tp_size if dp_size == -1 else dp_size
    if inferred * tp_size != world_size:
        raise ValueError(
            f"dp_size({inferred}) * tp_size({tp_size}) = {inferred * tp_size} "
            f"!= world_size({world_size}). Fix dp_size/tp_size in the config."
        )
    return inferred


def build_device_mesh(
    tp_size: int,
    dp_size: int = -1,
    device_type: str | None = None,
) -> DeviceMesh:
    """Construct the 2D ``(dp, tp)`` ``DeviceMesh`` for this job.

    Args:
        tp_size: Tensor-parallel degree (``1`` disables TP).
        dp_size: Data-parallel degree, or ``-1`` to infer from world size.
        device_type: ``"cuda"`` or ``"cpu"``. Defaults to ``"cuda"`` when
            available else ``"cpu"`` (the Gloo test path).

    Returns:
        A 2D ``DeviceMesh`` of shape ``(dp_size, tp_size)`` with dim names
        ``("dp", "tp")``. ``mesh["dp"]`` and ``mesh["tp"]`` yield the per-axis
        sub-meshes whose ``.get_group()`` is the relevant ``ProcessGroup``.

    Raises:
        RuntimeError: If the default process group is not initialised.
        ValueError: Propagated from :func:`resolve_dp_size` on a bad config.

    Example:
        >>> # with world_size == 4 and an initialised PG:
        >>> mesh = build_device_mesh(tp_size=2)          # dp inferred = 2
        >>> mesh.mesh_dim_names
        ('dp', 'tp')
        >>> mesh["tp"].size()
        2
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "build_device_mesh requires an initialised default process group; "
            "call init_distributed() / dist.init_process_group first."
        )
    world_size = dist.get_world_size()
    resolved_dp = resolve_dp_size(world_size, tp_size, dp_size)

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    mesh = init_device_mesh(
        device_type,
        mesh_shape=(resolved_dp, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    return mesh


def get_parallel_dims(mesh: DeviceMesh) -> ParallelDims:
    """Derive :class:`ParallelDims` (sizes + this rank's coordinates) from a mesh.

    Args:
        mesh: A 2D ``(dp, tp)`` mesh from :func:`build_device_mesh`.

    Returns:
        A :class:`ParallelDims` for the calling rank.

    Notes:
        ``mesh["dp"].get_local_rank()`` returns this rank's coordinate along the
        ``dp`` axis; likewise for ``tp``. We avoid recomputing it from the global
        rank so the mapping stays correct even if a future layout changes.
    """
    tp_size = mesh["tp"].size()
    dp_size = mesh["dp"].size()
    return ParallelDims(
        tp_size=tp_size,
        dp_size=dp_size,
        world_size=tp_size * dp_size,
        dp_rank=mesh["dp"].get_local_rank(),
        tp_rank=mesh["tp"].get_local_rank(),
        global_rank=dist.get_rank(),
    )


def get_dp_group(mesh: DeviceMesh) -> dist.ProcessGroup:
    """Return the data-parallel (FSDP) ``ProcessGroup`` for this rank."""
    # A single-dim sub-mesh's get_group returns one ProcessGroup (not a list).
    return cast(dist.ProcessGroup, mesh["dp"].get_group())


def get_tp_group(mesh: DeviceMesh) -> dist.ProcessGroup:
    """Return the tensor-parallel ``ProcessGroup`` for this rank."""
    return cast(dist.ProcessGroup, mesh["tp"].get_group())


def format_mesh_layout(mesh: DeviceMesh) -> str:
    """Produce a human-readable map of ``(dp_rank, tp_rank) -> global_rank``.

    Logged once from rank 0 so an operator can see exactly which physical GPUs
    form each TP group — invaluable when diagnosing slow links or a bad
    ``CUDA_VISIBLE_DEVICES``. The underlying ``mesh.mesh`` tensor holds the
    global ranks laid out in ``(dp, tp)`` shape.

    Returns:
        A multi-line string table.
    """
    layout = mesh.mesh  # shape (dp_size, tp_size), values are global ranks
    dp_size, tp_size = layout.shape
    lines = [f"DeviceMesh layout (dp_size={dp_size}, tp_size={tp_size}):"]
    header = "        " + "".join(f"tp={t:<6}" for t in range(tp_size))
    lines.append(header)
    for d in range(dp_size):
        row = f"  dp={d:<3} " + "".join(
            f"r{int(layout[d, t]):<6}" for t in range(tp_size)
        )
        lines.append(row)
    return "\n".join(lines)
