"""Unit tests for mesh construction and rank mapping.

Pure-function tests (``resolve_dp_size`` and validation) run in-process; the real
2D-mesh tests spawn ``world_size`` Gloo processes via :func:`run_distributed` and
assert each rank computes the correct coordinates and group sizes.
"""

from __future__ import annotations

import pytest

from src.parallelism.mesh import (
    build_device_mesh,
    get_dp_group,
    get_parallel_dims,
    get_tp_group,
    resolve_dp_size,
)
from tests._dist_utils import run_distributed


def test_resolve_dp_size_infers_and_validates() -> None:
    assert resolve_dp_size(world_size=8, tp_size=2, dp_size=-1) == 4
    assert resolve_dp_size(world_size=8, tp_size=1, dp_size=-1) == 8
    assert resolve_dp_size(world_size=8, tp_size=8, dp_size=-1) == 1


def test_resolve_dp_size_rejects_non_divisor() -> None:
    with pytest.raises(ValueError, match="does not divide world_size"):
        resolve_dp_size(world_size=6, tp_size=4, dp_size=-1)


def test_resolve_dp_size_rejects_inconsistent_product() -> None:
    with pytest.raises(ValueError, match="!= world_size"):
        resolve_dp_size(world_size=8, tp_size=2, dp_size=3)


# --- multi-process mesh checks (top-level fns so they are picklable) --------- #


def _check_2x2(rank: int, world_size: int) -> None:
    mesh = build_device_mesh(tp_size=2, dp_size=-1, device_type="cpu")
    assert mesh.mesh_dim_names == ("dp", "tp")
    assert get_tp_group(mesh).size() == 2
    assert get_dp_group(mesh).size() == 2
    dims = get_parallel_dims(mesh)
    # Row-major layout: global_rank == dp_rank * tp_size + tp_rank.
    assert dims.global_rank == dims.dp_rank * dims.tp_size + dims.tp_rank, (
        f"rank {rank}: mapping mismatch dp={dims.dp_rank} tp={dims.tp_rank}"
    )
    assert dims.world_size == 4


def _check_pure_fsdp(rank: int, world_size: int) -> None:
    # tp_size == 1 -> degenerate TP dim, valid 1-rank tp group, dp group == world.
    mesh = build_device_mesh(tp_size=1, dp_size=-1, device_type="cpu")
    assert get_tp_group(mesh).size() == 1
    assert get_dp_group(mesh).size() == world_size


def _check_pure_tp(rank: int, world_size: int) -> None:
    # dp_size == 1 -> degenerate DP dim; tp group spans the world.
    mesh = build_device_mesh(tp_size=world_size, dp_size=-1, device_type="cpu")
    assert get_tp_group(mesh).size() == world_size
    assert get_dp_group(mesh).size() == 1


def _check_invalid(rank: int, world_size: int) -> None:
    with pytest.raises(ValueError):
        build_device_mesh(tp_size=3, dp_size=-1, device_type="cpu")  # 3 ∤ 4


def test_build_mesh_2x2() -> None:
    run_distributed(_check_2x2, world_size=4)


def test_build_mesh_pure_fsdp() -> None:
    run_distributed(_check_pure_fsdp, world_size=4)


def test_build_mesh_pure_tp() -> None:
    run_distributed(_check_pure_tp, world_size=4)


def test_build_mesh_invalid_topology_raises() -> None:
    run_distributed(_check_invalid, world_size=4)
