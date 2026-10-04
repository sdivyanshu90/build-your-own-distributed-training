"""communication_fraction must see collectives on CPU/gloo (and not read 0 because
of a renamed profiler attribute)."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from src.observability.profiler import communication_fraction


def test_communication_fraction_nonzero_for_collectives(single_process_pg: None) -> None:
    x = torch.randn(1 << 16)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            dist.all_reduce(x)
        torch.randn(256, 256) @ torch.randn(256, 256)
    b = communication_fraction(prof)
    assert b.total_cuda_us > 0
    assert b.comm_cuda_us > 0, "collective ops not classified as communication"
    assert 0.0 < b.comm_fraction <= 1.0
