"""Root pytest config: make ``src`` importable and provide dist fixtures.

Adds the repository root to ``sys.path`` so ``import src.<module>`` works under
plain ``pytest`` without an editable install, and exposes a single-process
distributed fixture used by the unit tests that need a (degenerate) process
group.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.distributed as dist  # noqa: E402


@pytest.fixture()
def single_process_pg():
    """Initialise a world_size=1 gloo process group for the test, then tear down.

    Lets unit tests exercise the autograd-aware TP collectives (which call into
    ``dist`` even though, over a 1-rank group, every collective is an identity).
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
