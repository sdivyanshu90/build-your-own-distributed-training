"""Reproducible, rank-aware seeding.

What this module does
---------------------
Seeds every RNG (Python ``random``, NumPy, PyTorch CPU and CUDA) deterministically
as a function of a base seed and the rank's *data-parallel* position.

Why the seed depends on dp_rank but NOT tp_rank
-----------------------------------------------
This is the single most important correctness rule for seeding under 2D
parallelism, and getting it wrong produces silent divergence:

  * **DP ranks must see different data** so the data-parallel average is over a
    real mini-batch, not the same sequence ``dp_size`` times. Hence the data
    seed includes ``dp_rank``.
  * **TP ranks (same dp_rank, different tp_rank) must stay bit-identical** in
    every stochastic decision that is *not* sharded — dropout masks on
    replicated activations, weight init of replicated modules (embeddings, the
    LM head), data ordering. If two TP ranks drew different dropout masks on the
    replicated residual stream, the ``all_reduce`` that stitches a RowParallel
    layer's output back together would sum mismatched activations and the model
    would diverge with no error. Hence the seed is constant across the TP group.

So: ``data/dropout seed = base + dp_rank`` (constant within a TP group), while
sharded weights are initialised correctly *because each TP rank slices its own
shard out of a tensor that was conceptually identical* — see the TP modules.

Determinism caveats
-------------------
``torch.use_deterministic_algorithms(True)`` makes many ops deterministic but
disables some fast kernels and raises if a non-deterministic op has no
deterministic fallback. We expose it behind a flag; for bit-exact convergence
tests it is on, for throughput benchmarks it is off.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(base_seed: int, dp_rank: int, deterministic: bool = False) -> int:
    """Seed all RNGs for the calling rank.

    Args:
        base_seed: The run-wide base seed from the config.
        dp_rank: This rank's data-parallel index (constant across its TP group).
            Use ``0`` for single-process / non-distributed contexts.
        deterministic: If True, enable ``torch.use_deterministic_algorithms`` and
            set the cuBLAS workspace env var required for deterministic matmuls.

    Returns:
        The effective per-rank seed (``base_seed + dp_rank``), so the caller can
        log exactly what was used.

    Side effects:
        Mutates global RNG state for ``random``, ``numpy``, ``torch`` (CPU+CUDA)
        and, if ``deterministic``, process-wide cuDNN/cuBLAS settings.

    Example:
        >>> eff = seed_everything(1234, dp_rank=2)
        >>> eff
        1236
    """
    effective = base_seed + dp_rank
    random.seed(effective)
    np.random.seed(effective % (2**32 - 1))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    if deterministic:
        # Required before enabling deterministic cuBLAS GEMMs.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return effective


def get_rng_state() -> dict[str, object]:
    """Snapshot all RNG states for checkpointing.

    Returns:
        A dict with ``python``, ``numpy``, ``torch`` and (if available)
        ``cuda`` RNG states. Stored per-rank in the checkpoint so a resume
        reproduces the exact stochastic stream.
    """
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict[str, object]) -> None:
    """Restore RNG states captured by :func:`get_rng_state`.

    Args:
        state: A dict as produced by :func:`get_rng_state`.

    Side effects:
        Overwrites the process's RNG state. CUDA state is only restored if CUDA
        is available *and* the checkpoint carried a ``cuda`` entry, so a
        GPU-saved checkpoint can be inspected on a CPU box without crashing.
    """
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])  # type: ignore[arg-type]
