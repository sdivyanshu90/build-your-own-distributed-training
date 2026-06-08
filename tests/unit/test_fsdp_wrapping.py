"""Unit tests for FSDP wrapping: param-count invariant, sharding, MP policy.

The parameter-count and sharding checks spawn 2 Gloo ranks (pure FSDP, which is
the CPU-supported path). The mixed-precision policy is checked in-process from the
config (the dtype mapping is what matters; live bf16 reductions need NCCL).
"""

from __future__ import annotations

import torch

from src.config import ModelConfig, ParallelConfig
from src.model.transformer import TransformerBlock, build_model
from src.parallelism.fsdp_utils import (
    count_unsharded_parameters,
    get_model_state_dict,
    wrap_model_with_fsdp,
)
from src.parallelism.process_groups import build_process_context
from src.utils.dtype import build_mixed_precision
from tests._dist_utils import run_distributed

_MODEL = ModelConfig(
    vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2,
    max_seq_len=32, tie_embeddings=True,
)


def _fp32_parallel() -> ParallelConfig:
    return ParallelConfig(
        tp_size=1, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32"
    )


def _check_param_count(rank: int, world_size: int) -> None:
    ctx = build_process_context(tp_size=1, backend="gloo")
    original = sum(p.numel() for p in build_model(_MODEL).parameters())
    model = build_model(_MODEL)
    wrapped = wrap_model_with_fsdp(model, ctx, _fp32_parallel(), {TransformerBlock})
    total = count_unsharded_parameters(wrapped, ctx)
    assert total == original, f"sharded total {total} != original {original}"


def _check_sharded_and_forward(rank: int, world_size: int) -> None:
    ctx = build_process_context(tp_size=1, backend="gloo")
    model = build_model(_MODEL)
    wrapped = wrap_model_with_fsdp(model, ctx, _fp32_parallel(), {TransformerBlock})
    # Each rank's local shard is strictly smaller than the full model.
    local = sum(p.numel() for p in wrapped.parameters())
    original = sum(p.numel() for p in build_model(_MODEL).parameters())
    assert 0 < local < original, f"rank {rank} local {local} not a proper shard of {original}"
    # Sharded state dict is non-empty per rank.
    sd = get_model_state_dict(wrapped, full=False)
    assert len(sd) > 0
    # Forward + backward run end-to-end.
    tokens = torch.randint(0, _MODEL.vocab_size, (2, 16))
    _, loss = wrapped(tokens, labels=tokens)
    loss.backward()
    assert torch.isfinite(loss)


def test_fsdp_param_count_invariant() -> None:
    run_distributed(_check_param_count, world_size=2)


def test_fsdp_sharding_and_forward() -> None:
    run_distributed(_check_sharded_and_forward, world_size=2)


def test_mixed_precision_policy_dtypes() -> None:
    cfg = ParallelConfig(param_dtype="bfloat16", reduce_dtype="float32", buffer_dtype="bfloat16")
    mp = build_mixed_precision(cfg)
    assert mp.param_dtype == torch.bfloat16
    # reduce_dtype MUST be fp32 for unbiased large-DP gradient reduction.
    assert mp.reduce_dtype == torch.float32
    assert mp.buffer_dtype == torch.bfloat16
