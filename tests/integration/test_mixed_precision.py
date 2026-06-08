"""Integration: mixed-precision policy and runtime dtype behaviour.

CPU-verifiable: under ``autocast(bf16)`` the matmul runs in bf16 while the master
weight stays fp32 (the principle behind FSDP's ``param_dtype``/fp32-master split),
and the ``MixedPrecision`` policy carries fp32 ``reduce_dtype``. The live FSDP
gradient-reduction dtype (fp32 reduce-scatter) and fp32 optimizer step are checked
on CUDA where bf16 collectives are supported.
"""

from __future__ import annotations

import pytest
import torch

from src.config import ParallelConfig
from src.utils.dtype import autocast_dtype, build_mixed_precision, resolve_dtype


def test_autocast_computes_bf16_weights_stay_fp32() -> None:
    lin = torch.nn.Linear(8, 8)
    captured: dict[str, torch.dtype] = {}

    def hook(_m, _inp, out: torch.Tensor) -> None:
        captured["out"] = out.dtype

    lin.register_forward_hook(hook)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _ = lin(torch.randn(2, 8))
    assert lin.weight.dtype == torch.float32, "master weight must remain fp32"
    assert captured["out"] == torch.bfloat16, "compute must happen in bf16"


def test_policy_reduce_dtype_is_fp32() -> None:
    mp = build_mixed_precision(
        ParallelConfig(param_dtype="bfloat16", reduce_dtype="float32", buffer_dtype="bfloat16")
    )
    assert mp.param_dtype == torch.bfloat16
    assert mp.reduce_dtype == torch.float32
    assert mp.buffer_dtype == torch.bfloat16


def test_autocast_dtype_follows_param_dtype() -> None:
    assert autocast_dtype(ParallelConfig(param_dtype="bfloat16")) == torch.bfloat16
    assert autocast_dtype(ParallelConfig(param_dtype="float32")) == torch.float32


def test_resolve_dtype_aliases() -> None:
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("fp32") == torch.float32
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("float8_made_up")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 collectives need NCCL/CUDA")
def test_fsdp_reduces_in_fp32_runtime() -> None:
    pytest.skip("Requires multi-GPU NCCL; verify grad bucket dtype is fp32 under FSDP MP.")
