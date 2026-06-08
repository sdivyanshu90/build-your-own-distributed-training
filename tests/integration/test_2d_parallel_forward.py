"""Integration: tensor-parallel forward correctness across real ranks.

Runs with 2 Gloo ranks. Three checks:
  1. The hand-rolled column/row parallel linears, given a broadcast reference
     weight sharded across ranks, reproduce a single-GPU ``nn.Linear`` exactly.
  2. A full model parallelised with :func:`apply_tensor_parallelism` produces the
     *same* logits as an identically-initialised non-parallel reference — the
     spec's "TP block numerically identical to single-GPU reference".
  3. Under pure FSDP, two DP ranks given the *same* input produce the same loss
     (the all-gathered forward is rank-invariant).

The composed 2D (FSDP+TP) path is exercised on real multi-GPU/NCCL hardware; on
the CPU/Gloo + torch-2.3 combo it hits a known FSDP-writeback-with-DTensor bug, so
the 2D end-to-end check lives in the GPU-gated convergence test.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from src.config import ModelConfig, ParallelConfig
from src.model.transformer import TransformerBlock, build_model
from src.parallelism.fsdp_utils import wrap_model_with_fsdp
from src.parallelism.mesh import build_device_mesh, get_tp_group
from src.parallelism.process_groups import build_process_context
from src.parallelism.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    apply_tensor_parallelism,
)
from tests._dist_utils import run_distributed

_MODEL = ModelConfig(
    vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
    max_seq_len=32, tie_embeddings=True,
)


def _check_handrolled_tp_linears(rank: int, world_size: int) -> None:
    mesh = build_device_mesh(tp_size=world_size, device_type="cpu")
    tp_group = get_tp_group(mesh)
    d_in, d_hidden, d_out = 8, 16, 8

    # Reference full linears, identical on all ranks (broadcast from rank 0).
    torch.manual_seed(0)
    ref1 = torch.nn.Linear(d_in, d_hidden)
    ref2 = torch.nn.Linear(d_hidden, d_out)
    for p in list(ref1.parameters()) + list(ref2.parameters()):
        dist.broadcast(p.data, src=0)

    col = ColumnParallelLinear(d_in, d_hidden, tp_group=tp_group, gather_output=False, bias=True)
    row = RowParallelLinear(d_hidden, d_out, tp_group=tp_group, input_is_parallel=True, bias=True)
    # Give each rank its shard of the reference weights.
    out_per = d_hidden // world_size
    col.weight.data.copy_(ref1.weight.data[rank * out_per : (rank + 1) * out_per])
    col.bias.data.copy_(ref1.bias.data[rank * out_per : (rank + 1) * out_per])
    in_per = d_hidden // world_size
    row.weight.data.copy_(ref2.weight.data[:, rank * in_per : (rank + 1) * in_per])
    row.bias.data.copy_(ref2.bias.data)

    torch.manual_seed(42)
    x = torch.randn(4, d_in)
    dist.broadcast(x, src=0)  # all ranks see the same (replicated) input
    ref_out = ref2(ref1(x))
    par_out = row(col(x))
    assert torch.allclose(ref_out, par_out, atol=1e-5), (
        f"rank {rank}: TP linear chain != reference (max diff "
        f"{(ref_out - par_out).abs().max().item():.2e})"
    )


def _check_tp_model_matches_reference(rank: int, world_size: int) -> None:
    # Same seed => identical full weights on every rank; parallelize_module then
    # only *redistributes* those weights, so the TP model is mathematically the
    # same function as the reference.
    torch.manual_seed(123)
    ref = build_model(_MODEL)
    torch.manual_seed(123)
    tp_model = build_model(_MODEL)
    mesh = build_device_mesh(tp_size=world_size, device_type="cpu")
    apply_tensor_parallelism(tp_model, mesh["tp"])

    torch.manual_seed(7)
    tokens = torch.randint(0, _MODEL.vocab_size, (2, 16))
    dist.broadcast(tokens, src=0)
    with torch.no_grad():
        ref_logits, _ = ref(tokens)
        tp_logits, _ = tp_model(tokens)
    assert torch.allclose(ref_logits, tp_logits, atol=1e-4), (
        f"rank {rank}: TP model logits != reference (max diff "
        f"{(ref_logits - tp_logits).abs().max().item():.2e})"
    )


def _check_fsdp_dp_loss_consistency(rank: int, world_size: int) -> None:
    ctx = build_process_context(tp_size=1, backend="gloo")
    torch.manual_seed(321)
    model = build_model(_MODEL)
    pc = ParallelConfig(tp_size=1, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32")
    wrapped = wrap_model_with_fsdp(model, ctx, pc, {TransformerBlock})

    # Identical input on all DP ranks -> identical loss (forward is rank-invariant).
    torch.manual_seed(7)
    tokens = torch.randint(0, _MODEL.vocab_size, (2, 16))
    dist.broadcast(tokens, src=0)
    _, loss = wrapped(tokens, labels=tokens)
    losses = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(losses, loss.detach())
    assert torch.allclose(losses[0], losses[1], atol=1e-5), (
        f"DP ranks disagree on loss for identical input: {[v.item() for v in losses]}"
    )


def test_handrolled_tp_linears_match_reference() -> None:
    run_distributed(_check_handrolled_tp_linears, world_size=2)


def test_tp_model_matches_single_gpu_reference() -> None:
    run_distributed(_check_tp_model_matches_reference, world_size=2)


def test_fsdp_dp_loss_consistency() -> None:
    run_distributed(_check_fsdp_dp_loss_consistency, world_size=2)
