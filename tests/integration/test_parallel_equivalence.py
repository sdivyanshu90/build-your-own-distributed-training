"""End-to-end equivalence: every parallel layout computes the same loss and the
same *global* gradient norm as an unparallelised single-process reference.

Each rank builds the reference locally (same seed => same init, no collectives),
feeds it the union of the micro-batches the DP ranks consume (``shuffle=False`` so
rank r takes indices ``r::dp``), and compares loss and grad-norm with what
``train_step`` reports for the real FSDP / TP / FSDP+TP model. This is the check
that catches wrong-group reductions in the clip / loss paths.
"""

from __future__ import annotations

import pytest
import torch

from src.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.model.transformer import build_model
from src.training.loop import train_step
from src.training.trainer import Trainer
from src.utils.seed import seed_everything
from tests._dist_utils import run_distributed

_MBS = 2


def _cfg(tp: int, sharding: str = "FULL_SHARD", sp: bool = False, hybrid: int = 0,
         accum: int = 1) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_hidden_size=64, max_seq_len=32, tie_embeddings=True,
    )
    cfg.parallel = ParallelConfig(
        tp_size=tp, sharding_strategy=sharding, param_dtype="float32",
        reduce_dtype="float32", buffer_dtype="float32",
        sequence_parallel=sp, hybrid_shard_size=hybrid,
    )
    cfg.optimizer = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0, fused=False)
    cfg.scheduler = SchedulerConfig(warmup_steps=2, max_steps=10, min_lr=1e-4)
    cfg.data = DataConfig(
        dataset_path="synthetic", seq_len=16, micro_batch_size=_MBS,
        global_batch_size=0, num_workers=0, shuffle=False,
    )
    cfg.grad_accum_steps = accum
    cfg.max_steps = 10
    cfg.max_grad_norm = 1e9  # never clips; we compare the reported pre-clip norm
    cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
    cfg.backend = "gloo"
    cfg.seed = 5
    return cfg


def _reference(trainer: Trainer, dp: int, accum: int = 1) -> tuple[float, float]:
    cfg = trainer.config
    seed_everything(cfg.seed, 0)
    ref = build_model(cfg.model)
    ds = trainer.train_dataset
    items = [ds[i] for i in range(_MBS * dp * accum)]
    ids = torch.stack([x["input_ids"] for x in items])
    labels = torch.stack([x["labels"] for x in items])
    _, loss = ref(ids, labels=labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), 1e9)
    return float(loss), float(norm)


def _check(rank: int, world_size: int, tp: int, sharding: str, sp: bool = False,
           hybrid: int = 0, accum: int = 1) -> None:
    cfg = _cfg(tp, sharding, sp, hybrid, accum)
    trainer = Trainer(cfg, gpu_type="cpu")
    dp = trainer.ctx.dims.dp_size
    ref_loss, ref_norm = _reference(trainer, dp, accum)
    window, _ = trainer._next_window()
    m = train_step(trainer.model, window, trainer.optimizer, trainer.scheduler,
                   trainer.ctx, cfg, gpu_type="cpu")
    assert m.loss == pytest.approx(ref_loss, rel=1e-4), (rank, m.loss, ref_loss)
    assert m.grad_norm == pytest.approx(ref_norm, rel=1e-3), (
        f"rank {rank} tp={tp} dp={dp}: grad_norm {m.grad_norm} != reference {ref_norm}"
    )


def test_fsdp_dp2_matches_reference() -> None:
    run_distributed(_check, 2, 1, "FULL_SHARD")


def test_tp2_matches_reference() -> None:
    run_distributed(_check, 2, 2, "FULL_SHARD")


def test_2d_fsdp_x_tp_matches_reference() -> None:
    run_distributed(_check, 4, 2, "FULL_SHARD")


def test_2d_with_grad_accumulation_matches_reference() -> None:
    """Regression: FSDP1 + DTensor-TP grads under no_sync crashed in
    ``_writeback_orig_params`` ("invalid python storage") from the 2nd micro-batch."""
    run_distributed(_check, 4, 2, "FULL_SHARD", False, 0, 2)


def test_fsdp_with_grad_accumulation_matches_reference() -> None:
    run_distributed(_check, 2, 1, "FULL_SHARD", False, 0, 3)


def test_sequence_parallel_tp2_matches_reference() -> None:
    run_distributed(_check, 2, 2, "FULL_SHARD", True)


def test_hybrid_shard_2x2_matches_reference() -> None:
    """Regression: HYBRID_SHARD used to raise (1-D mesh passed where 2-D required)."""
    run_distributed(_check, 4, 1, "HYBRID_SHARD", False, 2)


def _check_replicas_identical(rank: int, world_size: int) -> None:
    """NO_SHARD keeps a full replica per DP rank: they must start identical."""
    trainer = Trainer(_cfg(1, "NO_SHARD"), gpu_type="cpu")
    flat = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, flat)
    assert torch.equal(gathered[0], gathered[1]), "DP replicas initialised differently"


def test_no_shard_replicas_start_identical() -> None:
    run_distributed(_check_replicas_identical, 2)


def test_is_tp_sharded_param_matches_plan_names() -> None:
    from src.parallelism.tensor_parallel import is_tp_sharded_param

    assert is_tp_sharded_param("layers.0._fsdp_wrapped_module.attention.wq.weight")
    assert is_tp_sharded_param("layers.3.mlp.down_proj.weight")
    assert not is_tp_sharded_param("layers.0.attention_norm.weight")
    assert not is_tp_sharded_param("tok_embeddings.weight")
    assert not is_tp_sharded_param("norm.weight")


def _check_handrolled_backward(rank: int, world_size: int) -> None:
    """Col->Row chain: weight-shard grads and input grad match a dense reference.

    Exercises the backward collectives of the hand-written autograd Functions
    (all-reduce of the input grad in ``_CopyToTPRegion``, identity in
    ``_ReduceFromTPRegion``); the single-rank unit tests cannot, as every
    collective is an identity there.
    """
    import torch.distributed as dist

    from src.parallelism.mesh import build_device_mesh, get_tp_group
    from src.parallelism.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    g = get_tp_group(build_device_mesh(tp_size=world_size, device_type="cpu"))
    torch.manual_seed(0)
    ref1, ref2 = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)
    for p in [*ref1.parameters(), *ref2.parameters()]:
        dist.broadcast(p.data, src=0)
    col = ColumnParallelLinear(8, 16, g, bias=True)
    row = RowParallelLinear(16, 8, g, bias=True)
    n = 16 // world_size
    sl = slice(rank * n, (rank + 1) * n)
    col.weight.data.copy_(ref1.weight.data[sl])
    col.bias.data.copy_(ref1.bias.data[sl])
    row.weight.data.copy_(ref2.weight.data[:, sl])
    row.bias.data.copy_(ref2.bias.data)
    x = torch.randn(4, 8)
    dist.broadcast(x, src=0)
    x_ref, x_par = x.clone().requires_grad_(), x.clone().requires_grad_()
    (ref2(ref1(x_ref)) ** 2).sum().backward()
    (row(col(x_par)) ** 2).sum().backward()
    assert torch.allclose(x_ref.grad, x_par.grad, atol=1e-5), "input grad mismatch"
    assert torch.allclose(ref1.weight.grad[sl], col.weight.grad, atol=1e-5)
    assert torch.allclose(ref1.bias.grad[sl], col.bias.grad, atol=1e-5)
    assert torch.allclose(ref2.weight.grad[:, sl], row.weight.grad, atol=1e-5)
    # The replicated row bias gets the *full* gradient on every rank.
    assert torch.allclose(ref2.bias.grad, row.bias.grad, atol=1e-5)


def test_handrolled_tp_backward_matches_reference() -> None:
    run_distributed(_check_handrolled_backward, 2)
