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


def _cfg(tp: int, sharding: str = "FULL_SHARD") -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_hidden_size=64, max_seq_len=32, tie_embeddings=True,
    )
    cfg.parallel = ParallelConfig(
        tp_size=tp, sharding_strategy=sharding, param_dtype="float32",
        reduce_dtype="float32", buffer_dtype="float32",
    )
    cfg.optimizer = OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0, fused=False)
    cfg.scheduler = SchedulerConfig(warmup_steps=2, max_steps=10, min_lr=1e-4)
    cfg.data = DataConfig(
        dataset_path="synthetic", seq_len=16, micro_batch_size=_MBS,
        global_batch_size=0, num_workers=0, shuffle=False,
    )
    cfg.grad_accum_steps = 1
    cfg.max_steps = 10
    cfg.max_grad_norm = 1e9  # never clips; we compare the reported pre-clip norm
    cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
    cfg.backend = "gloo"
    cfg.seed = 5
    return cfg


def _reference(trainer: Trainer, dp: int) -> tuple[float, float]:
    cfg = trainer.config
    seed_everything(cfg.seed, 0)
    ref = build_model(cfg.model)
    ds = trainer.train_dataset
    items = [ds[i] for i in range(_MBS * dp)]
    ids = torch.stack([x["input_ids"] for x in items])
    labels = torch.stack([x["labels"] for x in items])
    _, loss = ref(ids, labels=labels)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), 1e9)
    return float(loss), float(norm)


def _check(rank: int, world_size: int, tp: int, sharding: str) -> None:
    cfg = _cfg(tp, sharding)
    trainer = Trainer(cfg, gpu_type="cpu")
    dp = trainer.ctx.dims.dp_size
    ref_loss, ref_norm = _reference(trainer, dp)
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
