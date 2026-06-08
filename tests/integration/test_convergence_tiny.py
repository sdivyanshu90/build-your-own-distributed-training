"""Integration: a tiny model converges under every runnable parallelism config.

The strongest end-to-end correctness signal: drive real optimizer steps through
the full stack (model + parallelism + optimizer + scheduler + data) and assert the
loss drops substantially on the learnable synthetic Markov corpus. We verify three
configs that run on the CPU/Gloo test path:

    * single process            (baseline, no parallelism)
    * pure FSDP   (tp=1, dp=2)
    * pure TP     (tp=2, dp=1)

The composed 2D (FSDP+TP) config is the same code; it is validated on real
multi-GPU/NCCL hardware (the CPU+torch-2.3 FSDP-writeback-with-DTensor bug blocks
it here — see :mod:`tests.integration.test_2d_parallel_forward`). Bit-identical
cross-topology curves additionally require a matched effective batch and identical
data ordering; here we assert convergence per config, which catches the real
failure modes (wrong group, broken TP math, no_sync misuse) that would prevent
learning at all.
"""

from __future__ import annotations

import torch

from src.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.training.loop import train_step
from src.training.trainer import Trainer
from tests._dist_utils import run_distributed

_MAX_STEPS = 60


def _convergence_config(tp_size: int) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=64, tie_embeddings=True,
    )
    cfg.parallel = ParallelConfig(
        tp_size=tp_size, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32"
    )
    cfg.optimizer = OptimizerConfig(name="adamw", lr=3e-3, weight_decay=0.0, fused=False)
    cfg.scheduler = SchedulerConfig(warmup_steps=5, max_steps=_MAX_STEPS, min_lr=3e-4)
    cfg.data = DataConfig(
        dataset_path="synthetic", seq_len=32, micro_batch_size=8,
        global_batch_size=0, num_workers=0,
    )
    cfg.grad_accum_steps = 1
    cfg.max_steps = _MAX_STEPS
    cfg.log_interval = 1000
    cfg.eval_interval = 0
    cfg.save_interval = 0
    cfg.profile_steps = 0
    cfg.backend = "gloo"
    cfg.run_id = f"converge_tp{tp_size}"
    return cfg


def _run_convergence(rank: int, world_size: int, tp_size: int) -> None:
    cfg = _convergence_config(tp_size)
    trainer = Trainer(cfg, gpu_type="cpu")
    losses: list[float] = []
    for _ in range(cfg.max_steps):
        window, data_wait = trainer._next_window()
        metrics = train_step(
            trainer.model, window, trainer.optimizer, trainer.scheduler,
            trainer.ctx, cfg, gpu_type="cpu", data_wait_s=data_wait,
        )
        losses.append(metrics.loss)
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert torch.isfinite(torch.tensor(final)), f"rank {rank}: loss went non-finite"
    assert final < initial - 1.0, (
        f"[tp={tp_size}] loss did not converge: initial≈{initial:.3f} final≈{final:.3f}"
    )


def test_convergence_single_process() -> None:
    run_distributed(_run_convergence, 1, 1)


def test_convergence_pure_fsdp() -> None:
    run_distributed(_run_convergence, 2, 1)


def test_convergence_pure_tp() -> None:
    run_distributed(_run_convergence, 2, 2)
