"""Fault injection: recover from a crash by resuming the last valid checkpoint.

Simulates a job that checkpointed at step 10, then died while writing step 20
(leaving an unmarked/incomplete checkpoint). On restart, recovery must skip the
broken step-20 checkpoint, find step 10, and resume training from it — the core
of elastic fault tolerance. Runs single-process on CPU; the multi-rank elastic
re-join (a killed rank reloading its shard) is the same code path validated on
NCCL hardware.
"""

from __future__ import annotations

import os

import torch

from src.checkpointing.checkpoint import save_checkpoint
from src.checkpointing.recovery import find_latest_valid_checkpoint
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


def _cfg(tmp: str) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(vocab_size=96, d_model=48, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    cfg.parallel = ParallelConfig(tp_size=1, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32")
    cfg.optimizer = OptimizerConfig(name="adamw", lr=2e-3, weight_decay=0.0, fused=False)
    cfg.scheduler = SchedulerConfig(warmup_steps=3, max_steps=40, min_lr=2e-4)
    cfg.data = DataConfig(dataset_path="synthetic", seq_len=32, micro_batch_size=4, global_batch_size=0, num_workers=0)
    cfg.grad_accum_steps = 1
    cfg.max_steps = 40
    cfg.eval_interval = 0
    cfg.save_interval = 0
    cfg.profile_steps = 0
    cfg.log_interval = 1000
    cfg.backend = "gloo"
    cfg.checkpoint_dir = tmp
    cfg.run_id = "recover"
    cfg.seed = 11
    return cfg


def _drive(trainer: Trainer, n: int) -> None:
    for _ in range(n):
        window, dw = trainer._next_window()
        train_step(
            trainer.model, window, trainer.optimizer, trainer.scheduler,
            trainer.ctx, trainer.config, gpu_type="cpu", data_wait_s=dw,
        )
        trainer.step += 1


def _run_recovery(rank: int, world_size: int, tmp: str) -> None:
    cfg = _cfg(tmp)
    trainer = Trainer(cfg, gpu_type="cpu")

    # Good checkpoint at step 10.
    _drive(trainer, 10)
    good = save_checkpoint(trainer.model, trainer.optimizer, trainer.scheduler, 10, cfg, trainer.ctx)

    # Crash while writing step 20: save then strip the _SUCCESS marker.
    _drive(trainer, 10)
    bad = save_checkpoint(trainer.model, trainer.optimizer, trainer.scheduler, 20, cfg, trainer.ctx)
    os.remove(os.path.join(bad, "_SUCCESS"))

    # Restart: recovery skips the broken step-20 and finds step-10.
    run_dir = os.path.join(cfg.checkpoint_dir, cfg.run_id)
    latest = find_latest_valid_checkpoint(run_dir)
    assert latest is not None and latest.endswith("step_10"), f"recovery picked {latest}"
    assert latest == good

    resume_cfg = _cfg(tmp)
    resume_cfg.resume_from = latest
    resumed = Trainer(resume_cfg, gpu_type="cpu")
    assert resumed.step == 10, "restarted process must resume at the checkpoint step"

    # It can keep training (re-joined, shard loaded).
    window, dw = resumed._next_window()
    from src.training.loop import train_step as ts

    m = ts(resumed.model, window, resumed.optimizer, resumed.scheduler, resumed.ctx, resume_cfg, gpu_type="cpu")
    assert torch.isfinite(torch.tensor(m.loss)), "resumed training produced non-finite loss"


def test_recover_from_crash(tmp_path) -> None:
    run_distributed(_run_recovery, 1, str(tmp_path))
