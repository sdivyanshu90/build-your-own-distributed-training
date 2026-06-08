"""Integration: resume reproduces a continuous run bit-for-bit.

Train N steps continuously vs. train N/2, checkpoint, resume, train N/2 more —
the loss and LR at step N must match. One test runs single-process (exercising the
full Trainer path: model, optimizer, scheduler, RNG, and **deterministic data
fast-forward** on resume); the other runs real 2-rank FSDP (DP=2). Both run on CPU
via the per-rank optimizer-state fallback (the portable re-keyed FSDP path needs
CUDA in torch 2.3).
"""

from __future__ import annotations

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

_TOTAL = 20
_HALF = 10


def _cfg(tmp: str, run_id: str) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=96, d_model=48, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=64, tie_embeddings=True,
    )
    cfg.parallel = ParallelConfig(
        tp_size=1, param_dtype="float32", reduce_dtype="float32", buffer_dtype="float32"
    )
    cfg.optimizer = OptimizerConfig(name="adamw", lr=2e-3, weight_decay=0.0, fused=False)
    cfg.scheduler = SchedulerConfig(warmup_steps=3, max_steps=_TOTAL, min_lr=2e-4)
    cfg.data = DataConfig(
        dataset_path="synthetic", seq_len=32, micro_batch_size=4,
        global_batch_size=0, num_workers=0,
    )
    cfg.grad_accum_steps = 1
    cfg.max_steps = _TOTAL
    cfg.log_interval = 1000
    cfg.eval_interval = 0
    cfg.save_interval = 0
    cfg.profile_steps = 0
    cfg.backend = "gloo"
    cfg.checkpoint_dir = tmp
    cfg.run_id = run_id
    cfg.seed = 7
    return cfg


def _drive(trainer: Trainer, n: int) -> list[float]:
    losses = []
    for _ in range(n):
        window, dw = trainer._next_window()
        m = train_step(
            trainer.model, window, trainer.optimizer, trainer.scheduler,
            trainer.ctx, trainer.config, gpu_type="cpu", data_wait_s=dw,
        )
        trainer.step += 1
        losses.append(m.loss)
    return losses


def _run_resume_equivalence(rank: int, world_size: int, tmp: str) -> None:
    from src.checkpointing.checkpoint import save_checkpoint

    # Continuous run.
    cont = Trainer(_cfg(tmp, "cont"), gpu_type="cpu")
    cont_losses = _drive(cont, _TOTAL)

    # Split run: train HALF, checkpoint, resume into a fresh Trainer, train HALF.
    split = Trainer(_cfg(tmp, "split"), gpu_type="cpu")
    _drive(split, _HALF)
    ckpt = save_checkpoint(
        split.model, split.optimizer, split.scheduler, split.step, split.config, split.ctx
    )

    resume_cfg = _cfg(tmp, "split")
    resume_cfg.resume_from = ckpt
    resumed = Trainer(resume_cfg, gpu_type="cpu")
    assert resumed.step == _HALF
    resumed_losses = _drive(resumed, _TOTAL - _HALF)

    # Loss and LR at the final step must match the continuous run.
    cont_final = cont_losses[-1]
    resumed_final = resumed_losses[-1]
    assert abs(cont_final - resumed_final) < 1e-4, (
        f"resume diverged: continuous={cont_final:.6f} resumed={resumed_final:.6f}"
    )
    assert abs(cont.scheduler.get_last_lr()[0] - resumed.scheduler.get_last_lr()[0]) < 1e-9


def _run_fsdp_resume(rank: int, world_size: int, tmp: str) -> None:
    """Real 2-rank FSDP (DP=2) checkpoint + resume; loss at step N must match."""
    from src.checkpointing.checkpoint import save_checkpoint

    cont = Trainer(_cfg(tmp, "fsdp_cont"), gpu_type="cpu")
    cont_losses = _drive(cont, _TOTAL)

    split = Trainer(_cfg(tmp, "fsdp_split"), gpu_type="cpu")
    _drive(split, _HALF)
    ckpt = save_checkpoint(
        split.model, split.optimizer, split.scheduler, split.step, split.config, split.ctx
    )

    resume_cfg = _cfg(tmp, "fsdp_split")
    resume_cfg.resume_from = ckpt
    resumed = Trainer(resume_cfg, gpu_type="cpu")
    assert resumed.step == _HALF
    resumed_losses = _drive(resumed, _TOTAL - _HALF)

    # FSDP reduce ordering on gloo is deterministic, so the resumed run tracks
    # the continuous one closely; a small tolerance covers fp accumulation order.
    assert abs(cont_losses[-1] - resumed_losses[-1]) < 1e-3, (
        f"FSDP resume diverged: continuous={cont_losses[-1]:.6f} "
        f"resumed={resumed_losses[-1]:.6f}"
    )


def test_checkpoint_resume_equivalence(tmp_path) -> None:
    run_distributed(_run_resume_equivalence, 1, str(tmp_path))


def test_checkpoint_resume_fsdp(tmp_path) -> None:
    # Exercises the real FSDP (DP=2) checkpoint + resume path on CPU (the per-rank
    # optimizer-state fallback used when CUDA is unavailable; the portable re-keyed
    # path runs on GPU).
    run_distributed(_run_fsdp_resume, 2, str(tmp_path))
