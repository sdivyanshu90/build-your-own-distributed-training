"""Checkpoint cost + resume-exactness benchmark (CPU + gloo).

Inside one ``torchrun`` job (world size = FSDP degree, tp=1) this:

1. trains ``--steps`` steps continuously (run A) recording the per-step loss;
2. trains ``--steps//2`` steps (run B), saves a sharded checkpoint (timed, sized);
3. builds a fresh ``Trainer`` with ``resume_from`` (run C; timed load), trains the
   remaining half and compares C's per-step losses with A's second half.

Reported: save seconds, load seconds, bytes on disk, max |loss_A - loss_C| over the
second half, and the full curves so the claim can be inspected, not trusted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

from _common import REPO_ROOT, apply_overrides, emit

from src.checkpointing.checkpoint import save_checkpoint
from src.config import TrainingConfig
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer


def _dir_bytes(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _cfg(a: argparse.Namespace, run_id: str, ckpt_dir: str) -> TrainingConfig:
    cfg = TrainingConfig.from_yaml(f"{REPO_ROOT}/config/test_tiny.yaml")
    apply_overrides(cfg, preset=a.preset, seq_len=None, micro_batch_size=None,
                    grad_accum=None, tp=1)
    cfg.max_steps = cfg.scheduler.max_steps = a.steps
    cfg.scheduler.warmup_steps = 3
    cfg.checkpoint_dir = ckpt_dir
    cfg.run_id = run_id
    return cfg


def _drive(t: Trainer, n: int) -> list[float]:
    out = []
    for _ in range(n):
        w, dw = t._next_window()
        out.append(train_step(t.model, w, t.optimizer, t.scheduler, t.ctx, t.config,
                              gpu_type="cpu", data_wait_s=dw).loss)
        t.step += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=["tiny", "small", "mid"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    half = a.steps // 2
    # Shared tmp dir across ranks: rank 0 creates it, path broadcast via env-free file.
    import torch.distributed as dist
    tmp_holder = [tempfile.mkdtemp(prefix="dt_bench_ckpt_") if int(os.environ.get("RANK", 0)) == 0 else None]
    from src.parallelism.process_groups import init_distributed
    init_distributed("gloo")
    dist.broadcast_object_list(tmp_holder, src=0)
    tmp = tmp_holder[0]
    try:
        A = Trainer(_cfg(a, "A", tmp), gpu_type="cpu")
        curve_a = _drive(A, a.steps)
        B = Trainer(_cfg(a, "B", tmp), gpu_type="cpu")
        _drive(B, half)
        dist.barrier()
        t0 = time.perf_counter()
        ckpt = save_checkpoint(B.model, B.optimizer, B.scheduler, B.step, B.config, B.ctx)
        save_s = time.perf_counter() - t0
        size = _dir_bytes(ckpt)
        cfg_c = _cfg(a, "B", tmp)
        cfg_c.resume_from = ckpt
        dist.barrier()
        t0 = time.perf_counter()
        C = Trainer(cfg_c, gpu_type="cpu")  # includes model build + load
        build_and_load_s = time.perf_counter() - t0
        curve_c = _drive(C, a.steps - half)
        diffs = [abs(x - y) for x, y in zip(curve_a[half:], curve_c, strict=True)]
        emit({
            "bench": "resume", "preset": a.preset, "world_size": B.ctx.world_size,
            "dp": B.ctx.dims.dp_size, "steps": a.steps, "resume_at": half,
            "n_params": B.config.model.num_parameters(),
            "ckpt_save_s": round(save_s, 4), "trainer_build_plus_load_s": round(build_and_load_s, 4),
            "ckpt_bytes_total": size, "max_abs_loss_diff": max(diffs),
            "bit_identical_steps": sum(1 for d in diffs if d == 0.0), "compared_steps": len(diffs),
            "loss_continuous": [round(x, 6) for x in curve_a],
            "loss_resumed_second_half": [round(x, 6) for x in curve_c],
        }, a.out)
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(tmp, ignore_errors=True)
        destroy_distributed()


if __name__ == "__main__":
    main()
