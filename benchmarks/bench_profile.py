"""Capture a short real training trace; profiling timings are separate from throughput."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import torch
from _common import REPO_ROOT, apply_overrides, emit

from src.config import TrainingConfig
from src.observability.profiler import communication_fraction, text_summary
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    out = Path(args.out_dir)
    cfg = TrainingConfig.from_yaml(f"{REPO_ROOT}/config/test_tiny.yaml")
    apply_overrides(cfg, preset="small", seq_len=128, micro_batch_size=4, grad_accum=2, tp=1)
    cfg.backend = args.backend
    if args.backend == "nccl":
        cfg.parallel.param_dtype = cfg.parallel.buffer_dtype = args.dtype
    trainer = Trainer(cfg, gpu_type="unlisted_gpu")
    ctx = trainer.ctx
    try:
        for _ in range(5):
            window, wait = trainer._next_window()
            train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, cfg, gpu_type="unlisted_gpu", data_wait_s=wait)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if ctx.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            for _ in range(3):
                window, wait = trainer._next_window()
                train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, cfg, gpu_type="unlisted_gpu", data_wait_s=wait)
        if ctx.is_rank0:
            stem = f"profile-{args.backend}"
            path = out / f"{stem}.json"
            prof.export_chrome_trace(str(path))
            with path.open("rb") as source, gzip.open(str(path) + ".gz", "wb") as dest:
                dest.write(source.read())
            path.unlink()
            (out / f"{stem}.txt").write_text(text_summary(prof, 30))
            breakdown = communication_fraction(prof)
            ops = [{"name": e.key, "count": e.count, "self_cpu_us": e.self_cpu_time_total,
                    "self_device_us": getattr(e, "self_device_time_total", 0)} for e in prof.key_averages()]
            emit({"bench": "profile", "world_size": ctx.world_size, "backend": args.backend,
                  "steps": 3, "communication_fraction": breakdown.comm_fraction,
                  "total_attributed_us": breakdown.total_cuda_us,
                  "communication_attributed_us": breakdown.comm_cuda_us, "operators": ops},
                 str(out / "profiles.jsonl"))
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
