"""Training throughput / memory / convergence benchmark (CPU + gloo).

Launch (world size = nproc_per_node; parallel layout from --tp, dp inferred)::

    OMP_NUM_THREADS=3 torchrun --standalone --nproc_per_node=2 \
        benchmarks/bench_train.py --preset small --tp 1 --steps 20 --out benchmarks/results/train.jsonl

Reports, per run (one JSONL record, rank 0): median/mean step time, global
tokens/sec, per-rank peak RSS (MiB), per-rank resident model-state bytes
(params / grads / optimizer state), and optionally the per-step loss curve.
It drives the real ``Trainer`` + ``train_step`` -- the same code path as ``train.py``.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from _common import REPO_ROOT, apply_overrides, emit, gather_obj, peak_rss_mb, state_bytes

from src.config import TrainingConfig
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=["tiny", "small", "mid"])
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--config", default=None, help="Use a complete model config instead of a preset")
    p.add_argument("--sharding", default=None, help="FULL_SHARD | NO_SHARD | SHARD_GRAD_OP")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--mbs", type=int, default=None, help="micro batch size per rank")
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--curve", action="store_true", help="record per-step loss")
    p.add_argument("--label", default="")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = TrainingConfig.from_yaml(a.config or f"{REPO_ROOT}/config/test_tiny.yaml")
    if a.config:
        if a.seq_len:
            cfg.data.seq_len = a.seq_len
        if a.mbs:
            cfg.data.micro_batch_size = a.mbs
        if a.accum:
            cfg.grad_accum_steps = a.accum
        cfg.parallel.tp_size = a.tp
        cfg.data.global_batch_size = 0
        cfg.data.num_workers = 0
        cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
    else:
        apply_overrides(cfg, preset=a.preset, seq_len=a.seq_len, micro_batch_size=a.mbs,
                        grad_accum=a.accum, tp=a.tp, sharding=a.sharding)
    cfg.backend = a.backend
    cfg.parallel.param_dtype = a.dtype
    cfg.parallel.buffer_dtype = a.dtype
    cfg.parallel.reduce_dtype = "float32"
    cfg.parallel.activation_checkpointing = a.activation_checkpointing
    if a.lr:
        cfg.optimizer.lr = a.lr
        cfg.scheduler.min_lr = a.lr / 10
    cfg.run_id = f"bench_{a.label or a.preset}"
    gpu_type = "cpu" if a.backend == "gloo" else "unlisted_gpu"
    trainer = Trainer(cfg, gpu_type=gpu_type)
    ctx = trainer.ctx
    rss_after_build = peak_rss_mb()

    times: list[float] = []
    losses: list[float] = []
    wall0 = 0.0
    try:
        for i in range(a.warmup + a.steps):
            if i == a.warmup:
                if ctx.device.type == "cuda":
                    torch.cuda.synchronize(ctx.device)
                    torch.cuda.reset_peak_memory_stats(ctx.device)
                wall0 = time.perf_counter()
            window, dw = trainer._next_window()
            t0 = time.perf_counter()
            m = train_step(trainer.model, window, trainer.optimizer, trainer.scheduler,
                           ctx, cfg, gpu_type=gpu_type, data_wait_s=dw)
            dt = time.perf_counter() - t0
            if i >= a.warmup:
                times.append(dt)
            losses.append(m.loss)
        wall = time.perf_counter() - wall0
        tokens_step = (cfg.data.micro_batch_size * cfg.data.seq_len
                       * cfg.grad_accum_steps * ctx.dims.dp_size)
        sb = state_bytes(trainer.model, trainer.optimizer)
        per_rank = gather_obj({"rank": ctx.rank, "peak_rss_mb": round(peak_rss_mb(), 1),
                               "rss_after_build_mb": round(rss_after_build, 1),
                               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(ctx.device) if ctx.device.type == "cuda" else 0,
                               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(ctx.device) if ctx.device.type == "cuda" else 0,
                               **sb})
        rec = {
            "bench": "train",
            "label": a.label,
            "preset": a.config or a.preset,
            "backend": a.backend, "dtype": a.dtype,
            "activation_checkpointing": a.activation_checkpointing,
            "world_size": ctx.world_size, "tp": ctx.dims.tp_size, "dp": ctx.dims.dp_size,
            "sharding": cfg.parallel.sharding_strategy,
            "seq_len": cfg.data.seq_len, "micro_batch_size": cfg.data.micro_batch_size,
            "grad_accum": cfg.grad_accum_steps, "tokens_per_step": tokens_step,
            "n_params": cfg.model.num_parameters(),
            "actual_n_params": sum(p.numel() for p in trainer.model.parameters()) if ctx.world_size == 1 else None,
            "steps": a.steps, "warmup": a.warmup,
            "step_time_median_s": round(statistics.median(times), 5),
            "step_time_mean_s": round(statistics.fmean(times), 5),
            "step_time_stdev_s": round(statistics.pstdev(times), 5),
            "tokens_per_s": round(tokens_step * a.steps / wall, 1),
            "tokens_per_s_median_step": round(tokens_step / statistics.median(times), 1),
            "per_rank": per_rank,
            "loss_first": losses[0], "loss_last": losses[-1],
        }
        if a.curve:
            rec["loss_curve"] = [round(x, 5) for x in losses]
        emit(rec, a.out)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
