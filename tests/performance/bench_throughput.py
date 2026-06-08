"""Throughput benchmark: tokens/sec, scaling efficiency, MFU, peak memory.

Launched under ``torchrun`` (one launch per parallelism config). Measures
steady-state tokens/sec for the launched ``(tp, dp)`` topology, reports MFU and
peak GPU memory, and can fail the run if MFU falls below a threshold — useful as a
performance regression gate in CI on a fixed GPU.

Example:
    # measure tp=2 on 8 GPUs, fail if MFU < 0.40
    torchrun --standalone --nproc_per_node=8 tests/performance/bench_throughput.py \\
        --config config/125m.yaml --tp-size 2 --steps 50 --assert-min-mfu 0.40

Scaling efficiency (throughput(N)/ (N * throughput(1))) is computed across runs by
the caller; this script prints the per-config tokens/sec so a wrapper can divide.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import sys
import time

import torch

from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer
from train import build_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Throughput benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--gpu-type", type=str, default="a100")
    p.add_argument("--assert-min-mfu", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ns = argparse.Namespace(
        config=args.config, tp_size=args.tp_size, dp_size=None, backend=args.backend,
        run_id="bench_tput", resume_from=None, max_steps=None,
    )
    config = build_config(ns)
    config.eval_interval = 0
    config.save_interval = 0
    config.profile_steps = 0
    trainer = Trainer(config, gpu_type=args.gpu_type)
    ctx = trainer.ctx

    try:
        for _ in range(args.warmup):
            window, _ = trainer._next_window()
            train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, config, gpu_type=args.gpu_type)

        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
            torch.cuda.synchronize(ctx.device)
        t0 = time.perf_counter()
        total_tokens = 0
        mfu_sum = 0.0
        for _ in range(args.steps):
            window, _ = trainer._next_window()
            m = train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, config, gpu_type=args.gpu_type)
            total_tokens += config.data.micro_batch_size * config.data.seq_len * config.grad_accum_steps * ctx.dims.dp_size
            mfu_sum += m.mfu or 0.0
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        elapsed = time.perf_counter() - t0

        tokens_per_sec = total_tokens / elapsed
        avg_mfu = mfu_sum / args.steps
        peak_mb = (torch.cuda.max_memory_allocated(ctx.device) / 1e6) if ctx.device.type == "cuda" else 0.0

        if ctx.is_rank0:
            print(json.dumps({
                "event": "bench_throughput",
                "tp_size": ctx.dims.tp_size, "dp_size": ctx.dims.dp_size, "world_size": ctx.world_size,
                "tokens_per_sec": round(tokens_per_sec, 1),
                "tokens_per_sec_per_gpu": round(tokens_per_sec / ctx.world_size, 1),
                "avg_mfu": round(avg_mfu, 4),
                "peak_mem_mb": round(peak_mb, 1),
                "steps": args.steps,
            }))

        if args.assert_min_mfu is not None and avg_mfu < args.assert_min_mfu:
            if ctx.is_rank0:
                print(f"FAIL: MFU {avg_mfu:.4f} < threshold {args.assert_min_mfu}", file=sys.stderr)
            sys.exit(2)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
