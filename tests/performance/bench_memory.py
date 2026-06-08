"""Memory benchmark: peak GPU memory vs sequence length, AC, and TP degree.

Demonstrates the two headline memory properties:
  * Activation checkpointing trades compute for memory: peak memory grows much
    more slowly with sequence length when AC is on.
  * FSDP reduces per-GPU parameter memory from ``O(P)`` to ``O(P/dp_size)``.

Launched under ``torchrun``. On CPU it reports zeros for CUDA memory (the metric
is GPU-only) but still validates that the configurations construct and run.

Example:
    torchrun --standalone --nproc_per_node=8 tests/performance/bench_memory.py \\
        --config config/125m.yaml --tp-size 2 --seq-lens 512 1024 2048 4096
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json

import torch

from src.config import TrainingConfig
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--gpu-type", type=str, default="a100")
    return p.parse_args()


def _measure(config: TrainingConfig, gpu_type: str, ac: bool) -> float:
    config.parallel.activation_checkpointing = ac
    trainer = Trainer(config, gpu_type=gpu_type)
    ctx = trainer.ctx
    if ctx.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(ctx.device)
    for _ in range(3):
        window, _ = trainer._next_window()
        train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, config, gpu_type=gpu_type)
    peak = (torch.cuda.max_memory_allocated(ctx.device) / 1e6) if ctx.device.type == "cuda" else 0.0
    return peak


def main() -> None:
    args = parse_args()
    base = TrainingConfig.from_yaml(args.config)
    base.parallel.tp_size = args.tp_size
    if args.backend:
        base.backend = args.backend
    base.eval_interval = base.save_interval = base.profile_steps = 0

    # We must rebuild the trainer per seq_len (the model's RoPE cache and data
    # depend on it); each rebuild reuses the same process group.
    results = []
    for seq_len in args.seq_lens:
        for ac in (False, True):
            cfg = TrainingConfig.from_yaml(args.config)
            cfg.parallel.tp_size = args.tp_size
            if args.backend:
                cfg.backend = args.backend
            cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
            cfg.data.seq_len = seq_len
            cfg.model.max_seq_len = max(cfg.model.max_seq_len, seq_len)
            cfg.run_id = f"mem_s{seq_len}_ac{int(ac)}"
            peak = _measure(cfg, args.gpu_type, ac)
            results.append({"seq_len": seq_len, "activation_checkpointing": ac, "peak_mem_mb": round(peak, 1)})

    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps({"event": "bench_memory", "tp_size": args.tp_size, "results": results}, indent=2))
    destroy_distributed()


if __name__ == "__main__":
    main()
