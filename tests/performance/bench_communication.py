"""Communication benchmark: fraction of step time in collectives vs compute.

Uses ``torch.profiler`` to record a short steady-state window and attributes CUDA
time to NCCL/collective kernels (all-gather, reduce-scatter, all-reduce) vs.
compute. Also contrasts prefetch on vs off so you can quantify how much
communication is hidden behind compute.

Launched under ``torchrun``. Meaningful on CUDA only (NCCL kernels); on CPU it
reports a zero breakdown but still validates the profiler integration.

Example:
    torchrun --standalone --nproc_per_node=8 tests/performance/bench_communication.py \\
        --config config/125m.yaml --tp-size 2
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json

from src.config import TrainingConfig
from src.observability.profiler import build_profiler, communication_fraction
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Communication/overlap benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=8)
    return p.parse_args()


def _run(forward_prefetch: bool, backward_prefetch: str, args: argparse.Namespace) -> dict:
    cfg = TrainingConfig.from_yaml(args.config)
    cfg.parallel.tp_size = args.tp_size
    cfg.parallel.forward_prefetch = forward_prefetch
    cfg.parallel.backward_prefetch = backward_prefetch
    if args.backend:
        cfg.backend = args.backend
    cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
    cfg.run_id = f"comm_fp{int(forward_prefetch)}"
    trainer = Trainer(cfg)
    ctx = trainer.ctx

    for _ in range(args.warmup):
        window, _ = trainer._next_window()
        train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, cfg)

    prof = build_profiler(f"traces/{cfg.run_id}", warmup_steps=1, profile_steps=args.steps, record_memory=False)
    with prof:
        for _ in range(args.steps + 2):
            window, _ = trainer._next_window()
            train_step(trainer.model, window, trainer.optimizer, trainer.scheduler, ctx, cfg)
            prof.step()
    breakdown = communication_fraction(prof)
    return {
        "forward_prefetch": forward_prefetch,
        "backward_prefetch": backward_prefetch,
        "comm_fraction": round(breakdown.comm_fraction, 4),
        "comm_us": round(breakdown.comm_cuda_us, 1),
        "compute_us": round(breakdown.compute_cuda_us, 1),
    }


def main() -> None:
    args = parse_args()
    import torch.distributed as dist

    on = _run(True, "BACKWARD_PRE", args)
    off = _run(False, "BACKWARD_POST", args)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps({
            "event": "bench_communication",
            "tp_size": args.tp_size,
            "prefetch_on": on,
            "prefetch_off": off,
            "hidden_by_overlap": round(off["comm_fraction"] - on["comm_fraction"], 4),
        }, indent=2))
    destroy_distributed()


if __name__ == "__main__":
    main()
