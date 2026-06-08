"""Training entry point — launched under ``torchrun``.

Usage
-----
Single-node, 8 GPUs, TP=2 (so DP=4):

    torchrun --standalone --nproc_per_node=8 train.py \
        --config config/125m.yaml --tp-size 2

Multi-node (2 nodes x 8 GPUs), TP=8 intra-node, DP=2 across nodes:

    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
        train.py --config config/7b.yaml --tp-size 8

CPU functional run (no GPU, for local testing):

    torchrun --standalone --nproc_per_node=4 train.py \
        --config config/test_tiny.yaml --tp-size 2 --backend gloo

What this file does
-------------------
Parses CLI overrides, loads the YAML config, constructs the :class:`Trainer`, and
runs it — wrapping everything in a try/finally that always tears down the process
group so NCCL communicators are released even on failure. It deliberately does the
*minimum* orchestration; all real logic lives in the library so it is unit-tested.
"""

from __future__ import annotations

import argparse

from src.config import TrainingConfig
from src.parallelism.process_groups import destroy_distributed
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (a thin set of overrides on top of the YAML config)."""
    parser = argparse.ArgumentParser(description="2D-parallel (FSDP x TP) LM training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--tp-size", type=int, default=None, help="Override TP degree.")
    parser.add_argument("--dp-size", type=int, default=None, help="Override DP degree.")
    parser.add_argument("--backend", type=str, default=None, help="nccl | gloo.")
    parser.add_argument("--run-id", type=str, default=None, help="Override run id.")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint dir.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps.")
    parser.add_argument("--gpu-type", type=str, default="a100", help="GPU key for MFU.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Load the YAML config and apply CLI overrides."""
    config = TrainingConfig.from_yaml(args.config)
    if args.tp_size is not None:
        config.parallel.tp_size = args.tp_size
    if args.dp_size is not None:
        config.parallel.dp_size = args.dp_size
    if args.backend is not None:
        config.backend = args.backend
    if args.run_id is not None:
        config.run_id = args.run_id
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.max_steps is not None:
        config.max_steps = args.max_steps
        config.scheduler.max_steps = args.max_steps
    return config


def main() -> None:
    """Entry point: build config, run the trainer, always tear down the PG."""
    args = parse_args()
    config = build_config(args)
    try:
        trainer = Trainer(config, gpu_type=args.gpu_type)
        trainer.train()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
