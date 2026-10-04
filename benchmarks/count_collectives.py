"""Count the collectives one optimizer step actually issues (CPU + gloo).

Uses ``torch.profiler`` to list c10d / functional-collective ops that run inside a
single ``train_step`` and prints per-op call counts on rank 0. Used by
docs/PARALLELISM.md to verify the communication schedule against what the code
really does rather than against what the design says.

    torchrun --standalone --nproc_per_node=2 benchmarks/count_collectives.py --tp 2 --accum 2
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from _common import REPO_ROOT, apply_overrides, emit
from torch.profiler import ProfilerActivity, profile

from src.config import TrainingConfig
from src.parallelism.process_groups import destroy_distributed
from src.training.loop import train_step
from src.training.trainer import Trainer

_PAT = re.compile(r"all_?reduce|all_?gather|reduce_?scatter|broadcast|barrier|all_to_all", re.I)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = TrainingConfig.from_yaml(f"{REPO_ROOT}/config/test_tiny.yaml")
    apply_overrides(cfg, preset="tiny", seq_len=None, micro_batch_size=None,
                    grad_accum=a.accum, tp=a.tp)
    cfg.model.n_layers = a.layers
    trainer = Trainer(cfg, gpu_type="cpu")
    ctx = trainer.ctx
    for _ in range(2):  # warm up (lazy FSDP init issues extra broadcasts/collectives)
        w, _ = trainer._next_window()
        train_step(trainer.model, w, trainer.optimizer, trainer.scheduler, ctx, cfg)
    w, _ = trainer._next_window()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        train_step(trainer.model, w, trainer.optimizer, trainer.scheduler, ctx, cfg)
    counts: Counter[str] = Counter()
    for evt in prof.events():
        if _PAT.search(evt.name) and not evt.name.startswith("aten::") and evt.device_type.name == "CPU":
            counts[evt.name] += 1
    emit({"bench": "collective_count", "world_size": ctx.world_size, "dp": ctx.dims.dp_size,
          "tp": ctx.dims.tp_size, "grad_accum": a.accum, "n_layers": a.layers,
          "counts": dict(counts)}, a.out)
    destroy_distributed()
    _ = json


if __name__ == "__main__":
    main()
