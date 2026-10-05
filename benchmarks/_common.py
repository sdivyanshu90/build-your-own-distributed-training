"""Shared helpers for the CPU/gloo benchmark harness.

Every benchmark is launched with ``torchrun`` (or plain ``python`` for world
size 1) and appends one JSON object per run to a JSONL file under
``benchmarks/results/``. Nothing here is GPU specific: the harness measures what
the *code* does on CPU + gloo and says so in every record (``hardware`` field).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from src.config import TrainingConfig  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

# Named model sizes used by the benchmarks (all fp32, gloo-friendly).
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                 "ffn_hidden_size": 128, "max_seq_len": 64},
    "small": {"vocab_size": 4096, "d_model": 256, "n_layers": 4, "n_heads": 8, "n_kv_heads": 4,
                  "ffn_hidden_size": 704, "max_seq_len": 128},
    "mid": {"vocab_size": 8192, "d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4,
                "ffn_hidden_size": 1408, "max_seq_len": 128},
}


def apply_overrides(cfg: TrainingConfig, *, preset: str, seq_len: int | None,
                    micro_batch_size: int | None, grad_accum: int | None,
                    tp: int, sharding: str | None = None) -> TrainingConfig:
    """Turn the ``test_tiny`` config into a benchmark config (in place)."""
    for k, v in MODEL_PRESETS[preset].items():
        setattr(cfg.model, k, v)
    if seq_len:
        cfg.data.seq_len = seq_len
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, cfg.data.seq_len)
    if micro_batch_size:
        cfg.data.micro_batch_size = micro_batch_size
    if grad_accum:
        cfg.grad_accum_steps = grad_accum
    cfg.parallel.tp_size = tp
    if sharding:
        cfg.parallel.sharding_strategy = sharding
    cfg.data.global_batch_size = 0
    cfg.eval_interval = cfg.save_interval = cfg.profile_steps = 0
    cfg.log_interval = 10**9
    cfg.backend = "gloo"
    cfg.scheduler.max_steps = cfg.max_steps = 10**6  # constant-ish LR over the run
    cfg.scheduler.warmup_steps = 5
    return cfg


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux: ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tensor_bytes(t: Any) -> int:
    if t is None:
        return 0
    if hasattr(t, "to_local"):
        t = t.to_local()
    return int(t.numel() * t.element_size())


def state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Bytes of parameters / gradients / optimizer state *resident on this rank*."""
    params = sum(tensor_bytes(p) for p in model.parameters())
    grads = sum(tensor_bytes(p.grad) for p in model.parameters())
    opt = 0
    for st in optimizer.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                opt += tensor_bytes(v)
    return {"params": params, "grads": grads, "optimizer": opt}


def hardware_info() -> dict[str, Any]:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "backend": "gloo",
        "device": "cpu",
        "loadavg_1m_at_end": round(os.getloadavg()[0], 2),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
    }


def emit(record: dict[str, Any], out: str | None) -> None:
    """Rank-0 only: print and append one JSON record."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    record = {**record, "hardware": hardware_info()}
    line = json.dumps(record)
    print(line, flush=True)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "a") as f:
            f.write(line + "\n")


def gather_obj(obj: Any) -> list[Any]:
    if not dist.is_initialized():
        return [obj]
    out: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out
