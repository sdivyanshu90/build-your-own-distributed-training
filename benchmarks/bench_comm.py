"""Collective-communication micro-benchmark on gloo (CPU).

Measures all_reduce / all_gather / reduce_scatter latency and bandwidth versus
message size across the *world* group. ``algbw`` = message_bytes / time (the
convention of nccl-tests); ``busbw`` applies the standard ring correction
factors (all_reduce 2(n-1)/n, all_gather & reduce_scatter (n-1)/n) so numbers are
comparable across world sizes. Message size = bytes of the per-rank *input*
tensor for all_reduce / reduce_scatter-output-times-n convention noted below.

    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 benchmarks/bench_comm.py \
        --out benchmarks/results/comm.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.distributed as dist
from _common import emit


def _time(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    dist.barrier()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-kb", type=int, nargs="+",
                   default=[1, 4, 16, 64, 256, 1024, 4096, 16384])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    dist.init_process_group("gloo")
    n = dist.get_world_size()
    rows = []
    for kb in a.sizes_kb:
        numel = kb * 1024 // 4  # fp32 elements; message = per-rank full buffer
        numel -= numel % n
        nbytes = numel * 4
        x = torch.randn(numel)
        shard = torch.randn(numel // n)
        out_full = torch.empty(numel)
        ops = {
            "all_reduce": (lambda x=x: dist.all_reduce(x), 2 * (n - 1) / n),
            "all_gather": (lambda o=out_full, s=shard: dist.all_gather_into_tensor(o, s), (n - 1) / n),
            "reduce_scatter": (lambda o=shard, i=x: dist.reduce_scatter_tensor(o, i), (n - 1) / n),
        }
        for name, (fn, factor) in ops.items():
            ts = _time(fn, a.iters, a.warmup)
            med = statistics.median(ts)
            rows.append({
                "op": name, "message_bytes": nbytes, "world_size": n,
                "latency_median_us": round(med * 1e6, 1),
                "latency_p90_us": round(sorted(ts)[int(0.9 * len(ts)) - 1] * 1e6, 1),
                "algbw_MBps": round(nbytes / med / 1e6, 1),
                "busbw_MBps": round(nbytes * factor / med / 1e6, 1),
            })
    emit({"bench": "comm", "world_size": n, "rows": rows,
          "note": "message = full per-rank buffer; all_gather output / reduce_scatter input are this size"},
         a.out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
