"""Fixed CPU-thread-budget scaling, GPU resume, and trace checks after the full matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    with open("/tmp/distributed-training-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        failures = []

        def run(name: str, cmd: list[str], threads: int = 1, gpu: bool = False) -> None:
            start = time.time()
            print("START", name, flush=True)
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES="0" if gpu else "")
            with (out / f"{name}.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    rc = 124
            with (out / "supplement-execution.jsonl").open("a") as record:
                record.write(json.dumps({"case": name, "exit_code": rc, "duration_s": time.time() - start,
                                         "started_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
                                         "command": cmd, "omp_threads": threads, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"]}) + "\n")
            if rc:
                failures.append(name)
            print("END", name, "exit", rc, flush=True)

        for world, threads in [(1, 4), (2, 2), (4, 1)]:
            run(f"budget4-dp{world}", [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                                       "benchmarks/bench_train.py", "--preset", "small", "--mbs", "4", "--seq-len", "128",
                                       "--steps", "20", "--warmup", "3", "--tp", "1", "--label", f"dp{world}_budget4",
                                       "--out", str(out / "thread-budget.jsonl")], threads=threads)
        for dtype in ["float32", "bfloat16"]:
            run(f"cuda-resume-{dtype}", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                                         "benchmarks/bench_resume.py", "--backend", "nccl", "--dtype", dtype,
                                         "--preset", "small", "--steps", "30", "--out", str(out / "cuda-resume.jsonl")], gpu=True)
        for backend, world in [("gloo", 2), ("nccl", 1)]:
            run(f"profile-{backend}", [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                                       "benchmarks/bench_profile.py", "--backend", backend, "--out-dir", str(out)], gpu=backend == "nccl")
        print(json.dumps({"failed_cases": failures}), flush=True)
        raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
