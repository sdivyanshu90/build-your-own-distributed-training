"""Run the complete CPU matrix and single-GPU measurements into an isolated directory."""
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

import psutil

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out / "execution.jsonl"
    if status.exists():
        raise SystemExit("Use a fresh result directory; execution record already exists")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", BENCH_RESULTS_DIR=str(out))
    with open("/tmp/distributed-training-benchmark.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.wait_pid:
            while psutil.pid_exists(args.wait_pid):
                try:
                    if "pytest" not in " ".join(psutil.Process(args.wait_pid).cmdline()):
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(3)
        failures = []

        def run(name: str, cmd: list[str], timeout: int = 2400, gpu: bool = False) -> None:
            print(f"START {name}", flush=True)
            start = time.time()
            if psutil.virtual_memory().available < 1500 * 1024**2:
                raise RuntimeError("Less than 1500 MiB RAM available")
            case_env = dict(env, CUDA_VISIBLE_DEVICES="0" if gpu else "")
            with open(out / f"{name}.log", "w") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=case_env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                try:
                    rc = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    rc = 124
            row = {"case": name, "started_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
                   "duration_s": round(time.time() - start, 3), "exit_code": rc,
                   "timeout": timed_out, "command": cmd, "cuda_visible_devices": case_env["CUDA_VISIBLE_DEVICES"]}
            with open(status, "a") as record:
                record.write(json.dumps(row) + "\n")
            if rc:
                failures.append(name)
            print(f"END {name}: exit={rc} duration={row['duration_s']}s", flush=True)

        for stage in ["scaling", "memory", "comm", "resume", "fault", "convergence"]:
            run(stage, ["bash", str(ROOT / "benchmarks/run_all.sh"), stage, sys.executable])
        for world, tp, accum, layers in [(2, 1, 1, 2), (2, 1, 2, 2), (2, 2, 1, 2),
                                         (2, 2, 1, 4), (4, 2, 1, 2), (4, 2, 2, 2)]:
            run(f"count-w{world}-tp{tp}-k{accum}-l{layers}",
                [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                 "benchmarks/count_collectives.py", "--tp", str(tp), "--accum", str(accum),
                 "--layers", str(layers), "--out", str(out / "collectives.jsonl")], timeout=300)
        # Three independent paired scaling observations, retaining the complete first sweep.
        for repeat in [2, 3]:
            for world in [1, 2, 4]:
                run(f"repeat{repeat}-dp{world}",
                    [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                     "benchmarks/bench_train.py", "--preset", "small", "--mbs", "4", "--seq-len", "128",
                     "--steps", "20", "--warmup", "3", "--tp", "1", "--label", f"dp{world}_r{repeat}",
                     "--out", str(out / "repeats.jsonl")], timeout=300)
        if not args.cpu_only:
            for dtype in ["float32", "bfloat16"]:
                for preset in ["small", "mid"]:
                    run(f"cuda-{preset}-{dtype}",
                        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                         "benchmarks/bench_train.py", "--backend", "nccl", "--dtype", dtype,
                         "--preset", preset, "--seq-len", "128", "--mbs", "4", "--steps", "50",
                         "--warmup", "10", "--label", f"cuda_{preset}_{dtype}",
                         "--out", str(out / "cuda.jsonl")], timeout=300, gpu=True)
            for seq in [128, 512, 1024]:
                for ac in [False, True]:
                    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                           "benchmarks/bench_train.py", "--backend", "nccl", "--dtype", "bfloat16",
                           "--preset", "mid", "--seq-len", str(seq), "--mbs", "1", "--accum", "1",
                           "--steps", "20", "--warmup", "5", "--label", f"cuda_mem_s{seq}_ac{int(ac)}",
                           "--out", str(out / "cuda-memory.jsonl")]
                    if ac:
                        cmd.append("--activation-checkpointing")
                    run(f"cuda-memory-s{seq}-ac{int(ac)}", cmd, timeout=300, gpu=True)
            for ac in [False, True]:
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                       "benchmarks/bench_train.py", "--config", "config/125m.yaml", "--backend", "nccl",
                       "--dtype", "bfloat16", "--seq-len", "128", "--mbs", "1", "--accum", "1",
                       "--steps", "10", "--warmup", "3", "--label", f"cuda_125m_s128_ac{int(ac)}",
                       "--out", str(out / "cuda-125m.jsonl")]
                if ac:
                    cmd.append("--activation-checkpointing")
                run(f"cuda-125m-ac{int(ac)}", cmd, timeout=300, gpu=True)
        run("report", [sys.executable, "benchmarks/make_report.py", "--plots"])
        print(json.dumps({"failed_cases": failures}), flush=True)
        raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
