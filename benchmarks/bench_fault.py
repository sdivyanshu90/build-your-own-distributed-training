"""Rank-failure recovery benchmark (CPU + gloo, real processes, real SIGKILL).

Orchestrator (run with plain ``python``, it launches ``torchrun`` itself)::

    python benchmarks/bench_fault.py --nproc 2 --out benchmarks/results/fault.jsonl

Timeline measured with wall clock:
  1. launch ``torchrun train.py`` (test_tiny, FSDP dp=nproc) with periodic checkpoints;
  2. once a checkpoint ``step_K`` is committed (``_SUCCESS`` present) and training has
     moved past it, SIGKILL one worker;
  3. ``detect_s``  = kill -> torchrun exits (the agent notices and tears the job down);
  4. relaunch with ``--resume-from <run dir>`` (newest *valid* checkpoint is chosen);
  5. ``recover_s`` = relaunch -> first post-resume metric line;  also reports the
     step resumed from, and the number of steps of work lost.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _launch(nproc: int, cfg: str, ckpt: str, run_id: str, log: str, resume: str | None):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={nproc}", os.path.join(REPO, "train.py"),
           "--config", cfg, "--backend", "gloo", "--run-id", run_id]
    if resume:
        cmd += ["--resume-from", resume]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": REPO}
    with open(log, "w") as out:
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=REPO,
                                env=env, start_new_session=True)


def _wait_for(pred, timeout: float, what: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        time.sleep(0.05)
    raise TimeoutError(what)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--save-interval", type=int, default=10)
    ap.add_argument("--kill-after-step", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="dt_bench_fault_")
    try:
        with open(os.path.join(REPO, "config/test_tiny.yaml")) as fh:
            raw = yaml.safe_load(fh)
        raw.update(log_interval=1, eval_interval=0, save_interval=a.save_interval,
                   max_steps=200, checkpoint_dir=os.path.join(tmp, "ckpt"))
        raw["scheduler"]["max_steps"] = 200
        cfg = os.path.join(tmp, "cfg.yaml")
        with open(cfg, "w") as fh:
            yaml.safe_dump(raw, fh)
        run_dir = os.path.join(tmp, "ckpt", "fault")
        log1, log2 = os.path.join(tmp, "run1.log"), os.path.join(tmp, "run2.log")

        t_start = time.time()
        p = _launch(a.nproc, cfg, tmp, "fault", log1, None)

        def last_step() -> int:
            try:
                with open(log1) as fh:
                    steps = [json.loads(line)["step"] for line in fh
                             if line.startswith("{") and '"event": "step"' in line]
                return max(steps) if steps else 0
            except (OSError, ValueError, KeyError):
                return 0

        _wait_for(lambda: last_step() >= a.kill_after_step, 300, "training did not reach kill step")
        step_at_kill = last_step()
        workers = [c for c in psutil.Process(p.pid).children(recursive=True)
                   if "train.py" in " ".join(c.cmdline())]
        victim = workers[-1]
        t_kill = time.time()
        os.kill(victim.pid, signal.SIGKILL)
        p.wait(timeout=300)
        detect_s = time.time() - t_kill
        for c in psutil.Process().children(recursive=True):  # belt and braces
            if "train.py" in " ".join(c.cmdline()):
                c.kill()
        committed = sorted(d for d in os.listdir(run_dir) if os.path.exists(os.path.join(run_dir, d, "_SUCCESS")))

        t_relaunch = time.time()
        p2 = _launch(a.nproc, cfg, tmp, "fault", log2, run_dir)

        def resumed_metric() -> bool:
            try:
                with open(log2) as fh:
                    return any(line.startswith("{") and '"event": "step"' in line for line in fh)
            except OSError:
                return False

        _wait_for(resumed_metric, 300, "no metric after resume")
        recover_s = time.time() - t_relaunch
        resumed_from = None
        with open(log2) as fh:
            for line in fh:
                if line.startswith("{") and '"event": "resumed"' in line:
                    resumed_from = json.loads(line)["step"]
        os.killpg(p2.pid, signal.SIGTERM)
        p2.wait(timeout=60)
        rec = {
            "bench": "fault", "world_size": a.nproc, "save_interval": a.save_interval,
            "step_at_kill": step_at_kill, "committed_checkpoints": committed,
            "resumed_from_step": resumed_from,
            "steps_of_work_lost": None if resumed_from is None else step_at_kill - resumed_from,
            "detect_s": round(detect_s, 2), "recover_s_relaunch_to_first_step": round(recover_s, 2),
            "time_to_kill_s": round(t_kill - t_start, 2),
        }
        from _common import hardware_info
        rec["hardware"] = hardware_info()
        print(json.dumps(rec))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
