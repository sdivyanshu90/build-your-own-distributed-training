"""Record low-frequency host and GPU resource samples during benchmark runs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    with Path(args.out).open("a", buffering=1) as stream:
        while psutil.pid_exists(args.pid):
            memory = psutil.virtual_memory()
            row = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "loadavg": os.getloadavg(),
                   "cpu_pct": psutil.cpu_percent(), "available_mib": round(memory.available / 1024**2, 1),
                   "swap_used_mib": round(psutil.swap_memory().used / 1024**2, 1)}
            sample = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            row["nvidia_smi"] = sample.stdout.strip()
            stream.write(json.dumps(row) + "\n")
            time.sleep(5)


if __name__ == "__main__":
    main()
