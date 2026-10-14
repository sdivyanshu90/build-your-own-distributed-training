"""Render the completed local CPU/CUDA run without mixing historical observations."""
from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import make_report as tables
from colab_report import section as colab_section

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--date", default="2026-10-14")
    parser.add_argument("--cloud-results", help="Validated user-executed Colab supplement")
    args = parser.parse_args()
    results = Path(args.results).resolve()
    tables.RES = str(results)
    load = tables.load
    md = tables.md
    system = json.loads((results / "system.json").read_text())
    execution = load("execution.jsonl") + load("supplement-execution.jsonl")
    assert len(execution) == 38, "The full matrix and supplement must finish before reporting"
    tests = ET.parse(results / "tests.xml").find("testsuite")
    assert tests is not None
    runs = load("scaling.jsonl")
    repeats = runs + load("repeats.jsonl")
    gpu = load("cuda.jsonl")
    gpu_memory = load("cuda-memory.jsonl")
    large = load("cuda-125m.jsonl")
    gpu_resume = load("cuda-resume.jsonl")
    fault = load("fault.jsonl")[0]
    collectives = load("collectives.jsonl")
    profiles = load("profiles.jsonl")
    monitor = load("monitor.jsonl")
    peak_temperature = max(float(r["nvidia_smi"].split(",")[0]) for r in monitor if r.get("nvidia_smi"))
    ram_floor = min(r["available_mib"] for r in monitor)
    failed = [r["case"] for r in execution if r["exit_code"]]
    out: list[str] = []

    def add(title: str, body: str) -> None:
        out.append(f"## {title}\n\n{body}")

    cloud = Path(args.cloud_results).resolve() if args.cloud_results else None
    cloud_scope = "The user-executed Colab Tesla T4 supplement is reported separately below." if cloud else "No cloud GPU result is claimed."
    out.append(f"# Full distributed-training benchmark: {args.date}\n\nSource commit: `{system['source_commit']}`. Training implementation, tests, and shipped configs are unchanged. Benchmark instrumentation adds CUDA measurements, actual-device metadata, isolated result directories, failure accounting, and complete execution records. CPU/Gloo and CUDA/NCCL results below are separate measurements. {cloud_scope}")
    add("Findings", f"The complete correctness suite passes: **97 passed, 1 skipped** in {float(tests.attrib['time']):.2f} seconds. All six CPU scaling layouts, eight memory layouts, collectives at 2 and 4 ranks, both CPU resume layouts, four convergence layouts, and rank-kill recovery complete. The remaining skip is the existing multi-GPU mixed-precision test stub.\n\nThe single local RTX 3050 Laptop GPU runs real CUDA training in fp32 and bf16 autocast, including the 151.9M-parameter model shape at reduced batch/sequence settings. Single-GPU checkpoint/resume is bit-exact in both precisions. Multi-GPU NCCL, GPU FSDP/TP/2D, GPU optimizer re-keying, 7B, and multi-node behavior remain unmeasured. Execution failures: {', '.join(failed) if failed else 'none'}.")
    add("Hardware and isolation", md([
        ["CPU", "AMD Ryzen 7 8845HS; WSL2 exposes 6 logical CPUs / 3 reported physical cores"],
        ["Host/WSL RAM", "Windows 16 GiB; WSL 8.7 GiB visible, 4 GiB swap"],
        ["GPU", "NVIDIA RTX 3050 Laptop, 4 GiB VRAM; one CUDA device"],
        ["Runtime", f"Python {system['python']}; PyTorch 2.6.0+cu124; CUDA 12.4; NCCL 2.21.5"],
        ["CPU cases", "CUDA hidden, Gloo over loopback; 1 thread/rank unless explicitly stated"],
        ["CPU budget control", "OMP/MKL requests total 4 threads: 1×4, 2×2, 4×1; observed PyTorch totals 3, 4, 4"],
        ["Resource monitoring", f"5-second samples from the memory phase onward; sampled GPU max {peak_temperature:.0f}°C; minimum available RAM {ram_floor:.0f} MiB"],
        ["NPU", "No /dev/accel in WSL; project uses CPU/Gloo or CUDA/NCCL and has no NPU backend"],
    ], ["Item", "Observed configuration"]) + "\n\nEach data point runs serially under the benchmark lock. Four CPU ranks share three WSL-reported physical cores, so SMT and scheduling matter. Windows activity and laptop power/thermal limits are not fully isolated. The CUDA-enabled PyTorch build also increases CPU-process runtime footprint relative to the older CPU-only build. [System and dependency snapshot](../benchmarks/results/20261014/system.json).")
    add("Method", "All datasets are synthetic. CPU scaling uses the real Trainer/train_step, `small` model (4,000,000 actual parameters), sequence 128, micro-batch 4, accumulation 2, 3 warm-up + 20 timed steps. DP keeps per-rank batch fixed (weak scaling); TP keeps global batch fixed. CPU DP observations repeat three times; within-run step medians are not confidence intervals. The comparison requesting four total threads is a separate single sweep; actual PyTorch counts are recorded.\n\nMemory uses `mid` (27,795,968 actual parameters), sequence 128, micro-batch 1, accumulation 2, 2 warm-up + 4 measured steps. Resident state is parameters + gradients + Adam tensors after the step; RSS also includes Python/PyTorch, allocator caches, and transient full-model construction; summing per-process RSS can double-count shared library pages.\n\nCUDA throughput uses sequence 128, micro-batch 4, accumulation 2, 10 warm-up + 50 timed steps. The memory sweep uses micro-batch 1, accumulation 1, 5 warm-up + 20 timed steps. CUDA timing synchronizes the device; reported tokens/s includes data-window overhead. GPU points are single runs. On one GPU the Trainer uses a bare model: bf16 means autocast with fp32 persistent parameters/Adam, not a GPU FSDP mixed-precision reduction test. MFU is omitted because this GPU has no calibrated peak in the project table. `n_params` in raw records is the config's documented analytical estimate; `actual_n_params` and resident bytes include norm parameters.")
    add("Figures", "![CPU scaling, memory, convergence, and CUDA measurements](../benchmarks/results/20261014/overview.png)\n\n[Vector figure](../benchmarks/results/20261014/overview.svg).")
    add("CPU throughput and scaling", tables.scaling_section()[0] + "\n\nThe table is the first complete sweep. Its per-rank batch and thread count are fixed; total CPU threads and global DP batch increase with DP degree.")
    rows = []
    medians = {}
    for world in [1, 2, 4]:
        vals = [r["tokens_per_s"] for r in repeats if r["tp"] == 1 and r["world_size"] == world]
        assert len(vals) == 3
        medians[world] = statistics.median(vals)
        rows.append([world, ", ".join(f"{v:.1f}" for v in vals), f"{statistics.median(vals):.1f}", f"{min(vals):.1f}–{max(vals):.1f}"])
    add("Three independent CPU DP observations", md(rows, ["Ranks", "Tokens/s samples", "Median tokens/s", "Range"]) + f"\n\nMedian speedup versus one rank: dp=2 **{medians[2]/medians[1]:.2f}×**, dp=4 **{medians[4]/medians[1]:.2f}×**. Dividing by rank count gives {100*medians[2]/medians[1]/2:.1f}% / {100*medians[4]/medians[1]/4:.1f}% efficiency for this weak-scaling workload; this is not GPU scaling efficiency.")
    add("CPU thread-budget control", md([[r["world_size"], r["hardware"]["omp_num_threads"], r["hardware"]["torch_num_threads"], r["tokens_per_step"], r["tokens_per_s"], f"{r['step_time_median_s']*1000:.2f}"] for r in load("thread-budget.jsonl")], ["Ranks", "Requested OMP threads/rank", "Observed PyTorch threads/rank", "Global tokens/step", "Tokens/s", "Median step ms"]) + "\n\nOMP/MKL requests total four threads in each run. PyTorch reports 3 threads for the one-rank baseline, 2 per rank for two ranks, and 1 per rank for four ranks: observed totals are 3, 4, and 4. These measurements therefore do not establish equal actual thread counts. Global DP batch still grows with rank count; this does not establish fixed-global-batch strong scaling.")
    add("CPU memory per rank", tables.memory_section() + "\n\nMeasured state follows sharding: 424.1 MiB unsharded, 212.1 MiB/rank with FULL_SHARD dp=2, 106.0 MiB/rank with dp=4. TP retains replicated embeddings/norms/head, giving 244.1 MiB/rank at tp=2 and 154.1 at tp=4; 2D dp=2,tp=2 gives 122.1 MiB/rank. SHARD_GRAD_OP also has 212.1 MiB/rank at this post-step snapshot: it retains unsharded parameters through forward/backward, then reshards; equal endpoint state does not imply equal transient peaks or gather schedules.")
    count_rows = []
    for r in collectives:
        counts = r["counts"]
        count_rows.append([r["world_size"], r["dp"], r["tp"], r["n_layers"], r["grad_accum"], counts.get("c10d::allgather_", 0), counts.get("c10d::_reduce_scatter_base_", 0), counts.get("c10d::allreduce_", 0), counts.get("_c10d_functional::all_reduce", 0)])
    add("Measured collective schedule", md(count_rows, ["World", "DP", "TP", "Layers", "Accumulation", "FSDP all-gather", "FSDP reduce-scatter", "c10d all-reduce", "Functional TP all-reduce"]) + "\n\nCounts are rank-0 profiler events for one real train_step after two warm-up steps. In 2D, accumulation deliberately synchronizes each micro-step because the deferred DTensor-gradient path is incompatible with this FSDP1 combination. Operator names/counts remain available in [collectives.jsonl](../benchmarks/results/20261014/collectives.jsonl).")
    add("CPU collective latency and bandwidth", tables.comm_section() + "\n\nFP32 buffers; 5 warm-up + 30 timed iterations per operator/size. Message bytes denote the full per-rank buffer (all-gather output / reduce-scatter input); bandwidth applies the script's ring correction. These are local-loopback Gloo measurements, not NVLink, PCIe, network, or NCCL link bandwidth.")
    add("CPU checkpoint and resume", tables.resume_section() + "\n\nBoth jobs compare all 15 post-resume losses against a continuous run; every comparison is bit-identical. The load measurement includes constructing the new Trainer. CPU FSDP uses the raw per-rank optimizer-state fallback and remains topology-locked. Checkpoint durability against power loss is not established by these process-level tests.")
    add("Rank-failure recovery", tables.fault_section() + f"\n\nA real worker receives SIGKILL. Torchrun detects the exit in {fault['detect_s']:.2f} seconds; relaunch-to-first-metric takes {fault['recover_s_relaunch_to_first_step']:.2f} seconds. Recovery selects step {fault['resumed_from_step']} after step {fault['step_at_kill']} was reached, losing {fault['steps_of_work_lost']} steps. This is explicit whole-job relaunch, not automatic elastic repair of a running job.")
    add("CPU convergence", tables.convergence_section() + "\n\nAll four layouts use the tiny synthetic workload, global batch 16, 150 steps. The 2D accumulation path now completes, and its loss curve matches the FSDP curve closely; pure TP tracks the single-process curve. These measurements do not establish language-model quality on a real corpus.")
    gpu_rows = [[r["label"], r["actual_n_params"], r["dtype"], r["tokens_per_s"], f"{1000*r['step_time_median_s']:.2f}", f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}", f"{r['per_rank'][0]['cuda_peak_reserved_bytes']/1024**2:.1f}"] for r in gpu]
    add("Single-GPU CUDA throughput", "Model training runs on the RTX 3050 Laptop through the NCCL/CUDA launch path, world size 1.\n\n" + md(gpu_rows, ["Run", "Actual parameters", "Autocast policy", "Tokens/s", "Median step ms", "Peak allocated MiB", "Peak reserved MiB"]) + "\n\nThe small model is overhead-sensitive: bf16 is not automatically faster. Mid-model bf16 improves throughput in this run. Device peak counters and host process RSS measure different memory. No inter-GPU communication or sharding is exercised at world size 1.")
    add("GPU activation checkpointing", md([[r["seq_len"], r["activation_checkpointing"], r["tokens_per_s"], f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}", f"{r['per_rank'][0]['cuda_peak_reserved_bytes']/1024**2:.1f}"] for r in gpu_memory], ["Sequence", "Activation checkpointing", "Tokens/s", "Peak allocated MiB", "Peak reserved MiB"]) + "\n\nMid model, bf16 autocast, micro-batch 1, accumulation 1. At sequence 1024 checkpointing lowers the measured allocated peak while reducing throughput; persistent fp32 training state limits savings at short sequences.")
    add("151.9M-parameter model shape", md([[r["actual_n_params"], r["seq_len"], r["micro_batch_size"], r["grad_accum"], r["activation_checkpointing"], r["tokens_per_s"], f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}"] for r in large], ["Actual parameters", "Sequence", "Micro-batch", "Accumulation", "Checkpointing", "Tokens/s", "Peak allocated MiB"]) + "\n\nUses the architecture from config/125m.yaml, whose actual size is 151,898,880 parameters. To fit 4 GiB VRAM, sequence is reduced from 1024 to 128, micro-batch from 16 to 1, and accumulation from 4 to 1; data workers are 0. It is a 3-warm-up + 10-step computational measurement, not the shipped full-size workload or a completed pretraining run. The 7B config is not attempted on this hardware.")
    add("Single-GPU checkpoint and resume", md([[r["dtype"], r["ckpt_save_s"], r["trainer_build_plus_load_s"], f"{r['ckpt_bytes_total']/1e6:.2f}", r["max_abs_loss_diff"], f"{r['bit_identical_steps']}/{r['compared_steps']}"] for r in gpu_resume], ["Autocast policy", "Save seconds", "Build+load seconds", "Checkpoint MB", "Max loss difference", "Identical steps"]) + "\n\nSmall model, world size 1, 30 steps with resume at 15. Both policies retain fp32 persistent model/optimizer state. This validates the unsharded CUDA path; GPU FSDP optimizer re-keying requires a multi-GPU host and remains untested.")
    add("Profiler traces", md([[r["backend"], r["world_size"], r["steps"], f"{r['total_attributed_us']:.1f}" if r["total_attributed_us"] else "unavailable", f"{r['communication_fraction']:.4f}" if r["total_attributed_us"] else "unavailable"] for r in profiles], ["Backend", "Ranks", "Profiled steps", "Attributed device/CPU microseconds", "Collective fraction"]) + "\n\n[CPU trace](../benchmarks/results/20261014/profile-gloo.json.gz), [CPU operator summary](../benchmarks/results/20261014/profile-gloo.txt), [CUDA trace](../benchmarks/results/20261014/profile-nccl.json.gz), [CUDA operator summary](../benchmarks/results/20261014/profile-nccl.txt). Gloo breakdown uses CPU self-time. The CUDA trace contains host/API activity but no GPU kernel timing events, and all device self-times are zero; its reported raw 0.0 communication fraction is **unavailable**, not evidence of zero communication. An isolated profiler probe reproduces missing kernel events while CUDA event timing confirms actual device execution. Fractions sum attributed operator times and are not elapsed wall-time fractions. The single-GPU trace cannot establish distributed communication overlap. Profiling measurements are separate from throughput results.")
    if cloud:
        add("Colab Tesla T4 supplement", colab_section(cloud, results))
    add("Execution and verification", md([[r["case"], r["exit_code"], f"{r['duration_s']:.2f}"] for r in execution], ["Case/block", "Exit status", "Wall seconds"]) + "\n\n[Tests log](../benchmarks/results/20261014/tests.log), [JUnit record](../benchmarks/results/20261014/tests.xml), [lint](../benchmarks/results/20261014/ruff.log), [type checks](../benchmarks/results/20261014/mypy.log), [resource snapshots](../benchmarks/results/20261014/monitor.jsonl), [artifact index](../benchmarks/results/20261014/README.md), and [SHA-256 manifest](../benchmarks/results/20261014/sha256.txt). Shell exit status is checked separately from numerical invariants and complete result counts.")
    add("Reproducing", "```bash\n# Use Python 3.12 and the project's supported torch version.\npython3.12 -m venv .venv\n.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124\n.venv/bin/pip install numpy pyyaml pytest ruff mypy psutil matplotlib\nCUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q\n\ntraining_bench_results=benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)\n.venv/bin/python benchmarks/rebenchmark.py --out \"$training_bench_results\"\n.venv/bin/python benchmarks/supplement.py --out \"$training_bench_results\"\n# Reports are generated only after every measurement finishes.\n.venv/bin/python benchmarks/full_report.py --results \"$training_bench_results\"\n```\n\nThe captured run uses a dedicated environment under ~/.local/share/distributed-training-benchmark/. CPU-only hosts can use a CPU PyTorch wheel and `rebenchmark.py --cpu-only`; the CUDA supplement/report sections require corresponding GPU artifacts. The complete report renderer also requires system/test snapshots. For a new GPU model, inspect capacity first and adjust the reduced 125M-shape workload if necessary; it is not a universal memory fit. PyTorch wheel sources: [official version-specific installation instructions](https://pytorch.org/get-started/previous-versions/).")
    if cloud:
        cloud_path = cloud.relative_to(ROOT).as_posix()
        local_path = results.relative_to(ROOT).as_posix()
        add("Regenerating the imported Colab reports", f"Keep the original Colab archive and its manifest unchanged. The imported files are in `{cloud_path}`.\n\n```bash\n.venv/bin/python benchmarks/plot_full.py --results {local_path} --cloud-results {cloud_path}\n.venv/bin/python benchmarks/colab_report.py --results {cloud_path} --local-results {local_path}\n.venv/bin/python benchmarks/full_report.py --results {local_path} --cloud-results {cloud_path}\n```\n\nThe Colab renderer validates the captured 15-case matrix and matching local workload fields before publishing tables. The current prepared notebook uses native BF16 detection and may select a different precision policy on T4; preserve the executed notebook as the provenance for this captured run.")
    cloud_limit = "The laptop and Colab each expose one GPU in separate environments; they do not form a distributed GPU benchmark. The Colab supplement adds no cloud pytest suite, fault test or multi-GPU evidence." if cloud else "Only one GPU and one WSL host are available. No cloud measurement is included."
    add("Limits and historical context", cloud_limit + " GPU multi-rank FSDP/TP/2D, NCCL link bandwidth, inter-node behavior, hybrid GPU sharding, CUDA FSDP optimizer re-keying, CPU offload, fp16 loss scaling, 7B, and real-corpus convergence are not established. The GPU mixed-precision reduction test remains a skipped placeholder. Sequence-parallel numerical equivalence passes on CPU, but its speed/memory benefit is not benchmarked.\n\nThe [October 4/5 report](archive/BENCHMARKS-20261004.md) and unlabelled result files retain their original observations. CPU, Python, kernel, runtime build, resources, and background load differ, so these results are not a controlled before/after estimate of code changes.")
    text = "\n\n".join(out) + "\n"
    for name in ["BENCHMARKS.md", f"BENCHMARKS-{args.date.replace('-', '')}.md"]:
        (ROOT / "docs" / name).write_text(text)
    files = sorted(p for p in results.iterdir() if p.is_file() and p.name != "README.md")
    (results / "README.md").write_text("# Benchmark artifacts: " + args.date + "\n\nCurrent report: [BENCHMARKS.md](../../../docs/BENCHMARKS.md). CPU/Gloo and CUDA/NCCL measurements are distinct. Files outside this dated directory are historical.\n\n" + "\n".join(f"- [{p.name}]({p.name})" for p in files) + "\n")
    print("Wrote current and dated full reports")


if __name__ == "__main__":
    main()
