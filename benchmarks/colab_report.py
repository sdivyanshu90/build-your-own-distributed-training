"""Validate the supplied Colab artifacts and render a separate single-GPU supplement."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from make_report import md

ROOT = Path(__file__).resolve().parent.parent
BF16_SOURCE = "https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/__init__.py#L177-L207"


def load(results: Path, stem: str) -> list[dict]:
    return [json.loads(line) for line in (results / f"{stem}.jsonl").read_text().splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(results: Path, local: Path) -> dict:
    """Check the captured two-policy matrix, original checksums, and numerical evidence."""
    manifest = (results / "sha256.txt").read_text().splitlines()
    for line in manifest:
        digest, name = line.split("  ", 1)
        require(Path(name).name == name, f"Unexpected manifest path: {name}")
        require(hashlib.sha256((results / name).read_bytes()).hexdigest() == digest, f"Checksum mismatch: {name}")
    require(len(manifest) == 24, "Expected 24 checksummed original Colab artifacts")
    execution = load(results, "execution")
    require(len(execution) == 15, "Expected 15 Colab execution records")
    require(len({r['case'] for r in execution}) == 15, "Duplicate Colab case")
    require(all(r['exit_code'] == 0 for r in execution), "A Colab case failed")
    counts = {"cuda": 4, "cuda-memory": 6, "cuda-125m": 2, "cuda-resume": 2, "profiles": 1}
    data = {name: load(results, name) for name in counts}
    for name, count in counts.items():
        require(len(data[name]) == count, f"Incomplete Colab results: {name}")
    for stem in ["cuda", "cuda-memory", "cuda-125m"]:
        reference = {r['label']: r for r in load(local, stem)}
        require({r['label'] for r in data[stem]} == set(reference), f"Workload labels differ: {stem}")
        for row in data[stem]:
            require(row['world_size'] == row['dp'] == row['tp'] == 1, "Expected single-GPU measurements")
            require(row['hardware']['device'] == 'cuda' and row['hardware']['gpu'] == 'Tesla T4', "Unexpected device")
            for key in ['preset', 'dtype', 'seq_len', 'micro_batch_size', 'grad_accum', 'steps', 'warmup', 'actual_n_params', 'activation_checkpointing']:
                require(row[key] == reference[row['label']][key], f"Workload differs: {row['label']} / {key}")
            for key in ['tokens_per_s', 'step_time_median_s', 'loss_first', 'loss_last']:
                require(math.isfinite(row[key]) and row[key] > 0, f"Non-finite/non-positive metric: {key}")
            require(row['tokens_per_step'] == row['seq_len'] * row['micro_batch_size'] * row['grad_accum'], "Token count mismatch")
            rank = row['per_rank'][0]
            require(len(row['per_rank']) == 1, "Unexpected rank records")
            require(rank['params'] == row['actual_n_params'] * 4, "Persistent parameters must remain fp32")
            require(rank['cuda_peak_reserved_bytes'] >= rank['cuda_peak_allocated_bytes'] > 0, "Invalid device memory counters")
    for row in data['cuda-resume']:
        require(row['max_abs_loss_diff'] == 0, "Resume loss mismatch")
        require(row['bit_identical_steps'] == row['compared_steps'] == 15, "Incomplete resume comparison")
        require(row['loss_continuous'][15:] == row['loss_resumed_second_half'], "Resume curves differ")
    trace = json.loads(gzip.decompress((results / 'profile-nccl.json.gz').read_bytes()))
    categories = Counter(event.get('cat', '') for event in trace['traceEvents'])
    device = trace['deviceProperties'][0]
    profile = data['profiles'][0]
    require(categories['kernel'] > 0 and profile['total_attributed_us'] > 0, "No usable GPU kernel timings")
    require(math.isclose(sum(r['self_device_us'] for r in profile['operators']), profile['total_attributed_us']), "Profiler operator totals differ")
    system = json.loads((results / 'system.json').read_text())
    provenance = json.loads((results / 'provenance.json').read_text())
    require(system['gpu_count'] == 1 and system['gpu'] == device['name'] == 'Tesla T4', "Device metadata mismatch")
    require(hashlib.sha256((results / 'executed-notebook.ipynb').read_bytes()).hexdigest() == provenance['executed_notebook_sha256'], "Notebook checksum mismatch")
    # PyTorch 2.6's CUDA native-support branch requires CUDA >= 11 and capability >= 8.
    native = int(system['cuda'].split('.')[0]) >= 11 and device['computeMajor'] >= 8
    summary = {
        'run_id': provenance['run_id'], 'execution_cases': len(execution), 'execution_failures': 0,
        'measurement_records': sum(counts.values()), 'original_checksums_verified': len(manifest),
        'source_commit_pinned_by_notebook': provenance['source_commit_pinned_by_notebook'],
        'matching_local_workload_fields': True, 'resume_identical_steps_each_policy': 15,
        'gpu': device['name'], 'compute_capability': [device['computeMajor'], device['computeMinor']],
        'gpu_memory_bytes_from_trace': device['totalGlobalMem'], 'bf16_probe_including_emulation': system['bf16'],
        'native_bf16_inferred_from_pytorch_2_6_check': native, 'bf16_check_source': BF16_SOURCE,
        'trace_kernel_events': categories['kernel'], 'profiler_device_self_us': profile['total_attributed_us'],
        'scope': 'Single-GPU synthetic measurements; no Colab pytest suite or multi-GPU verification',
    }
    (results / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def section(results: Path, local: Path) -> str:
    evidence = validate(results, local)
    prefix = '../' + results.relative_to(ROOT).as_posix()
    system = json.loads((results / 'system.json').read_text())
    provenance = json.loads((results / 'provenance.json').read_text())
    rows = load(results, 'cuda')
    reference = {r['label']: r for r in load(local, 'cuda')}
    parts = [
        f"The user-executed Colab run **{evidence['run_id']}** completed **15/15 cases** on one **Tesla T4**. "
        f"The trace reports compute capability 7.5 and {evidence['gpu_memory_bytes_from_trace']/1024**3:.2f} GiB visible GPU memory. "
        f"Runtime: Python {system['python']}, PyTorch {system['torch']}, CUDA {system['cuda']}, NCCL {'.'.join(map(str, system['nccl']))}. "
        "The host reports 2 logical Intel Xeon CPUs. These are additional single-device measurements; no distributed GPU scaling or cloud pytest run is claimed.",
        f"Source commit `{provenance['source_commit_pinned_by_notebook']}` is pinned by the executed notebook. Its four embedded benchmark helpers match the local instrumented files at import. "
        f"All 24 original artifact checksums pass. [Import provenance]({prefix}/provenance.json), [executed notebook]({prefix}/executed-notebook.ipynb), "
        f"[validation]({prefix}/validation.json), [artifact index]({prefix}/README.md).",
        "**Precision interpretation:** the original notebook called `torch.cuda.is_bf16_supported()` with its default `including_emulation=True`. "
        f"Its raw `bf16: true` does not mean native BF16 arithmetic. The T4's captured capability 7.5 fails [PyTorch 2.6's native BF16 check]({BF16_SOURCE}). "
        "T4 `bfloat16` rows below are successful BF16-autocast runs on the software-supported path, without native BF16 acceleration. Persistent weights and Adam state remain fp32. "
        "The prepared notebook now distinguishes native support from emulation and selects BF16 automatically only with native support. Existing measurements are preserved unchanged.",
        "### Colab throughput and laptop comparison\n\n" + md([
            [r['preset'], r['dtype'] + (' (non-native)' if r['dtype'] == 'bfloat16' else ''),
             reference[r['label']]['tokens_per_s'], r['tokens_per_s'], f"{r['tokens_per_s']/reference[r['label']]['tokens_per_s']:.2f}×",
             f"{1000*r['step_time_median_s']:.2f}", f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}"]
            for r in rows
        ], ['Model', 'T4 autocast policy', 'RTX 3050 tokens/s', 'T4 tokens/s', 'T4 / laptop', 'T4 median step ms', 'T4 peak allocated MiB']),
        "Same model shapes, batch, sequence, warm-up and measured steps as the laptop CUDA table: 10 warm-up + 50 timed steps, sequence 128, micro-batch 4, accumulation 2. "
        "FP32 mid-model throughput is 1.71× the laptop observation; T4 BF16-autocast mid throughput is lower than its own FP32 result. "
        "Both hosts contribute one observation per point. CPU/data-window overhead, available attention kernels, scheduling and power conditions differ. "
        "Ratios compare observed end-to-end training throughput, not isolated GPU compute or native BF16 hardware performance.",
        f"![Laptop and Colab single-GPU comparison]({prefix}/comparison.png)\n\n[Vector figure]({prefix}/comparison.svg).",
        "### Colab activation memory\n\n" + md([
            [r['seq_len'], r['activation_checkpointing'], r['tokens_per_s'],
             f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}", f"{r['per_rank'][0]['cuda_peak_reserved_bytes']/1024**2:.1f}"]
            for r in load(results, 'cuda-memory')
        ], ['Sequence', 'Checkpointing', 'T4 tokens/s', 'T4 peak allocated MiB', 'T4 peak reserved MiB']),
        "Mid model, software-supported BF16 autocast, micro-batch 1, accumulation 1, 5 warm-up + 20 measured steps. "
        "At sequence 1024, checkpointing lowers peak allocated memory from 1,015.4 to 603.4 MiB while throughput falls from 7,610.0 to 6,031.7 tokens/s. "
        "The different device peaks are measured separately; no particular attention kernel cause is established by these tables.",
        "### Colab reduced 151.9M architecture\n\n" + md([
            [r['actual_n_params'], r['activation_checkpointing'], r['tokens_per_s'], f"{r['per_rank'][0]['cuda_peak_allocated_bytes']/1024**2:.1f}"]
            for r in load(results, 'cuda-125m')
        ], ['Actual parameters', 'Checkpointing', 'T4 tokens/s', 'T4 peak allocated MiB']),
        "The architecture from config/125m.yaml uses the same reduced sequence 128, micro-batch 1 and accumulation 1 as the laptop run, with 3 warm-up + 10 measured steps and software-supported BF16 autocast. "
        "The larger T4 memory does not turn this into a measurement of the shipped sequence-1024, micro-batch-16 workload. No 7B or full pretraining run was attempted.",
        "### Colab checkpoint/resume\n\n" + md([
            [r['dtype'] + (' (non-native)' if r['dtype'] == 'bfloat16' else ''), r['ckpt_save_s'], r['trainer_build_plus_load_s'],
             f"{r['ckpt_bytes_total']/1e6:.2f}", r['max_abs_loss_diff'], f"{r['bit_identical_steps']}/{r['compared_steps']}"]
            for r in load(results, 'cuda-resume')
        ], ['Autocast policy', 'Save seconds', 'Build+load seconds', 'Checkpoint MB', 'Max loss difference', 'Identical losses']),
        "Both policies match their own continuous 30-step run exactly after resuming at step 15. Cross-device or cross-policy bit identity is not implied. "
        "World size 1 uses the bare model, so GPU FSDP optimizer conversion remains untested.",
        "### Colab GPU profiler\n\n"
        f"The 3-step trace contains **{evidence['trace_kernel_events']:,} GPU kernel events** and **{evidence['profiler_device_self_us']/1000:.2f} ms** of attributed device self-time. "
        "Unlike the laptop trace, GPU timing is available. The raw collective fraction is 0.0 for a single-device workload with no inter-GPU communication; it says nothing about distributed overlap. "
        f"Operator self-time is not elapsed wall time. [Trace]({prefix}/profile-nccl.json.gz), [operator summary]({prefix}/profile-nccl.txt).",
        "### Colab execution status\n\n" + md([
            [r['case'], r['exit_code'], r['duration_s']] for r in load(results, 'execution')
        ], ['Case', 'Exit status', 'Wall seconds']),
    ]
    return '\n\n'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--local-results', required=True)
    args = parser.parse_args()
    results, local = Path(args.results).resolve(), Path(args.local_results).resolve()
    body = section(results, local)
    (ROOT / 'docs/BENCHMARKS-COLAB-20261014.md').write_text('# Colab Tesla T4 benchmark: 2026-10-14\n\n' + body + '\n')
    files = sorted(p for p in results.iterdir() if p.is_file() and p.name != 'README.md')
    (results / 'README.md').write_text(
        '# Colab Tesla T4 artifacts: 2026-10-14\n\n'
        '[Report](../../../docs/BENCHMARKS-COLAB-20261014.md). Original files and their `sha256.txt` are preserved. '
        'Additional import evidence, validation, figures and the executed notebook are covered by `artifact-sha256.txt`. '
        'The source archive remains in the user\'s Downloads folder; its SHA-256 is recorded in provenance.json.\n\n'
        + '\n'.join(f'- [{p.name}]({p.name})' for p in files) + '\n'
    )
    print('Validated and rendered Colab supplement')


if __name__ == '__main__':
    main()
