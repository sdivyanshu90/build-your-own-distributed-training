"""Turn benchmarks/results/*.jsonl into results/SUMMARY.md (+ PNG plots).

    python benchmarks/make_report.py            # tables only needs the stdlib
    python benchmarks/make_report.py --plots    # also writes results/*.png (matplotlib)
"""

from __future__ import annotations

import argparse
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name: str) -> list[dict]:
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def md(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scaling_section() -> tuple[str, list[dict]]:
    recs = load("scaling.jsonl")
    if not recs:
        return "", []
    base = {}
    for r in recs:
        if r["label"] == "dp1_pr":
            base["pr"] = r["tokens_per_s"]
        if r["label"] == "dp1_budget":
            base["budget"] = r["tokens_per_s"]
    rows = []
    for r in recs:
        pol = "pr" if r["label"].endswith("_pr") else "budget"
        b = base.get(pol)
        kind = "2D" if r["tp"] > 1 and r["dp"] > 1 else ("TP" if r["tp"] > 1 else ("FSDP" if r["dp"] > 1 else "single"))
        speedup = r["tokens_per_s"] / b if b else float("nan")
        rows.append([r["label"], kind, r["world_size"], r["dp"], r["tp"],
                     r["hardware"]["torch_num_threads"], r["tokens_per_step"],
                     f'{r["step_time_median_s"] * 1000:.0f}', f'{r["tokens_per_s"]:.0f}',
                     f"{speedup:.2f}x", f"{100 * speedup / r['world_size']:.0f}%"])
    hdr = ["run", "layout", "world", "dp", "tp", "threads/rank", "tokens/step",
           "median step (ms)", "tokens/s", "vs 1 rank", "efficiency"]
    return md(rows, hdr), recs


def memory_section() -> str:
    recs = load("memory.jsonl")
    if not recs:
        return ""
    rows = []
    for r in recs:
        mb = 1024 * 1024
        pr = r["per_rank"]
        rows.append([r["label"], r["world_size"], r["dp"], r["tp"], r["sharding"] if r["dp"] > 1 else "-",
                     f'{max(x["params"] for x in pr) / mb:.1f}',
                     f'{max(x["grads"] for x in pr) / mb:.1f}',
                     f'{max(x["optimizer"] for x in pr) / mb:.1f}',
                     f'{sum(x["params"] + x["grads"] + x["optimizer"] for x in pr) / len(pr) / mb:.1f}',
                     f'{max(x["peak_rss_mb"] for x in pr):.0f}',
                     f'{sum(x["peak_rss_mb"] for x in pr):.0f}'])
    hdr = ["run", "world", "dp", "tp", "sharding", "params MiB/rank", "grads MiB/rank",
           "adam MiB/rank", "state MiB/rank (mean)", "peak RSS MiB (max rank)", "peak RSS MiB (sum ranks)"]
    return md(rows, hdr)


def comm_section() -> str:
    recs = load("comm.jsonl")
    if not recs:
        return ""
    out = []
    for r in recs:
        rows = [[x["op"], x["message_bytes"], x["latency_median_us"], x["latency_p90_us"],
                 x["algbw_MBps"], x["busbw_MBps"]] for x in r["rows"]]
        out.append(f'**world size {r["world_size"]}**\n\n' + md(
            rows, ["op", "bytes/rank", "median latency (us)", "p90 (us)", "algbw (MB/s)", "busbw (MB/s)"]))
    return "\n\n".join(out)


def resume_section() -> str:
    recs = load("resume.jsonl")
    if not recs:
        return ""
    rows = [[r["world_size"], r["dp"], r["n_params"], r["steps"], r["resume_at"], r["ckpt_save_s"],
             r["trainer_build_plus_load_s"], f'{r["ckpt_bytes_total"] / 1e6:.2f}',
             r["max_abs_loss_diff"], f'{r["bit_identical_steps"]}/{r["compared_steps"]}'] for r in recs]
    return md(rows, ["world", "dp", "params", "steps", "resumed at", "save (s)", "build+load (s)",
                     "ckpt size (MB, all ranks)", "max |dloss|", "bit-identical steps"])


def fault_section() -> str:
    recs = load("fault.jsonl")
    if not recs:
        return ""
    rows = [[r["world_size"], r["save_interval"], r["step_at_kill"], ",".join(r["committed_checkpoints"]),
             r["resumed_from_step"], r["steps_of_work_lost"], r["detect_s"],
             r["recover_s_relaunch_to_first_step"]] for r in recs]
    return md(rows, ["world", "save every", "step at kill", "committed ckpts", "resumed from",
                     "steps lost", "detect (s)", "relaunch->first step (s)"])


def convergence_section() -> str:
    recs = load("convergence.jsonl")
    if not recs:
        return ""
    rows = []
    for r in recs:
        c = r["loss_curve"]
        pick = [c[0], c[24], c[49], c[99], c[-1]]
        rows.append([r["label"], r["world_size"], r["dp"], r["tp"]] + [f"{x:.3f}" for x in pick])
    return md(rows, ["run", "world", "dp", "tp", "step 1", "step 25", "step 50", "step 100", "step 150"])


def plots() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conv = load("convergence.jsonl")
    if conv:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for r in conv:
            ax.plot(range(1, len(r["loss_curve"]) + 1), r["loss_curve"], label=r["label"])
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("train loss")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "convergence.png"), dpi=130)
    comm = load("comm.jsonl")
    if comm:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for r in comm:
            for op in ("all_reduce", "all_gather", "reduce_scatter"):
                xs = [x["message_bytes"] for x in r["rows"] if x["op"] == op]
                ys = [x["busbw_MBps"] for x in r["rows"] if x["op"] == op]
                ax.plot(xs, ys, marker="o", label=f'{op} n={r["world_size"]}')
        ax.set_xscale("log")
        ax.set_xlabel("message bytes per rank")
        ax.set_ylabel("bus bandwidth (MB/s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "comm_busbw.png"), dpi=130)
    _, sc = scaling_section()
    if sc:
        pr = [r for r in sc if r["label"].endswith("_pr")]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar([r["label"] for r in pr], [r["tokens_per_s"] for r in pr])
        ax.set_ylabel("tokens/s (CPU, gloo, 1 thread/rank)")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "scaling.png"), dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    sc, recs = scaling_section()
    hw = recs[0]["hardware"] if recs else {}
    parts = ["# Benchmark summary (generated by benchmarks/make_report.py)", "",
             f"Hardware/software of the first record: `{json.dumps(hw)}`", ""]
    for title, body in [("Throughput / scaling", sc), ("Memory per rank", memory_section()),
                        ("Collectives (gloo)", comm_section()), ("Checkpoint + resume", resume_section()),
                        ("Fault recovery", fault_section()), ("Convergence (loss at step)", convergence_section())]:
        parts += [f"## {title}", "", body or "_no data_", ""]
    with open(os.path.join(RES, "SUMMARY.md"), "w") as f:
        f.write("\n".join(parts))
    if a.plots:
        plots()
    print("wrote", os.path.join(RES, "SUMMARY.md"))


if __name__ == "__main__":
    main()
