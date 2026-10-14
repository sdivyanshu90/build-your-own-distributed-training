"""Export publication-ready figures for the captured CPU and single-GPU matrix."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--cloud-results", help="Optional separate Colab run for a comparison figure")
    args = parser.parse_args()
    root = Path(args.results)

    def load(name: str) -> list[dict]:
        return [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]

    scaling = load("scaling")
    repeated = scaling + load("repeats")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Distributed training — Ryzen 7 8845HS / RTX 3050 Laptop, 2026-10-14", fontsize=16)
    ax = axes[0, 0]
    labels, values, lower, upper = [], [], [], []
    for row in scaling:
        samples = [r["tokens_per_s"] for r in repeated if r["tp"] == 1 and r["world_size"] == row["world_size"]] if row["tp"] == 1 else [row["tokens_per_s"]]
        center = statistics.median(samples)
        labels.append({"dp1_pr": "1 rank", "dp2_pr": "FSDP 2", "dp4_pr": "FSDP 4", "tp2_pr": "TP 2", "tp4_pr": "TP 4", "2d_dp2tp2_pr": "DP2×TP2"}[row["label"]])
        values.append(center)
        lower.append(center - min(samples))
        upper.append(max(samples) - center)
    ax.bar(labels, values, yerr=[lower, upper], capsize=4, color=["#2563eb"] * 3 + ["#d97706"] * 2 + ["#059669"])
    ax.set(title="CPU/Gloo: 1 thread/rank (DP median/range of 3)", ylabel="Global tokens / second")
    memory = load("memory")
    labels = ["Single", "NO_SHARD 2", "ZeRO2 2", "FSDP 2", "FSDP 4", "TP 2", "TP 4", "DP2×TP2"]
    values = [statistics.mean(sum(rank[k] for k in ["params", "grads", "optimizer"]) for rank in row["per_rank"]) / 1024**2 for row in memory]
    axes[0, 1].bar(labels, values, color="#2563eb")
    axes[0, 1].set(title="CPU resident training state after the step", ylabel="MiB per rank (mid model)")
    for row in load("convergence"):
        label = row["label"].removeprefix("conv_")
        axes[0, 2].plot(range(1, 151), row["loss_curve"], label=label, alpha=0.85)
    axes[0, 2].set(title="All four CPU layouts converge", xlabel="Optimizer step", ylabel="Synthetic training loss")
    axes[0, 2].legend(fontsize=8)
    gpu = load("cuda")
    for index, dtype in enumerate(["float32", "bfloat16"]):
        rows = [r for r in gpu if r["dtype"] == dtype]
        axes[1, 0].bar([i + (index - 0.5) * 0.35 for i in range(2)], [r["tokens_per_s"] for r in rows], width=0.35, label=dtype, color=["#2563eb", "#059669"][index])
    axes[1, 0].set(xticks=[0, 1], xticklabels=["Small (4M)", "Mid (27.8M)"], title="Single-GPU throughput: fp32 / bf16 autocast", ylabel="Tokens / second")
    axes[1, 0].legend(fontsize=8)
    for ac, color in [(False, "#2563eb"), (True, "#059669")]:
        rows = [r for r in load("cuda-memory") if r["activation_checkpointing"] == ac]
        axes[1, 1].plot([r["seq_len"] for r in rows], [r["per_rank"][0]["cuda_peak_allocated_bytes"] / 1024**2 for r in rows], "o-", label=f"Checkpointing {'on' if ac else 'off'}", color=color)
    axes[1, 1].set(title="Single-GPU activation memory tradeoff", xlabel="Sequence length (micro-batch 1)", ylabel="Peak allocated MiB")
    axes[1, 1].legend(fontsize=8)
    budget = load("thread-budget")
    axes[1, 2].plot([1, 2, 4], [r["tokens_per_s"] for r in budget], "o-", label="4 requested (observed 3/4/4)", color="#059669")
    axes[1, 2].plot([1, 2, 4], [statistics.median(r["tokens_per_s"] for r in repeated if r["tp"] == 1 and r["world_size"] == world) for world in [1, 2, 4]], "o-", label="1 thread/rank", color="#2563eb")
    axes[1, 2].set(title="CPU thread budget changes the baseline", xlabel="FSDP ranks (DP global batch grows)", ylabel="Global tokens / second", xticks=[1, 2, 4])
    axes[1, 2].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[0, :2]:
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    fig.savefig(root / "overview.png", dpi=180)
    fig.savefig(root / "overview.svg")
    print(root / "overview.png")
    plt.close(fig)
    if args.cloud_results:
        plot_cloud(root, Path(args.cloud_results))


def plot_cloud(local: Path, cloud: Path) -> None:
    def load_at(path: Path, name: str) -> list[dict]:
        return [json.loads(line) for line in (path / f"{name}.jsonl").read_text().splitlines()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Single-GPU training: laptop RTX 3050 and Colab Tesla T4 — 2026-10-14", fontsize=14)
    order = ["cuda_small_float32", "cuda_mid_float32", "cuda_small_bfloat16", "cuda_mid_bfloat16"]
    for offset, path, label, color in [(-0.18, local, "RTX 3050 Laptop", "#2563eb"), (0.18, cloud, "Tesla T4 (Colab)", "#059669")]:
        by_label = {r["label"]: r for r in load_at(path, "cuda")}
        axes[0].bar([i + offset for i in range(4)], [by_label[k]["tokens_per_s"] for k in order], width=0.36, label=label, color=color)
        for ac, style in [(False, "-"), (True, "--")]:
            rows = [r for r in load_at(path, "cuda-memory") if r["activation_checkpointing"] == ac]
            axes[1].plot([r["seq_len"] for r in rows], [r["per_rank"][0]["cuda_peak_allocated_bytes"] / 1024**2 for r in rows],
                         marker="o", linestyle=style, color=color, label=f"{label}, AC {'on' if ac else 'off'}")
    axes[0].set(title="Small / mid throughput (one observation per point)", ylabel="Global tokens / second",
                xticks=range(4), xticklabels=["Small\nfp32", "Mid\nfp32", "Small\nbf16*", "Mid\nbf16*"])
    axes[1].set(title="Mid-model peak allocated memory, bf16 autocast*", xlabel="Sequence length (micro-batch 1)", ylabel="MiB", xticks=[128, 512, 1024])
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.035, "*T4 BF16 autocast uses software support without native BF16 acceleration. Persistent parameters/Adam are fp32.\nDifferent CPU hosts, kernels and scheduling affect end-to-end throughput; these are not multi-GPU scaling results.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(cloud / "comparison.png", dpi=180)
    fig.savefig(cloud / "comparison.svg")
    plt.close(fig)
    print(cloud / "comparison.png")


if __name__ == "__main__":
    main()
