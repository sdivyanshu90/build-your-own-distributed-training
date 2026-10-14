# Colab Tesla T4 benchmark: 2026-10-14

The user-executed Colab run **20261014T133332Z** completed **15/15 cases** on one **Tesla T4**. The trace reports compute capability 7.5 and 14.56 GiB visible GPU memory. Runtime: Python 3.12.15, PyTorch 2.6.0+cu124, CUDA 12.4, NCCL 2.21.5. The host reports 2 logical Intel Xeon CPUs. These are additional single-device measurements; no distributed GPU scaling or cloud pytest run is claimed.

Source commit `3f96b82778688f27f87f710592b1912103939921` is pinned by the executed notebook. Its four embedded benchmark helpers match the local instrumented files at import. All 24 original artifact checksums pass. [Import provenance](../benchmarks/results/20261014-colab-t4/provenance.json), [executed notebook](../benchmarks/results/20261014-colab-t4/executed-notebook.ipynb), [validation](../benchmarks/results/20261014-colab-t4/validation.json), [artifact index](../benchmarks/results/20261014-colab-t4/README.md).

**Precision interpretation:** the original notebook called `torch.cuda.is_bf16_supported()` with its default `including_emulation=True`. Its raw `bf16: true` does not mean native BF16 arithmetic. The T4's captured capability 7.5 fails [PyTorch 2.6's native BF16 check](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/__init__.py#L177-L207). T4 `bfloat16` rows below are successful BF16-autocast runs on the software-supported path, without native BF16 acceleration. Persistent weights and Adam state remain fp32. The prepared notebook now distinguishes native support from emulation and selects BF16 automatically only with native support. Existing measurements are preserved unchanged.

### Colab throughput and laptop comparison

| Model | T4 autocast policy | RTX 3050 tokens/s | T4 tokens/s | T4 / laptop | T4 median step ms | T4 peak allocated MiB |
|---|---|---|---|---|---|---|
| small | float32 | 19716.6 | 18270.0 | 0.93× | 36.31 | 152.9 |
| mid | float32 | 6538.0 | 11172.1 | 1.71× | 72.70 | 684.1 |
| small | bfloat16 (non-native) | 19404.5 | 16475.7 | 0.85× | 44.54 | 149.3 |
| mid | bfloat16 (non-native) | 9368.7 | 8816.1 | 0.94× | 97.20 | 701.4 |

Same model shapes, batch, sequence, warm-up and measured steps as the laptop CUDA table: 10 warm-up + 50 timed steps, sequence 128, micro-batch 4, accumulation 2. FP32 mid-model throughput is 1.71× the laptop observation; T4 BF16-autocast mid throughput is lower than its own FP32 result. Both hosts contribute one observation per point. CPU/data-window overhead, available attention kernels, scheduling and power conditions differ. Ratios compare observed end-to-end training throughput, not isolated GPU compute or native BF16 hardware performance.

![Laptop and Colab single-GPU comparison](../benchmarks/results/20261014-colab-t4/comparison.png)

[Vector figure](../benchmarks/results/20261014-colab-t4/comparison.svg).

### Colab activation memory

| Sequence | Checkpointing | T4 tokens/s | T4 peak allocated MiB | T4 peak reserved MiB |
|---|---|---|---|---|
| 128 | False | 2350.3 | 485.6 | 524.0 |
| 128 | True | 1473.3 | 483.3 | 508.0 |
| 512 | False | 7210.7 | 642.4 | 682.0 |
| 512 | True | 5568.2 | 495.8 | 534.0 |
| 1024 | False | 7610.0 | 1015.4 | 1064.0 |
| 1024 | True | 6031.7 | 603.4 | 654.0 |

Mid model, software-supported BF16 autocast, micro-batch 1, accumulation 1, 5 warm-up + 20 measured steps. At sequence 1024, checkpointing lowers peak allocated memory from 1,015.4 to 603.4 MiB while throughput falls from 7,610.0 to 6,031.7 tokens/s. The different device peaks are measured separately; no particular attention kernel cause is established by these tables.

### Colab reduced 151.9M architecture

| Actual parameters | Checkpointing | T4 tokens/s | T4 peak allocated MiB |
|---|---|---|---|
| 151898880 | False | 941.9 | 2611.4 |
| 151898880 | True | 825.7 | 2610.2 |

The architecture from config/125m.yaml uses the same reduced sequence 128, micro-batch 1 and accumulation 1 as the laptop run, with 3 warm-up + 10 measured steps and software-supported BF16 autocast. The larger T4 memory does not turn this into a measurement of the shipped sequence-1024, micro-batch-16 workload. No 7B or full pretraining run was attempted.

### Colab checkpoint/resume

| Autocast policy | Save seconds | Build+load seconds | Checkpoint MB | Max loss difference | Identical losses |
|---|---|---|---|---|---|
| float32 | 0.0711 | 0.2208 | 48.06 | 0.0 | 15/15 |
| bfloat16 (non-native) | 0.07 | 0.2333 | 48.06 | 0.0 | 15/15 |

Both policies match their own continuous 30-step run exactly after resuming at step 15. Cross-device or cross-policy bit identity is not implied. World size 1 uses the bare model, so GPU FSDP optimizer conversion remains untested.

### Colab GPU profiler

The 3-step trace contains **6,147 GPU kernel events** and **199.34 ms** of attributed device self-time. Unlike the laptop trace, GPU timing is available. The raw collective fraction is 0.0 for a single-device workload with no inter-GPU communication; it says nothing about distributed overlap. Operator self-time is not elapsed wall time. [Trace](../benchmarks/results/20261014-colab-t4/profile-nccl.json.gz), [operator summary](../benchmarks/results/20261014-colab-t4/profile-nccl.txt).

### Colab execution status

| Case | Exit status | Wall seconds |
|---|---|---|
| cuda-small-float32 | 0 | 17.8 |
| cuda-mid-float32 | 0 | 12.76 |
| cuda-small-bfloat16 | 0 | 10.3 |
| cuda-mid-bfloat16 | 0 | 14.56 |
| memory-s128-ac0 | 0 | 8.8 |
| memory-s128-ac1 | 0 | 9.15 |
| memory-s512-ac0 | 0 | 8.65 |
| memory-s512-ac1 | 0 | 9.1 |
| memory-s1024-ac0 | 0 | 10.29 |
| memory-s1024-ac1 | 0 | 11.35 |
| 125m-ac0 | 0 | 11.05 |
| 125m-ac1 | 0 | 11.4 |
| resume-float32 | 0 | 9.44 |
| resume-bfloat16 | 0 | 9.9 |
| profile | 0 | 24.69 |
