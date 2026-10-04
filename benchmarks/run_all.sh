#!/usr/bin/env bash
# Reproducible CPU/gloo benchmark driver.
#
#   benchmarks/run_all.sh <stage> [python-executable]
#
# Stages (run each under your own lock if the machine is shared):
#   scaling      tokens/s + step time vs world size, DP(FSDP) / TP / 2D
#   memory       per-rank peak RSS + resident state bytes per strategy
#   comm         all_reduce / all_gather / reduce_scatter latency + bandwidth
#   resume       checkpoint save/load cost + resume exactness
#   fault        SIGKILL a rank, measure detection + recovery
#   convergence  loss curves (single / FSDP / TP / 2D) on the synthetic corpus
#
# Safety rails: never more than 4 ranks, OMP/MKL threads chosen so that
# world_size * threads <= 6 (policy "pr": 1 thread per rank; policy "budget":
# ~6 threads in total), every run wrapped in `timeout`, and a free-RAM guard
# before each launch. Results append to benchmarks/results/*.jsonl.
set -euo pipefail
STAGE=${1:?stage}
PY=${2:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
RES="$HERE/results"; mkdir -p "$RES"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$HERE/..:$HERE"
MIN_FREE_MB=${MIN_FREE_MB:-1500}
RUN_TIMEOUT=${RUN_TIMEOUT:-300}

wait_for_ram() {
  while [ "$(free -m | awk '/^Mem:/{print $7}')" -lt "$MIN_FREE_MB" ]; do
    echo "[guard] available RAM < ${MIN_FREE_MB} MB, waiting..." >&2; sleep 10
  done
}

# run <nproc> <threads> <script> [args...]
run() {
  local n=$1 th=$2 script=$3; shift 3
  [ "$n" -le 4 ] || { echo "refusing >4 ranks" >&2; exit 1; }
  [ $((n * th)) -le 6 ] || { echo "threads*ranks > 6" >&2; exit 1; }
  wait_for_ram
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th timeout "$RUN_TIMEOUT" \
    nice -n 15 "$PY" -m torch.distributed.run --standalone --nproc_per_node="$n" \
    "$HERE/$script" "$@"
}

case "$STAGE" in
  scaling)
    OUT="$RES/scaling.jsonl"
    COMMON=(--preset small --mbs 4 --seq-len 128 --steps 20 --warmup 3 --out "$OUT")
    # Policy "pr": 1 thread per rank -> isolates parallel overhead/scaling.
    for cfg in "1 1" "2 1" "4 1"; do set -- $cfg
      run "$1" 1 bench_train.py "${COMMON[@]}" --tp 1 --label "dp$1_pr"; done
    for tp in 2 4; do run "$tp" 1 bench_train.py "${COMMON[@]}" --tp "$tp" --label "tp${tp}_pr"; done
    run 4 1 bench_train.py "${COMMON[@]}" --tp 2 --label "2d_dp2tp2_pr"
    # Policy "budget": ~6 threads in total (world 1 -> 6, 2 -> 3, 4 -> 1).
    run 1 6 bench_train.py "${COMMON[@]}" --tp 1 --label "dp1_budget"
    run 2 3 bench_train.py "${COMMON[@]}" --tp 1 --label "dp2_budget"
    run 2 3 bench_train.py "${COMMON[@]}" --tp 2 --label "tp2_budget"
    ;;
  memory)
    OUT="$RES/memory.jsonl"
    COMMON=(--preset mid --mbs 1 --seq-len 128 --steps 4 --warmup 2 --out "$OUT")
    run 1 1 bench_train.py "${COMMON[@]}" --tp 1 --label "mem_single"
    run 2 1 bench_train.py "${COMMON[@]}" --tp 1 --sharding NO_SHARD --label "mem_noshard_dp2"
    run 2 1 bench_train.py "${COMMON[@]}" --tp 1 --sharding SHARD_GRAD_OP --label "mem_zero2_dp2"
    run 2 1 bench_train.py "${COMMON[@]}" --tp 1 --sharding FULL_SHARD --label "mem_fsdp_dp2"
    run 4 1 bench_train.py "${COMMON[@]}" --tp 1 --sharding FULL_SHARD --label "mem_fsdp_dp4"
    run 2 1 bench_train.py "${COMMON[@]}" --tp 2 --label "mem_tp2"
    run 4 1 bench_train.py "${COMMON[@]}" --tp 4 --label "mem_tp4"
    run 4 1 bench_train.py "${COMMON[@]}" --tp 2 --label "mem_2d_dp2tp2"
    ;;
  comm)
    for n in 2 4; do run "$n" 1 bench_comm.py --out "$RES/comm.jsonl"; done
    ;;
  resume)
    run 1 6 bench_resume.py --preset small --steps 30 --out "$RES/resume.jsonl"
    run 2 3 bench_resume.py --preset small --steps 30 --out "$RES/resume.jsonl"
    ;;
  fault)
    wait_for_ram
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 nice -n 15 "$PY" "$HERE/bench_fault.py" \
      --nproc 2 --out "$RES/fault.jsonl"
    ;;
  convergence)
    OUT="$RES/convergence.jsonl"
    # Global batch = 16 sequences in every layout (mbs * dp = 16).
    COMMON=(--preset tiny --seq-len 32 --steps 150 --warmup 0 --curve --lr 3e-3 --out "$OUT")
    run 1 2 bench_train.py "${COMMON[@]}" --mbs 16 --tp 1 --label "conv_single"
    run 2 1 bench_train.py "${COMMON[@]}" --mbs 8 --tp 1 --label "conv_fsdp2"
    run 2 1 bench_train.py "${COMMON[@]}" --mbs 16 --tp 2 --label "conv_tp2"
    run 4 1 bench_train.py "${COMMON[@]}" --mbs 8 --tp 2 --label "conv_2d"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 1;;
esac
