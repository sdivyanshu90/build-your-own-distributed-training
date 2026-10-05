# Parallelism deep-dive

Notation: `P` = total parameters, `L` layers, `d` model width, `B·S` tokens per
micro-batch per rank, `n = world_size = dp · tp`, `b_w` bytes per parameter element
held (4 for fp32 master weights), `b_a` bytes per activation element. All formulas
describe this code; where I measured something, the measurement is cited from
[BENCHMARKS.md](BENCHMARKS.md) (CPU/gloo, torch 2.6.0, 2026-10-04).

## 1. The four layouts

| Layout | Mesh | What is sharded | In this repo |
|---|---|---|---|
| Single process | `(1,1)` | nothing | bare module, no FSDP |
| Data parallel via FSDP | `(dp,1)` | params, grads, optimizer state across `dp` (FULL_SHARD) | `wrap_model_with_fsdp`; DDP is **not** implemented |
| Tensor parallel | `(1,tp)` | each block's 7 linears across `tp`; activations inside attention/MLP | DTensor plan; no FSDP wrap at `dp=1` |
| 2D | `(dp,tp)` | both | TP first, then FSDP over the DTensor params |

Why FSDP instead of DDP: DDP replicates `P·(b_w + 4 + 8)` (params + grads + two
Adam moments in fp32) on every rank; FSDP `FULL_SHARD` leaves `1/dp` of it resident
plus transient gathers. `NO_SHARD` (FSDP with replication) is the closest thing to
DDP here and is used as the "replicated" baseline in the memory benchmark.

## 2. What each parallelism communicates

### FSDP (dp axis), per optimizer step with `K` micro-batches

* `FULL_SHARD` re-shards parameters after use, so every micro-batch performs an
  all-gather of each FSDP unit in **forward** and again in **backward**:
  `2K` all-gathers per unit. The reduce-scatter of gradients happens once per unit,
  in the last micro-batch (earlier ones run under `no_sync`).
* Bytes moved per rank per unit-collective: `(dp-1)/dp · bytes(unit)`; the gather is
  in `param_dtype`, the scatter in `reduce_dtype`.
* Per step and rank, for `P_dp` parameters in the FSDP groups:
  `V_fsdp ≈ (dp-1)/dp · P_dp · (2K·b_param + b_reduce)`. With the fp32 test config
  `b_param = b_reduce = 4`; with the production bf16/fp32 policy `b_param = 2`,
  `b_reduce = 4`.
* `no_sync` removes `K-1` reduce-scatters but **not** the all-gathers (measured
  counts in §3).
* Units: each `TransformerBlock` plus one root unit holding the embedding, final
  norm and (if untied) LM head.

### Tensor parallelism (tp axis), per micro-batch

* Column-parallel `wq,wk,wv,gate_proj,up_proj`: input replicated, output sharded; no
  forward collective. Row-parallel `wo,down_proj`: sharded input, partial sums →
  **one all-reduce of the `B·S·d` output** in forward. Forward: 2 all-reduces per
  block (attention out, MLP out).
* Backward mirrors it: the input gradient of every column-parallel layer is partial
  across TP ranks and must be summed. The hand-written version shares one all-reduce
  per sub-layer input (`_CopyToTPRegion`); the DTensor plan issues them per op. The
  measured count of collectives (§3) is the authority.
* Volume per all-reduce per rank (ring): `2(tp-1)/tp · B·S·d·b_a`.
* Not sharded: embedding, final norm, LM head, RMSNorm weights (replicated); the
  `B·S·V` logits are computed redundantly on every TP rank.
* Attention heads: `n_heads/tp` query heads and `n_kv_heads/tp` KV heads per rank
  (validated up front), reshaped with `-1`.

### 2D

Both sets of collectives run, on different groups: FSDP collectives on the DP group
(`tp` independent groups of `dp` ranks, each moving `1/tp` of the parameters), TP
all-reduces on the TP group. The TP group is the *fast-varying* mesh axis so it maps
to adjacent ranks/NVLink on a real cluster.

## 3. Measured collective schedule (profiler counts, one `train_step`)

Source: `benchmarks/count_collectives.py`, raw data in
`benchmarks/results/collectives.jsonl`. (Filled from the run; see the table in
BENCHMARKS.md §Collectives per step.)

| world | dp | tp | layers | grad_accum K | FSDP all-gather | FSDP reduce-scatter | all_reduce (c10d, total) | of which TP (DTensor functional) |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 1 | 2 | 1 | 5 | 3 | 3 | 0 |
| 2 | 2 | 1 | 2 | 2 | 10 | 3 | 3 | 0 |
| 2 | 1 | 2 | 2 | 1 | 0 | 0 | 16 | 14 |
| 2 | 1 | 2 | 4 | 1 | 0 | 0 | 30 | 28 |

Reading the counts (2 layers = 2 FSDP block units + 1 root unit):

* **FSDP, K=1:** 5 all-gathers = 2 per block (forward + backward re-gather, because
  `FULL_SHARD` reshards after forward) + 1 for the root unit (kept unsharded between
  forward and backward); 3 reduce-scatters = one per unit; 3 plain all-reduces = finite
  check, clip-norm, loss aggregation.
* **FSDP, K=2:** all-gathers double (10), reduce-scatters stay 3: `no_sync` removes the
  extra reduce-scatters but not the gathers, as stated in §2.
* **TP=2:** 14 TP all-reduces for 2 layers and 28 for 4 layers, i.e. **7 per layer per
  micro-batch** (2 forward, so 5 backward: the DTensor plan reduces the input gradient of
  each of `wq, wk, wv, gate_proj, up_proj` separately). The remaining 2 plain
  all-reduces are the finite check and the clip norm. No TP collectives occur outside the
  blocks (replicated embedding/head).
* The 2D (4-rank) count was not obtained: the attempt crashed on the 2D gradient
  accumulation bug (finding #5) and the re-run did not complete in the time available.

## 4. Memory per rank

Resident model state per rank, with `P_tp` = TP-sharded parameters (the seven
linears per block), `P_rep` = replicated parameters (embedding, norms, untied head):

```
state_bytes(rank) = (P_tp / tp + P_rep) / dp_shard · (b_w + b_grad + 2·b_adam)
                    [FULL_SHARD; dp_shard = 1 for NO_SHARD, = hybrid shard size for HYBRID]
```

* FSDP `FULL_SHARD`: params, grads and Adam moments all `/dp` (ZeRO-3).
* `SHARD_GRAD_OP` (ZeRO-2): params stay full between steps; grads + moments `/dp`.
* Transient: one FSDP unit gathered at a time (`limit_all_gathers`), i.e.
  `P_block/tp` elements in `param_dtype`, plus its unsharded gradient before
  reduce-scatter. Under `no_sync` (accumulation) unsharded gradients of all units can
  accumulate (PyTorch documentation; not measured here).
* Activations: `O(L·B·S·d)` per rank without checkpointing, `O(B·S·d)` plus one
  block with it; the logits `B·S·V` are not reduced by TP (see MODEL.md).
* Mixed precision (bf16 param, fp32 reduce): FSDP keeps fp32 sharded master params,
  a bf16 copy exists only for the gathered unit; Adam moments fp32.

Measured resident state bytes and RSS for the `mid` preset (27.8M parameters) are in
BENCHMARKS.md §Memory; the check of the formula against them is below.

Measured vs formula: see the memory table and the formula-check lines in BENCHMARKS.md §Memory per rank.

Formula check: `P·16 B` for the `mid` model is 424.1 MiB; the single-process run holds 424.1 MiB of parameters+gradients+Adam state, and `NO_SHARD` at dp=2 holds the same 424.1 MiB on each rank (replication, no saving).
* SHARD_GRAD_OP (ZeRO-2) dp=2: not measured (run did not complete on the shared machine).
* FULL_SHARD dp=2: not measured (run did not complete on the shared machine).
* FULL_SHARD dp=4: not measured (run did not complete on the shared machine).
* TP=2: not measured (run did not complete on the shared machine).
* 2D dp=2 x tp=2: not measured (run did not complete on the shared machine).

## 5. Sequence parallelism

`parallel.sequence_parallel: true` adds `SequenceParallel` to the two norms per block,
makes the attention/MLP inputs `Shard(1)` → `Replicate()` (all-gather along the
sequence) and the row-parallel outputs `Shard(1)` (reduce-scatter instead of
all-reduce), shrinking norm activations by `tp`. **Status (audit):** the original plan crashed at construction (`PrepareModuleInput` was given one layout for the three-argument attention) and never scattered/gathered the residual stream. It now scatters the embedding output along the sequence, keeps the stream as local `S/tp` shards between blocks, and gathers before the final norm; `test_sequence_parallel_tp2_matches_reference` shows loss and global grad norm equal to the dense single-process reference at tp=2 (CPU/gloo, one forward/backward). Throughput/memory benefit were not measured.

## 6. Gradient clipping across shards

Global squared norm = sum over every distinct gradient element once. Per rank the
resident grad is a `1/dp` slice of the TP-local tensor, so:

```
‖g‖² = Σ_dp-shards Σ_tp-ranks  w · ‖g_local‖²,   w = 1 (TP-sharded param), 1/tp (TP-replicated)
```

`FSDP.clip_grad_norm_` computes only the `Σ_dp-shards` part; `_fsdp_tp_clip` adds the TP
sum and the weights (name-based rule `is_tp_sharded_param`). Verified: at `tp=2,dp=2`
the reported norm was 1.088 before the fix versus 1.120 for the single-process
reference; after the fix the equivalence test passes at rtol 1e-3.

## 7. Choosing a layout (qualitative; CPU numbers do not settle this on GPUs)

* Fits with FSDP alone → use FSDP alone: TP adds latency-bound all-reduces on the
  critical path of every layer.
* Needs TP when one layer's working set or activations do not fit, or the global batch
  cannot grow; keep `tp ≤ GPUs per NVLink domain`.
* Multi-node: TP inside the node, FSDP across; `HYBRID_SHARD` to keep the per-layer
  all-gathers inside the node.
