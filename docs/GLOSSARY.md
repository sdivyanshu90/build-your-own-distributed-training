# Glossary

| Term | Meaning in this repository |
|---|---|
| **DP / dp axis** | Data-parallel axis of the mesh. Ranks along it see *different* micro-batches. Implemented by FSDP, never by DDP. `dp_rank`, `dp_size`. |
| **TP / tp axis** | Tensor-parallel axis. Ranks along it hold *different slices of the same layer* and see the *same* tokens. |
| **2D parallelism** | `world_size = dp_size × tp_size`; the mesh is `(dp, tp)` row-major, `global_rank = dp_rank·tp_size + tp_rank`. |
| **DeviceMesh** | `torch.distributed.device_mesh.DeviceMesh`; named N-D grid of ranks; `mesh["tp"]` is the 1-D sub-mesh containing this rank. |
| **FSDP** | `FullyShardedDataParallel` (the original "FSDP1" class). ZeRO-3 style: shards params, grads, optimizer state. |
| **FULL_SHARD / SHARD_GRAD_OP / NO_SHARD / HYBRID_SHARD** | FSDP sharding strategies (≈ ZeRO-3 / ZeRO-2 / DDP-like replication / shard-in-group + replicate-across-groups). |
| **ZeRO-1/2/3** | Shard optimizer state / + gradients / + parameters. |
| **all-gather / reduce-scatter / all-reduce** | Collectives. FSDP: all-gather params, reduce-scatter grads. TP: all-reduce activations. |
| **no_sync** | FSDP context manager that skips the gradient reduce-scatter; used for the first `K-1` micro-steps. |
| **Column-parallel / Row-parallel** | Linear sharded along output / input features. Column outputs are sharded; a row-parallel layer consumes sharded input and all-reduces. Pairing them gives one all-reduce per sub-layer. |
| **DTensor** | PyTorch's distributed tensor with placements (`Shard(d)`, `Replicate()`, `Partial()`). `parallelize_module` converts linears to DTensor parameters. |
| **Sequence parallelism** | Shards norm activations along the sequence dimension within a TP group. |
| **GQA** | Grouped-query attention: `n_kv_heads < n_heads`. |
| **RoPE** | Rotary position embedding (rotate-half layout here). |
| **SwiGLU** | `down(silu(gate(x))·up(x))` MLP. |
| **MFU** | Model FLOPs utilisation: achieved model FLOP/s ÷ hardware peak. Meaningless on CPU here. |
| **gloo / NCCL** | Collective backends: CPU-capable / NVIDIA GPU. All measured results here are gloo. |
| **algbw / busbw** | `bytes/time` and the same scaled by the ring-algorithm factor (all-reduce `2(n-1)/n`, all-gather and reduce-scatter `(n-1)/n`) following nccl-tests. |
| **RSS** | Resident set size of a process (what the CPU memory benchmark reports). |
| **`_SUCCESS` marker** | File written last (rank 0, after a barrier) that commits a checkpoint. |
| **Shard (checkpoint)** | `rank_N/{model,optim,rng}.pt`: that rank's slice; only valid on the same `(world, tp, dp)` topology. |
| **Weak / strong scaling** | Work per rank constant (global batch grows) / total work constant. DP runs here are weak, TP runs strong. |
