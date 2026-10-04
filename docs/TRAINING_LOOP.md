# Training loop

Files: `src/training/trainer.py` (orchestration), `loop.py` (one step),
`grad_utils.py`, `optimizer.py`, `scheduler.py`, `src/utils/{dtype,seed,env}.py`.

## Trainer construction order (`Trainer.__init__`)

1. `build_process_context(tp, dp, backend)`: init the default group (30-minute
   collective timeout), pick the device, build the `(dp, tp)` mesh, fetch the groups.
2. `config.validate(world_size, dp_size)`.
3. `seed_everything(seed, 0)` then `build_model(...)` — **same init on every rank** —
   then `seed_everything(seed, dp_rank)` for per-rank stochastic state.
4. `apply_tensor_parallelism` (no-op at `tp=1`) → optional activation checkpointing
   → FSDP wrap **iff `dp_size > 1`** (pure-TP and single-process use the bare module).
5. `build_optimizer` (after wrapping) and `build_scheduler`.
6. Datasets, samplers, loaders; startup all-reduce check that batch settings agree.
7. If `resume_from`: resolve (run dir → newest valid `step_*`), validate, load,
   fast-forward data.

## One optimizer step (`loop.train_step`)

```mermaid
sequenceDiagram
    participant T as Trainer
    participant S as train_step
    participant M as model (FSDP/TP)
    T->>S: window of K micro-batches
    S->>S: zero_grad(set_to_none)
    loop i in 0..K-1
        S->>M: no_sync() if i < K-1 (FSDP only)
        S->>M: forward (autocast on CUDA) -> loss
        S->>M: (loss / K).backward()
        Note over M: TP all-reduces run every micro-step;<br/>FSDP all-gather every micro-step;<br/>reduce-scatter only on the last
    end
    S->>S: grad_finite_check  (all_reduce MIN, world group)
    S->>S: clip_grad_norm_  (dp [+ tp] reduction)
    S->>S: optimizer.step(); scheduler.step()
    S->>S: aggregate_loss (all_reduce SUM / dp over dp group)
    S-->>T: StepMetrics
```

Details, each verified in `src/training/loop.py`:

* **Loss scaling for accumulation.** Each micro-batch loss is the token-mean loss of
  that micro-batch; the step backpropagates `loss / K`, so the accumulated gradient
  equals the gradient of the mean over the whole window (tests:
  `tests/unit/test_grad_accumulation.py`). Across DP ranks, FSDP's reduce-scatter
  averages the gradients, so the effective gradient is the mean over
  `K · dp` micro-batches; the **reported** loss is the mean over the window and over
  DP ranks. Every sequence has the same number of tokens (no padding, no
  `ignore_index` hits in the shipped datasets), so mean-of-means equals the global
  token mean. With variable-length or masked batches it would not.
* **`no_sync`.** `_maybe_no_sync` uses `model.no_sync()` for micro-steps `0..K-2`
  when the model has that attribute (i.e. it is FSDP). For plain modules (pure TP,
  single process) nothing is skipped and nothing needs to be: there is no DP
  collective. Under `FULL_SHARD`, PyTorch FSDP's `no_sync` accumulates gradients in
  *unsharded* form until the last micro-step (per the PyTorch documentation; this
  repository does not measure it), trading a full-size gradient buffer for fewer
  reduce-scatters. The all-gathers still run every micro-step (see PARALLELISM.md,
  measured counts).
* **Mixed precision.** On CUDA the forward runs under `torch.autocast(param_dtype)`
  and, if FSDP-wrapped, FSDP's `MixedPrecision(param, reduce, buffer)` also applies
  with `cast_forward_inputs=True`. On **CPU there is no autocast**
  (`use_autocast = device.type == "cuda"`); a non-FSDP CPU run is therefore plain
  fp32 whatever `param_dtype` says. `test_tiny.yaml` pins fp32 everywhere.
  bf16 needs no loss scaling (fp32 exponent range), and there is no `GradScaler`; fp16
  would not be safe here.
* **Finite check, then clip.** `grad_finite_check` all-reduces a 0/1 flag with `MIN`
  over the *world* group and raises `GradNotFiniteError` on every rank together.
  `clip_grad_norm_` returns the pre-clip global norm:
  * FSDP only: `FSDP.clip_grad_norm_` (reduces over the shard group).
  * FSDP + TP (2D): `_fsdp_tp_clip` — squared norms of TP-sharded grads are summed
    over both groups, TP-replicated grads are weighted `1/tp` (see PARALLELISM.md).
    **This replaced a bug**: FSDP's own clip missed the other TP ranks' shards
    (measured grad norm 1.088 vs a reference 1.120 at tp=2, dp=2; different TP ranks
    could also clip by different factors).
  * TP only: `_tp_aware_clip`, same weighting.
  * Plain: `torch.nn.utils.clip_grad_norm_`.
  The clip scales by `max_norm / (norm + 1e-6)` only if that coefficient is `< 1`.
* **Step/metrics.** `step_time_s` covers the micro-batch loop, clip and optimizer
  (`data_wait_s` measures `next(loader)` separately, outside it). Tokens per second
  are `local_tokens · dp_size / step_time`; MFU uses `6N + 12·L·d·max_seq_len` and a
  per-GPU peak table (meaningless on CPU).

## Optimizer (`optimizer.py`)

`build_param_groups` puts params with `dim() >= 2` in a decay group and everything
else (norm gains, biases) in a no-decay group; shared (tied) parameters are
deduplicated by `id`. Because it runs after FSDP wrapping with
`use_orig_params=True`, it sees the original parameter objects (local shards).
`adamw` is `torch.optim.AdamW` (fused only on CUDA). `lamb` is a small in-repo
implementation whose trust ratio uses the norms of the **local** tensor: under FSDP
that is a per-shard ratio, not the per-layer ratio of the LAMB paper (documented
approximation; not covered by a distributed test).

## Scheduler (`scheduler.py`)

`LambdaLR` with: `step < warmup`: `(step+1)/warmup` (so the first update uses
`lr/warmup`, never 0); `step >= max_steps`: `min_lr/lr`; otherwise cosine from 1 to
`min_lr/lr` over `[warmup, max_steps)`. `scheduler.step()` is called once per
optimizer step, so the LR used for update *n* (0-based) is `factor(n)` and
`get_last_lr()` after the step is `factor(n+1)` (what `StepMetrics.learning_rate`
logs). Edge cases are checked by `tests/unit/test_scheduler.py`; the scheduler state
(`last_epoch`) is saved and restored by the checkpointer. `lr_lambda_factory` raises
if `max_steps <= warmup_steps` or `min_lr > peak_lr`.

## Evaluation

`eval_step` runs under `torch.inference_mode`, averages the loss over the supplied
batches and over DP ranks. All ranks must pass the same number of batches (they
do: `_eval_windows` takes the first `eval_steps` batches of an equal-length sharded
val loader).

## Seeding and determinism (`src/utils/seed.py`)

`seed_everything(base, dp_rank)` seeds `random`, NumPy and torch with
`base + dp_rank`; `deterministic=True` additionally enables deterministic
algorithms, but the trainer always passes `False`. The RNG state of every rank
(Python, NumPy, torch, CUDA) is saved in `rank_N/rng.pt`. Without dropout the RNG
does not influence training after init, so resume exactness comes from the model,
optimizer, scheduler and data replay, not from the RNG blob.

## Shutdown

`train.py` wraps the trainer in `try/finally: destroy_distributed()`. On
`KeyboardInterrupt` the trainer saves a checkpoint and re-raises. A `SIGTERM`
(what `torchrun` sends to surviving ranks, and what schedulers send on pre-emption)
is **not** handled, so no checkpoint is written in that case; the previous periodic
checkpoint is what you resume from. No checkpoint is written at the end of
training unless `max_steps` is a multiple of `save_interval`.
