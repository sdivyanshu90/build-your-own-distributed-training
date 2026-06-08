"""The ``Trainer``: wiring, the train loop, eval, checkpointing, profiling.

What this module does
---------------------
Owns the full lifecycle of a run. The constructor builds the world in the *one
correct order* (the ordering bugs it avoids are called out below); ``train`` runs
the outer loop with gradient accumulation, periodic eval, checkpointing, metric
logging, and a scoped profiler, and saves a checkpoint on Ctrl-C.

The construction order (and why it is non-negotiable)
-----------------------------------------------------
1. **Process context / mesh** first — everything else needs the groups.
2. **Seed** per ``dp_rank`` (DP ranks differ, TP ranks match) before any module
   is constructed so weight init is reproducible.
3. **Build model** on CPU/meta, then **apply TP**, then optional **activation
   checkpointing**, then **FSDP wrap** — strictly TP-before-FSDP so FSDP shards
   the already-TP DTensor params; and AC inside FSDP so recompute uses gathered
   params.
4. **Optimizer after FSDP** — a pre-wrap optimizer references unsharded params
   and its state is incompatible with FSDP (self-review checklist).
5. **Scheduler** on that optimizer.
6. **Resume** (if any) restores model+optimizer+scheduler+RNG+step together.

Invariants enforced at startup
------------------------------
  * Config validated against the runtime ``(world_size, dp_size)``.
  * The global-batch identity is cross-checked across ranks with an all_reduce so
    a per-rank config skew fails fast instead of corrupting the average.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from src.checkpointing.checkpoint import load_checkpoint, save_checkpoint
from src.checkpointing.recovery import require_valid_checkpoint
from src.config import TrainingConfig
from src.data.dataloader import build_dataloader
from src.data.dataset import PackedTokenDataset, SyntheticTokenDataset
from src.model.transformer import TransformerBlock, build_model
from src.observability.logging import build_logger
from src.observability.metrics import reset_peak_memory
from src.observability.profiler import build_profiler
from src.parallelism.fsdp_utils import (
    apply_activation_checkpointing,
    wrap_model_with_fsdp,
)
from src.parallelism.process_groups import ProcessContext, build_process_context
from src.parallelism.tensor_parallel import apply_tensor_parallelism
from src.training.loop import eval_step, train_step
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.seed import seed_everything


class Trainer:
    """End-to-end 2D-parallel trainer.

    Args:
        config: The full training config.
        ctx: An existing process context (tests inject one); if ``None`` a context
            is built from the config's parallel dims and backend.
        train_dataset / val_dataset: Optional datasets; if ``None`` a synthetic
            corpus is built (CI/local default).
        gpu_type: GPU key for MFU reporting.

    Attributes:
        model: The FSDP(+TP)-wrapped model.
        optimizer / scheduler: Built post-wrap.
        step: The current global optimizer step.
    """

    def __init__(
        self,
        config: TrainingConfig,
        ctx: ProcessContext | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        gpu_type: str = "a100",
    ) -> None:
        self.config = config
        self.gpu_type = gpu_type
        # 1. Process context / mesh.
        self.ctx = ctx or build_process_context(
            tp_size=config.parallel.tp_size,
            dp_size=config.parallel.dp_size,
            backend=config.backend,
        )
        config.validate(self.ctx.world_size, self.ctx.dims.dp_size)

        # 2. Seed per dp_rank (constant across the TP group).
        seed_everything(config.seed, self.ctx.dims.dp_rank, deterministic=False)

        self.logger = build_logger(self.ctx.rank, config.run_id)

        # 3. Build -> TP -> (AC) -> FSDP.
        model = build_model(config.model).to(self.ctx.device)
        apply_tensor_parallelism(
            model,
            self.ctx.mesh["tp"],
            sequence_parallel=config.parallel.sequence_parallel,
        )
        if config.parallel.activation_checkpointing:
            apply_activation_checkpointing(model, {TransformerBlock})
        # FSDP shards across the DP axis. With dp_size == 1 there is nothing to
        # shard — a 1-process FSDP unit only adds bookkeeping and per-step
        # collectives over a trivial group. So for pure-TP (dp_size == 1) we use
        # the TP-parallelised model directly. ``no_sync``/clip/state-dict helpers
        # all detect the non-FSDP case, so the rest of the loop is unchanged.
        if self.ctx.dims.dp_size > 1:
            self.model: torch.nn.Module = wrap_model_with_fsdp(
                model, self.ctx, config.parallel, {TransformerBlock}
            )
        else:
            self.model = model

        # 4./5. Optimizer + scheduler (post-wrap).
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer, config.scheduler, config.optimizer.lr
        )

        # Data.
        self.train_dataset = train_dataset or self._default_dataset(split="train")
        self.val_dataset = val_dataset or self._default_dataset(split="val")
        self.train_loader = self._build_loader(self.train_dataset, shuffle=config.data.shuffle)
        self.val_loader = self._build_loader(self.val_dataset, shuffle=False)
        self._train_iter: Iterator[Any] | None = None
        self._epoch = 0

        # Cross-rank consistency check of the global-batch identity.
        self._assert_consistent_global_batch()

        # 6. Resume.
        self.step = 0
        if config.resume_from:
            require_valid_checkpoint(
                config.resume_from, expected_config=config.to_dict(), deep=False
            )
            self.step = load_checkpoint(
                self.model, self.optimizer, self.scheduler, config.resume_from, self.ctx
            )
            self._fast_forward_data(self.step)
            self.logger.info("resumed", step=self.step, path=config.resume_from)

    # ----------------------------- setup helpers ----------------------------- #

    def _default_dataset(self, split: str) -> Dataset:
        cfg = self.config
        if cfg.data.dataset_path == "synthetic":
            n = 4096 if split == "train" else 256
            # Distinct seed per split so val != train; same across ranks (sampler
            # does the per-rank partition).
            return SyntheticTokenDataset(
                vocab_size=cfg.model.vocab_size,
                seq_len=cfg.data.seq_len,
                num_samples=n,
                seed=cfg.seed + (0 if split == "train" else 7),
            )
        path = cfg.data.dataset_path
        return PackedTokenDataset(path=path, seq_len=cfg.data.seq_len)

    def _build_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return build_dataloader(
            dataset,
            micro_batch_size=self.config.data.micro_batch_size,
            dp_size=self.ctx.dims.dp_size,
            dp_rank=self.ctx.dims.dp_rank,
            shuffle=shuffle,
            seed=self.config.seed,
            num_workers=self.config.data.num_workers,
        )

    def _assert_consistent_global_batch(self) -> None:
        """All ranks must agree on the (micro_bs, accum, global_bs) triple."""
        local = torch.tensor(
            [
                self.config.data.micro_batch_size,
                self.config.grad_accum_steps,
                self.config.data.global_batch_size,
            ],
            device=self.ctx.device,
            dtype=torch.long,
        )
        if dist.is_initialized() and self.ctx.world_size > 1:
            gathered = local.clone()
            dist.all_reduce(gathered, op=dist.ReduceOp.MAX)
            if not torch.equal(gathered, local):
                raise RuntimeError(
                    f"[rank {self.ctx.rank}] batch config disagrees across ranks: "
                    f"local={local.tolist()} max-across-ranks={gathered.tolist()}. "
                    f"All ranks must use identical batch settings."
                )

    # ------------------------------- iteration ------------------------------- #

    def _batch_iterator(self) -> Iterator[Any]:
        """Infinite iterator over the train loader, bumping the epoch each pass."""
        while True:
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)  # type: ignore[union-attr]
            yield from self.train_loader
            self._epoch += 1

    def _fast_forward_data(self, step: int) -> None:
        """Advance the data iterator to the position a continuous run would be at.

        On resume, a fresh iterator would restart from epoch 0, so the resumed run
        would see *different* data than a run that never stopped — breaking
        bit-exact resume. We replay the window stream so step ``S`` consumes the
        same sequences regardless of restarts.

        Args:
            step: The global step being resumed at.

        Performance note:
            This replays ``step * grad_accum_steps`` window fetches. For the
            synthetic/packed datasets a fetch is cheap (deterministic indexing),
            but for a very large resume point you would instead persist the
            sampler position and seek directly. The replay is exact and
            sufficient here.
        """
        self._epoch = 0
        self._train_iter = self._batch_iterator()
        for _ in range(step * self.config.grad_accum_steps):
            next(self._train_iter)

    def _next_window(self) -> tuple[list[Any], float]:
        """Pull ``grad_accum_steps`` micro-batches; return them + data-wait seconds."""
        if self._train_iter is None:
            self._train_iter = self._batch_iterator()
        window: list[Any] = []
        t0 = time.perf_counter()
        for _ in range(self.config.grad_accum_steps):
            window.append(next(self._train_iter))
        return window, time.perf_counter() - t0

    def _eval_windows(self) -> list[Any]:
        batches: list[Any] = []
        for i, batch in enumerate(self.val_loader):
            if i >= self.config.eval_steps:
                break
            batches.append(batch)
        return batches

    # --------------------------------- loop ---------------------------------- #

    def train(self) -> None:
        """Run the training loop from ``self.step`` to ``config.max_steps``.

        Side effects:
            Logs metrics (rank 0), writes checkpoints, exports a profiler trace,
            and on ``KeyboardInterrupt`` saves a final checkpoint before exiting.
        """
        cfg = self.config
        self.logger.info(
            "train_start",
            start_step=self.step,
            max_steps=cfg.max_steps,
            tp_size=self.ctx.dims.tp_size,
            dp_size=self.ctx.dims.dp_size,
            world_size=self.ctx.world_size,
        )
        reset_peak_memory(self.ctx.device)

        profiler_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        prof = None
        if cfg.profile_steps > 0 and self.ctx.is_rank0:
            prof = build_profiler(
                output_dir=f"traces/{cfg.run_id}",
                warmup_steps=cfg.warmup_profile_steps,
                profile_steps=cfg.profile_steps,
            )
            profiler_ctx = prof

        try:
            with profiler_ctx:
                while self.step < cfg.max_steps:
                    window, data_wait = self._next_window()
                    metrics = train_step(
                        self.model,
                        window,
                        self.optimizer,
                        self.scheduler,
                        self.ctx,
                        cfg,
                        gpu_type=self.gpu_type,
                        data_wait_s=data_wait,
                    )
                    self.step += 1
                    if prof is not None:
                        prof.step()

                    if self.step % cfg.log_interval == 0:
                        self.logger.metric("step", step=self.step, **_metric_fields(metrics))
                        reset_peak_memory(self.ctx.device)
                    if cfg.eval_interval > 0 and self.step % cfg.eval_interval == 0:
                        val_loss = eval_step(self.model, self._eval_windows(), self.ctx, cfg)
                        self.logger.metric("eval", step=self.step, val_loss=val_loss)
                    if cfg.save_interval > 0 and self.step % cfg.save_interval == 0:
                        path = save_checkpoint(
                            self.model, self.optimizer, self.scheduler, self.step, cfg, self.ctx
                        )
                        self.logger.info("checkpoint", step=self.step, path=path)
        except KeyboardInterrupt:
            self.logger.warning("interrupted", step=self.step)
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.step, cfg, self.ctx
            )
            self.logger.info("checkpoint_on_interrupt", step=self.step)
            raise
        self.logger.info("train_done", step=self.step)


def _metric_fields(m: Any) -> dict[str, Any]:
    """Flatten a ``StepMetrics`` into rounded JSON-friendly fields for logging."""
    return {
        "loss": round(m.loss, 5),
        "ppl": round(m.perplexity(), 3),
        "grad_norm": round(m.grad_norm, 4),
        "lr": m.learning_rate,
        "tok_per_s": round(m.tokens_per_second, 1),
        "mfu": round(m.mfu, 4) if m.mfu is not None else None,
        "step_time_s": round(m.step_time_s, 4),
        "data_wait_s": round(m.data_wait_s, 4),
        "peak_mem_mb": round(m.peak_memory_bytes / 1e6, 1),
    }
