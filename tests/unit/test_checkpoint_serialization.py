"""Unit tests for checkpoint save/load round-trips and corruption detection.

Uses a *plain* (non-FSDP) tiny model on a world_size=1 Gloo group so the full
save/load/validate/recover logic — atomic writes, the ``_SUCCESS`` marker, RNG and
scheduler round-trip, corruption detection — runs on CPU. (FSDP sharded optimizer
state-dict needs CUDA in torch 2.3; that path is covered by the gated integration
test.)
"""

from __future__ import annotations

import contextlib
import os

import torch

from src.checkpointing.checkpoint import load_checkpoint, save_checkpoint
from src.checkpointing.recovery import (
    find_latest_valid_checkpoint,
    validate_checkpoint,
)
from src.config import ModelConfig, SchedulerConfig, TrainingConfig
from src.model.transformer import build_model
from src.parallelism.process_groups import build_process_context
from src.training.scheduler import build_scheduler


def _tiny_config(tmp: str) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.model = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=32, tie_embeddings=True,
    )
    cfg.scheduler = SchedulerConfig(warmup_steps=2, max_steps=20, min_lr=1e-4)
    cfg.checkpoint_dir = tmp
    cfg.run_id = "unit_ckpt"
    cfg.backend = "gloo"
    return cfg


def _build(tmp: str):
    cfg = _tiny_config(tmp)
    ctx = build_process_context(tp_size=1, dp_size=-1, backend="gloo")
    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, cfg.scheduler, peak_lr=1e-3)
    return cfg, ctx, model, opt, sched


def test_save_load_identical_outputs(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    tokens = torch.randint(0, cfg.model.vocab_size, (2, 16))
    # Take a couple of optimizer steps so weights are non-trivial.
    for _ in range(2):
        _, loss = model(tokens, labels=tokens)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
    with torch.no_grad():
        before, _ = model(tokens)

    path = save_checkpoint(model, opt, sched, step=2, config=cfg, ctx=ctx)

    fresh = build_model(cfg.model)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = build_scheduler(fresh_opt, cfg.scheduler, peak_lr=1e-3)
    step = load_checkpoint(fresh, fresh_opt, fresh_sched, path, ctx)
    assert step == 2
    with torch.no_grad():
        after, _ = fresh(tokens)
    assert torch.allclose(before, after, atol=1e-6)


def test_save_load_optimizer_state(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    tokens = torch.randint(0, cfg.model.vocab_size, (2, 16))
    _, loss = model(tokens, labels=tokens)
    loss.backward()
    opt.step()
    opt.zero_grad()
    path = save_checkpoint(model, opt, sched, step=1, config=cfg, ctx=ctx)

    fresh = build_model(cfg.model)
    fresh.load_state_dict(model.state_dict())
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    fresh_sched = build_scheduler(fresh_opt, cfg.scheduler, peak_lr=1e-3)
    load_checkpoint(fresh, fresh_opt, fresh_sched, path, ctx)

    # One more identical step on both must move weights identically.
    for m, o in [(model, opt), (fresh, fresh_opt)]:
        _, ll = m(tokens, labels=tokens)
        ll.backward()
        o.step()
        o.zero_grad()
    for p1, p2 in zip(model.parameters(), fresh.parameters(), strict=False):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_truncated_checkpoint_detected(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    path = save_checkpoint(model, opt, sched, step=1, config=cfg, ctx=ctx)
    # Simulate a crash that truncated the rank-0 model shard to zero bytes.
    shard = os.path.join(path, "rank_0", "model.pt")
    with open(shard, "w"):
        pass  # truncate
    result = validate_checkpoint(path, deep=True)
    assert not result.is_valid
    assert any("zero-byte" in e or "corrupt" in e for e in result.errors)


def test_find_latest_skips_corrupt(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    save_checkpoint(model, opt, sched, step=10, config=cfg, ctx=ctx)
    path20 = save_checkpoint(model, opt, sched, step=20, config=cfg, ctx=ctx)
    # Corrupt the newer checkpoint by removing its success marker.
    os.remove(os.path.join(path20, "_SUCCESS"))
    run_dir = os.path.join(cfg.checkpoint_dir, cfg.run_id)
    latest = find_latest_valid_checkpoint(run_dir)
    assert latest is not None and latest.endswith("step_10")


def test_config_mismatch_detected(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    path = save_checkpoint(model, opt, sched, step=1, config=cfg, ctx=ctx)
    # Pretend the current model now has a different d_model.
    bad = cfg.to_dict()
    bad["model"]["d_model"] = 999
    result = validate_checkpoint(path, expected_config=bad)
    assert not result.is_valid
    assert any("d_model" in e for e in result.errors)


def test_resave_clears_stale_success_marker(single_process_pg: None, tmp_path) -> None:
    """Regression: re-saving a step must not leave the old marker vouching for
    files being overwritten (marker is removed before any write)."""
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    path = save_checkpoint(model, opt, sched, step=3, config=cfg, ctx=ctx)
    marker = os.path.join(path, "_SUCCESS")
    assert os.path.exists(marker)
    # Simulate a crash mid-resave: wrap one write so it raises after the marker is cleared.
    import src.checkpointing.checkpoint as ck

    real = ck._atomic_torch_save
    seen: dict[str, bool] = {}

    def boom(obj, p):  # type: ignore[no-untyped-def]
        seen["marker_present"] = os.path.exists(marker)
        raise RuntimeError("crash during save")

    ck._atomic_torch_save = boom
    try:
        with contextlib.suppress(RuntimeError):
            save_checkpoint(model, opt, sched, step=3, config=cfg, ctx=ctx)
    finally:
        ck._atomic_torch_save = real
    assert seen["marker_present"] is False
    assert not validate_checkpoint(path).is_valid


def test_load_rejects_topology_mismatch(single_process_pg: None, tmp_path) -> None:
    """Regression: a sharded checkpoint from another (world,tp,dp) must not load
    silently into whatever rank_N dir happens to exist."""
    import json

    import pytest

    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    path = save_checkpoint(model, opt, sched, step=1, config=cfg, ctx=ctx)
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["dp_size"], meta["world_size"] = 4, 4
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="topology"):
        load_checkpoint(model, opt, sched, path, ctx)


def test_validate_flags_missing_optimizer_shard(single_process_pg: None, tmp_path) -> None:
    cfg, ctx, model, opt, sched = _build(str(tmp_path))
    path = save_checkpoint(model, opt, sched, step=1, config=cfg, ctx=ctx)
    os.remove(os.path.join(path, "rank_0", "optim.pt"))
    res = validate_checkpoint(path)
    assert not res.is_valid and any("optim.pt" in e for e in res.errors)
