"""Optimizer construction: AdamW / LAMB with FSDP-compatible param groups.

What this module does
---------------------
Builds the optimizer on an *already-FSDP-wrapped* model. Two pieces matter:

1. **Param groups built post-wrap.** Weight matrices (ndim >= 2) get decoupled
   weight decay; norms and biases (ndim < 2) get none. Decaying a LayerNorm gain
   or a bias pulls it toward zero for no benefit and measurably hurts. Because
   FSDP is created with ``use_orig_params=True``, ``named_parameters()`` still
   exposes the original parameter shapes, so we can group by ``dim()`` correctly
   even though the underlying storage is sharded. **The optimizer must be built
   after wrapping** — a pre-wrap optimizer holds references to the unsharded
   parameters and its state is incompatible with FSDP's sharded params.

2. **Optimizer choice.** AdamW is the default and the right call for FSDP +
   pretraining. LAMB is provided for large-batch regimes but carries an
   FSDP caveat documented on the class.

Invariant
---------
Every rank builds the *same* param-group structure (same keys, same
hyperparameters) over its *disjoint* shard of each parameter. This is what lets
``FSDP.optim_state_dict`` re-key and reshard the optimizer state on resume.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.config import OptimizerConfig


def build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups.

    Args:
        model: The FSDP-wrapped model (``use_orig_params=True`` preserves shapes).
        weight_decay: Decay applied to the >=2D group.

    Returns:
        A two-element list of param-group dicts: ``{"params": ..., "weight_decay":
        weight_decay}`` for matrices and ``{"params": ..., "weight_decay": 0.0}``
        for norms/biases. Empty groups are omitted.

    Note:
        Only parameters with ``requires_grad`` are included so frozen params do
        not allocate optimizer state.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        # Tensors with >=2 dims are weight matrices -> decay; norms/biases -> no.
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments optimizer for large-batch training.

    LAMB is AdamW with a per-parameter *trust ratio* ``||w|| / ||update||`` that
    rescales each layer's step so large batches stay stable. See You et al.,
    "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes".

    Args:
        params: Iterable of parameters or param-group dicts.
        lr: Learning rate.
        betas: Moment decay coefficients.
        eps: Denominator epsilon.
        weight_decay: Decoupled weight decay.

    FSDP caveat:
        The trust ratio uses ``||w||`` and ``||update||``. Under FSDP each rank
        holds only a *shard* of ``w``, so the norms here are per-shard, making the
        trust ratio an approximation of the true global LAMB ratio. For bitwise
        LAMB under sharding you must all_reduce the squared norms across the DP/TP
        groups per parameter (costly). AdamW has no such dependency and is the
        recommended default; use LAMB only when its large-batch behaviour is
        specifically needed and the approximation is acceptable.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"LAMB: lr must be > 0, got {lr}.")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        """Perform one optimization step.

        Args:
            closure: Optional closure that re-evaluates the model and returns the
                loss (standard ``Optimizer`` contract).

        Returns:
            The loss from ``closure`` if provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_c1 = 1 - beta1**step
                bias_c2 = 1 - beta2**step
                update = (exp_avg / bias_c1) / ((exp_avg_sq / bias_c2).sqrt() + group["eps"])
                if group["weight_decay"] != 0:
                    update = update.add(p, alpha=group["weight_decay"])

                w_norm = p.norm(2)
                u_norm = update.norm(2)
                # trust = ||w||/||u|| where both are positive, else 1.0.
                trust = torch.where(
                    w_norm > 0,
                    torch.where(u_norm > 0, w_norm / u_norm, torch.ones_like(w_norm)),
                    torch.ones_like(w_norm),
                )
                p.add_(update * trust, alpha=-group["lr"])
        return loss


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    """Construct the optimizer from config on the wrapped model's parameters.

    Args:
        model: The FSDP-wrapped model.
        cfg: Optimizer config (name, lr, betas, eps, weight_decay, fused).

    Returns:
        An ``AdamW`` or :class:`LAMB` instance with decay/no-decay param groups.

    Raises:
        ValueError: If ``cfg.name`` is not ``"adamw"`` or ``"lamb"``.

    Performance note:
        ``fused=True`` AdamW fuses the elementwise update into one CUDA kernel,
        a meaningful speedup at scale; it is silently ignored on CPU.
    """
    groups = build_param_groups(model, cfg.weight_decay)
    name = cfg.name.lower()
    if name == "adamw":
        use_fused = cfg.fused and torch.cuda.is_available()
        return torch.optim.AdamW(
            groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=use_fused
        )
    if name == "lamb":
        return LAMB(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    raise ValueError(f"Unknown optimizer name={cfg.name!r}; valid: 'adamw', 'lamb'.")
