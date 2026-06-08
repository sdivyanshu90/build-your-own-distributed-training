"""TP-aware SwiGLU MLP (column-parallel in, row-parallel out).

What this module does
---------------------
Implements the LLaMA SwiGLU feed-forward block with plain ``nn.Linear`` layers so
TP can shard them: ``gate_proj`` and ``up_proj`` are column-parallel (their
output hidden dim is split across TP ranks), and ``down_proj`` is row-parallel
(its input hidden dim is sharded and the output is all_reduced back to the full
residual width).

Why SwiGLU
----------
SwiGLU (``down(silu(gate(x)) * up(x))``) consistently outperforms a plain
GeLU/ReLU MLP at equal parameter count in LM pretraining. It uses three matrices
instead of two; LLaMA compensates by sizing the hidden dim to ``8/3 * d_model``
so the parameter count matches a ``4x`` GeLU MLP.

Why the column/row split is communication-optimal
--------------------------------------------------
Making the two input projections column-parallel and the output projection
row-parallel means the *only* TP collective is a single all_reduce at the end of
``down_proj`` (the row-parallel g operator). The intermediate ``silu(gate)*up``
activation stays sharded across TP ranks with no communication — the elementwise
product is computed independently per shard because gate and up are sharded the
same way. This is the canonical Megatron MLP and matches the attention block's
one-all_reduce-per-sublayer cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """LLaMA-style SwiGLU feed-forward network.

    Args:
        config: Model config; uses ``d_model``, ``ffn_hidden_size`` and
            ``mlp_bias``.

    Shape:
        input  ``(batch, seq, d_model)``
        output ``(batch, seq, d_model)``

    Attributes:
        gate_proj / up_proj: ``d_model -> ffn_hidden_size`` (column-parallel).
        down_proj: ``ffn_hidden_size -> d_model`` (row-parallel).

    Example:
        >>> from src.config import ModelConfig
        >>> mlp = SwiGLUMLP(ModelConfig(d_model=64, ffn_hidden_size=128))
        >>> import torch
        >>> mlp(torch.randn(2, 8, 64)).shape
        torch.Size([2, 8, 64])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.ffn_hidden_size is not None
        hidden = config.ffn_hidden_size
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=config.mlp_bias)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate and up are sharded identically under TP, so their product is a
        # local elementwise op needing no communication.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
