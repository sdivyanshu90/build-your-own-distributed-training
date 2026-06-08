"""TP-aware grouped-query attention with RoPE and fused SDPA.

What this module does
---------------------
Implements multi-head / grouped-query attention using plain ``nn.Linear``
projections (``wq``, ``wk``, ``wv``, ``wo``) so that
:func:`src.parallelism.tensor_parallel.apply_tensor_parallelism` can shard them
column/row-wise via DTensor plans. The attention math itself is written to be
*tensor-parallel-agnostic*.

The key TP trick: reshape with ``-1`` for the head count
--------------------------------------------------------
After ``ColwiseParallel`` shards ``wq``/``wk``/``wv``, each TP rank's projection
outputs only ``n_heads / tp_size`` heads' worth of features (the library returns
the *local* tensor, not a DTensor). If the reshape hard-coded ``self.n_heads`` it
would be wrong on every rank. Instead we reshape with ``-1`` for the head axis::

    xq = xq.view(bsz, seqlen, -1, self.head_dim)

so the local head count is inferred from the (already-sharded) tensor width. The
same applies to the KV heads. ``wo`` is ``RowwiseParallel``: it consumes the
sharded attention output and all_reduces back to the full residual stream. The
residual stream therefore stays replicated across the TP group; only the
per-head activations are sharded.

Grouped-query attention (GQA)
-----------------------------
With ``n_kv_heads < n_heads`` each KV head is shared by ``n_heads / n_kv_heads``
query heads, shrinking the KV cache. Under TP both counts are divided by
``tp_size``, so the repeat factor is preserved — we compute it from the runtime
shapes rather than the config to stay TP-correct.

Communication
-------------
This module itself issues no collectives. Under TP, the ``all_reduce`` lives in
``wo`` (RowParallel). Under no TP it is a plain attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.embeddings import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    Args:
        x: KV tensor of shape ``(bsz, seqlen, n_kv_heads, head_dim)``.
        n_rep: Repeat factor ``n_heads // n_kv_heads``.

    Returns:
        Tensor of shape ``(bsz, seqlen, n_kv_heads * n_rep, head_dim)`` where each
        KV head is repeated ``n_rep`` times contiguously.
    """
    if n_rep == 1:
        return x
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Causal multi-head / grouped-query attention.

    Args:
        config: The :class:`~src.config.ModelConfig`. Uses ``d_model``,
            ``n_heads``, ``n_kv_heads``, ``head_dim``, ``attention_bias`` and
            ``dropout``.

    Shape:
        input  ``(batch, seq, d_model)`` (replicated residual stream)
        output ``(batch, seq, d_model)``

    Attributes:
        wq/wk/wv: Q/K/V projections (column-parallel under TP).
        wo: Output projection (row-parallel under TP).

    Example:
        >>> from src.config import ModelConfig
        >>> attn = Attention(ModelConfig(d_model=64, n_heads=4, n_kv_heads=2))
        >>> import torch
        >>> cos, sin = torch.ones(8, 16), torch.zeros(8, 16)
        >>> y = attn(torch.randn(2, 8, 64), cos, sin)
        >>> y.shape
        torch.Size([2, 8, 64])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_kv_heads is not None
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        kv_dim = self.n_kv_heads * self.head_dim
        self.wq = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.wk = nn.Linear(config.d_model, kv_dim, bias=config.attention_bias)
        self.wv = nn.Linear(config.d_model, kv_dim, bias=config.attention_bias)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Run causal attention.

        Args:
            x: Input ``(bsz, seqlen, d_model)``.
            cos / sin: RoPE cache slices ``(seqlen, head_dim)``.

        Returns:
            ``(bsz, seqlen, d_model)`` attention output (replicated under TP via
            ``wo``'s row-parallel all_reduce).
        """
        bsz, seqlen, _ = x.shape

        # Reshape with -1 for the head axis so the LOCAL (TP-sharded) head count
        # is inferred from the tensor width rather than hard-coded.
        xq = self.wq(x).view(bsz, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, -1, self.head_dim)

        xq = apply_rotary_emb(xq, cos, sin)
        xk = apply_rotary_emb(xk, cos, sin)

        # GQA repeat factor from runtime shapes (TP-correct).
        n_local_q_heads = xq.shape[2]
        n_local_kv_heads = xk.shape[2]
        n_rep = n_local_q_heads // n_local_kv_heads
        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

        # (bsz, heads, seqlen, head_dim) for SDPA.
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Fused scaled-dot-product attention; is_causal applies the mask without
        # materialising an (seq x seq) tensor.
        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)
