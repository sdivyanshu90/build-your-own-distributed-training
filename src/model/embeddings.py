"""Token embeddings and rotary position embeddings (RoPE).

What this module does
---------------------
Holds the input token embedding and the rotary position embedding machinery.
RoPE is precomputed once (a ``(max_seq_len, head_dim)`` cos/sin cache) and applied
to queries and keys inside attention.

Why RoPE (over learned absolute / ALiBi)
----------------------------------------
RoPE encodes *relative* position by rotating each (q, k) pair in 2D subspaces,
which (a) requires no learned parameters, (b) extrapolates better to longer
contexts than learned absolute embeddings, and (c) injects position into the
attention dot-product directly, so no position vector is added to the residual
stream. It is the LLaMA/most-modern-LM convention.

Why the embedding is NOT tensor-parallel by default
----------------------------------------------------
A vocab-parallel embedding shards the ``(vocab, d_model)`` matrix across TP ranks
and needs an all_reduce to combine partial lookups, plus a matching
vocab-parallel cross-entropy at the head. For the model sizes here, the embedding
is cheap relative to the transformer blocks, and FSDP already shards it across
the DP axis. So we keep it replicated across TP and reserve vocab parallelism for
the optional ``loss_parallel`` path. (A :class:`VocabParallelEmbedding` is
provided for users who opt in.)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Replicated token embedding table.

    Args:
        vocab_size: Number of tokens.
        d_model: Embedding width.

    Shape:
        input  ``(batch, seq)`` int64 token ids
        output ``(batch, seq, d_model)``
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(tokens, self.weight)


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the RoPE cosine/sine cache.

    Args:
        head_dim: Per-head dimension (must be even).
        max_seq_len: Longest position to cache.
        theta: Base period (LLaMA default 10000).
        device: Where to allocate the cache.

    Returns:
        ``(cos, sin)`` each of shape ``(max_seq_len, head_dim)`` in float32. The
        first ``head_dim/2`` columns are duplicated into the second half so they
        broadcast against the "rotate-half" layout used by
        :func:`apply_rotary_emb`.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: ``[x1, x2] -> [-x2, x1]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to a ``(batch, seq, heads, head_dim)`` tensor.

    Args:
        x: Query or key tensor, shape ``(b, s, h, d)``.
        cos: Cosine cache slice, shape ``(s, d)``.
        sin: Sine cache slice, shape ``(s, d)``.

    Returns:
        The rotated tensor, same shape and dtype as ``x``. The cache is upcast to
        ``x``'s dtype and broadcast over batch and head dims.

    Note:
        This is TP-agnostic: the head count ``h`` may be the sharded local count;
        RoPE acts per-head on the ``head_dim`` axis only.
    """
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    return (x * cos) + (_rotate_half(x) * sin)


class VocabParallelEmbedding(nn.Module):
    """Optional vocabulary-parallel embedding (TP-sharded on the vocab axis).

    Each TP rank owns a contiguous slice of the vocabulary and zeros out tokens
    outside its slice; an all_reduce sums the partial lookups. Provided for the
    ``loss_parallel`` configuration; not used by the default model.

    Args:
        vocab_size: Full vocabulary size (must be divisible by ``tp_size``).
        d_model: Embedding width.
        tp_group: Tensor-parallel process group.

    Raises:
        ValueError: If ``vocab_size`` is not divisible by the TP degree.
    """

    def __init__(
        self, vocab_size: int, d_model: int, tp_group: dist.ProcessGroup | None
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if vocab_size % self.tp_size != 0:
            raise ValueError(
                f"VocabParallelEmbedding: vocab_size={vocab_size} not divisible "
                f"by tp_size={self.tp_size}."
            )
        self.vocab_per_partition = vocab_size // self.tp_size
        self.vocab_start = self.tp_rank * self.vocab_per_partition
        self.vocab_end = self.vocab_start + self.vocab_per_partition
        self.weight = nn.Parameter(torch.empty(self.vocab_per_partition, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens < self.vocab_start) | (tokens >= self.vocab_end)
        local = tokens - self.vocab_start
        local = local.masked_fill(mask, 0)
        out = nn.functional.embedding(local, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out
