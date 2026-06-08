"""LLaMA-style causal transformer language model.

What this module does
---------------------
Assembles the full model: token embedding -> N x ``TransformerBlock`` (RMSNorm ->
attention -> residual -> RMSNorm -> SwiGLU MLP -> residual) -> final RMSNorm ->
LM head. ``TransformerBlock`` is the unit FSDP wraps and TP parallelises, so its
submodule names (``attention``, ``mlp``, ``attention_norm``, ``mlp_norm``) are part
of the public contract consumed by
:func:`src.parallelism.tensor_parallel.apply_tensor_parallelism`.

Architecture choices and why
----------------------------
  * **Pre-norm** (norm before each sublayer, residual around it): stabilises deep
    transformer training; post-norm needs careful warmup and diverges at depth.
  * **RMSNorm** over LayerNorm: no mean-subtraction or bias, ~cheaper, and
    empirically as good — the LLaMA convention.
  * **RoPE** for position (see :mod:`src.model.embeddings`).
  * **Tied embeddings** (LM head reuses the embedding matrix): saves
    ``vocab*d_model`` params and tends to help small models. Both live in the
    *root* FSDP unit (not per-block), so weight-tying survives sharding.
  * **Scaled residual init**: output projections (``wo``, ``down_proj``) are
    initialised with std ``0.02/sqrt(2*n_layers)`` (GPT-2 trick) so the residual
    stream variance does not grow with depth.

Memory footprint (per GPU, with FSDP FULL_SHARD across ``dp_size``)
-------------------------------------------------------------------
  * Resident params: ``num_parameters / dp_size`` in ``param_dtype`` + an fp32
    master copy in the optimizer shard.
  * Transient (one block all-gathered at a time): ``P_block`` in ``param_dtype``.
  * Activations: dominated by the per-block residual + attention scores; bounded
    by activation checkpointing when enabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.model.attention import Attention
from src.model.embeddings import TokenEmbedding, precompute_rope_cache
from src.model.mlp import SwiGLUMLP


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias).

    Args:
        dim: Normalised feature dimension.
        eps: Numerical-stability epsilon.

    Shape:
        input/output ``(*, dim)``.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 for stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


class TransformerBlock(nn.Module):
    """A single pre-norm transformer block (the FSDP/TP wrapping unit).

    Args:
        config: The model config.
        layer_id: Index of this block (for scaled-residual init).

    Shape:
        input/output ``(batch, seq, d_model)``.

    Attributes:
        attention_norm / attention: First sublayer (norm + attention).
        mlp_norm / mlp: Second sublayer (norm + SwiGLU MLP).
    """

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class Transformer(nn.Module):
    """The full causal LM.

    Args:
        config: The :class:`~src.config.ModelConfig`.

    Attributes:
        tok_embeddings: Replicated input embedding.
        layers: ``nn.ModuleList`` of ``TransformerBlock`` (the wrapping units).
        norm: Final RMSNorm.
        lm_head: Output projection to vocab logits (weight-tied to the embedding
            when ``config.tie_embeddings``).

    Example:
        >>> from src.config import ModelConfig
        >>> m = Transformer(ModelConfig(vocab_size=128, d_model=64, n_layers=2,
        ...                             n_heads=4, max_seq_len=32))
        >>> import torch
        >>> tokens = torch.randint(0, 128, (2, 16))
        >>> logits, loss = m(tokens, labels=tokens)
        >>> logits.shape, bool(loss > 0)
        (torch.Size([2, 16, 128]), True)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = TokenEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(config, i) for i in range(config.n_layers)
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Share storage; both modules sit in the root FSDP unit so tying
            # survives sharding.
            self.lm_head.weight = self.tok_embeddings.weight

        # RoPE cache as non-persistent buffers (recomputed on load, not saved).
        cos, sin = precompute_rope_cache(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._scale_residual_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_init(self) -> None:
        """GPT-2 scaled init for residual output projections (depth stability)."""
        std = 0.02 / (2 * self.config.n_layers) ** 0.5
        for block in self.layers:
            nn.init.normal_(block.attention.wo.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=std)

    def forward(
        self, tokens: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model and optionally compute the cross-entropy loss.

        Args:
            tokens: Input token ids ``(batch, seq)``.
            labels: Target token ids ``(batch, seq)``; positions equal to
                ``-100`` are ignored. If ``None``, only logits are returned.

        Returns:
            ``(logits, loss)`` where ``logits`` is ``(batch, seq, vocab)`` and
            ``loss`` is a scalar tensor (or ``None`` if no labels).
        """
        _, seqlen = tokens.shape
        cos = self.rope_cos[:seqlen]
        sin = self.rope_sin[:seqlen]
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def build_model(config: ModelConfig) -> Transformer:
    """Factory that constructs a :class:`Transformer` from a model config."""
    return Transformer(config)
