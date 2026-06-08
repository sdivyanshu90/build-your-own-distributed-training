"""Datasets: a learnable synthetic corpus and a packed-token memmap corpus.

What this module does
---------------------
Two map-style datasets that both yield ``{"input_ids", "labels"}`` pairs where
``labels`` is ``input_ids`` shifted by one (next-token prediction):

  * :class:`SyntheticTokenDataset` — deterministic, *learnable* sequences drawn
    from a fixed sparse Markov chain. Used by CI/convergence tests: a tiny model
    can actually fit it, so the loss demonstrably drops, which is what makes the
    cross-parallelism convergence test meaningful.
  * :class:`PackedTokenDataset` — the production path: a single contiguous
    memmapped token array (``uint16``) chunked into ``seq_len`` windows, the
    standard LM-pretraining layout that avoids per-document padding.

Why next-token labels live in the dataset, not the loss
-------------------------------------------------------
Producing ``labels = tokens[1:]`` here (rather than shifting inside the model)
keeps the model's ``forward`` agnostic to the task and lets the dataloader pack
windows without a special last-token case. Positions to ignore use ``-100`` to
match ``F.cross_entropy(ignore_index=-100)``.

Sharding note
-------------
These datasets are *rank-agnostic*: the per-DP-rank partition is the
:class:`~src.data.dataloader.ShardedSampler`'s job, keeping the dataset reusable
across any mesh.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTokenDataset(Dataset):
    """Deterministic, learnable next-token sequences from a sparse Markov chain.

    Each token maps to ``k`` possible successors (a fixed random table seeded
    once), so sequences have low conditional entropy (~``ln k``) and a tiny model
    can learn them — the loss drops from ``ln(vocab)`` toward ``ln(k)``. Every
    sample is reproducible from ``(seed, idx)`` so runs are bit-identical.

    Args:
        vocab_size: Token vocabulary size.
        seq_len: Sequence length (the model sees ``seq_len`` inputs).
        num_samples: Number of sequences (the epoch length).
        seed: Base seed for the transition table and per-sample generation.
        k: Number of successors per token (lower => easier to learn).

    Example:
        >>> ds = SyntheticTokenDataset(vocab_size=64, seq_len=16, num_samples=8)
        >>> item = ds[0]
        >>> item["input_ids"].shape, item["labels"].shape
        (torch.Size([16]), torch.Size([16]))
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        seed: int = 0,
        k: int = 4,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed
        self.k = min(k, vocab_size)
        gen = torch.Generator().manual_seed(seed)
        # Fixed successor table: row t lists the k tokens that may follow token t.
        self.next_tokens = torch.randint(
            0, vocab_size, (vocab_size, self.k), generator=gen
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        seq = torch.empty(self.seq_len + 1, dtype=torch.long)
        seq[0] = torch.randint(0, self.vocab_size, (1,), generator=gen)
        for t in range(1, self.seq_len + 1):
            choice = int(torch.randint(0, self.k, (1,), generator=gen).item())
            seq[t] = self.next_tokens[seq[t - 1], choice]
        return {"input_ids": seq[:-1].contiguous(), "labels": seq[1:].contiguous()}


class PackedTokenDataset(Dataset):
    """Contiguous packed-token corpus chunked into fixed-length windows.

    The corpus is a flat ``uint16`` token array on disk (memmapped, so it is not
    loaded into RAM). Window ``i`` is ``tokens[i*seq_len : (i+1)*seq_len + 1]``;
    the trailing ``+1`` provides the shifted label. The final partial window is
    dropped (standard; < ``seq_len`` tokens, negligible).

    Args:
        path: Path to the ``.bin`` token file (``uint16``).
        seq_len: Window length.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the corpus has fewer than ``seq_len + 1`` tokens.
    """

    def __init__(self, path: str, seq_len: int) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Token corpus not found: {path}")
        self.path = path
        self.seq_len = seq_len
        self._data: np.memmap | None = None
        n_tokens = os.path.getsize(path) // np.dtype(np.uint16).itemsize
        if n_tokens < seq_len + 1:
            raise ValueError(
                f"Corpus has {n_tokens} tokens < seq_len+1={seq_len + 1}."
            )
        # Number of non-overlapping windows leaving room for the shifted label.
        self.num_windows = (n_tokens - 1) // seq_len

    def _memmap(self) -> np.memmap:
        # Lazily open per worker process (memmaps must not cross fork).
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self._memmap()
        start = idx * self.seq_len
        chunk = np.asarray(data[start : start + self.seq_len + 1], dtype=np.int64)
        seq = torch.from_numpy(chunk)
        return {"input_ids": seq[:-1].contiguous(), "labels": seq[1:].contiguous()}
