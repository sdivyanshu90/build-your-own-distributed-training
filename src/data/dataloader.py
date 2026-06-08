"""Per-DP-rank sharded sampler and dataloader construction.

What this module does
---------------------
Provides :class:`ShardedSampler` (an exact, no-duplication partition of the
dataset across the *data-parallel* ranks) and :func:`build_dataloader`.

Why a custom sampler instead of ``DistributedSampler``
------------------------------------------------------
PyTorch's ``DistributedSampler`` *pads* the index list so it divides evenly across
ranks — i.e. it **duplicates** a few samples every epoch. For LM pretraining that
silently lets some tokens be trained on twice per epoch, biasing the gradient.
:class:`ShardedSampler` instead uses a strided partition
(``indices[rank::num_replicas]``): the union over ranks is exactly the dataset,
the per-rank subsets are disjoint, and sizes differ by at most one — no padding,
no duplication.

Sharding is over the DP axis only
---------------------------------
The sampler is constructed with ``num_replicas = dp_size`` and
``rank = dp_rank``. TP ranks within a group share the same ``dp_rank`` and thus
read the **identical** data — required so the TP all_reduce combines matching
activations. Passing ``world_rank``/``world_size`` here would be the wrong-group
bug at the data layer.

Step-alignment vs. no-drop
--------------------------
Training uses ``drop_last=True`` at the *batch* level so every DP rank performs an
identical number of optimizer steps (FSDP collectives require lock-step). The
handful of leftover samples are not duplicated — they are simply seen in a later
epoch under a different shuffle. The exact-partition property the tests check is a
*sampler-level* guarantee, independent of batch drop_last.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ShardedSampler(Sampler[int]):
    """Exact, non-duplicating index partition across DP ranks.

    Args:
        dataset_len: Number of samples in the dataset.
        num_replicas: Number of DP ranks (the partition count).
        rank: This rank's DP index in ``[0, num_replicas)``.
        shuffle: Whether to permute indices each epoch.
        seed: Base seed; the per-epoch permutation uses ``seed + epoch``.

    Raises:
        ValueError: If ``rank`` is out of range for ``num_replicas``.

    Example:
        >>> # union of all ranks == full set, pairwise disjoint
        >>> s0 = list(ShardedSampler(10, num_replicas=2, rank=0, shuffle=False))
        >>> s1 = list(ShardedSampler(10, num_replicas=2, rank=1, shuffle=False))
        >>> sorted(s0 + s1) == list(range(10)) and set(s0).isdisjoint(s1)
        True
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank={rank} out of range for num_replicas={num_replicas}."
            )
        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs across epochs (same seed+epoch =>
        same order). Call once per epoch before iterating."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=gen).tolist()
        else:
            indices = list(range(self.dataset_len))
        # Strided partition: disjoint, union == full, sizes differ by <= 1.
        return iter(indices[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        return len(range(self.rank, self.dataset_len, self.num_replicas))


def build_dataloader(
    dataset: Dataset,
    micro_batch_size: int,
    dp_size: int,
    dp_rank: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DP-sharded ``DataLoader``.

    Args:
        dataset: A map-style dataset yielding ``{"input_ids", "labels"}``.
        micro_batch_size: Per-rank per-step batch size.
        dp_size: Data-parallel degree (sampler ``num_replicas``).
        dp_rank: This rank's DP index (sampler ``rank``).
        shuffle: Shuffle each epoch.
        seed: Sampler seed.
        num_workers: Dataloader worker processes.
        drop_last: Drop the final partial batch so all DP ranks take the same
            number of steps (required for FSDP lock-step). Defaults to True.
        pin_memory: Pin host memory for faster H2D copies (CUDA only).

    Returns:
        A configured ``DataLoader``. Its ``.sampler`` is a :class:`ShardedSampler`
        whose ``set_epoch`` the trainer calls each epoch.

    Performance note:
        ``pin_memory=True`` plus ``.to(device, non_blocking=True)`` in the step
        overlaps the host->device copy with compute. ``num_workers>0`` prefetches
        batches so ``data_wait_s`` stays near zero.
    """
    sampler = ShardedSampler(
        dataset_len=len(dataset),  # type: ignore[arg-type]
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
