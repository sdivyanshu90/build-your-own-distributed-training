"""Integration: the sharded sampler partitions the dataset exactly.

These are properties of :class:`ShardedSampler` (no live process group needed):
every index is seen exactly once per epoch across DP ranks, no index is seen by
two ranks, and shuffling is per-epoch-deterministic.
"""

from __future__ import annotations

from src.data.dataloader import ShardedSampler


def _all_rank_indices(n: int, dp_size: int, shuffle: bool, seed: int, epoch: int) -> list[list[int]]:
    out = []
    for r in range(dp_size):
        s = ShardedSampler(n, num_replicas=dp_size, rank=r, shuffle=shuffle, seed=seed)
        s.set_epoch(epoch)
        out.append(list(s))
    return out


def test_every_index_seen_exactly_once() -> None:
    # Dataset size deliberately NOT divisible by dp_size (37 across 4 ranks).
    n, dp = 37, 4
    per_rank = _all_rank_indices(n, dp, shuffle=True, seed=0, epoch=0)
    flat = [i for r in per_rank for i in r]
    assert sorted(flat) == list(range(n)), "union of shards must be the whole dataset"
    assert len(flat) == n, "no duplicated or dropped indices"


def test_ranks_are_disjoint() -> None:
    n, dp = 37, 4
    per_rank = _all_rank_indices(n, dp, shuffle=True, seed=0, epoch=0)
    seen: set[int] = set()
    for indices in per_rank:
        s = set(indices)
        assert seen.isdisjoint(s), "an index appears on more than one rank"
        seen |= s


def test_shuffle_differs_across_epochs_same_within_seed() -> None:
    n, dp = 50, 2
    e0 = _all_rank_indices(n, dp, shuffle=True, seed=5, epoch=0)
    e1 = _all_rank_indices(n, dp, shuffle=True, seed=5, epoch=1)
    e0_again = _all_rank_indices(n, dp, shuffle=True, seed=5, epoch=0)
    assert e0 != e1, "different epochs must shuffle differently"
    assert e0 == e0_again, "same seed+epoch must reproduce the same order"


def test_sizes_differ_by_at_most_one() -> None:
    n, dp = 37, 4
    per_rank = _all_rank_indices(n, dp, shuffle=False, seed=0, epoch=0)
    sizes = [len(r) for r in per_rank]
    assert max(sizes) - min(sizes) <= 1, f"shard sizes too uneven: {sizes}"
