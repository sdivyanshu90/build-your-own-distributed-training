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


def test_every_index_seen_at_most_once_and_only_remainder_dropped() -> None:
    n, dp = 37, 4
    per_rank = _all_rank_indices(n, dp, shuffle=True, seed=0, epoch=0)
    flat = [i for r in per_rank for i in r]
    assert len(flat) == len(set(flat)), "no duplicated indices"
    assert len(flat) == n - n % dp, "only the n % dp remainder may be dropped"
    assert set(flat) <= set(range(n))
    # Evenly divisible: the union is the whole dataset.
    even = [i for r in _all_rank_indices(36, dp, shuffle=True, seed=0, epoch=0) for i in r]
    assert sorted(even) == list(range(36))


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


def test_sizes_equal_across_ranks() -> None:
    """Regression: unequal shards => unequal batch counts => collective hang."""
    for n, dp in ((37, 4), (41, 3), (7, 8), (64, 4)):
        for shuffle in (False, True):
            per_rank = _all_rank_indices(n, dp, shuffle=shuffle, seed=0, epoch=0)
            sizes = {len(r) for r in per_rank}
            assert sizes == {n // dp}, f"n={n} dp={dp}: shard sizes {sizes}"
            for r in range(dp):
                s = ShardedSampler(n, num_replicas=dp, rank=r, shuffle=shuffle)
                assert len(s) == len(list(s))
