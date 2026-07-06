from config.storage_layout import SHARDING, shard_index, shard_root


def test_shard_index_deterministic_across_calls():
    assert shard_index("raw123") == shard_index("raw123")


def test_shard_index_in_range():
    for key in ["a", "b", "some_raw_id", "0123456789abcdef"]:
        idx = shard_index(key)
        assert 0 <= idx < SHARDING["n_shards"]


def test_shard_index_reasonably_uniform():
    # Not a strict statistical test -- just a sanity check that a few hundred
    # distinct keys don't all collapse onto one shard (e.g. a broken hash that
    # always returns 0).
    n_shards = SHARDING["n_shards"]
    counts = [0] * n_shards
    for i in range(600):
        counts[shard_index(f"key_{i}")] += 1
    assert all(c > 0 for c in counts)


def test_shard_root_matches_configured_roots():
    for key in ["x", "y", "z"]:
        idx = shard_index(key)
        assert str(shard_root(key)) == SHARDING["shard_roots"][idx]
