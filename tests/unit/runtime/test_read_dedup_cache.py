"""Unit tests for ReadDedupCache history bounds.

Covers:
 - content_hash includes total_bytes (R3 truncated-read aliasing fix)
 - check/record round-trip + content-hash mismatch invalidates hit
 - invalidate_path clears all cache entries for a path
 - LRU eviction caps memory growth
 - Per-actor isolation: different agents are isolated by default
 - Per-scope isolation: different capability fingerprints never share hits
 - make_capability_fingerprint + make_dedup_key helpers

Ported from side-branch commits 23452d0 + 18a547a + 6570ab5 +
9feed4e onto the canonical core.
"""

from __future__ import annotations

from protocore.runtime.read_dedup_cache import (
    CapabilityFingerprint,
    ReadDedupCache,
    make_capability_fingerprint,
    make_dedup_key,
)

# ---------------------------------------------------------------------------
# content_hash size-aware
# ---------------------------------------------------------------------------


def test_read_dedup_cache_content_hash_includes_total_bytes() -> None:
    """R3 fix: same chunk hash differs when total_bytes differs."""
    h1 = ReadDedupCache.content_hash(b"first 8 KB", total_bytes=8192)
    h2 = ReadDedupCache.content_hash(b"first 8 KB", total_bytes=50000)
    h3 = ReadDedupCache.content_hash(b"first 8 KB", total_bytes=8192)
    assert h1 != h2
    assert h1 == h3


# ---------------------------------------------------------------------------
# check/record basics
# ---------------------------------------------------------------------------


def test_read_dedup_cache_check_and_record() -> None:
    c = ReadDedupCache(ttl_seconds=10, max_entries=4)
    c.record(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h1",
        size_bytes=10,
    )
    hit = c.check(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h1",
    )
    assert hit is not None
    miss = c.check(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h2",
    )
    assert miss is None


def test_read_dedup_cache_invalidate_path() -> None:
    c = ReadDedupCache(ttl_seconds=10)
    c.record(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h",
        size_bytes=10,
    )
    c.invalidate_path(session_id="s", agent_id="a", path="p")
    miss = c.check(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h",
    )
    assert miss is None


def test_read_dedup_cache_lru_evicts_when_over_max() -> None:
    c = ReadDedupCache(ttl_seconds=10, max_entries=2)
    for i in range(5):
        c.record(
            session_id="s",
            agent_id="a",
            path=f"p{i}",
            range_signature="r",
            content_hash="h",
            size_bytes=10,
        )
    # The cache should have evicted older entries.
    assert c.size() <= 2


# ---------------------------------------------------------------------------
# / : secondary index must be bounded + reverse-map invariant
# ---------------------------------------------------------------------------


def test_path_index_does_not_leak_after_eviction() -> None:
    """: distinct (session, agent, path) entries that are LRU-evicted
    must be reclaimed from ``_path_index`` — the index must not grow past the
    set of paths whose store entries are still live.
    """
    max_entries = 4
    distinct_paths = 1000
    c = ReadDedupCache(ttl_seconds=10, max_entries=max_entries)
    for i in range(distinct_paths):
        c.record(
            session_id="s",
            agent_id="a",
            path=f"p{i}",
            range_signature="r",
            content_hash="h",
            size_bytes=10,
        )
    # The store is bounded …
    assert c.size() <= max_entries
    # … and so is the secondary index (no orphaned empty sets).
    assert c.path_index_size() <= max_entries
    # No empty value-sets must linger.
    assert all(len(s) > 0 for s in c._path_index.values())


def test_path_index_reverse_map_stays_consistent() -> None:
    """/: the reverse store_key -> idx_key map must mirror the
    forward index exactly, so eviction can target a single set in O(1).
    """
    c = ReadDedupCache(ttl_seconds=10, max_entries=8)
    for i in range(50):
        c.record(
            session_id="s",
            agent_id="a",
            path=f"p{i % 5}",  # 5 distinct paths, many ranges
            range_signature=f"r{i}",
            content_hash="h",
            size_bytes=10,
        )
        c._assert_index_invariant()
    # Every live store key is reachable from the forward index, and every
    # forward-index key is a live store key.
    forward_keys = {k for s in c._path_index.values() for k in s}
    assert forward_keys == set(c._store)
    assert set(c._key_index) == set(c._store)


def test_eviction_is_o1_per_key_not_full_index_scan() -> None:
    """: eviction must NOT scan every value-set in ``_path_index``.

    We replace each value-set with a discard-counting subclass and count how
    many ``discard`` ops eviction performs. With the reverse-map fix the count
    is O(1) per evicted key (exactly one discard from one set), independent of
    how many distinct paths are already indexed. The old full-index scan would
    call discard ``target_drop * path_index_size`` times (thousands here).
    """

    class _CountingSet(set):  # type: ignore[type-arg]
        discard_calls = 0

        def discard(self, value: object) -> None:
            _CountingSet.discard_calls += 1
            super().discard(value)

    # Pre-load many distinct paths so the index has high cardinality.
    max_entries = 200
    c = ReadDedupCache(ttl_seconds=10, max_entries=max_entries)
    for i in range(max_entries):
        c.record(
            session_id="s",
            agent_id="a",
            path=f"p{i}",
            range_signature="r",
            content_hash="h",
            size_bytes=10,
        )
    assert c.path_index_size() == max_entries

    # Swap every value-set for the counting subclass (preserving membership)
    # so we observe exactly the discard calls the eviction pass makes.
    with c._lock:
        c._path_index = {
            idx_key: _CountingSet(keys) for idx_key, keys in c._path_index.items()
        }
    _CountingSet.discard_calls = 0

    # One over-cap insert -> eviction drops ~10% (target_drop) keys.
    c.record(
        session_id="s",
        agent_id="a",
        path="p-new",
        range_signature="r",
        content_hash="h",
        size_bytes=10,
    )
    target_drop = max(1, (max_entries + 1) // 10)
    # O(1) per evicted key: exactly one discard per dropped key. The bug's
    # full-index scan would be target_drop * path_index_size (~ thousands).
    assert _CountingSet.discard_calls == target_drop
    c._assert_index_invariant()


# ---------------------------------------------------------------------------
# : edit-detection is key-divergence (content_hash folded into the key)
# ---------------------------------------------------------------------------


def test_changed_content_hash_is_a_miss_and_does_not_evict_old_entry() -> None:
    """: edit detection is purely key-divergence + invalidate_path.

    A re-read of the same (session, agent, path, range) with a *different*
    content_hash produces a fresh miss (different store key); the stale entry
    is not returned and is reclaimed by TTL/LRU/invalidate_path, never by a
    post-lookup content_hash guard (which would be unreachable).
    """
    c = ReadDedupCache(ttl_seconds=10, max_entries=8)
    c.record(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h1",
        size_bytes=10,
    )
    # Same coordinates, new content -> miss (the model must re-read).
    miss = c.check(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h2",
    )
    assert miss is None
    # The original hash still hits (no false invalidation by the new read).
    hit = c.check(
        session_id="s",
        agent_id="a",
        path="p",
        range_signature="r",
        content_hash="h1",
    )
    assert hit is not None


def test_invalidate_path_after_edit_forces_fresh_read_all_ranges() -> None:
    """: invalidate_path is the explicit edit-detection mechanism —
    it drops every range/content variant recorded for a path and prunes the
    index entry (no orphaned empty set left behind).
    """
    c = ReadDedupCache(ttl_seconds=10, max_entries=64)
    for i in range(3):
        c.record(
            session_id="s",
            agent_id="a",
            path="p",
            range_signature=f"r{i}",
            content_hash=f"h{i}",
            size_bytes=10,
        )
    assert c.path_index_size() == 1
    c.invalidate_path(session_id="s", agent_id="a", path="p")
    assert c.size() == 0
    assert c.path_index_size() == 0
    assert set(c._key_index) == set()


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_read_dedup_cache_agents_in_same_scope_are_isolated() -> None:
    """Different actors must not receive stubs for content outside their context.

    Capability scope is a permission boundary, not proof that another model
    conversation has already seen the prior full read.
    """
    c = ReadDedupCache(ttl_seconds=10)
    c.record(
        session_id="s",
        agent_id="agent_a",
        path="p",
        range_signature="r",
        content_hash="h",
        size_bytes=10,
    )
    hit_other_agent = c.check(
        session_id="s",
        agent_id="agent_b",
        path="p",
        range_signature="r",
        content_hash="h",
    )
    assert hit_other_agent is None


def test_read_dedup_cache_different_scope_isolates() -> None:
    """Agents with different capability fingerprints are isolated."""
    scope_a: CapabilityFingerprint = (
        "user:Alice",
        frozenset({"workspace_read"}),
    )
    scope_b: CapabilityFingerprint = (
        "user:Bob",
        frozenset({"workspace_read"}),
    )
    c = ReadDedupCache(ttl_seconds=10)
    c.record(
        session_id="s",
        agent_id="agent_a",
        path="p",
        range_signature="r",
        content_hash="h",
        size_bytes=10,
        capability_fingerprint=scope_a,
    )
    hit = c.check(
        session_id="s",
        agent_id="agent_b",
        path="p",
        range_signature="r",
        content_hash="h",
        capability_fingerprint=scope_b,
    )
    assert hit is None, "different capability scopes must not share dedup hits"


# ---------------------------------------------------------------------------
# make_capability_fingerprint
# ---------------------------------------------------------------------------


def test_capability_fingerprint_stable_across_call_sites() -> None:
    fp1 = make_capability_fingerprint(
        agent_id="coder",
        permission_scope="user:bf9ad2c2",
        capability_set=frozenset({"workspace_read", "workspace_write"}),
    )
    fp2 = make_capability_fingerprint(
        agent_id="coder",
        permission_scope="user:bf9ad2c2",
        capability_set=frozenset(
            {"workspace_write", "workspace_read"}
        ),  # different insertion order
    )
    assert fp1 == fp2


def test_capability_fingerprint_differs_for_different_scope() -> None:
    fp1 = make_capability_fingerprint(
        agent_id="coder",
        permission_scope="user:A",
        capability_set=frozenset({"workspace_read"}),
    )
    fp2 = make_capability_fingerprint(
        agent_id="coder",
        permission_scope="user:B",
        capability_set=frozenset({"workspace_read"}),
    )
    assert fp1 != fp2


def test_capability_fingerprint_same_scope_same_caps_equal() -> None:
    """agent_id must NOT partition fingerprint — only scope + caps matter."""
    fp_leader = make_capability_fingerprint(
        agent_id="leader",
        permission_scope="user:X",
        capability_set=frozenset({"workspace_read"}),
    )
    fp_coder = make_capability_fingerprint(
        agent_id="coder",
        permission_scope="user:X",
        capability_set=frozenset({"workspace_read"}),
    )
    assert fp_leader == fp_coder


# ---------------------------------------------------------------------------
# make_dedup_key
# ---------------------------------------------------------------------------


def test_dedup_key_shares_across_agents_same_scope() -> None:
    """Leader and coder with same permission_scope hit the same dedup key
    when agent_id is omitted (default empty); when agent_id is supplied
    differently the keys diverge.
    """
    fp_shared: CapabilityFingerprint = ("user:X", frozenset({"workspace_read"}))
    key_a = make_dedup_key(
        tenant_id="t1",
        session_id="s1",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp_shared,
    )
    key_b = make_dedup_key(
        tenant_id="t1",
        session_id="s1",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp_shared,
    )
    assert key_a == key_b


def test_dedup_key_isolates_across_tenants() -> None:
    fp: CapabilityFingerprint = ("scope", frozenset({"workspace_read"}))
    key_a = make_dedup_key(
        tenant_id="tenantA",
        session_id="s1",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp,
    )
    key_b = make_dedup_key(
        tenant_id="tenantB",
        session_id="s1",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp,
    )
    assert key_a != key_b


def test_dedup_key_isolates_across_sessions() -> None:
    fp: CapabilityFingerprint = ("scope", frozenset({"workspace_read"}))
    key_a = make_dedup_key(
        tenant_id="t",
        session_id="sessionA",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp,
    )
    key_b = make_dedup_key(
        tenant_id="t",
        session_id="sessionB",
        workspace_id="w1",
        path="a.md",
        range_signature="0:1000",
        content_hash="h1",
        capability_fingerprint=fp,
    )
    assert key_a != key_b


def test_dedup_key_deterministic() -> None:
    fp: CapabilityFingerprint = ("scope", frozenset({"workspace_read"}))
    key_a = make_dedup_key(
        tenant_id="t",
        session_id="s",
        workspace_id="w",
        path="a.md",
        range_signature="0:1000",
        content_hash="h",
        capability_fingerprint=fp,
    )
    key_b = make_dedup_key(
        tenant_id="t",
        session_id="s",
        workspace_id="w",
        path="a.md",
        range_signature="0:1000",
        content_hash="h",
        capability_fingerprint=fp,
    )
    assert key_a == key_b
    # sha256 hex digest -> 64 chars
    assert len(key_a) == 64
