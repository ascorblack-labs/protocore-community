# ruff: noqa: RUF002 — Module docstring uses unicode multiplication + en-dash for clarity.
"""Per-agent workspace read deduplication.

The same coder loop can repeatedly re-read the
same 14 KB README and each read added ~3 200 tokens to the conversation
history. After 5-9 re-reads the prompt exceeded the context window.

The cache fingerprints a multi-dimensional key (tenant_id, session_id,
workspace_id, path, range_signature, content_hash, capability_fingerprint) →
``_ReadCacheEntry``. On a repeat read with an identical content hash the
workspace tool returns ``unchanged=true`` with empty content. The agent's
system prompt instructs it to re-use the prior read's content. This
linearises history growth: re-reads add ~constant tokens instead of ~constant
× payload size.

The capability fingerprint captures (permission_scope, capability_set), while
``agent_id`` remains a separate cache-key dimension. Permissions alone do not
prove that another model conversation has seen the prior full read; without the
actor dimension a fresh subagent can receive an empty unchanged-read stub for
content that only a different actor saw.

Lifetime: process-local TTL cache. The cache is per-instance of the workspace
executor (one per pod). Keys include ``session_id`` so different runs on the
same pod do not collide. TTL eviction handles stale entries from completed
runs; LRU eviction bounds memory. The secondary ``(session, agent, path)``
index is kept bounded by the live store via a reverse ``store_key → idx_key``
map: eviction reclaims an evicted key from exactly one value-set in O(1) and
prunes the idx_key once its set empties, so neither the index nor the
per-eviction cost grows with pod uptime (/ ).

Edit detection: ``content_hash`` is folded into the store key (see
``make_dedup_key``), so an in-place edit produces a *different* key — a fresh
miss the next read records — and a callable cannot return a stale hit (a hit
requires presenting the matching hash). Detection is therefore purely
key-divergence plus an explicit ``invalidate_path`` on a write; there is no
post-lookup ``content_hash`` discriminator because, given the hash is already
in the key, such a guard would be structurally unreachable .

For horizontal scaling: this is acceptable in-memory state because each run is
sticky to a single orchestrator pod. Cross-pod dedup would require Redis but
is unnecessary for the failure class targeted here (re-reads inside one
subagent loop).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Final

# ---------------------------------------------------------------------------
# Multi-dimensional key helpers (Plan 4 / Task 1)
# ---------------------------------------------------------------------------

#: Type alias for a capability fingerprint tuple: (permission_scope, capability_set).
CapabilityFingerprint = tuple[str, frozenset[str]]


def make_capability_fingerprint(
    *,
    agent_id: str,
    permission_scope: str,
    capability_set: frozenset[str],
) -> CapabilityFingerprint:
    """Return a normalised capability identity for dedup-key construction.

    Captures ``permission_scope`` (e.g. the user/session boundary) and the
    sorted set of capability strings. ``agent_id`` is accepted as a keyword
    argument for call-site clarity but is intentionally excluded from the
    returned tuple because it is handled separately by the read-dedup key.
    """
    return (permission_scope, capability_set)


def make_dedup_key(
    *,
    tenant_id: str,
    session_id: str,
    agent_id: str = "",
    workspace_id: str,
    path: str,
    range_signature: str,
    content_hash: str,
    capability_fingerprint: CapabilityFingerprint,
) -> str:
    """Compose the multi-dimensional dedup cache key as a single sha-256 hex digest.

    Dimensions:
    - ``tenant_id`` — top-level isolation boundary (prevents cross-tenant aliasing)
    - ``session_id`` — run isolation (different sessions on the same pod don't collide)
    - ``agent_id`` — conversation visibility boundary; a different actor has not
      necessarily seen the prior full read even when permissions match
    - ``workspace_id`` — workspace namespace (multiple workspaces per session)
    - ``path`` — the file path within the workspace
    - ``range_signature`` — read window (offset+limit or start_line+end_line)
    - ``content_hash`` — file content fingerprint (detects in-place edits)
    - ``capability_fingerprint`` — (permission_scope, sorted capability_set)
    """
    scope, caps = capability_fingerprint
    raw = "|".join([
        tenant_id,
        session_id,
        agent_id,
        workspace_id,
        path,
        range_signature,
        content_hash,
        scope,
        "+".join(sorted(caps)),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _ReadCacheEntry:
    size_bytes: int
    recorded_at: float


_DEFAULT_TTL_SECONDS: Final[int] = 300
_DEFAULT_MAX_ENTRIES: Final[int] = 256

# Sentinel fingerprint used when callers don't supply capability context.
# Produces a consistent hash segment so old call sites don't gain
# cross-capability aliasing by sharing the same empty-string segment.
_NO_CAPABILITY: CapabilityFingerprint = ("", frozenset())


def _build_cache_key(
    *,
    session_id: str,
    agent_id: str,
    path: str,
    range_signature: str,
    content_hash: str,
    tenant_id: str = "",
    workspace_id: str = "",
    capability_fingerprint: CapabilityFingerprint = _NO_CAPABILITY,
) -> str:
    """Derive the cache store key as a sha-256 hex string.

    Callers that supply all seven dimensions get full multi-dim isolation.
    Legacy callers that omit the new optional params get isolation by
    (session_id, agent_id, path, range_signature, content_hash), which is
    equivalent to the old _ReadCacheKey dataclass behaviour.
    """
    return make_dedup_key(
        tenant_id=tenant_id,
        session_id=session_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        path=path,
        range_signature=range_signature,
        content_hash=content_hash,
        capability_fingerprint=capability_fingerprint,
    )


class ReadDedupCache:
    """Bounded per-agent read content-hash cache.

    Thread-safe: workspace tools are typically dispatched concurrently across
    requests within a pod. A single ``threading.Lock`` is sufficient — read
    operations finish in microseconds and contention is low.

    The internal store key is a sha-256 hex digest produced by
    ``make_dedup_key``. Callers that provide ``tenant_id``, ``workspace_id``,
    and ``capability_fingerprint`` get full multi-dimensional isolation.
    Callers that omit those params (legacy call sites) get session+agent+path
    isolation, identical to the previous dataclass-key behaviour.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, _ReadCacheEntry] = {}
        # Secondary index: (session_id, agent_id, path) → set of store keys.
        # Maintained by record() to allow O(1) invalidate_path() without
        # iterating the full store or parsing opaque hash keys.
        self._path_index: dict[tuple[str, str, str], set[str]] = {}
        # Reverse index: store_key → its (session_id, agent_id, path) idx_key.
        # Lets eviction (/) discard an evicted key from exactly
        # one value-set in O(1) and prune the idx_key when its set empties,
        # so _path_index stays bounded by the live store (no orphaned empty
        # sets) and eviction never scans every value-set.
        self._key_index: dict[str, tuple[str, str, str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def range_signature(
        *,
        start_line: int | None,
        end_line: int | None,
        offset: int | None,
        limit: int | None,
        encoding: str,
    ) -> str:
        """Stable signature for a read range, independent of None ordering."""
        return (
            f"sl={start_line}|el={end_line}|off={offset}|lim={limit}|enc={encoding}"
        )

    @staticmethod
    def content_hash(content: bytes, *, total_bytes: int | None = None) -> str:
        """Stable hash over (content, total_bytes).

        Hashing the chunk alone exposes a truncated-read
        aliasing bug. If the agent reads ``offset=0 limit=8192`` on a 10 KB
        file (returns first 8 KB + ``truncated=True``), then the file grows
        to 50 KB and the agent re-reads the same range, the first 8 KB are
        byte-identical and the cache would falsely report ``unchanged=True``
        — masking the new tail. Mixing total_bytes into the hash makes any
        size change invalidate the dedup hit even when the chunk content
        matches.
        """
        hasher = hashlib.sha256()
        hasher.update(content)
        if total_bytes is not None:
            hasher.update(b"||total_bytes=")
            hasher.update(str(total_bytes).encode("ascii"))
        return hasher.hexdigest()

    @staticmethod
    def tool_key(tool_name: str, args: dict[str, Any] | None) -> str:
        """Canonical cache ``path`` key for a non-workspace tool.

        The cache's ``path`` dimension is just an opaque string, so a
        non-workspace read tool (e.g. a remote-read / remote-list /
        read-only ``/bin/*`` exec) can share the same
        ``ReadDedupCache`` the workspace tools use by passing
        ``path=ReadDedupCache.tool_key(tool_name, args)``. The args are
        serialised with sorted keys so semantically-identical calls collide
        regardless of argument ordering. Pair with ``range_signature=""`` for
        non-ranged reads, and call ``invalidate_path(..., path=tool_key(...))``
        (or ``clear()``) on any mutating exec / write / delete so a stale read
        cannot survive a state change.

        Non-JSON-serialisable arg values fall back to ``repr`` so the key is
        always stable and never raises.
        """
        try:
            canonical = json.dumps(
                args or {}, sort_keys=True, ensure_ascii=False, default=repr
            )
        except (TypeError, ValueError):
            canonical = repr(args)
        return f"{tool_name}:{canonical}"

    def check(
        self,
        *,
        session_id: str,
        agent_id: str,
        path: str,
        range_signature: str,
        content_hash: str,
        tenant_id: str = "",
        workspace_id: str = "",
        capability_fingerprint: CapabilityFingerprint = _NO_CAPABILITY,
    ) -> _ReadCacheEntry | None:
        """Return the prior entry iff its hash matches the new content_hash.

        Returns ``None`` when there is no prior entry or the content has
        changed. A non-None return means "you already read this, return
        unchanged=true".

        New callers should supply ``tenant_id``, ``workspace_id``, and
        ``capability_fingerprint`` for full multi-dimensional isolation.
        Legacy callers may omit them; isolation degrades gracefully to
        (session_id, agent_id, path, range_signature) scope.
        """
        if not session_id or not agent_id or not path:
            return None
        key = _build_cache_key(
            session_id=session_id,
            agent_id=agent_id,
            path=path,
            range_signature=range_signature,
            content_hash=content_hash,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            capability_fingerprint=capability_fingerprint,
        )
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now - entry.recorded_at > self._ttl_seconds:
                self._pop_unlocked(key)
                return None
            # : content_hash is folded into the store key (see
            # make_dedup_key), so any entry returned here was recorded with
            # exactly this content_hash. Edit detection is therefore purely
            # key-divergence (a changed file yields a different key → fresh
            # miss) plus explicit invalidate_path() on a write. There is no
            # post-lookup content_hash guard because it would be unreachable.
            return entry

    def record(
        self,
        *,
        session_id: str,
        agent_id: str,
        path: str,
        range_signature: str,
        content_hash: str,
        size_bytes: int,
        tenant_id: str = "",
        workspace_id: str = "",
        capability_fingerprint: CapabilityFingerprint = _NO_CAPABILITY,
    ) -> None:
        """Record (or refresh) the latest content hash for a (path, range).

        New callers should supply ``tenant_id``, ``workspace_id``, and
        ``capability_fingerprint`` to match the multi-dim key used in
        ``check()``. Legacy callers may omit them.
        """
        if not session_id or not agent_id or not path:
            return
        key = _build_cache_key(
            session_id=session_id,
            agent_id=agent_id,
            path=path,
            range_signature=range_signature,
            content_hash=content_hash,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            capability_fingerprint=capability_fingerprint,
        )
        idx_key = (session_id, agent_id, path)
        now = time.monotonic()
        with self._lock:
            self._store[key] = _ReadCacheEntry(
                size_bytes=size_bytes,
                recorded_at=now,
            )
            self._path_index.setdefault(idx_key, set()).add(key)
            self._key_index[key] = idx_key
            if len(self._store) > self._max_entries:
                self._evict_oldest_unlocked()

    def _pop_unlocked(self, key: str) -> None:
        """Remove ``key`` from the store and both indexes (caller holds lock).

 Uses the reverse ``_key_index`` to discard the key from exactly one
 value-set in O(1) and prunes the now-empty idx_key from
 ``_path_index`` so the secondary index never accumulates orphaned
 empty sets .
 """
        self._store.pop(key, None)
        idx_key = self._key_index.pop(key, None)
        if idx_key is None:
            return
        key_set = self._path_index.get(idx_key)
        if key_set is None:
            return
        key_set.discard(key)
        if not key_set:
            del self._path_index[idx_key]

    def _evict_oldest_unlocked(self) -> None:
        # Remove ~10% oldest entries to amortize eviction cost. Each evicted
        # key is reclaimed from the store and both indexes in O(1) via the
        # reverse map (: no full-_path_index scan per evicted key).
        target_drop = max(1, len(self._store) // 10)
        ordered = sorted(self._store.items(), key=lambda kv: kv[1].recorded_at)
        for evict_key, _ in ordered[:target_drop]:
            self._pop_unlocked(evict_key)

    def invalidate_path(self, *, session_id: str, agent_id: str, path: str) -> None:
        """Clear all cache entries for a path after a write that may have changed it.

 The store key is an opaque sha-256 hash, so invalidation cannot
 re-derive every (range_signature, content_hash) combination that could
 be stored. Instead the secondary ``_path_index`` maps
 ``(session_id, agent_id, path)`` to the set of store keys currently
 cached for that path; we drop them all and prune the index entry.

 This is the explicit edit-detection mechanism (see ): a write
 forces a fresh read on the *next* read of the same path regardless of
 range or content hash.
 """
        idx_key = (session_id, agent_id, path)
        with self._lock:
            for key in list(self._path_index.get(idx_key, ())):
                self._pop_unlocked(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._path_index.clear()
            self._key_index.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def path_index_size(self) -> int:
        """Number of distinct (session, agent, path) keys in the secondary
 index. Used by tests to assert the index stays bounded by the live
 store and never accumulates orphaned empty value-sets.
 """
        with self._lock:
            return len(self._path_index)

    def _assert_index_invariant(self) -> None:
        """Debug/test guard: the forward index, reverse index, and store must
        agree, and no value-set may be empty. Raises AssertionError otherwise.
        """
        with self._lock:
            forward_keys = {k for s in self._path_index.values() for k in s}
            if not all(self._path_index.values()):
                raise AssertionError("empty value-set leaked into _path_index")
            if forward_keys != set(self._store):
                raise AssertionError("forward _path_index out of sync with store")
            if set(self._key_index) != set(self._store):
                raise AssertionError("reverse _key_index out of sync with store")
            for key, idx_key in self._key_index.items():
                if key not in self._path_index.get(idx_key, set()):
                    raise AssertionError(
                        "reverse _key_index points at a missing forward entry"
                    )


__all__ = [
    "CapabilityFingerprint",
    "ReadDedupCache",
    "make_capability_fingerprint",
    "make_dedup_key",
]
