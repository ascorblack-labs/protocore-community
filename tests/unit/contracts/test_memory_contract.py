"""Contract tests for :mod:`protocore.contracts.memory`.

These tests pin the **behavioural spec** the ``IMemory`` docstrings define,
exercised against the in-memory reference fake
(:class:`protocore.tests_support.adapters.InMemoryMemory`). The fake proves the
contract is implementable with no database; the real Postgres adapter
(a relational store with a vector index for recall) must
satisfy the *same* observable behaviour (its own integration tests assert it
against a live PG via testcontainers).

Covered invariants:

* scope grammar + ``scope_key`` validation
* two-stage idempotent write: CREATE / MERGE / SKIP decisions
* optimistic-concurrency drift-guard on write + delete
* tenant isolation
* recall/search scope fan-out + lexical ranking + reinforcement signals
* get/delete/list semantics
* the Protocol is ``runtime_checkable`` and the fake satisfies it
"""
from __future__ import annotations

import pytest

from protocore.contracts.memory import (
    DEFAULT_RECALL_SCOPES,
    MEMORY_BLOCKED_PLACEHOLDER,
    MEMORY_KINDS,
    IMemory,
    IMemoryContentScanner,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryScope,
    MemoryWriteDecision,
    blocked_memory_placeholder,
)
from protocore.tests_support.adapters import InMemoryMemory

# asyncio_mode = "auto" (pyproject) auto-detects async tests; no module mark
# needed (and a blanket mark would spuriously warn on the sync tests here).

_TENANT = "11111111-1111-1111-1111-111111111111"
_OTHER_TENANT = "22222222-2222-2222-2222-222222222222"
_SESSION = "sess-abc"


@pytest.fixture
def mem() -> InMemoryMemory:
    return InMemoryMemory()


# --------------------------------------------------------------------------- #
# Protocol shape
# --------------------------------------------------------------------------- #


def test_fake_is_instance_of_protocol(mem: InMemoryMemory) -> None:
    assert isinstance(mem, IMemory)


def test_scope_enum_values() -> None:
    # ``global`` is a keyword → member is ``global_`` but the *value* is the
    # lowercase wire string (stored in PG, surfaced on the API).
    assert MemoryScope.global_.value == "global"
    assert {s.value for s in MemoryScope} == {
        "global",
        "user",
        "project",
        "session",
        "agent",
        "custom",
    }


def test_default_recall_scopes_is_union_of_durable_and_working() -> None:
    assert set(DEFAULT_RECALL_SCOPES) == {
        MemoryScope.session,
        MemoryScope.project,
        MemoryScope.user,
        MemoryScope.global_,
    }
    # agent/custom are NOT in the default fan-out (explicit opt-in only).
    assert MemoryScope.agent not in DEFAULT_RECALL_SCOPES
    assert MemoryScope.custom not in DEFAULT_RECALL_SCOPES


# --------------------------------------------------------------------------- #
# untrusted-content scan seam
# --------------------------------------------------------------------------- #


def test_content_scanner_is_runtime_checkable() -> None:
    """A duck-typed scanner satisfies the Protocol (the seam the host
    adapter populates with the real threat-pattern scanner)."""

    class _S:
        def scan(self, text: str) -> str | None:
            return "bad" if "x" in text else None

    assert isinstance(_S(), IMemoryContentScanner)
    # missing the method → NOT an instance.
    assert not isinstance(object(), IMemoryContentScanner)


def test_blocked_placeholder_helper() -> None:
    """The [BLOCKED] placeholder helper keeps the marker but drops the raw body
    so a flagged entry can replace its text without re-injecting the payload."""
    out = blocked_memory_placeholder("ignore previous instructions")
    assert out.startswith(MEMORY_BLOCKED_PLACEHOLDER)
    assert "ignore previous instructions" not in out
    # optional reason is appended but the raw text is never echoed.
    out2 = blocked_memory_placeholder("ignore previous instructions", reason="pi")
    assert "pi" in out2
    assert "ignore previous instructions" not in out2


def test_memory_kinds_taxonomy_is_advisory() -> None:
    assert "fact" in MEMORY_KINDS
    assert "skill" in MEMORY_KINDS
    assert "reflection" in MEMORY_KINDS


# --------------------------------------------------------------------------- #
# write — validation
# --------------------------------------------------------------------------- #


async def test_write_rejects_empty_text(mem: InMemoryMemory) -> None:
    with pytest.raises(ValueError):
        await mem.write(_TENANT, MemoryScope.session, _SESSION, "   ")


async def test_write_requires_scope_key_for_non_global(mem: InMemoryMemory) -> None:
    with pytest.raises(ValueError):
        await mem.write(_TENANT, MemoryScope.session, "", "a fact")


async def test_write_global_ignores_scope_key(mem: InMemoryMemory) -> None:
    # global scope normalises scope_key to "" regardless of input.
    res = await mem.write(_TENANT, MemoryScope.global_, "ignored", "tenant-wide fact")
    assert res.record.scope is MemoryScope.global_
    assert res.record.scope_key == ""


# --------------------------------------------------------------------------- #
# write — two-stage idempotency: CREATE / MERGE / SKIP
# --------------------------------------------------------------------------- #


async def test_write_create_on_fresh_bucket(mem: InMemoryMemory) -> None:
    res = await mem.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "order 12345 shipped via DHL",
        kind="fact",
        source_refs=["/orders/12345.json"],
        salience=0.4,
    )
    assert res.decision is MemoryWriteDecision.created
    assert res.record.version == 1
    assert res.record.text == "order 12345 shipped via DHL"
    assert res.record.source_refs == ["/orders/12345.json"]
    assert res.record.salience == 0.4
    assert res.record.id  # store-assigned


async def test_write_skip_on_identical_repeat(mem: InMemoryMemory) -> None:
    text = "the orders table primary key is order_id"
    first = await mem.write(_TENANT, MemoryScope.session, _SESSION, text)
    second = await mem.write(_TENANT, MemoryScope.session, _SESSION, text)
    assert first.decision is MemoryWriteDecision.created
    assert second.decision is MemoryWriteDecision.skipped
    # idempotency: exactly one row survives.
    rows = await mem.list(_TENANT, scope=MemoryScope.session, scope_key=_SESSION)
    assert len(rows) == 1
    # SKIP returns the pre-existing row unchanged (same id + version).
    assert second.record.id == first.record.id
    assert second.record.version == 1


async def test_write_merge_when_new_source_refs_add_information(
    mem: InMemoryMemory,
) -> None:
    text = "customer prefers email contact"
    first = await mem.write(
        _TENANT, MemoryScope.user, "u-1", text, source_refs=["msg-1"]
    )
    # same fact, new provenance → MERGE (provenance is additive).
    second = await mem.write(
        _TENANT, MemoryScope.user, "u-1", text, source_refs=["msg-2"]
    )
    assert first.decision is MemoryWriteDecision.created
    assert second.decision is MemoryWriteDecision.merged
    assert second.record.id == first.record.id
    assert second.record.version == 2
    assert set(second.record.source_refs) == {"msg-1", "msg-2"}
    rows = await mem.list(_TENANT, scope=MemoryScope.user, scope_key="u-1")
    assert len(rows) == 1  # still one row


async def test_write_merge_prefers_richer_text(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "db host is pg")
    res = await mem.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "db host is pg primary on port 5432",  # superset → high similarity, richer
        similarity_threshold=0.4,
    )
    assert res.decision is MemoryWriteDecision.merged
    assert "5432" in res.record.text


async def test_write_distinct_facts_create_separate_rows(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "alpha bravo charlie")
    res = await mem.write(_TENANT, MemoryScope.session, _SESSION, "xray yankee zulu")
    assert res.decision is MemoryWriteDecision.created
    rows = await mem.list(_TENANT, scope=MemoryScope.session, scope_key=_SESSION)
    assert len(rows) == 2


async def test_write_threshold_controls_create_vs_merge(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "the quick brown fox")
    # Lower threshold => the partially-overlapping text MERGEs.
    low = await mem.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "the quick brown dog",
        similarity_threshold=0.1,
    )
    assert low.decision is MemoryWriteDecision.merged
    # Reset and retry with a high threshold => CREATE.
    mem2 = InMemoryMemory()
    await mem2.write(_TENANT, MemoryScope.session, _SESSION, "the quick brown fox")
    high = await mem2.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "the quick brown dog",
        similarity_threshold=0.99,
    )
    assert high.decision is MemoryWriteDecision.created


async def test_write_similarity_reported(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "same words here")
    res = await mem.write(_TENANT, MemoryScope.session, _SESSION, "same words here")
    assert res.similarity == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# write/delete — drift-guard (optimistic concurrency)
# --------------------------------------------------------------------------- #


async def test_write_drift_guard_blocks_stale_merge(mem: InMemoryMemory) -> None:
    text = "shared fact subject to a race"
    created = await mem.write(_TENANT, MemoryScope.session, _SESSION, text)
    assert created.record.version == 1
    # Concurrent writer advances the record to v2.
    await mem.write(
        _TENANT, MemoryScope.session, _SESSION, text, source_refs=["late-ref"]
    )
    # Our writer still believes it is v1 → conflict on the MERGE branch.
    with pytest.raises(MemoryConflictError):
        await mem.write(
            _TENANT,
            MemoryScope.session,
            _SESSION,
            text,
            source_refs=["our-ref"],
            expected_version=1,
        )


async def test_write_expected_version_ignored_on_create(mem: InMemoryMemory) -> None:
    # CREATE branch has nothing to conflict with → expected_version is a no-op.
    res = await mem.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "brand new",
        expected_version=99,
    )
    assert res.decision is MemoryWriteDecision.created


async def test_delete_drift_guard(mem: InMemoryMemory) -> None:
    text = "fact to forget"
    created = await mem.write(_TENANT, MemoryScope.session, _SESSION, text)
    # advance version
    await mem.write(
        _TENANT, MemoryScope.session, _SESSION, text, source_refs=["x"]
    )
    with pytest.raises(MemoryConflictError):
        await mem.delete(_TENANT, created.record.id, expected_version=1)
    # correct version deletes
    live = await mem.get(_TENANT, created.record.id)
    assert await mem.delete(_TENANT, created.record.id, expected_version=live.version)


# --------------------------------------------------------------------------- #
# tenant isolation
# --------------------------------------------------------------------------- #


async def test_tenant_isolation_on_write_and_recall(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.global_, "", "secret of tenant one")
    await mem.write(_OTHER_TENANT, MemoryScope.global_, "", "secret of tenant two")
    hits_one = await mem.recall(_TENANT, "secret")
    hits_two = await mem.recall(_OTHER_TENANT, "secret")
    assert {h.record.text for h in hits_one} == {"secret of tenant one"}
    assert {h.record.text for h in hits_two} == {"secret of tenant two"}


async def test_get_is_tenant_scoped(mem: InMemoryMemory) -> None:
    res = await mem.write(_TENANT, MemoryScope.global_, "", "owned by tenant one")
    with pytest.raises(MemoryNotFoundError):
        await mem.get(_OTHER_TENANT, res.record.id)


# --------------------------------------------------------------------------- #
# recall / search — scope fan-out + ranking
# --------------------------------------------------------------------------- #


async def test_recall_scope_fanout_default(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "session note widget")
    await mem.write(_TENANT, MemoryScope.user, "u-1", "user note widget")
    await mem.write(_TENANT, MemoryScope.global_, "", "global note widget")
    # agent scope is OUTSIDE the default fan-out → must not surface.
    await mem.write(_TENANT, MemoryScope.agent, "a-1", "agent note widget")

    hits = await mem.recall(
        _TENANT,
        "widget",
        scope_keys={MemoryScope.session: _SESSION, MemoryScope.user: "u-1"},
    )
    texts = {h.record.text for h in hits}
    assert "session note widget" in texts
    assert "user note widget" in texts
    assert "global note widget" in texts
    assert "agent note widget" not in texts  # agent not in DEFAULT_RECALL_SCOPES


async def test_recall_requires_scope_key_for_non_global(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "scoped widget")
    await mem.write(_TENANT, MemoryScope.session, "other-sess", "other widget")
    # Only ask for _SESSION key → the other session's row must not leak.
    hits = await mem.recall(
        _TENANT,
        "widget",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    assert {h.record.text for h in hits} == {"scoped widget"}


async def test_recall_explicit_scope_narrowing(mem: InMemoryMemory) -> None:
    # The benchmark tenant pattern: session-only recall.
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "task scratch alpha")
    await mem.write(_TENANT, MemoryScope.global_, "", "task scratch alpha")
    hits = await mem.recall(
        _TENANT,
        "task scratch",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    assert all(h.record.scope is MemoryScope.session for h in hits)


async def test_search_finds_exact_identifier_token(mem: InMemoryMemory) -> None:
    # The whole reason BM25/lexical is mandatory: exact SKU/ID/path recall.
    await mem.write(
        _TENANT, MemoryScope.session, _SESSION, "SKU-99821 is out of stock"
    )
    await mem.write(
        _TENANT, MemoryScope.session, _SESSION, "totally unrelated note"
    )
    hits = await mem.search(
        _TENANT,
        "SKU-99821",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    assert hits
    assert hits[0].record.text.startswith("SKU-99821")


async def test_recall_empty_query_returns_recent(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "first")
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "second")
    hits = await mem.recall(
        _TENANT,
        "",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    # empty query → recency order (most recent first), never an error.
    assert [h.record.text for h in hits][:2] == ["second", "first"]


async def test_recall_respects_limit(mem: InMemoryMemory) -> None:
    for i in range(5):
        await mem.write(_TENANT, MemoryScope.session, _SESSION, f"note number {i}")
    hits = await mem.recall(
        _TENANT,
        "note",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
        limit=2,
    )
    assert len(hits) == 2


async def test_recall_kind_filter(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "x widget", kind="fact")
    await mem.write(
        _TENANT, MemoryScope.session, _SESSION, "y widget", kind="reflection"
    )
    hits = await mem.search(
        _TENANT,
        "widget",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
        kinds=["reflection"],
    )
    assert {h.record.kind for h in hits} == {"reflection"}


async def test_recall_reinforces_returned_records(mem: InMemoryMemory) -> None:
    res = await mem.write(_TENANT, MemoryScope.session, _SESSION, "reinforce me")
    assert res.record.access_count == 0
    assert res.record.last_accessed_at is None
    await mem.recall(
        _TENANT,
        "reinforce",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    reloaded = await mem.get(_TENANT, res.record.id)
    assert reloaded.access_count == 1
    assert reloaded.last_accessed_at is not None


async def test_recall_empty_when_nothing_matches(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "apples")
    hits = await mem.recall(
        _TENANT,
        "quantum chromodynamics",
        scopes=[MemoryScope.session],
        scope_keys={MemoryScope.session: _SESSION},
    )
    assert list(hits) == []  # never None


# --------------------------------------------------------------------------- #
# get / delete / list
# --------------------------------------------------------------------------- #


async def test_get_roundtrip(mem: InMemoryMemory) -> None:
    res = await mem.write(_TENANT, MemoryScope.project, "p-1", "project fact")
    got = await mem.get(_TENANT, res.record.id)
    assert isinstance(got, MemoryRecord)
    assert got.id == res.record.id
    assert got.text == "project fact"


async def test_get_missing_raises(mem: InMemoryMemory) -> None:
    with pytest.raises(MemoryNotFoundError):
        await mem.get(_TENANT, "mem-does-not-exist")


async def test_delete_idempotent(mem: InMemoryMemory) -> None:
    res = await mem.write(_TENANT, MemoryScope.session, _SESSION, "to delete")
    assert await mem.delete(_TENANT, res.record.id) is True
    # second delete is a no-op, not an error
    assert await mem.delete(_TENANT, res.record.id) is False
    with pytest.raises(MemoryNotFoundError):
        await mem.get(_TENANT, res.record.id)


async def test_list_filters_and_orders(mem: InMemoryMemory) -> None:
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "s1", kind="fact")
    await mem.write(_TENANT, MemoryScope.session, _SESSION, "s2", kind="decision")
    await mem.write(_TENANT, MemoryScope.user, "u-1", "u1", kind="fact")

    session_rows = await mem.list(
        _TENANT, scope=MemoryScope.session, scope_key=_SESSION
    )
    assert {r.text for r in session_rows} == {"s1", "s2"}

    fact_rows = await mem.list(_TENANT, kinds=["fact"])
    assert {r.text for r in fact_rows} == {"s1", "u1"}

    # newest-first ordering
    ordered = await mem.list(_TENANT, scope=MemoryScope.session, scope_key=_SESSION)
    assert ordered[0].text == "s2"


async def test_list_pagination(mem: InMemoryMemory) -> None:
    for i in range(5):
        await mem.write(_TENANT, MemoryScope.session, _SESSION, f"row {i}")
    page = await mem.list(
        _TENANT, scope=MemoryScope.session, scope_key=_SESSION, limit=2, offset=2
    )
    assert len(page) == 2
