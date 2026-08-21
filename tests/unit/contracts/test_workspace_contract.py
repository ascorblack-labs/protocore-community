"""Contract tests for :mod:`protocore.contracts.workspace`.

These tests pin the **behavioural spec** the ``IWorkspace`` docstrings define,
exercised against the in-memory reference fake
(:class:`protocore.tests_support.adapters.InMemoryWorkspace`). The fake proves
the contract is implementable with no database / no object store; the real
durable-byte-store + Postgres-FTS adapter
(a durable byte store plus a full-text manifest index) must
satisfy the *same* observable behaviour (its own integration tests assert it
against a live PG via testcontainers).

Covered invariants:

* scope grammar + ``scope_key`` validation + path safety
* atomic idempotent write: CREATE / REPLACE / UNCHANGED decisions
* hard quota refusal (per-unit + per-scope byte budget)
* scratch soft-cap LRU GC (durable units never evicted)
* optimistic-concurrency drift-guard on write + delete
* tenant isolation
* read returns the body; list/search return manifest-only (files-over-inline)
* search scope fan-out + lexical ranking + exact-identifier recall + reinforcement
* lifecycle teardown (clear_scope)
* the Protocol is ``runtime_checkable`` and the fake satisfies it
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from protocore.contracts.workspace import (
    DEFAULT_WORKSPACE_LIFECYCLE,
    IWorkspace,
    WorkspaceConflictError,
    WorkspaceLifecycle,
    WorkspaceNotFoundError,
    WorkspacePathError,
    WorkspaceQuotaExceededError,
    WorkspaceScope,
    WorkspaceUnit,
    WorkspaceWriteDecision,
)
from protocore.tests_support.adapters import InMemoryWorkspace

_TENANT = "11111111-1111-1111-1111-111111111111"
_OTHER_TENANT = "22222222-2222-2222-2222-222222222222"
_SESSION = "sess-abc"


@pytest.fixture
def ws() -> InMemoryWorkspace:
    return InMemoryWorkspace()


def _b(text: str) -> bytes:
    return text.encode("utf-8")


# --------------------------------------------------------------------------- #
# Protocol shape
# --------------------------------------------------------------------------- #


def test_fake_is_instance_of_protocol(ws: InMemoryWorkspace) -> None:
    assert isinstance(ws, IWorkspace)


def test_scope_enum_values() -> None:
    assert {s.value for s in WorkspaceScope} == {
        "session",
        "task",
        "project",
        "knowledge_base",
    }


def test_knowledge_base_scope_round_trips_through_its_value() -> None:
    """The knowledge-base scope is addressable by its stored/wire string.

    The enum *value* is what lands in Postgres and on the wire, so a scope
    rehydrated from a stored row must resolve back to the same member.
    """
    assert WorkspaceScope.knowledge_base.value == "knowledge_base"
    assert WorkspaceScope("knowledge_base") is WorkspaceScope.knowledge_base


def test_lifecycle_default_is_scratch() -> None:
    assert DEFAULT_WORKSPACE_LIFECYCLE is WorkspaceLifecycle.scratch
    assert {lc.value for lc in WorkspaceLifecycle} == {"scratch", "durable"}


# --------------------------------------------------------------------------- #
# write — validation + path safety
# --------------------------------------------------------------------------- #


async def test_write_requires_scope_key(ws: InMemoryWorkspace) -> None:
    with pytest.raises(WorkspacePathError):
        await ws.write(_TENANT, WorkspaceScope.session, "", "x.txt", _b("data"))


@pytest.mark.parametrize(
    "bad_path",
    ["/abs.txt", "../escape.txt", "a/../../etc", "back\\slash.txt", "", "a//b.txt"],
)
async def test_write_rejects_unsafe_paths(
    ws: InMemoryWorkspace, bad_path: str
) -> None:
    with pytest.raises(WorkspacePathError):
        await ws.write(_TENANT, WorkspaceScope.session, _SESSION, bad_path, _b("d"))


async def test_write_rejects_control_chars(ws: InMemoryWorkspace) -> None:
    with pytest.raises(WorkspacePathError):
        await ws.write(
            _TENANT, WorkspaceScope.session, _SESSION, "a\r\nb.txt", _b("d")
        )


async def test_write_accepts_nested_and_unicode_paths(
    ws: InMemoryWorkspace,
) -> None:
    out = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "notes/схема.md", _b("данные")
    )
    assert out.unit.path == "notes/схема.md"


# --------------------------------------------------------------------------- #
# write — atomic idempotency: CREATE / REPLACE / UNCHANGED
# --------------------------------------------------------------------------- #


async def test_write_create_on_fresh_path(ws: InMemoryWorkspace) -> None:
    out = await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "orders.json",
        _b('[{"id": 1}]'),
        summary="orders dump",
        source_refs=["/q/123"],
    )
    assert out.decision is WorkspaceWriteDecision.created
    assert out.unit.version == 1
    assert out.unit.size_bytes == len('[{"id": 1}]')
    assert out.unit.summary == "orders dump"
    assert out.unit.source_refs == ["/q/123"]
    assert out.unit.lifecycle is WorkspaceLifecycle.scratch
    # manifest-only on the write outcome (files-over-inline).
    assert out.unit.content is None
    assert out.unit.id


async def test_write_unchanged_on_identical_bytes(ws: InMemoryWorkspace) -> None:
    body = _b("the orders table primary key is order_id")
    first = await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "k.txt", body)
    second = await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "k.txt", body)
    assert first.decision is WorkspaceWriteDecision.created
    assert second.decision is WorkspaceWriteDecision.unchanged
    # idempotency: exactly one unit survives, version not bumped.
    rows = await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    assert len(rows) == 1
    assert second.unit.id == first.unit.id
    assert second.unit.version == 1


async def test_write_replace_overwrites_in_place(ws: InMemoryWorkspace) -> None:
    first = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "data.json", _b("old")
    )
    second = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "data.json", _b("new richer body")
    )
    assert first.decision is WorkspaceWriteDecision.created
    assert second.decision is WorkspaceWriteDecision.replaced
    assert second.unit.id == first.unit.id  # SAME unit (no foo(1) copy)
    assert second.unit.version == 2
    rows = await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    assert len(rows) == 1  # still one unit
    # the body is the new one.
    got = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "data.json")
    assert got.content == _b("new richer body")


async def test_write_replace_on_metadata_change_same_bytes(
    ws: InMemoryWorkspace,
) -> None:
    body = _b("same body")
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", body)
    # identical bytes but a new summary → REPLACE (not unchanged).
    out = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "x", body, summary="now summarised"
    )
    assert out.decision is WorkspaceWriteDecision.replaced
    assert out.unit.summary == "now summarised"


async def test_write_lifecycle_durable(ws: InMemoryWorkspace) -> None:
    out = await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "keep.txt",
        _b("important"),
        lifecycle=WorkspaceLifecycle.durable,
    )
    assert out.unit.lifecycle is WorkspaceLifecycle.durable


# --------------------------------------------------------------------------- #
# write — hard quota
# --------------------------------------------------------------------------- #


async def test_write_refuses_oversize_unit(ws: InMemoryWorkspace) -> None:
    with pytest.raises(WorkspaceQuotaExceededError):
        await ws.write(
            _TENANT,
            WorkspaceScope.session,
            _SESSION,
            "big.bin",
            _b("x" * 50),
            max_bytes=10,
        )


async def test_write_refuses_when_scope_budget_exceeded_by_durable(
    ws: InMemoryWorkspace,
) -> None:
    # Fill the scope with a DURABLE unit that cannot be GC'd, then a second write
    # that would exceed the budget must be refused (GC cannot reclaim durable).
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "a",
        _b("x" * 80),
        lifecycle=WorkspaceLifecycle.durable,
        max_scope_bytes=100,
    )
    with pytest.raises(WorkspaceQuotaExceededError):
        await ws.write(
            _TENANT,
            WorkspaceScope.session,
            _SESSION,
            "b",
            _b("y" * 80),
            lifecycle=WorkspaceLifecycle.durable,
            max_scope_bytes=100,
        )


# --------------------------------------------------------------------------- #
# write — scratch soft-cap GC (LRU, durable never evicted)
# --------------------------------------------------------------------------- #


async def test_gc_evicts_lru_scratch_on_unit_cap(ws: InMemoryWorkspace) -> None:
    # cap = 2 units. Write 3 scratch units; the least-recently-accessed is
    # evicted. Access the first so it survives over the second.
    await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "u1", _b("a"), max_units_per_scope=2
    )
    await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "u2", _b("b"), max_units_per_scope=2
    )
    # touch u1 so u2 becomes the least-recently-accessed.
    await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "u1")
    out = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "u3", _b("c"), max_units_per_scope=2
    )
    rows = {u.path for u in await ws.list(_TENANT, WorkspaceScope.session, _SESSION)}
    assert rows == {"u1", "u3"}
    assert "u2" in out.evicted_paths


async def test_gc_never_evicts_durable(ws: InMemoryWorkspace) -> None:
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "keep",
        _b("a"),
        lifecycle=WorkspaceLifecycle.durable,
        max_units_per_scope=1,
    )
    # A durable unit fills the cap; a scratch write has no OTHER scratch victim
    # and the just-written unit is never evicted → the write is REFUSED,
    # the durable one stays, and the refused scratch never leaks into the scope.
    with pytest.raises(WorkspaceQuotaExceededError):
        await ws.write(
            _TENANT,
            WorkspaceScope.session,
            _SESSION,
            "scratch1",
            _b("b"),
            max_units_per_scope=1,
        )
    rows = {u.path for u in await ws.list(_TENANT, WorkspaceScope.session, _SESSION)}
    assert rows == {"keep"}  # durable survived; refused scratch rolled back


async def test_gc_never_returns_just_written_evicted_unit(
    ws: InMemoryWorkspace,
) -> None:
    # The just-written scratch survives GC when an OLDER scratch is the eligible
    # victim (the new unit is excluded from the victim set, never
    # returned-then-evicted).
    await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "old", _b("a"),
        max_units_per_scope=1,
    )
    out = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "new", _b("b"),
        max_units_per_scope=1,
    )
    assert out.decision is WorkspaceWriteDecision.created
    assert "old" in out.evicted_paths
    rows = {u.path for u in await ws.list(_TENANT, WorkspaceScope.session, _SESSION)}
    assert rows == {"new"}  # the survivor is the just-written unit
    # and it is genuinely present (not a phantom).
    got = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "new")
    assert got.content == _b("b")


# --------------------------------------------------------------------------- #
# write — searchable-text cap is threaded per call (SH1)
# --------------------------------------------------------------------------- #


async def test_searchable_text_max_bytes_caps_indexed_prefix(
    ws: InMemoryWorkspace,
) -> None:
    # SH1 — the per-call searchable_text_max_bytes bounds the indexed prefix; the
    # full body is still stored + readable. A tenant override (e.g. 5) shortens
    # the indexed prefix vs the default.
    body = _b("abcdefghijklmnopqrstuvwxyz")
    short = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "s.txt", body,
        searchable_text_max_bytes=5,
    )
    assert short.unit.searchable_text == "abcde"
    # the body is untouched by the searchable cap.
    got = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "s.txt")
    assert got.content == body
    # 0 = index the whole body (no bound).
    whole = await ws.write(
        _TENANT, WorkspaceScope.session, "other-sess", "w.txt", body,
        searchable_text_max_bytes=0,
    )
    assert whole.unit.searchable_text == body.decode("utf-8")


# --------------------------------------------------------------------------- #
# write / delete — drift-guard (optimistic concurrency)
# --------------------------------------------------------------------------- #


async def test_write_drift_guard_blocks_stale_replace(ws: InMemoryWorkspace) -> None:
    created = await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "race.txt", _b("v1")
    )
    assert created.unit.version == 1
    # concurrent writer advances to v2.
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "race.txt", _b("v2"))
    # our writer still believes it's v1 → conflict.
    with pytest.raises(WorkspaceConflictError):
        await ws.write(
            _TENANT,
            WorkspaceScope.session,
            _SESSION,
            "race.txt",
            _b("ours"),
            expected_version=1,
        )


async def test_write_expected_version_ignored_on_create(
    ws: InMemoryWorkspace,
) -> None:
    out = await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "fresh.txt",
        _b("new"),
        expected_version=99,
    )
    assert out.decision is WorkspaceWriteDecision.created


async def test_delete_drift_guard(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "d.txt", _b("v1"))
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "d.txt", _b("v2"))
    with pytest.raises(WorkspaceConflictError):
        await ws.delete(
            _TENANT, WorkspaceScope.session, _SESSION, "d.txt", expected_version=1
        )
    # correct version deletes.
    live = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "d.txt")
    assert await ws.delete(
        _TENANT, WorkspaceScope.session, _SESSION, "d.txt", expected_version=live.version
    )


# --------------------------------------------------------------------------- #
# tenant + scope isolation
# --------------------------------------------------------------------------- #


async def test_tenant_isolation(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("tenant one"))
    await ws.write(
        _OTHER_TENANT, WorkspaceScope.session, _SESSION, "x", _b("tenant two")
    )
    one = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")
    two = await ws.read(_OTHER_TENANT, WorkspaceScope.session, _SESSION, "x")
    assert one.content == _b("tenant one")
    assert two.content == _b("tenant two")


async def test_scope_isolation(ws: InMemoryWorkspace) -> None:
    # same path in session vs task scope are distinct units.
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("session"))
    await ws.write(_TENANT, WorkspaceScope.task, "task-1", "x", _b("task"))
    s = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")
    t = await ws.read(_TENANT, WorkspaceScope.task, "task-1", "x")
    assert s.content == _b("session")
    assert t.content == _b("task")


async def test_read_missing_raises(ws: InMemoryWorkspace) -> None:
    with pytest.raises(WorkspaceNotFoundError):
        await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "ghost.txt")


# --------------------------------------------------------------------------- #
# read returns body; list/search return manifest-only
# --------------------------------------------------------------------------- #


async def test_read_populates_body_list_does_not(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("the body"))
    got = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")
    assert got.content == _b("the body")
    rows = await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    assert rows[0].content is None  # manifest-only


async def test_read_reinforces(ws: InMemoryWorkspace) -> None:
    created = await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("d"))
    assert created.unit.access_count == 0
    await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")
    again = await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")
    assert again.access_count == 2
    assert again.last_accessed_at is not None


# --------------------------------------------------------------------------- #
# search — fan-out + lexical ranking + exact identifier recall
# --------------------------------------------------------------------------- #


async def test_search_finds_exact_identifier_token(ws: InMemoryWorkspace) -> None:
    # The whole reason lexical/BM25 is mandatory: exact SKU/ID/path recall.
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "stock.txt",
        _b("SKU-99821 is out of stock at warehouse 3"),
    )
    await ws.write(
        _TENANT, WorkspaceScope.session, _SESSION, "misc.txt", _b("totally unrelated")
    )
    hits = await ws.search(
        _TENANT,
        "SKU-99821",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
    )
    assert hits
    assert hits[0].unit.path == "stock.txt"
    # manifest-only on hits.
    assert hits[0].unit.content is None


async def test_search_matches_summary(ws: InMemoryWorkspace) -> None:
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "dump1.bin",
        _b("\x00\x01\x02 binary blob"),  # binary → empty searchable_text
        summary="top revenue orders Q1",
    )
    hits = await ws.search(
        _TENANT,
        "revenue orders",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
    )
    # matched via the summary even though the body is binary.
    assert {h.unit.path for h in hits} == {"dump1.bin"}


async def test_search_scope_fanout_requires_key(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "a", _b("widget alpha"))
    await ws.write(
        _TENANT, WorkspaceScope.session, "other-sess", "b", _b("widget beta")
    )
    # only ask for _SESSION key → other session's unit must not leak.
    hits = await ws.search(
        _TENANT,
        "widget",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
    )
    assert {h.unit.path for h in hits} == {"a"}


async def test_search_lifecycle_filter(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "s", _b("note x"))
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "d",
        _b("note y"),
        lifecycle=WorkspaceLifecycle.durable,
    )
    hits = await ws.search(
        _TENANT,
        "note",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
        lifecycles=[WorkspaceLifecycle.durable],
    )
    assert {h.unit.path for h in hits} == {"d"}


async def test_search_empty_query_returns_recent(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "first", _b("a"))
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "second", _b("b"))
    hits = await ws.search(
        _TENANT,
        "",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
    )
    assert [h.unit.path for h in hits][:2] == ["second", "first"]


async def test_search_respects_limit(ws: InMemoryWorkspace) -> None:
    for i in range(5):
        await ws.write(
            _TENANT, WorkspaceScope.session, _SESSION, f"note{i}", _b(f"note body {i}")
        )
    hits = await ws.search(
        _TENANT,
        "note",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
        limit=2,
    )
    assert len(hits) == 2


async def test_search_empty_when_nothing_matches(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("apples"))
    hits = await ws.search(
        _TENANT,
        "quantum chromodynamics",
        scopes=[WorkspaceScope.session],
        scope_keys={WorkspaceScope.session: _SESSION},
    )
    assert list(hits) == []  # never None


# --------------------------------------------------------------------------- #
# list / delete / clear_scope
# --------------------------------------------------------------------------- #


async def test_list_orders_and_filters(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "s1", _b("a"))
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "s2",
        _b("b"),
        lifecycle=WorkspaceLifecycle.durable,
    )
    rows = await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    assert {r.path for r in rows} == {"s1", "s2"}
    assert rows[0].path == "s2"  # newest first
    durable = await ws.list(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        lifecycles=[WorkspaceLifecycle.durable],
    )
    assert {r.path for r in durable} == {"s2"}


async def test_list_pagination(ws: InMemoryWorkspace) -> None:
    for i in range(5):
        await ws.write(_TENANT, WorkspaceScope.session, _SESSION, f"r{i}", _b("x"))
    page = await ws.list(
        _TENANT, WorkspaceScope.session, _SESSION, limit=2, offset=2
    )
    assert len(page) == 2


async def test_list_requires_scope_key(ws: InMemoryWorkspace) -> None:
    with pytest.raises(WorkspacePathError):
        await ws.list(_TENANT, WorkspaceScope.session, "")


async def test_delete_idempotent(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "x", _b("d"))
    assert await ws.delete(_TENANT, WorkspaceScope.session, _SESSION, "x") is True
    assert await ws.delete(_TENANT, WorkspaceScope.session, _SESSION, "x") is False
    with pytest.raises(WorkspaceNotFoundError):
        await ws.read(_TENANT, WorkspaceScope.session, _SESSION, "x")


async def test_clear_scope_all(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "a", _b("1"))
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "b", _b("2"))
    removed = await ws.clear_scope(_TENANT, WorkspaceScope.session, _SESSION)
    assert removed == 2
    assert list(await ws.list(_TENANT, WorkspaceScope.session, _SESSION)) == []
    # idempotent: clearing an empty scope removes 0.
    assert await ws.clear_scope(_TENANT, WorkspaceScope.session, _SESSION) == 0


async def test_clear_scope_lifecycle_filter(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "s", _b("1"))
    await ws.write(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        "d",
        _b("2"),
        lifecycle=WorkspaceLifecycle.durable,
    )
    # clear only scratch (session-end teardown that keeps durable results).
    removed = await ws.clear_scope(
        _TENANT,
        WorkspaceScope.session,
        _SESSION,
        lifecycles=[WorkspaceLifecycle.scratch],
    )
    assert removed == 1
    remaining = await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    assert {r.path for r in remaining} == {"d"}


async def test_read_roundtrip_unit_type(ws: InMemoryWorkspace) -> None:
    await ws.write(_TENANT, WorkspaceScope.project, "p-1", "f.txt", _b("project body"))
    got = await ws.read(_TENANT, WorkspaceScope.project, "p-1", "f.txt")
    assert isinstance(got, WorkspaceUnit)
    assert got.scope is WorkspaceScope.project
    assert got.content == _b("project body")


async def test_knowledge_base_scope_is_separately_addressable(
    ws: InMemoryWorkspace,
) -> None:
    """A knowledge-base unit is a distinct address from the same path elsewhere."""
    await ws.write(_TENANT, WorkspaceScope.session, _SESSION, "guide.md", _b("scratch"))
    await ws.write(
        _TENANT,
        WorkspaceScope.knowledge_base,
        "kb-1",
        "wiki/guide.md",
        _b("kb body"),
        lifecycle=WorkspaceLifecycle.durable,
    )
    got = await ws.read(
        _TENANT, WorkspaceScope.knowledge_base, "kb-1", "wiki/guide.md"
    )
    assert got.scope is WorkspaceScope.knowledge_base
    assert got.content == _b("kb body")
    assert got.lifecycle is WorkspaceLifecycle.durable
    session_rows = {
        u.path for u in await ws.list(_TENANT, WorkspaceScope.session, _SESSION)
    }
    assert session_rows == {"guide.md"}


# --------------------------------------------------------------------------- #
# WorkspaceUnit binary content JSON round-trip (losslessness)
# --------------------------------------------------------------------------- #


def _binary_unit(content: bytes | None) -> WorkspaceUnit:
    now = datetime.now(UTC)
    return WorkspaceUnit(
        id="u",
        tenant_id="t",
        scope=WorkspaceScope.session,
        scope_key="s",
        path="b.bin",
        size_bytes=len(content or b""),
        sha256="0" * 64,
        content=content,
        created_at=now,
        updated_at=now,
    )


def test_workspace_unit_non_utf8_content_json_round_trip() -> None:
    """A non-UTF8 binary body must JSON-round-trip losslessly.

    Before the fix, ``model_dump_json()`` on a unit holding non-UTF8 bytes
    raised ``PydanticSerializationError: invalid utf-8 sequence`` (Pydantic v2
    UTF-8-decodes a ``bytes`` field in JSON mode). The unit documents binary
    bodies as supported and the contract invariant is a lossless
    serialize -> deserialize round-trip, so the body is now base64-encoded on
    JSON output and decoded back on input.
    """
    raw = bytes([0xFF, 0xFE, 0x00, 0x80])
    unit = _binary_unit(raw)
    # Must not raise.
    blob = unit.model_dump_json()
    rebuilt = WorkspaceUnit.model_validate_json(blob)
    assert rebuilt.content == raw


def test_workspace_unit_text_content_json_round_trip() -> None:
    """A UTF-8-decodable body still round-trips byte-for-byte."""
    unit = _binary_unit(b"hello world")
    rebuilt = WorkspaceUnit.model_validate_json(unit.model_dump_json())
    assert rebuilt.content == b"hello world"


def test_workspace_unit_none_content_json_round_trip() -> None:
    """The manifest-only (``content=None``) case stays ``None`` over JSON."""
    unit = _binary_unit(None)
    rebuilt = WorkspaceUnit.model_validate_json(unit.model_dump_json())
    assert rebuilt.content is None


def test_workspace_unit_python_mode_keeps_raw_bytes() -> None:
    """``model_dump()`` (python mode) leaves the body as raw ``bytes`` — only
    JSON output is base64-encoded (the host adapter writes to a BYTEA
    column, not via JSON, so it must keep the raw bytes)."""
    raw = bytes([0xFF, 0xFE, 0x00, 0x80])
    dumped = _binary_unit(raw).model_dump()
    assert dumped["content"] == raw
