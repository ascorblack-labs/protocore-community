"""A snapshot states its schema, and a reader that cannot place it refuses.

The snapshot crosses a process boundary, and the two processes need not be the
same build. Without a version in the payload the reader has no way to tell a
field that was absent because the writer never had it from a field that was
absent because the run legitimately had nothing there — and the two lead to
opposite resumes. One carries a spent budget forward; the other hands the run a
fresh one.

So the version is written into every snapshot and checked before anything is
restored. The check is fail-closed and it runs FIRST: earlier than the delivery
binding, earlier than the run identity, earlier than any mutation of the engine.
A refused snapshot leaves the run exactly where it was, which is the outcome an
operator can act on.
"""
from __future__ import annotations

import pytest

from protocore.contracts.snapshot import (
    RUN_SCOPED_STATE_SNAPSHOT_KEY,
    SNAPSHOT_SCHEMA_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotSchemaError,
    migrate_snapshot,
    read_schema_version,
)
from protocore.contracts.types import Message, MessageRole, TextBlock


def test_a_written_snapshot_states_its_schema(engine_factory) -> None:
    engine = engine_factory()

    assert engine.snapshot()[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION


def test_a_snapshot_of_this_build_is_accepted(engine_factory) -> None:
    engine = engine_factory()
    payload = engine.snapshot()

    assert migrate_snapshot(payload)[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION


def test_a_payload_without_a_version_reads_as_version_one() -> None:
    """The field was added on top of exactly one shape, and a payload without
    it is that shape — the only one a store can hold with no version in it."""
    assert read_schema_version({"run_id": "run-1"}) == 1


@pytest.mark.parametrize("value", ["1", 1.0, None, [1], True])
def test_a_version_that_is_not_an_integer_is_refused(value: object) -> None:
    """``True`` is in this list on purpose: ``bool`` is an ``int`` subclass, so
    a payload carrying ``schema_version: true`` would otherwise compare equal to
    version 1 and be accepted by the build that happens to sit there."""
    with pytest.raises(SnapshotSchemaError):
        read_schema_version({SNAPSHOT_SCHEMA_KEY: value})


@pytest.mark.parametrize("value", [0, -1])
def test_a_version_below_one_is_refused(value: int) -> None:
    with pytest.raises(SnapshotSchemaError, match="at least 1"):
        read_schema_version({SNAPSHOT_SCHEMA_KEY: value})


def test_a_non_mapping_is_refused() -> None:
    with pytest.raises(SnapshotSchemaError, match="mapping"):
        read_schema_version(["run_id", "run-1"])  # type: ignore[arg-type]


def test_a_newer_schema_is_refused_rather_than_read_partially() -> None:
    """The fields a newer build added are precisely the ones this reader would
    drop on the floor, and dropping them is the failure the version prevents."""
    with pytest.raises(SnapshotSchemaError, match="newer build"):
        migrate_snapshot({SNAPSHOT_SCHEMA_KEY: SNAPSHOT_SCHEMA_VERSION + 1})


async def test_resume_reads_a_snapshot_written_before_the_version_field(engine_factory) -> None:
    """The shape every store already held. A run paused when this shipped is
    waiting on a person, and refusing its payload would leave it paused for
    good — no later build would accept it either."""
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="whose turn?")])
    )
    source.turn_count = 7
    payload = source.snapshot()
    # The shape that predates the version field named the run's two allowances
    # at the top level.
    run_state_payload = payload.pop(RUN_SCOPED_STATE_SNAPSHOT_KEY)
    payload["run_work_ledger"] = run_state_payload["run_work_ledger"]
    payload["subagent_tree_budget"] = run_state_payload["subagent_tree_budget"]
    del payload[SNAPSHOT_SCHEMA_KEY]

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    await resumed.resume_from_snapshot(payload)

    assert resumed.turn_count == 7
    assert len(resumed.history) == 1


async def test_resume_refuses_a_snapshot_from_a_newer_build(engine_factory) -> None:
    source = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    source.turn_count = 3
    payload = source.snapshot()
    payload[SNAPSHOT_SCHEMA_KEY] = SNAPSHOT_SCHEMA_VERSION + 5

    resumed = engine_factory(run_id="run-1", tenant_id="scope-1", session_id="session-1")
    with pytest.raises(SnapshotSchemaError, match="newer build"):
        await resumed.resume_from_snapshot(payload)

    assert resumed.turn_count == 0


async def test_the_schema_is_checked_before_the_run_identity(engine_factory) -> None:
    """Order matters. Identity is read by field name, and what a field name
    means is a property of the schema — so an unreadable schema must not get as
    far as being told its identity is wrong, which would send an operator
    looking for the wrong fault."""
    source = engine_factory(run_id="source-run", tenant_id="scope-1", session_id="session-1")
    payload = source.snapshot()
    payload[SNAPSHOT_SCHEMA_KEY] = SNAPSHOT_SCHEMA_VERSION + 1

    resumed = engine_factory(run_id="other-run", tenant_id="scope-2", session_id="session-2")
    with pytest.raises(SnapshotSchemaError) as excinfo:
        await resumed.resume_from_snapshot(payload)

    assert "identity" not in str(excinfo.value)
