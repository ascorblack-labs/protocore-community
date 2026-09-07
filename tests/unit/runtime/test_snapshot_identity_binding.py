"""A snapshot may only be restored into the run it belongs to.

Restoring used to check exactly one field — the verification delivery mode —
and the run identity only inside the optional ``verification`` key, which the
overwhelming majority of runs never carry. So a snapshot of run A restored into
run B was accepted in silence: B took A's history, turn count and usage, and
nothing on the wire or in the logs said so. In a multi-scope installation the
worst reading of that is one scope's conversation appearing in another's
context, and the only thing standing between the two was the correctness of the
key the caller read the snapshot under.

The binding is now unconditional and fail-closed, and it runs before anything
is applied.
"""
from __future__ import annotations

import pytest

from protocore.contracts.types import Message, MessageRole, TextBlock


def _identity_fields() -> tuple[str, ...]:
    return (
        "run_id",
        "tenant_id",
        "session_id",
        "root_run_id",
        "parent_run_id",
        "subagent_id",
    )


async def test_a_matching_snapshot_still_restores(engine_factory) -> None:
    source = engine_factory(
        run_id="run-1",
        tenant_id="scope-1",
        session_id="session-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    source.turn_count = 4

    resumed = engine_factory(
        run_id="run-1",
        tenant_id="scope-1",
        session_id="session-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.turn_count == 4


async def test_a_foreign_snapshot_is_refused(engine_factory) -> None:
    source = engine_factory(
        run_id="source-run",
        tenant_id="source-scope",
        session_id="source-session",
    )
    source.history = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="another tenant's words")],
        )
    ]
    source.turn_count = 7

    resumed = engine_factory(
        run_id="other-run",
        tenant_id="other-scope",
        session_id="other-session",
    )
    with pytest.raises(ValueError, match="run_id"):
        await resumed.resume_from_snapshot(source.snapshot())

    # Nothing crossed: no history, no counters, no usage.
    assert resumed.history == []
    assert resumed.turn_count == 0


@pytest.mark.parametrize(
    ("field_name", "foreign_value"),
    [
        ("run_id", "someone-elses-run"),
        ("tenant_id", "someone-elses-scope"),
        ("session_id", "someone-elses-session"),
        ("root_run_id", "someone-elses-root"),
        ("parent_run_id", "someone-elses-parent"),
        ("subagent_id", "someone-elses-worker"),
    ],
)
async def test_each_identity_field_is_checked(
    engine_factory, field_name: str, foreign_value: str
) -> None:
    source = engine_factory(
        run_id="run-1",
        tenant_id="scope-1",
        session_id="session-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    snapshot = source.snapshot()
    snapshot[field_name] = foreign_value

    resumed = engine_factory(
        run_id="run-1",
        tenant_id="scope-1",
        session_id="session-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    with pytest.raises(ValueError, match=field_name):
        await resumed.resume_from_snapshot(snapshot)


@pytest.mark.parametrize("field_name", _identity_fields())
async def test_a_missing_identity_field_is_refused(
    engine_factory, field_name: str
) -> None:
    """Absence is a mismatch: an unbound snapshot has not been bound."""
    source = engine_factory(
        run_id="run-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    snapshot = source.snapshot()
    del snapshot[field_name]

    resumed = engine_factory(
        run_id="run-1",
        root_run_id="root-1",
        parent_run_id="parent-1",
        subagent_id="worker-1",
    )
    with pytest.raises(ValueError, match=field_name):
        await resumed.resume_from_snapshot(snapshot)


async def test_a_child_run_snapshot_is_refused_by_a_sibling(engine_factory) -> None:
    """The tree axes bind too, with no ``verification`` key in the snapshot."""
    source = engine_factory(
        run_id="child-a",
        root_run_id="root-a",
        parent_run_id="root-a",
        subagent_id="worker-a",
    )
    snapshot = source.snapshot()
    assert "verification" not in snapshot

    sibling = engine_factory(
        run_id="child-b",
        root_run_id="root-b",
        parent_run_id="root-b",
        subagent_id="worker-b",
    )
    with pytest.raises(ValueError):
        await sibling.resume_from_snapshot(snapshot)

    assert sibling.config.root_run_id == "root-b"
    assert sibling.config.subagent_id == "worker-b"


async def test_the_model_name_is_not_an_identity_field(engine_factory) -> None:
    """It is the outcome of a one-way demotion, so it is applied, not compared."""
    source = engine_factory(model_name="demoted-model")

    resumed = engine_factory(model_name="whatever-this-process-started-on")
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.config.model_name == "demoted-model"
