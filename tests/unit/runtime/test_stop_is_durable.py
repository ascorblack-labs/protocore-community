"""A requested cancellation is a durable fact of the run.

``stop()`` used to set a per-process ``asyncio.Event`` and nothing else, so a
cancellation landing moments before the process died was simply lost: the run
was picked up elsewhere and carried on spending budget on work the operator had
already called off. The flag now rides in the snapshot alongside the other
one-way run facts, and the resumed turn's first stop-check acts on it.
"""
from __future__ import annotations

from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.events.envelope import EventType
from protocore.runtime.loop_state import LoopState


def test_stop_is_in_the_snapshot(engine_factory) -> None:
    engine = engine_factory()
    assert engine.snapshot()["stop_requested"] is False

    engine.stop()

    assert engine.snapshot()["stop_requested"] is True


async def test_stop_survives_a_cold_resume(engine_factory) -> None:
    source = engine_factory()
    source.stop()

    resumed = engine_factory()
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.stop_requested is True


async def test_a_run_without_a_stop_resumes_runnable(engine_factory) -> None:
    source = engine_factory()

    resumed = engine_factory()
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed.stop_requested is False


async def test_a_snapshot_taken_before_the_stop_does_not_clear_it(
    engine_factory,
) -> None:
    """The flag is one-way: restoring never un-cancels a live cancellation."""
    source = engine_factory()
    snapshot = source.snapshot()

    resumed = engine_factory()
    resumed.stop()
    await resumed.resume_from_snapshot(snapshot)

    assert resumed.stop_requested is True


async def test_the_resumed_turn_cancels_immediately(engine_factory) -> None:
    """Checked right after restore: the first turn ends CANCELLED, not working."""
    source = engine_factory()
    source.history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ]
    source.state = LoopState.RUNNING
    source.stop()

    resumed = engine_factory()
    await resumed.resume_from_snapshot(source.snapshot())

    from protocore.runtime.query import _query as query

    events = [evt async for evt in query(resumed)]

    assert resumed.state is LoopState.CANCELLED
    assert any(
        evt.type is EventType.MESSAGE_STOP
        and evt.payload.get("stop_reason") == StopReason.cancelled.value
        for evt in events
    )
