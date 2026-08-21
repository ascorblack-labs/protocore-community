"""Tests for :class:`EventBus`."""
from __future__ import annotations

import asyncio

from protocore.events import EventBus, EventName


async def test_subscribe_and_publish_sync_handler() -> None:
    bus = EventBus()
    received: list[dict[str, int]] = []

    def handler(payload: dict[str, int]) -> None:
        received.append(payload)

    bus.subscribe(EventName.tool_call_start, handler)
    await bus.publish(EventName.tool_call_start, {"x": 1})
    assert received == [{"x": 1}]


async def test_subscribe_and_publish_async_handler() -> None:
    bus = EventBus()
    received: list[dict[str, int]] = []

    async def handler(payload: dict[str, int]) -> None:
        received.append(payload)

    bus.subscribe(EventName.tool_call_end, handler)
    await bus.publish(EventName.tool_call_end, {"y": 2})
    assert received == [{"y": 2}]


async def test_handler_exception_isolated() -> None:
    bus = EventBus()
    received: list[int] = []

    def bad_handler(_payload: dict[str, int]) -> None:
        raise RuntimeError("boom")

    def good_handler(payload: dict[str, int]) -> None:
        received.append(payload["v"])

    bus.subscribe(EventName.error, bad_handler)
    bus.subscribe(EventName.error, good_handler)
    await bus.publish(EventName.error, {"v": 7})
    assert received == [7]


async def test_fanout_10_subscribers_1000_events() -> None:
    bus = EventBus()
    counts = [0] * 10

    def make_handler(idx: int):
        def handler(_payload: dict[str, int]) -> None:
            counts[idx] += 1

        return handler

    for i in range(10):
        bus.subscribe(EventName.turn_start, make_handler(i))

    await asyncio.gather(
        *[bus.publish(EventName.turn_start, {"i": i}) for i in range(1000)]
    )
    assert counts == [1000] * 10


def test_unsubscribe() -> None:
    bus = EventBus()

    def handler(_payload: dict[str, int]) -> None:
        pass

    bus.subscribe(EventName.session_start, handler)
    assert bus.subscriber_count(EventName.session_start) == 1
    assert bus.unsubscribe(EventName.session_start, handler) is True
    assert bus.subscriber_count(EventName.session_start) == 0
    assert bus.unsubscribe(EventName.session_start, handler) is False


def test_event_name_has_expected_categories() -> None:
    names = {n.value for n in EventName}
    # Lifecycle
    assert "session_start" in names
    assert "session_end" in names
    assert "run_started" in names
    # Streaming
    assert "message_start" in names
    assert "content_block_delta" in names
    # Compaction
    assert "compaction_routine_start" in names
    assert "compaction_emergency_start" in names
    # Sandbox
    assert "sandbox_starting" in names
    # Subagent
    assert "subagent_spawn" in names
    # Hooks
    assert "hook_fired" in names
    # GC
    assert "gc_started" in names
    # Audit
    assert "audit_emit" in names


def test_dead_v1_events_dropped() -> None:
    """v1 event names tied to dead subsystems must NOT exist in v2 enum."""
    names = {n.value for n in EventName}
    forbidden = {
        # Ensemble / consensus
        "ev_ensemble_consensus_reached",
        "ev_ensemble_consensus_failed",
        # Causal trace
        "ev_causal_trace_recorded",
        # Knowledge capture
        "ev_knowledge_recorded",
        # Pattern #N
        "ev_task_constraint_violation",
        "ev_invariant_violation",
        "ev_drift_detected",
        "ev_anchor_preserved",
        "ev_fallback_step_triggered",
        "ev_fallback_exhausted",
        "ev_result_salvaged",
        "ev_pipeline_warming_started",
        "ev_pipeline_warming_hit",
        # Multi-agent parallel
        "ev_parallel_conflict_detected",
    }
    leaked = names & forbidden
    assert not leaked, f"v1 dead event names leaked into v2: {leaked}"
