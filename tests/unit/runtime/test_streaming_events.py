"""Tests for :mod:`protocore.runtime.events` taxonomy + envelope."""
from __future__ import annotations

import pytest

from protocore.runtime.events import (
    EventType,
    InMemoryTurnEventBuffer,
    TurnEvent,
)


def test_event_type_anthropic_subset_present() -> None:
    """All 10 mandatory Anthropic-aligned types are members."""
    required = {
        "message_start",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "tool_use_start",
        "tool_use_input_delta",
        "tool_use_stop",
        "tool_result",
        "error",
    }
    members = {e.value for e in EventType}
    assert required.issubset(members)


def test_event_type_protocore_extensions_present() -> None:
    """7 Protocore extensions are members."""
    required = {
        "sandbox_starting",
        "sandbox_ready",
        "subagent_spawn",
        "subagent_complete",
        "hook_fired",
        "tool_call_pending",
        "state_changed",
    }
    members = {e.value for e in EventType}
    assert required.issubset(members)


def test_event_type_verification_lifecycle_present() -> None:
    required = {
        "candidate_ready",
        "verification_started",
        "verification_reported",
        "repair_requested",
        "release_decided",
    }
    assert required.issubset({event.value for event in EventType})


def test_turn_event_frozen() -> None:
    """TurnEvent envelope is frozen."""
    evt = TurnEvent(type=EventType.MESSAGE_START, run_id="r1", payload={"k": "v"})
    with pytest.raises(Exception):  # pydantic frozen → ValidationError
        evt.run_id = "other"  # type: ignore[misc]


def test_turn_event_to_event_preserves_payload() -> None:
    """TurnEvent.to_event() flattens payload + propagates name."""
    evt = TurnEvent(
        type=EventType.MESSAGE_START,
        run_id="r1",
        payload={"model": "qwen3.6-35b-a3b"},
    )
    durable = evt.to_event()
    assert durable.name == "message_start"
    assert durable.payload["model"] == "qwen3.6-35b-a3b"
    assert durable.payload["schema_version"] == 1
    assert "server_ts_ms" in durable.payload


def test_turn_event_default_id_is_uuid() -> None:
    evt = TurnEvent(type=EventType.HEARTBEAT, run_id="r1")
    assert evt.id  # non-empty
    assert "-" in evt.id  # uuid format


def test_inmem_buffer_captures_in_order() -> None:
    buf = InMemoryTurnEventBuffer()
    buf.append(TurnEvent(type=EventType.MESSAGE_START, run_id="r1"))
    buf.append(TurnEvent(type=EventType.MESSAGE_STOP, run_id="r1"))
    assert buf.types() == ["message_start", "message_stop"]
    assert len(buf.events) == 2


def test_inmem_buffer_clear() -> None:
    buf = InMemoryTurnEventBuffer()
    buf.append(TurnEvent(type=EventType.HEARTBEAT, run_id="r1"))
    buf.clear()
    assert buf.events == ()
