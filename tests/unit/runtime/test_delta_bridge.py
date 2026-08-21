"""Tests for :mod:`protocore.runtime.llm.delta_bridge`."""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMStreamEvent,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.runtime.events import EventType
from protocore.runtime.llm.delta_bridge import (
    delta_to_turn_events,
    stream_events_to_provider_deltas,
)


async def _async_iter(items: list[LLMStreamEvent]) -> AsyncIterator[LLMStreamEvent]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_translate_text_delta_event() -> None:
    upstream = _async_iter([
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": "Hello"}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    kinds = [d.kind for d in deltas]
    assert ProviderDeltaKind.text in kinds
    assert any(d.is_block_end for d in deltas)
    finish = next((d for d in deltas if d.kind is ProviderDeltaKind.finish), None)
    assert finish is not None
    assert finish.finish_reason == "stop"


@pytest.mark.asyncio
async def test_translate_thinking_delta() -> None:
    upstream = _async_iter([
        LLMStreamEvent(name="content_block_delta", payload={"text": "...", "kind": "thinking"}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    thinking = [d for d in deltas if d.kind is ProviderDeltaKind.thinking]
    assert len(thinking) == 1
    assert thinking[0].content == "..."


@pytest.mark.asyncio
async def test_translate_tool_use_sequence() -> None:
    upstream = _async_iter([
        LLMStreamEvent(name="tool_use_start", payload={
            "tool_call_id": "t1", "tool_name": "Read",
        }),
        LLMStreamEvent(name="tool_use_input_delta", payload={
            "tool_call_id": "t1", "partial_input_json": "{\"path\":",
        }),
        LLMStreamEvent(name="tool_use_stop", payload={
            "tool_call_id": "t1", "final_input": {"path": "/x"},
        }),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "tool_use"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]

    kinds = [d.kind for d in deltas]
    assert ProviderDeltaKind.tool_use_start in kinds
    assert ProviderDeltaKind.tool_use_input in kinds
    assert ProviderDeltaKind.tool_use_stop in kinds
    stop = next(d for d in deltas if d.kind is ProviderDeltaKind.tool_use_stop)
    assert stop.tool_input_final == {"path": "/x"}


@pytest.mark.asyncio
async def test_translate_usage_event() -> None:
    upstream = _async_iter(
        [
            LLMStreamEvent(
                name="usage",
                payload={
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "response_cost_usd": 0.00042,
                    "provider_request_id": "gen-usage",
                },
            ),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "stop"}),
        ]
    )
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]

    usage = next(d for d in deltas if d.kind is ProviderDeltaKind.usage)
    assert usage.usage == {
        "input_tokens": 10,
        "output_tokens": 3,
        "response_cost_usd": 0.00042,
        "provider_request_id": "gen-usage",
    }


@pytest.mark.asyncio
async def test_tool_use_stop_preserves_truncated_by_output_cap_flag() -> None:
    """``truncated_by_output_cap`` payload flag survives the LLMStreamEvent →
    ProviderDelta translation.

    Dropping this flag would mask the mid-tool-call recovery branch
    in :func:`protocore.runtime.query._stream_one_assistant_message` for
    any adapter that emits the legacy event shape (LiteLLM adapter, Phase
    1 fixture mocks).
    """
    upstream = _async_iter([
        LLMStreamEvent(name="tool_use_start", payload={
            "tool_call_id": "t1", "tool_name": "Write",
        }),
        LLMStreamEvent(name="tool_use_stop", payload={
            "tool_call_id": "t1",
            "final_input": {"path": "/x"},
            "truncated_by_output_cap": True,
            "args_partial_truncated": True,
        }),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "length"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    stop = next(d for d in deltas if d.kind is ProviderDeltaKind.tool_use_stop)
    assert stop.truncated_by_output_cap is True
    assert stop.args_partial_truncated is True


@pytest.mark.asyncio
async def test_tool_use_stop_preserves_args_partial_truncated_without_length() -> None:
    """``args_partial_truncated`` is independent of
    ``finish_reason="length"`` — local models emit ``stop`` after only
    ``{`` of args. Bridge must preserve it on its own.
    """
    upstream = _async_iter([
        LLMStreamEvent(name="tool_use_start", payload={
            "tool_call_id": "t2", "tool_name": "Write",
        }),
        LLMStreamEvent(name="tool_use_stop", payload={
            "tool_call_id": "t2",
            "final_input": {},
            "truncated_by_output_cap": False,
            "args_partial_truncated": True,
        }),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "stop"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    stop = next(d for d in deltas if d.kind is ProviderDeltaKind.tool_use_stop)
    assert stop.truncated_by_output_cap is False
    assert stop.args_partial_truncated is True


@pytest.mark.asyncio
async def test_tool_use_stop_defaults_truncation_flags_false_when_absent() -> None:
    """Backward-compat — payload without truncation flags yields defaults
    (``False``) on the synthesised :class:`ProviderDelta`. Older fixture
    mocks without truncation flags still work.
    """
    upstream = _async_iter([
        LLMStreamEvent(name="tool_use_start", payload={
            "tool_call_id": "t3", "tool_name": "Read",
        }),
        LLMStreamEvent(name="tool_use_stop", payload={
            "tool_call_id": "t3",
            "final_input": {"path": "/x"},
        }),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "tool_use"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    stop = next(d for d in deltas if d.kind is ProviderDeltaKind.tool_use_stop)
    assert stop.truncated_by_output_cap is False
    assert stop.args_partial_truncated is False


@pytest.mark.asyncio
async def test_unknown_event_dropped() -> None:
    upstream = _async_iter([
        LLMStreamEvent(name="some_future_event", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ])
    deltas = [d async for d in stream_events_to_provider_deltas(upstream)]
    # Only finish event should remain.
    assert len(deltas) == 1
    assert deltas[0].kind is ProviderDeltaKind.finish


def test_delta_to_turn_events_text() -> None:
    delta = ProviderDelta(kind=ProviderDeltaKind.text, content="hi")
    events = delta_to_turn_events(delta, run_id="r1", turn_id="t1", block_idx=0)
    assert len(events) == 1
    assert events[0].type is EventType.CONTENT_BLOCK_DELTA
    assert events[0].payload["delta"]["text"] == "hi"


def test_delta_to_turn_events_thinking() -> None:
    delta = ProviderDelta(kind=ProviderDeltaKind.thinking, content="thinking…")
    events = delta_to_turn_events(delta, run_id="r1", turn_id="t1", block_idx=0)
    assert events[0].payload["delta"]["type"] == "thinking_delta"


def test_delta_to_turn_events_tool_use_start() -> None:
    delta = ProviderDelta(
        kind=ProviderDeltaKind.tool_use_start,
        tool_call_id="t1",
        tool_name="Read",
    )
    events = delta_to_turn_events(delta, run_id="r1", turn_id="t1", block_idx=2)
    assert events[0].type is EventType.TOOL_USE_START
    assert events[0].payload["tool_call_id"] == "t1"
    assert events[0].payload["tool_name"] == "Read"


def test_delta_to_turn_events_empty_content_skipped() -> None:
    delta = ProviderDelta(kind=ProviderDeltaKind.text, content="")
    events = delta_to_turn_events(delta, run_id="r1", turn_id="t1", block_idx=0)
    assert events == []


def test_delta_to_turn_events_finish_yields_nothing() -> None:
    """The loop handles ``finish`` separately — bridge doesn't surface it."""
    delta = ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")
    events = delta_to_turn_events(delta, run_id="r1", turn_id="t1", block_idx=0)
    assert events == []
