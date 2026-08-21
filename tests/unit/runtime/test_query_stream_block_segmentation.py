"""Per-kind content-block segmentation in ``_drive_one_stream``.

A thinking delta and a text delta MUST land in SEPARATE typed content
blocks. The old single ``current_text_block_open`` boolean fixed the
block ``kind`` to whichever delta arrived first and never re-opened on a
kind transition, so the canonical reasoning-then-answer stream emitted
the visible answer's ``text_delta`` events under a block opened as
``kind=thinking`` — the chat reducer (which renders blocks strictly by
their ``content_block_start`` kind) silently dropped every answer delta
live, while durable history (per-kind buffers) stayed correct.

Contract guarded here: every ``content_block_delta`` between a
``content_block_start`` and its ``content_block_stop`` matches the kind
the block was opened with, and a kind transition closes the old block
and opens a NEW one with a fresh ``block_idx``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query import (
    _drive_one_stream,
    _rebuild_context_for_recovery,
    _StreamAttemptResult,
)

_DELTA_TYPE_FOR_KIND = {"thinking": "thinking_delta", "text": "text_delta"}


def _user_message(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _make_provider_deltas(
    deltas: list[ProviderDelta],
):
    """Return a ``stream_with_tools(request)`` stub yielding ``deltas``."""

    def _stream_with_tools(request: object) -> AsyncIterator[ProviderDelta]:
        del request

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in deltas:
                yield delta

        return _gen()

    return _stream_with_tools


async def _collect_events(engine) -> list[TurnEvent]:
    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    return [evt async for evt in _drive_one_stream(engine, context, result)]


def _block_events(events: list[TurnEvent]) -> list[TurnEvent]:
    return [
        evt
        for evt in events
        if evt.type
        in (
            EventType.CONTENT_BLOCK_START,
            EventType.CONTENT_BLOCK_DELTA,
            EventType.CONTENT_BLOCK_STOP,
        )
    ]


def _assert_blocks_are_single_kind(events: list[TurnEvent]) -> None:
    """Every delta inside a block must match the kind the block opened with."""
    open_kind_by_idx: dict[int, str] = {}
    for evt in _block_events(events):
        idx = evt.payload["block_idx"]
        if evt.type is EventType.CONTENT_BLOCK_START:
            assert idx not in open_kind_by_idx, f"block {idx} re-opened"
            open_kind_by_idx[idx] = evt.payload["kind"]
        elif evt.type is EventType.CONTENT_BLOCK_DELTA:
            assert idx in open_kind_by_idx, f"delta for unopened block {idx}"
            expected = _DELTA_TYPE_FOR_KIND[open_kind_by_idx[idx]]
            assert evt.payload["delta"]["type"] == expected, (
                f"block {idx} opened kind={open_kind_by_idx[idx]} but carries "
                f"{evt.payload['delta']['type']}"
            )
        elif evt.type is EventType.CONTENT_BLOCK_STOP:
            assert idx in open_kind_by_idx, f"stop for unopened block {idx}"
            del open_kind_by_idx[idx]
    assert not open_kind_by_idx, f"blocks left open: {sorted(open_kind_by_idx)}"


@pytest.mark.asyncio
async def test_thinking_then_text_lands_in_two_typed_blocks(engine_factory) -> None:
    """Canonical reasoning-then-answer shape — the answer MUST open its own
    ``kind=text`` block instead of riding the ``kind=thinking`` one."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="let me "),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="think"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="the "),
        ProviderDelta(kind=ProviderDeltaKind.text, content="answer"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    stops = [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]
    assert [e.payload["kind"] for e in starts] == ["thinking", "text"]
    assert starts[0].payload["block_idx"] != starts[1].payload["block_idx"]
    assert len(stops) == 2
    # The thinking block closes BEFORE the text block opens.
    block_seq = [
        (e.type, e.payload["block_idx"]) for e in _block_events(events)
    ]
    thinking_idx = starts[0].payload["block_idx"]
    text_idx = starts[1].payload["block_idx"]
    assert block_seq.index(
        (EventType.CONTENT_BLOCK_STOP, thinking_idx)
    ) < block_seq.index((EventType.CONTENT_BLOCK_START, text_idx))
    _assert_blocks_are_single_kind(events)


@pytest.mark.asyncio
async def test_text_then_thinking_does_not_retag_text_block(engine_factory) -> None:
    """Reverse order — already-streamed answer text must not be swallowed
    into a thinking block; the thinking deltas open a new block."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content="answer"),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="afterthought"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    assert [e.payload["kind"] for e in starts] == ["text", "thinking"]
    assert starts[0].payload["block_idx"] != starts[1].payload["block_idx"]
    _assert_blocks_are_single_kind(events)


@pytest.mark.asyncio
async def test_interleaved_kinds_open_a_block_per_run(engine_factory) -> None:
    """Each contiguous same-kind run gets exactly one block."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="t1"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="a1"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="a2"),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="t2"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="a3"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    events = [evt async for evt in _drive_one_stream(engine, context, result)]

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    assert [e.payload["kind"] for e in starts] == [
        "thinking",
        "text",
        "thinking",
        "text",
    ]
    assert len({e.payload["block_idx"] for e in starts}) == 4
    _assert_blocks_are_single_kind(events)
    # Durable buffers stay per-kind regardless of block segmentation.
    assert result.text_buffer == "a1a2a3"
    assert result.reasoning_buffer == "t1t2"


@pytest.mark.asyncio
async def test_single_kind_stream_keeps_one_block(engine_factory) -> None:
    """No kind transition — behaviour unchanged: one block, one stop."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content="a"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="b"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    stops = [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]
    assert len(starts) == 1
    assert starts[0].payload["kind"] == "text"
    assert len(stops) == 1
    _assert_blocks_are_single_kind(events)


@pytest.mark.asyncio
async def test_tool_use_after_thinking_closes_open_block(engine_factory) -> None:
    """A tool_use_start after thinking deltas still closes the open block."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="plan"),
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_start,
            tool_call_id="call-1",
            tool_name="Read",
        ),
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_stop,
            tool_call_id="call-1",
            tool_input_final={"path": "x"},
        ),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    stops = [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]
    assert len(starts) == 1
    assert starts[0].payload["kind"] == "thinking"
    assert len(stops) == 1
    assert stops[0].payload["block_idx"] == starts[0].payload["block_idx"]
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert len(tool_starts) == 1
    assert tool_starts[0].payload["block_idx"] != starts[0].payload["block_idx"]


@pytest.mark.asyncio
async def test_text_after_tool_use_start_uses_fresh_block_idx(
    engine_factory,
) -> None:
    """A text delta arriving after ``tool_use_start`` (and BEFORE the matching
    ``tool_use_stop``) must open its own content block with a FRESH
    ``block_idx`` — the OpenAI wire permits interleaved ``delta.content``
    while a tool call is buffered open, so the reopen path must not reuse
    the tool block's idx. Without this, the wire emits two
    ``content_block_start`` events for the same ``block_idx`` (block-model
    violation) and the chat reducer replaces the ``tool_use`` placeholder
    with the text block.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_start,
            tool_call_id="call-1",
            tool_name="Read",
        ),
        # Text delta arrives WHILE the tool call is buffered open —
        # no tool_use_stop, no finish yet. This is the canonical
        # OpenAI interleaved shape.
        ProviderDelta(kind=ProviderDeltaKind.text, content="oh "),
        ProviderDelta(kind=ProviderDeltaKind.text, content="hi"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    text_starts = [
        e
        for e in events
        if e.type is EventType.CONTENT_BLOCK_START
        and e.payload["kind"] == "text"
    ]
    assert len(tool_starts) == 1
    assert len(text_starts) == 1
    # The text block must have a DIFFERENT block_idx from the tool block.
    assert text_starts[0].payload["block_idx"] != tool_starts[0].payload["block_idx"]
    _assert_blocks_are_single_kind(events)


@pytest.mark.asyncio
async def test_thinking_after_tool_use_stop_uses_fresh_block_idx(
    engine_factory,
) -> None:
    """A thinking delta arriving after a COMPLETE tool call (post
    ``tool_use_stop``) also gets a fresh ``block_idx``. Guards the
    ``tool_use_stop`` branch's ``next_block_idx()`` advance for the
    "text after tool stop" variant.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_start,
            tool_call_id="call-1",
            tool_name="Read",
        ),
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_stop,
            tool_call_id="call-1",
            tool_input_final={"path": "x"},
        ),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="afterthought"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    events = await _collect_events(engine)

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    # Two typed content blocks (tool_use_start has no CONTENT_BLOCK_START).
    assert [e.payload["kind"] for e in starts] == ["thinking"]
    assert len(tool_starts) == 1
    # The thinking block's block_idx must differ from the tool's.
    assert starts[0].payload["block_idx"] != tool_starts[0].payload["block_idx"]
    _assert_blocks_are_single_kind(events)
