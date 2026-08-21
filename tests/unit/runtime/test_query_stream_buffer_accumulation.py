"""Streamed text/reasoning buffer accumulation correctness.

Guards the de-quadratic refactor of ``_drive_one_stream`` (which replaced
per-delta ``result.text_buffer += ...`` / ``result.reasoning_buffer += ...``
string concatenation with ``list[str]`` accumulation joined once at
end-of-stream). The observable contract MUST be unchanged: the final
``text_buffer`` / ``reasoning_buffer`` equal the concatenation of every
emitted ``text`` / ``thinking`` delta fragment, in order.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.query import (
    _drive_one_stream,
    _rebuild_context_for_recovery,
    _StreamAttemptResult,
)


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


@pytest.mark.asyncio
async def test_text_buffer_joins_many_fragments_in_order(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    fragments = [f"chunk-{i}-" for i in range(500)]
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content=frag) for frag in fragments
    ]
    deltas.append(ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"))
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    assert result.text_buffer == "".join(fragments)
    assert result.reasoning_buffer == ""
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_reasoning_buffer_joins_fragments_independently(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="re"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="A"),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="ason"),
        ProviderDelta(kind=ProviderDeltaKind.text, content="B"),
        ProviderDelta(kind=ProviderDeltaKind.thinking, content="ing"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    assert result.text_buffer == "AB"
    assert result.reasoning_buffer == "reasoning"


@pytest.mark.asyncio
async def test_empty_and_none_content_fragments_are_ignored(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content="x"),
        ProviderDelta(kind=ProviderDeltaKind.text, content=None),
        ProviderDelta(kind=ProviderDeltaKind.text, content=""),
        ProviderDelta(kind=ProviderDeltaKind.text, content="y"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    assert result.text_buffer == "xy"
