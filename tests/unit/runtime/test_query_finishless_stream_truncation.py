"""A finish-less stream (SSE tail-loss) must not COMPLETE as a clean
end_turn with the truncated prefix persisted as the final answer.

``_StreamAttemptResult.finish_reason`` is set ONLY by a
``ProviderDeltaKind.finish`` delta. The empirically-observed OpenRouter SSE
tail-loss shape ends the upstream iterator cleanly (``data: [DONE]`` / EOF)
with NO finish delta, so ``finish_reason`` stays ``None``. Before the fix
``_stream_one_assistant_message`` had no ``None`` handling and treated the
attempt as a normal completion, so a mid-sentence partial with no tool calls
fell into the no-tool ``end_turn`` branch and the run COMPLETED with the
truncated text scored as the final answer.

Secondary defect: ``_drive_one_stream`` left ``current_text_block`` open (no
``CONTENT_BLOCK_STOP``) on this exit (and on the ``stop_requested`` break).

Contract guarded here:
  * unit — a finish-less text stream leaves ``finish_reason is None`` AND every
    opened content block is closed (the dangling-block defect);
  * loop — a finish-less truncated no-tool turn drives the bounded resume
    recovery (re-stream so the model finishes) instead of completing with the
    truncated prefix;
  * loop — when the recovery budget is exhausted the run goes TERMINAL
    (``output_length_exhausted``), never a silent truncated completion.

Mirrors the ``ProviderDelta``/`_drive_one_stream` harness from
``test_query_stream_block_segmentation.py`` and the engine-builder + scripted
``stream_with_tools`` pattern from ``test_resilience_recovery_loop.py``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMRequest,
    LLMStreamEvent,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _drive_one_stream,
    _rebuild_context_for_recovery,
    _StreamAttemptResult,
)


def _user_message(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _make_provider_deltas(deltas: list[ProviderDelta]):
    """Return a ``stream_with_tools(request)`` stub yielding ``deltas``."""

    def _stream_with_tools(request: object) -> AsyncIterator[ProviderDelta]:
        del request

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in deltas:
                yield delta

        return _gen()

    return _stream_with_tools


def _final_assistant_text(engine) -> str:
    """Concatenated text of the LAST assistant turn in durable history."""
    for msg in reversed(engine.history):
        if msg.role is MessageRole.assistant:
            return "".join(
                b.text for b in msg.content_blocks if isinstance(b, TextBlock)
            )
    return ""


# ---------------------------------------------------------------------------
# Unit — _drive_one_stream on a finish-less stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finishless_stream_leaves_finish_reason_none(engine_factory) -> None:
    """A text stream that ends with NO ``finish`` delta (clean EOF / ``[DONE]``)
    leaves ``finish_reason is None`` — the signal the caller must treat as a
    truncated, incomplete turn (not a normal completion)."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("write a long answer"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content="The answer begins "),
        ProviderDelta(kind=ProviderDeltaKind.text, content="and then it is cut "),
        # NO ProviderDeltaKind.finish — the iterator just ends (tail-loss).
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    events = [evt async for evt in _drive_one_stream(engine, context, result)]

    assert result.finish_reason is None
    assert result.text_buffer == "The answer begins and then it is cut "
    assert result.tool_calls == []
    # Sanity: the events did carry the visible text deltas.
    assert any(e.type is EventType.CONTENT_BLOCK_DELTA for e in events)


@pytest.mark.asyncio
async def test_finishless_stream_closes_open_content_block(engine_factory) -> None:
    """Secondary defect — a finish-less stream must still emit a
    ``CONTENT_BLOCK_STOP`` for the block it opened. Before the fix the block
    was left dangling (no matching stop), so the chat reducer never closed it.

    FAILS before the fix: the lone CONTENT_BLOCK_START has no matching STOP.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    deltas = [
        ProviderDelta(kind=ProviderDeltaKind.text, content="partial "),
        ProviderDelta(kind=ProviderDeltaKind.text, content="answer"),
        # NO finish delta — clean EOF.
    ]
    engine.llm.stream_with_tools = _make_provider_deltas(deltas)  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    events = [evt async for evt in _drive_one_stream(engine, context, result)]

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    stops = [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]
    assert len(starts) == 1
    # Every opened block is closed exactly once (no dangling block).
    assert len(stops) == 1
    assert starts[0].payload["block_idx"] == stops[0].payload["block_idx"]


@pytest.mark.asyncio
async def test_stop_requested_break_closes_open_content_block(
    engine_factory,
) -> None:
    """The ``stop_requested`` per-delta break is the OTHER finish-less exit; it
    must also close the open block. The interrupt lands after the first text
    delta opened the block."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))

    def _stream_with_tools(request: object) -> AsyncIterator[ProviderDelta]:
        del request

        async def _gen() -> AsyncIterator[ProviderDelta]:
            yield ProviderDelta(kind=ProviderDeltaKind.text, content="opening...")
            # Cancel lands mid-stream; the next per-delta check breaks the loop.
            engine.stop()
            yield ProviderDelta(kind=ProviderDeltaKind.text, content="more")
            yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")

        return _gen()

    engine.llm.stream_with_tools = _stream_with_tools  # type: ignore[method-assign]

    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    events = [evt async for evt in _drive_one_stream(engine, context, result)]

    starts = [e for e in events if e.type is EventType.CONTENT_BLOCK_START]
    stops = [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]
    assert len(starts) == 1
    assert len(stops) == 1
    # The break happened before the finish delta — finish_reason stays None.
    assert result.finish_reason is None


# ---------------------------------------------------------------------------
# Loop — end-to-end via engine.run
# ---------------------------------------------------------------------------


class _TruncatedThenCompleteLLM:
    """Stream 1: a mid-sentence prefix then NO ``message_stop`` (SSE tail-loss).
    Stream 2 (after the resume nudge): a complete answer that finishes cleanly.
    """

    def __init__(self, *, prefix: str, full: str) -> None:
        self._prefix = prefix
        self._full = full
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if len(self.calls) == 1:
            # Truncated prefix, NO message_stop → finish-less tail-loss.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "text"}
            )
            yield LLMStreamEvent(
                name="content_block_delta", payload={"text": self._prefix}
            )
            # NO content_block_stop, NO message_stop — the iterator just ends.
            return
        # Resume stream — a complete answer that finishes cleanly.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._full}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_finishless_truncated_turn_recovers_instead_of_completing(
    engine_factory,
) -> None:
    """PRIMARY regression — a finish-less truncated no-tool turn must drive the
    bounded resume recovery and complete with the FULL answer, never the
    truncated prefix.

    FAILS before the fix: the run COMPLETES on stream 1 with the truncated
    prefix (only one LLM call), because ``finish_reason is None`` fell into the
    no-tool ``end_turn`` branch.
    """
    rc = RuntimeConstants(model_context_window=4_096, max_output_recovery_rounds=3)
    llm = _TruncatedThenCompleteLLM(
        prefix="The total is EUR 12",
        full="The total is EUR 1234.56 and the answer is complete.",
    )
    engine = engine_factory(rc=rc)
    engine.llm = llm  # type: ignore[assignment]

    events: list[TurnEvent] = []
    async for evt in engine.run(_user_message("compute the total")):
        events.append(evt)

    # The resume recovery fired (state-change marker).
    assert [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "max_output_token_recovery"
    ]
    # The model re-streamed: TWO LLM calls (truncated, then complete).
    assert len(llm.calls) == 2
    # The run completed with the COMPLETE answer, not the truncated prefix.
    assert engine.state is LoopState.COMPLETED
    assert (
        _final_assistant_text(engine)
        == "The total is EUR 1234.56 and the answer is complete."
    )
    assert "The total is EUR 12" != _final_assistant_text(engine)


class _AlwaysFinishlessLLM:
    """Every stream emits a truncated prefix then NO ``message_stop`` — the
    model never recovers."""

    def __init__(self, *, prefix: str = "partial answer cut off") -> None:
        self._prefix = prefix
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": self._prefix}
        )
        # NO content_block_stop, NO message_stop.

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_finishless_truncation_exhausted_goes_terminal_not_complete(
    engine_factory,
) -> None:
    """When the resume budget is exhausted on a persistently finish-less stream,
    the run goes TERMINAL (``output_length_exhausted``) — never a silent
    truncated completion.

    FAILS before the fix: with ``finish_reason is None`` ignored, the very first
    truncated prefix COMPLETED the run (no recovery, no terminal error).
    """
    # Wind-down off so the call count measures the recovery budget alone.
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_output_recovery_rounds=2,
        soft_stop_enabled=False,
    )
    llm = _AlwaysFinishlessLLM()
    engine = engine_factory(rc=rc)
    engine.llm = llm  # type: ignore[assignment]

    events: list[TurnEvent] = []
    async for evt in engine.run(_user_message("compute the total")):
        events.append(evt)

    # NOT a clean completion — the run is terminal-failed.
    assert engine.state is not LoopState.COMPLETED
    # The recovery was attempted up to the budget then surfaced terminal.
    assert len(llm.calls) == rc.max_output_recovery_rounds + 1
    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors
    assert any(
        e.payload.get("kind") == "output_length_exhausted" for e in errors
    )


@pytest.mark.asyncio
async def test_clean_end_turn_still_completes_unchanged(engine_factory) -> None:
    """Guard — a stream that DOES finish cleanly (``message_stop`` →
    ``finish_reason="stop"``) with no tool calls still completes on the no-tool
    ``end_turn`` path (the fix is scoped to the finish-less / length shapes).
    """

    class _CleanEndTurnLLM:
        def __init__(self) -> None:
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self.calls.append(request)
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="content_block_start", payload={"kind": "text"}
            )
            yield LLMStreamEvent(
                name="content_block_delta",
                payload={"text": "Here is the complete answer."},
            )
            yield LLMStreamEvent(name="content_block_stop", payload={})
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.end_turn.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            from protocore.contracts.llm import LLMResponse

            return LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[]),
                stop_reason=StopReason.end_turn,
            )

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    llm = _CleanEndTurnLLM()
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.llm = llm  # type: ignore[assignment]

    events: list[TurnEvent] = []
    async for evt in engine.run(_user_message("q")):
        events.append(evt)

    # No resume recovery fired; exactly one LLM call; clean completion.
    assert [
        e
        for e in events
        if e.payload.get("reason") == "max_output_token_recovery"
    ] == []
    assert len(llm.calls) == 1
    assert engine.state is LoopState.COMPLETED
    assert _final_assistant_text(engine) == "Here is the complete answer."
