"""``visibility`` on the content-block frames of a live turn.

An agent that uses tools writes two very different kinds of text, and until now
the stream shipped them identically. Between calls it narrates ("Let me try more
targeted searches…"); at the end it answers. A client receiving
``content_block_start`` with ``kind: "text"`` and nothing else had to guess, and
the only guessable thing is shape — which fails on the case that matters, since
a reply about a config format looks exactly like a model reasoning about one.
The result was a message bubble, complete with feedback controls, for every
intermediate thought, kept across reloads.

The durable transcript has always made this judgement per part. These tests pin
the same judgement on the live wire, and pin the two properties that make it
safe to act on:

* the run's actual answer is never marked non-public — losing a reply is far
  worse than showing one bubble too many, so every ambiguous case resolves to
  ``public``;
* the field is purely additive, so a consumer that has never heard of it sees
  precisely the stream it saw before.
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

_TERMINAL_TOOL = "Finalize"

# Ordered weakest-to-strongest, so "the settled value never relaxes" is an
# index comparison rather than a pile of pairwise cases.
_VISIBILITY_RANK = {"public": 0, "collapsed": 1, "hidden": 2, "debug": 3}


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


async def _collect_events(engine) -> list[TurnEvent]:
    context = await _rebuild_context_for_recovery(engine)
    result = _StreamAttemptResult()
    return [evt async for evt in _drive_one_stream(engine, context, result)]


def _starts(events: list[TurnEvent]) -> list[TurnEvent]:
    return [e for e in events if e.type is EventType.CONTENT_BLOCK_START]


def _stops(events: list[TurnEvent]) -> list[TurnEvent]:
    return [e for e in events if e.type is EventType.CONTENT_BLOCK_STOP]


def _tool_call_deltas(
    *, tool_call_id: str, tool_name: str | None
) -> list[ProviderDelta]:
    return [
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_start,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        ),
        ProviderDelta(
            kind=ProviderDeltaKind.tool_use_stop,
            tool_call_id=tool_call_id,
            tool_input_final={},
            is_block_end=True,
        ),
    ]


# ---------------------------------------------------------------------------
# The answer stays public.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_shaped_text_block_is_public(engine_factory) -> None:
    """A turn that only speaks is answering, and is marked as such throughout.

    Nothing in this stream suggests the run continues, so both the opening
    frame and the settled one say ``public``. This is the case a false
    ``collapsed`` would destroy — the user would be left with no reply at all —
    so it is pinned at both ends of the block, not just at the open.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("what is a TOML table?"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(kind=ProviderDeltaKind.text, content="A TOML table "),
            ProviderDelta(kind=ProviderDeltaKind.text, content="is a section."),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
        ]
    )

    events = await _collect_events(engine)

    assert [e.payload["visibility"] for e in _starts(events)] == ["public"]
    assert [e.payload["visibility"] for e in _stops(events)] == ["public"]


@pytest.mark.asyncio
async def test_terminal_gate_call_does_not_collapse_the_answer(
    engine_factory,
) -> None:
    """The one tool that shares a message with the answer must not mark it.

    Under the background terminal gate the model writes its reply and calls the
    run's terminal tool in the SAME assistant message, and that call is stripped
    from every reader-facing view. Treating it like an ordinary tool would
    collapse exactly the text the user asked for, and would contradict the
    durable transcript, which shows this message as prose and nothing else.
    """
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool=_TERMINAL_TOOL,
    )
    engine.history.append(_user_message("summarise the findings"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(
                kind=ProviderDeltaKind.text,
                content="Three sources agree on the deprecation date.",
            ),
            *_tool_call_deltas(tool_call_id="call-fin", tool_name=_TERMINAL_TOOL),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    assert [e.payload["visibility"] for e in _starts(events)] == ["public"]
    assert [e.payload["visibility"] for e in _stops(events)] == ["public"]


@pytest.mark.asyncio
async def test_unnamed_tool_call_leaves_the_answer_public(engine_factory) -> None:
    """An unidentifiable call is not proof of anything, so nothing is marked.

    Some providers withhold the tool name until the arguments have finished
    streaming, by which point the text block is long closed. A run that HAS a
    terminal tool therefore cannot rule out that this nameless call is it — and
    the cheap error is the missed collapse, not the swallowed answer.
    """
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool=_TERMINAL_TOOL,
    )
    engine.history.append(_user_message("summarise the findings"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(kind=ProviderDeltaKind.text, content="Here it is."),
            *_tool_call_deltas(tool_call_id="call-anon", tool_name=None),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    assert [e.payload["visibility"] for e in _stops(events)] == ["public"]


# ---------------------------------------------------------------------------
# Narration is marked, in both the orders it occurs in.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narration_before_a_tool_call_settles_on_the_stop(
    engine_factory,
) -> None:
    """The reported shape: the model says what it is about to do, then does it.

    At the moment this block OPENS nothing distinguishes it from an answer, and
    the stream cannot see its own future, so it opens ``public`` — the honest
    value. The tool call is the proof, and it arrives at the same instant the
    block closes, so the stop frame is the first one able to say ``collapsed``.
    A client that renders live and reconciles on the stop ends up agreeing with
    what the transcript will say after a reload.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("find the deprecation date"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(
                kind=ProviderDeltaKind.text,
                content="Let me try more targeted searches…",
            ),
            *_tool_call_deltas(tool_call_id="call-1", tool_name="CatalogSearch"),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    assert [e.payload["visibility"] for e in _starts(events)] == ["public"]
    assert [e.payload["visibility"] for e in _stops(events)] == ["collapsed"]


@pytest.mark.asyncio
async def test_narration_after_a_tool_call_opens_collapsed(engine_factory) -> None:
    """Text interleaved after a call is provably narration the moment it opens.

    The model has already committed to a tool this message, whose result must be
    fed back, so another assistant message is coming and this text cannot be the
    run's reply. That is the only thing the open frame can ever prove, and here
    it can.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("find the deprecation date"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            *_tool_call_deltas(tool_call_id="call-1", tool_name="CatalogSearch"),
            ProviderDelta(
                kind=ProviderDeltaKind.text,
                content="Excellent! I found highly relevant sources…",
            ),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    assert [e.payload["visibility"] for e in _starts(events)] == ["collapsed"]
    assert [e.payload["visibility"] for e in _stops(events)] == ["collapsed"]


@pytest.mark.asyncio
async def test_reasoning_is_never_prose(engine_factory) -> None:
    """Thinking is collapsed unconditionally — it is not addressed to anyone.

    No ambiguity to resolve here and none of the answer-loss risk: a reasoning
    block is the model's working by definition, which is also how the durable
    transcript projects it (``reasoning_summary`` → ``collapsed``).
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hi"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(kind=ProviderDeltaKind.thinking, content="weighing it up"),
            ProviderDelta(kind=ProviderDeltaKind.text, content="Yes."),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop"),
        ]
    )

    events = await _collect_events(engine)

    assert [
        (e.payload["kind"], e.payload["visibility"]) for e in _starts(events)
    ] == [("thinking", "collapsed"), ("text", "public")]
    assert [e.payload["visibility"] for e in _stops(events)] == [
        "collapsed",
        "public",
    ]


@pytest.mark.asyncio
async def test_settled_value_never_relaxes_across_a_mixed_message(
    engine_factory,
) -> None:
    """A block that opened ``collapsed`` can never close ``public``.

    The whole design rests on the marking only ever tightening: a client is told
    to trust the stop frame, which is only safe if the stop cannot undo a
    collapse the start already published. The message below opens a block on
    each side of a tool call so both directions are exercised at once.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("find the deprecation date"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(kind=ProviderDeltaKind.text, content="Searching now…"),
            *_tool_call_deltas(tool_call_id="call-1", tool_name="CatalogSearch"),
            ProviderDelta(kind=ProviderDeltaKind.text, content="…and again."),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    opened = {
        e.payload["block_idx"]: e.payload["visibility"] for e in _starts(events)
    }
    assert list(opened.values()) == ["public", "collapsed"]
    for stop in _stops(events):
        idx = stop.payload["block_idx"]
        assert _VISIBILITY_RANK[stop.payload["visibility"]] >= _VISIBILITY_RANK[
            opened[idx]
        ], f"block {idx} relaxed on close"
    assert [e.payload["visibility"] for e in _stops(events)] == [
        "collapsed",
        "collapsed",
    ]


# ---------------------------------------------------------------------------
# The field is additive.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_consumer_that_ignores_the_field_sees_todays_stream(
    engine_factory,
) -> None:
    """Drop ``visibility`` and the frames are byte-for-byte what they always were.

    The point of a default and of adding rather than replacing: nobody has to
    ship a client change to keep working. Asserting the residue exactly — rather
    than just that the old keys survive — also catches a frame being added,
    removed or reordered on the way in.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("find the deprecation date"))
    engine.llm.stream_with_tools = _make_provider_deltas( # type: ignore[method-assign]
        [
            ProviderDelta(kind=ProviderDeltaKind.thinking, content="hmm"),
            ProviderDelta(kind=ProviderDeltaKind.text, content="Searching now…"),
            *_tool_call_deltas(tool_call_id="call-1", tool_name="CatalogSearch"),
            ProviderDelta(kind=ProviderDeltaKind.text, content="Found it."),
            ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="tool_use"),
        ]
    )

    events = await _collect_events(engine)

    turn_id = engine.turn_id()
    legacy = [
        (evt.type, {k: v for k, v in evt.payload.items() if k != "visibility"})
        for evt in events
        if evt.type
        in (EventType.CONTENT_BLOCK_START, EventType.CONTENT_BLOCK_STOP)
    ]
    assert legacy == [
        (EventType.CONTENT_BLOCK_START, {"turn_id": turn_id, "block_idx": 0, "kind": "thinking"}),
        (EventType.CONTENT_BLOCK_STOP, {"turn_id": turn_id, "block_idx": 0}),
        (EventType.CONTENT_BLOCK_START, {"turn_id": turn_id, "block_idx": 1, "kind": "text"}),
        (EventType.CONTENT_BLOCK_STOP, {"turn_id": turn_id, "block_idx": 1}),
        (EventType.CONTENT_BLOCK_START, {"turn_id": turn_id, "block_idx": 3, "kind": "text"}),
        (EventType.CONTENT_BLOCK_STOP, {"turn_id": turn_id, "block_idx": 3}),
    ]
