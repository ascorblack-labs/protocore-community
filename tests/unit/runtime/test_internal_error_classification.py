"""A crash of this process is not reported as a failure of the upstream.

Before this split, everything the loop's catch-all saw became
``llm_provider_error``: a ``RecursionError`` in a metadata walk, a parser bug, a
``KeyError`` in the dispatch path. Two things followed from that and both are
what these tests hold. The provider-failure metric counted our own bugs, so the
number that decides whether to blame an endpoint was not measuring endpoints.
And the ONE durable record of a crash — the error event, the run row, one log
line — said the upstream had failed, with no traceback anywhere, so the place
the run died was not recoverable from anything the system kept.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMStreamEvent,
    LLMStreamIdleError,
    LLMTimeoutError,
    MaxOutputTokensExhausted,
)
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import INTERNAL_ERROR_KIND, _emit_llm_terminal


class _RaisingLLM:
    """LLM stub whose stream raises ``exc`` before yielding anything."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def calls(self):  # type: ignore[no-untyped-def]
        return []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        if False:  # pragma: no cover — generator protocol marker
            yield LLMStreamEvent(name="never", payload={})
        raise self._exc

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise RuntimeError("unused")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def _drive(engine) -> list[TurnEvent]:  # type: ignore[no-untyped-def]
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    return [evt async for evt in engine.run(user_msg)]


def _error_kind(events: list[TurnEvent]) -> str:
    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors, [e.type.value for e in events]
    return str(errors[0].payload["kind"])


def _final_stop_reason(events: list[TurnEvent]) -> str:
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops
    return str(stops[-1].payload["stop_reason"])


# ----------------------------------------------------------------------
# Which class the failure is reported as
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        RecursionError("maximum recursion depth exceeded"),
        RuntimeError("parser blew up mid-stream"),
        AttributeError("'NoneType' object has no attribute 'text'"),
        KeyError("tool_call_id"),
        ValueError("metadata is nested deeper than 200 levels"),
    ],
    ids=["recursion", "runtime", "attribute", "key", "value"],
)
async def test_an_untyped_exception_is_reported_as_an_internal_error(
    engine_factory, exc
) -> None:
    engine = engine_factory()
    engine.llm = _RaisingLLM(exc)  # type: ignore[assignment]

    events = await _drive(engine)

    assert _error_kind(events) == INTERNAL_ERROR_KIND
    assert engine.state is LoopState.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_kind"),
    [
        (LLMProviderError("503 from upstream"), "llm_provider_error"),
        (LLMRateLimitError("429"), "llm_rate_limit"),
        (LLMTimeoutError("read timeout"), "llm_timeout"),
    ],
    ids=["provider", "rate_limit", "timeout"],
)
async def test_a_typed_upstream_failure_keeps_its_own_class(
    engine_factory, exc, expected_kind
) -> None:
    """The taxonomy the recovery dispatcher routes on is left exactly as it was."""
    engine = engine_factory()
    engine.llm = _RaisingLLM(exc)  # type: ignore[assignment]

    events = await _drive(engine)

    assert _error_kind(events) == expected_kind


@pytest.mark.asyncio
async def test_an_internal_error_still_stops_the_run_with_stop_reason_error(
    engine_factory,
) -> None:
    """Only the CLASS moves. A crash is still a run that ended in error."""
    engine = engine_factory()
    engine.llm = _RaisingLLM(RecursionError("maximum recursion depth exceeded"))  # type: ignore[assignment]

    events = await _drive(engine)

    assert _final_stop_reason(events) == StopReason.error.value


@pytest.mark.asyncio
async def test_the_state_change_to_failed_carries_the_same_class(
    engine_factory,
) -> None:
    """The reason on the FAILED transition is the kind, so both records agree."""
    engine = engine_factory()
    engine.llm = _RaisingLLM(RecursionError("boom"))  # type: ignore[assignment]

    events = await _drive(engine)

    failed = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("to") == LoopState.FAILED.value
    ]
    assert failed, [e.payload for e in events if e.type is EventType.STATE_CHANGED]
    assert failed[-1].payload["reason"] == INTERNAL_ERROR_KIND


# ----------------------------------------------------------------------
# The traceback
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_untyped_exception_is_logged_with_its_traceback(
    engine_factory, caplog
) -> None:
    """Without this, the record of the crash names no line of code at all.

    The incident this was written for left ``maximum recursion depth exceeded``
    in the run row and one traceback-less log line; which of several recursive
    walks over model data had blown the stack was not answerable from anything
    that had been kept.
    """
    engine = engine_factory()
    engine.llm = _RaisingLLM(RecursionError("maximum recursion depth exceeded"))  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        await _drive(engine)

    with_traceback = [r for r in caplog.records if r.exc_info is not None]
    assert with_traceback, [r.getMessage() for r in caplog.records]
    assert any(
        r.exc_info is not None and r.exc_info[0] is RecursionError
        for r in with_traceback
    )


@pytest.mark.asyncio
async def test_the_crash_is_logged_from_the_loop_and_from_the_terminal(
    engine_factory, caplog
) -> None:
    """Two records, because the terminal emitter can itself fail to run."""
    engine = engine_factory()
    engine.llm = _RaisingLLM(RuntimeError("kaboom"))  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        await _drive(engine)

    messages = [r.getMessage() for r in caplog.records if r.exc_info is not None]
    assert any("query.stream_crashed" in m for m in messages), messages
    assert any("query.internal_error" in m for m in messages), messages


@pytest.mark.asyncio
async def test_a_typed_upstream_failure_is_not_logged_as_an_internal_crash(
    engine_factory, caplog
) -> None:
    """A 503 is an ordinary operational event, not something to page on."""
    engine = engine_factory()
    engine.llm = _RaisingLLM(LLMProviderError("503 from upstream"))  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING):
        await _drive(engine)

    assert not [
        r for r in caplog.records if "query.internal_error" in r.getMessage()
    ]


# ----------------------------------------------------------------------
# The classification is made once, where every terminal passes through
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        LLMProviderError("upstream"),
        LLMRateLimitError("429"),
        LLMTimeoutError("timeout"),
        LLMStreamIdleError("idle"),
        LLMContextWindowExceeded("too long"),
        MaxOutputTokensExhausted("out of output"),
    ],
    ids=["provider", "rate_limit", "timeout", "idle", "context", "output"],
)
async def test_the_terminal_emitter_never_reclassifies_a_typed_failure(
    engine_factory, exc
) -> None:
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)

    events = [
        evt async for evt in _emit_llm_terminal(engine, exc, kind="caller_chosen_kind")
    ]

    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors[0].payload["kind"] == "caller_chosen_kind"


@pytest.mark.asyncio
async def test_the_terminal_emitter_overrides_a_caller_that_asked_for_the_wrong_class(
    engine_factory,
) -> None:
    """A call site that has to remember to classify is one that will forget."""
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)

    events = [
        evt
        async for evt in _emit_llm_terminal(
            engine, RecursionError("boom"), kind="llm_provider_error"
        )
    ]

    errors = [e for e in events if e.type is EventType.ERROR]
    assert errors[0].payload["kind"] == INTERNAL_ERROR_KIND


@pytest.mark.asyncio
async def test_terminal_hooks_still_run_after_an_internal_error(
    engine_factory,
) -> None:
    """The death-spiral guard is for a broken provider, not for our own bugs.

    Skipping Stop / SessionEnd hooks exists so a run against a dead endpoint
    does not cascade through error-only hooks. A crash in this process is the
    failure a deployment most wants its teardown to observe, so the guard stays
    off for it.
    """
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)

    async for _ in _emit_llm_terminal(engine, RecursionError("boom"), kind="anything"):
        pass

    assert engine.skip_terminal_hooks is False


@pytest.mark.asyncio
async def test_the_guard_is_still_engaged_for_a_provider_failure(
    engine_factory,
) -> None:
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)
    assert engine.config.rc.skip_terminal_hooks_on_llm_error is True

    async for _ in _emit_llm_terminal(
        engine, LLMProviderError("503"), kind="llm_provider_error"
    ):
        pass

    assert engine.skip_terminal_hooks is True
