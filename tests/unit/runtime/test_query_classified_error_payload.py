"""Terminal error payload surfaces ``classified_reason``.

The adapters (Anthropic / OpenAI) attach a ``ClassifiedError`` to every
raised :class:`LLMError` via the dynamic ``classified`` attribute. The
protocore loop forwards the verdict downstream by including it in the
terminal ``error`` TurnEvent payload (``classified_reason``,
``retryable``, ``should_compress``, ``should_fallback``). Host-
side ``RecoveryDispatcher`` consumes that payload.

This file tests the protocore side: the loop reads the dynamic attribute
and surfaces the fields. The host dispatcher itself is tested in
``protocore-the host/tests/unit/test_recovery_dispatcher.py``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    LLMStreamEvent,
)
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType, TurnEvent


@dataclass(frozen=True, slots=True)
class _FakeClassifiedError:
    """Stand-in for the host ``ClassifiedError``.

    Mirrors the attribute names the loop reads — keeps the protocore
    test free of upward imports.
    """

    reason: str
    retryable: bool = True
    should_compress: bool = False
    should_fallback: bool = False
    message: str = ""


class _FailingLLM:
    """LLM mock that raises an LLMProviderError with a classifier verdict."""

    def __init__(self, classified: _FakeClassifiedError) -> None:
        self._classified = classified

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        exc = LLMProviderError("provider blew up")
        object.__setattr__(exc, "classified", self._classified)
        raise exc
        yield  # type: ignore[unreachable]


@pytest.mark.asyncio
async def test_terminal_error_payload_carries_classified_reason(
    engine_factory,
    in_memory_runtime: dict[str, object],
) -> None:
    """LLM raises classified exception → ERROR payload exposes the reason."""
    in_memory_runtime["llm"] = _FailingLLM(
        _FakeClassifiedError(
            reason="provider_policy_blocked",
            retryable=False,
            should_fallback=False,
            message="endpoint rejected by data policy",
        )
    )
    engine = engine_factory()
    # Re-wire the engine with the failing LLM.
    engine.llm = in_memory_runtime["llm"]  # type: ignore[attr-defined]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    error_events = [e for e in events if e.type is EventType.ERROR]
    assert error_events, "expected an ERROR event on terminal failure"
    payload = error_events[0].payload
    assert payload["classified_reason"] == "provider_policy_blocked"
    assert payload["retryable"] is False
    assert payload["should_fallback"] is False


@pytest.mark.asyncio
async def test_terminal_error_payload_omits_fields_when_no_classifier(
    engine_factory,
    in_memory_runtime: dict[str, object],
) -> None:
    """Exception without ``.classified`` → payload has no ``classified_reason``.

    The legacy path (vLLM adapter pre-W8, raw httpx errors, unit-test
    stubs) raises without the classifier. The loop must not crash and
    must not surface stale / fake values.
    """

    class _LegacyLLM:
        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            raise LLMProviderError("legacy boom")
            yield  # type: ignore[unreachable]

    in_memory_runtime["llm"] = _LegacyLLM()
    engine = engine_factory()
    engine.llm = in_memory_runtime["llm"]  # type: ignore[attr-defined]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    error_events = [e for e in events if e.type is EventType.ERROR]
    assert error_events
    payload = error_events[0].payload
    assert "classified_reason" not in payload
    assert "retryable" not in payload


@pytest.mark.asyncio
async def test_terminal_error_payload_handles_str_reason(
    engine_factory,
    in_memory_runtime: dict[str, object],
) -> None:
    """Classifier ``reason`` may be a plain string (forward compat).

    The W8 classifier uses ``FailoverReason`` StrEnum; tests may stub
    with plain strings. The loop accepts both shapes — extracting
    ``.value`` when present, falling back to the string directly.
    """
    in_memory_runtime["llm"] = _FailingLLM(
        _FakeClassifiedError(reason="rate_limit", retryable=True)
    )
    engine = engine_factory()
    engine.llm = in_memory_runtime["llm"]  # type: ignore[attr-defined]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    error_events = [e for e in events if e.type is EventType.ERROR]
    payload = error_events[0].payload
    assert payload["classified_reason"] == "rate_limit"
    assert payload["retryable"] is True
