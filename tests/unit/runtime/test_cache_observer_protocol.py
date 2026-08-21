"""Cache-observer protocol shape + engine wire-up.

Verifies:

* :class:`protocore.contracts.observability.CacheObserverProtocol` is
  :func:`typing.runtime_checkable` and recognises a minimal stub.
* :class:`protocore.runtime.query_engine.QueryEngineConfig` carries an
  optional ``cache_observer`` field.
* :func:`protocore.runtime.query._stream_one_assistant_message` calls
  ``cache_observer.record_run_cache_hit_rate(...)`` with the right kwargs
  once per ``ProviderDeltaKind.usage`` envelope and forwards the
  ``cache_breakpoint_count`` from ``LLMRequest.extra["cache_breakpoints"]``.
* When the observer is ``None`` (default) the loop runs unchanged — no
  attribute access on a missing observer.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.llm import (
    ILLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.contracts.observability import CacheObserverProtocol
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.events import TurnEvent
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig


@dataclass
class _Observation:
    tenant_id: str
    cache_read_tokens: int
    prompt_tokens: int
    cache_breakpoint_count: int


@dataclass
class _RecordingObserver:
    """Test-only :class:`CacheObserverProtocol` implementation."""

    observations: list[_Observation] = field(default_factory=list)

    def record_run_cache_hit_rate(
        self,
        *,
        tenant_id: str,
        cache_read_tokens: int,
        prompt_tokens: int,
        cache_breakpoint_count: int,
    ) -> None:
        self.observations.append(
            _Observation(
                tenant_id=tenant_id,
                cache_read_tokens=cache_read_tokens,
                prompt_tokens=prompt_tokens,
                cache_breakpoint_count=cache_breakpoint_count,
            )
        )


class _UsageEmittingProvider(ILLMProvider):
    """Minimal ILLMProvider that emits a single text + usage + finish stream.

    Default :class:`~protocore.tests_support.adapters.InMemoryLLMProvider`
    only emits :class:`LLMStreamEvent` (no ``usage`` kind in the bridge
    mapping) — to exercise the observer wire we need a provider that
    yields :class:`ProviderDelta` with ``kind=usage`` natively, mirroring
    the production vLLM adapter shape.
    """

    def __init__(
        self,
        *,
        input_tokens: int = 100,
        cache_read_input_tokens: int = 75,
    ) -> None:
        self.input_tokens = input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[ProviderDelta | LLMStreamEvent]:
        self.calls.append(request)
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="ok")
        yield ProviderDelta(
            kind=ProviderDeltaKind.text,
            content="",
            is_block_end=True,
        )
        yield ProviderDelta(
            kind=ProviderDeltaKind.usage,
            usage={
                "input_tokens": self.input_tokens,
                "output_tokens": 5,
                "cache_read_input_tokens": self.cache_read_input_tokens,
                "cache_creation_input_tokens": 0,
            },
        )
        yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")

    async def complete_structured(
        self,
        request: LLMRequest,
        response_schema: dict[str, Any],
    ) -> LLMResponse:  # pragma: no cover - not exercised
        self.calls.append(request)
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    async def complete_text(
        self, request: LLMRequest
    ) -> LLMResponse:  # pragma: no cover - not exercised
        self.calls.append(request)
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(
        self, text: str, model: str | None = None
    ) -> int:  # pragma: no cover
        return max(1, len(text) // 4) if text else 0


def test_recording_observer_satisfies_protocol() -> None:
    """``CacheObserverProtocol`` is runtime-checkable; the stub conforms."""
    observer = _RecordingObserver()
    assert isinstance(observer, CacheObserverProtocol)


def test_query_engine_config_accepts_observer() -> None:
    """The ``cache_observer`` field is optional and defaults to ``None``."""
    cfg_default = QueryEngineConfig(
        run_id="r", tenant_id="t", session_id="s", model_name="m"
    )
    assert cfg_default.cache_observer is None

    observer = _RecordingObserver()
    cfg_with = QueryEngineConfig(
        run_id="r",
        tenant_id="t",
        session_id="s",
        model_name="m",
        cache_observer=observer,
    )
    assert cfg_with.cache_observer is observer


@pytest.mark.asyncio
async def test_engine_calls_observer_on_usage_delta(
    in_memory_runtime: dict[str, Any],
) -> None:
    """Engine forwards a single observation per provider usage envelope.

    ``cache_read_tokens`` / ``prompt_tokens`` mirror the provider delta;
    ``cache_breakpoint_count`` mirrors ``LLMRequest.extra["cache_breakpoints"]``
    length.
    """
    observer = _RecordingObserver()
    provider = _UsageEmittingProvider(
        input_tokens=400, cache_read_input_tokens=300
    )
    config = QueryEngineConfig(
        run_id="run-observer",
        tenant_id="tenant-observer",
        session_id="sess-observer",
        model_name="qwen3.6-35b-a3b",
        rc=RuntimeConstants(model_context_window=4_096),
        cache_observer=observer,
    )
    engine = QueryEngine(
        config=config,
        llm_provider=provider,
        tool_registry=in_memory_runtime["tools"],
        event_stream=in_memory_runtime["events"],
        hook_manager=in_memory_runtime["hooks"],
        skill_store=in_memory_runtime["skills"],
        blob_store=in_memory_runtime["blobs"],
    )

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="hi")]
    )
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert len(observer.observations) == 1, (
        f"expected exactly one observation, got {len(observer.observations)}"
    )
    obs = observer.observations[0]
    assert obs.tenant_id == "tenant-observer"
    assert obs.cache_read_tokens == 300
    assert obs.prompt_tokens == 400
    # ``apply_system_and_3`` returns the same list that landed on
    # ``LLMRequest.extra["cache_breakpoints"]`` — pull the actual count
    # from the provider's recorded request.
    expected_breakpoints = provider.calls[0].extra.get("cache_breakpoints", [])
    assert obs.cache_breakpoint_count == len(expected_breakpoints)


@pytest.mark.asyncio
async def test_engine_omits_observer_call_when_unset(
    in_memory_runtime: dict[str, Any],
) -> None:
    """Default ``cache_observer=None`` must not break the streaming loop."""
    provider = _UsageEmittingProvider()
    config = QueryEngineConfig(
        run_id="run-no-observer",
        tenant_id="tenant-no-observer",
        session_id="sess-no-observer",
        model_name="qwen3.6-35b-a3b",
        rc=RuntimeConstants(model_context_window=4_096),
    )
    assert config.cache_observer is None
    engine = QueryEngine(
        config=config,
        llm_provider=provider,
        tool_registry=in_memory_runtime["tools"],
        event_stream=in_memory_runtime["events"],
        hook_manager=in_memory_runtime["hooks"],
        skill_store=in_memory_runtime["skills"],
        blob_store=in_memory_runtime["blobs"],
    )
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="hi")]
    )
    # Run completes without raising even though the provider emits a
    # usage delta with cache stats.
    async for _ in engine.run(user_msg):
        pass
    # The provider was called.
    assert provider.calls
