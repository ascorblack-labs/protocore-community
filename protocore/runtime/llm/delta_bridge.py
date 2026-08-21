"""Translation between Anthropic-style :class:`LLMStreamEvent` and :class:`ProviderDelta`.

The loop is wired against :class:`ProviderDelta`, while
:class:`ILLMProvider` declares :meth:`stream_with_tools` returning
:class:`LLMStreamEvent`. This module bridges the two so the loop can
consume either pathway:

 1. Adapters that natively emit :class:`ProviderDelta` (a streaming
 provider client) — used directly.
 2. Adapters that emit :class:`LLMStreamEvent` (the in-memory test
 doubles) — translated to :class:`ProviderDelta` via
 :func:`stream_events_to_provider_deltas`.

Translation in the other direction (delta → turn event) drives the
:class:`TurnEvent` emission inside :func:`protocore.runtime.query.query`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from protocore.contracts.llm import (
    LLMStreamEvent,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.runtime.events import EventType, TurnEvent

_FINISH_REASON = Literal["stop", "tool_use", "length", "content_filter"]


async def stream_events_to_provider_deltas(
    upstream: AsyncIterator[LLMStreamEvent],
) -> AsyncIterator[ProviderDelta]:
    """Translate :class:`LLMStreamEvent` stream into :class:`ProviderDelta`.

 Mapping md`.
 Unknown event names are dropped silently — the loop only cares about
 the canonical kinds.
 """
    async for evt in upstream:
        name = evt.name
        payload = evt.payload or {}

        if name == "message_start":
            # Loop emits MESSAGE_START itself based on its own state; the
            # provider boundary event is informational here.
            continue

        if name == "content_block_start":
            # No-op at the provider-delta layer — content_block_delta
            # surfaces the kind.
            continue

        if name == "content_block_delta":
            text = payload.get("text", "")
            kind = payload.get("kind", "text")
            if kind == "thinking":
                yield ProviderDelta(kind=ProviderDeltaKind.thinking, content=text)
            else:
                yield ProviderDelta(kind=ProviderDeltaKind.text, content=text)
            continue

        if name == "content_block_stop":
            yield ProviderDelta(kind=ProviderDeltaKind.text, content="", is_block_end=True)
            continue

        if name == "tool_use_start":
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_start,
                tool_call_id=payload.get("tool_call_id"),
                tool_name=payload.get("tool_name"),
            )
            continue

        if name == "tool_use_input_delta":
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_input,
                tool_call_id=payload.get("tool_call_id"),
                tool_input_delta=payload.get("partial_input_json"),
            )
            continue

        if name == "tool_use_stop":
            yield ProviderDelta(
                kind=ProviderDeltaKind.tool_use_stop,
                tool_call_id=payload.get("tool_call_id"),
                tool_input_final=payload.get("final_input"),
                is_block_end=True,
                # Preserve the truncation signature the LiteLLM adapter
                # (and any other LLMStreamEvent emitter) attached to the
                # stop event. The ``truncated_by_output_cap`` flag drives
                # mid-tool-call recovery (re-stream once); the
                # ``args_partial_truncated`` flag drives the
                # synthetic-error tool_result branch. Dropping either
                # here would silently degrade the loop's recovery
                # behaviour for any adapter that emits the legacy event
                # shape (LiteLLM adapter, mocks).
                truncated_by_output_cap=bool(
                    payload.get("truncated_by_output_cap", False)
                ),
                args_partial_truncated=bool(
                    payload.get("args_partial_truncated", False)
                ),
            )
            continue

        if name == "usage":
            yield ProviderDelta(
                kind=ProviderDeltaKind.usage,
                usage=dict(payload),
            )
            continue

        if name == "message_stop":
            stop_reason = payload.get("stop_reason", "stop")
            yield ProviderDelta(
                kind=ProviderDeltaKind.finish,
                finish_reason=_normalise_finish_reason(stop_reason),
            )
            continue

        # Unknown events ignored — telemetry-only.


def _normalise_finish_reason(value: Any) -> _FINISH_REASON:
    """Map provider-specific stop_reason to ProviderDelta's Literal set."""
    normalised = str(value).lower()
    if normalised in {"stop", "end_turn"}:
        return "stop"
    if normalised in {"tool_use", "tool_call"}:
        return "tool_use"
    if normalised in {"length", "max_tokens", "max_output_tokens"}:
        return "length"
    if normalised in {"content_filter", "filter"}:
        return "content_filter"
    # Default to "stop" — pass-through unrecognized reasons.
    return "stop"


def delta_to_turn_events(
    delta: ProviderDelta,
    *,
    run_id: str,
    turn_id: str,
    block_idx: int,
) -> list[TurnEvent]:
    """Translate one :class:`ProviderDelta` to zero-or-more :class:`TurnEvent`.

 The function is pure — no engine
 mutation. Callers pass the current ``block_idx`` snapshot.
 """
    events: list[TurnEvent] = []

    if delta.kind is ProviderDeltaKind.text:
        if delta.content:
            events.append(
                TurnEvent(
                    type=EventType.CONTENT_BLOCK_DELTA,
                    run_id=run_id,
                    payload={
                        "turn_id": turn_id,
                        "block_idx": block_idx,
                        "delta": {"type": "text_delta", "text": delta.content},
                    },
                )
            )
        return events

    if delta.kind is ProviderDeltaKind.thinking:
        if delta.content:
            events.append(
                TurnEvent(
                    type=EventType.CONTENT_BLOCK_DELTA,
                    run_id=run_id,
                    payload={
                        "turn_id": turn_id,
                        "block_idx": block_idx,
                        "delta": {"type": "thinking_delta", "text": delta.content},
                    },
                )
            )
        return events

    if delta.kind is ProviderDeltaKind.tool_use_start:
        events.append(
            TurnEvent(
                type=EventType.TOOL_USE_START,
                run_id=run_id,
                payload={
                    "turn_id": turn_id,
                    "block_idx": block_idx,
                    "tool_call_id": delta.tool_call_id,
                    "tool_name": delta.tool_name,
                },
            )
        )
        return events

    if delta.kind is ProviderDeltaKind.tool_use_input:
        events.append(
            TurnEvent(
                type=EventType.TOOL_USE_INPUT_DELTA,
                run_id=run_id,
                payload={
                    "turn_id": turn_id,
                    "block_idx": block_idx,
                    "tool_call_id": delta.tool_call_id,
                    "partial_input_json": delta.tool_input_delta or "",
                },
            )
        )
        return events

    if delta.kind is ProviderDeltaKind.tool_use_stop:
        events.append(
            TurnEvent(
                type=EventType.TOOL_USE_STOP,
                run_id=run_id,
                payload={
                    "turn_id": turn_id,
                    "block_idx": block_idx,
                    "tool_call_id": delta.tool_call_id,
                    "final_input": delta.tool_input_final or {},
                },
            )
        )
        return events

    # finish / usage are loop-level — handled separately.
    return events


__all__ = [
    "delta_to_turn_events",
    "stream_events_to_provider_deltas",
]
