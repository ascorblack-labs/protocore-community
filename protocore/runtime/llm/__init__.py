"""In-core LLM helpers — :class:`MockLLMProvider` + stream-event translation.

The :class:`InMemoryLLMProvider` lives in
:mod:`protocore.tests_support.adapters`. This module ships the
stream-event ↔ provider-delta
translation helpers used by :func:`protocore.runtime.query.query` so the
loop works against any :class:`ILLMProvider` that emits the v1
:class:`LLMStreamEvent` API.
"""
from __future__ import annotations

from protocore.runtime.llm.delta_bridge import (
    delta_to_turn_events,
    stream_events_to_provider_deltas,
)

__all__ = ["delta_to_turn_events", "stream_events_to_provider_deltas"]
