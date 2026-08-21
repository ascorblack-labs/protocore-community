"""In-memory buffer for :class:`TurnEvent` — test fixture only.

Allows tests to iterate ``async for evt in query(engine)`` and assert
event order without a host transport. Not exported for production
use — that's :class:`protocore.tests_support.adapters.InMemoryEventStream`.
"""
from __future__ import annotations

from collections.abc import Sequence

from protocore.runtime.events.envelope import TurnEvent


class InMemoryTurnEventBuffer:
    """Capture every yielded :class:`TurnEvent` for test assertions."""

    def __init__(self) -> None:
        self._events: list[TurnEvent] = []

    def append(self, event: TurnEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> Sequence[TurnEvent]:
        return tuple(self._events)

    def types(self) -> list[str]:
        """Return ordered list of ``EventType`` values for ergonomic asserts."""
        return [e.type.value for e in self._events]

    def clear(self) -> None:
        self._events.clear()


__all__ = ["InMemoryTurnEventBuffer"]
