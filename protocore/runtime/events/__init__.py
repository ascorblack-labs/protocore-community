"""Per-turn streaming event taxonomy + envelope.

The :class:`EventType` StrEnum and :class:`TurnEvent` envelope
live in core; the Redis Stream adapter lives in the host.

Distinct from :class:`protocore.contracts.types.Event` (which is the
durable event record persisted via :class:`IEventStream`). :class:`TurnEvent`
is the in-flight, per-turn streaming primitive — once published via
:meth:`IEventStream.emit` it's converted to a durable :class:`Event` row.
"""
from __future__ import annotations

from protocore.runtime.events.envelope import TurnEvent
from protocore.runtime.events.memory import InMemoryTurnEventBuffer
from protocore.runtime.events.types import BlockVisibility, EventType

__all__ = [
    "BlockVisibility",
    "EventType",
    "InMemoryTurnEventBuffer",
    "TurnEvent",
]
