"""IEventStream Protocol — cross-pod event fanout.

Reference shape: a Redis stream driven with ``XADD``/``XREAD`` and a
``MAXLEN ~`` trim.

Distinct from in-process :class:`protocore.events.EventBus`:
    - ``EventBus``: in-process typed pub/sub for sibling-handler signalling
    - ``IEventStream``: durable cross-pod stream (SSE reconnect/replay)
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from protocore.contracts.types import Event


class EventStreamError(Exception):
    """Base for event-stream domain errors."""


@runtime_checkable
class IEventStream(Protocol):
    """Adapter Protocol over a per-run durable event stream."""

    async def emit(self, event: Event) -> None:
        """Append event to the per-run stream."""
        ...

    def subscribe(
        self,
        run_id: str,
        tenant_id: str,
        *,
        from_event_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Replay (if cursor) then tail. Yields until client disconnects.

        Implementations are async generators; declared here without
        ``async def`` so the Protocol typechecks against async iterators.
        """
        ...

    async def trim(self, run_id: str, tenant_id: str, *, max_len: int) -> None:
        """Apply ``MAXLEN ~`` trim. Called by GC on terminal runs."""
        ...


__all__ = ["EventStreamError", "IEventStream"]
