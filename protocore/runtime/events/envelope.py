"""``TurnEvent`` envelope — per-turn streaming event wrapper.

Frozen Pydantic model with discriminator (:class:`EventType`) + typed
payload dict. Schema version bumped per locked-decision when wire-format
breaks.

Distinct from :class:`protocore.contracts.types.Event` (the durable row
persisted in Redis Stream / S3 blob) — :class:`TurnEvent` is the
in-flight, in-loop primitive that the executor converts to :class:`Event`
prior to :meth:`IEventStream.emit`.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import Event
from protocore.runtime.events.types import EventType


def _utcnow_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


class TurnEvent(BaseModel):
    """Per-turn streaming envelope produced by ``query``.

 ``payload`` schema is type-discriminated; documented in
 . Callers should treat :class:`TurnEvent` as
 immutable once yielded — the loop owns lifetime.
 """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    run_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1
    server_ts_ms: int = Field(default_factory=_utcnow_ms)

    def to_event(self) -> Event:
        """Convert to the durable :class:`Event` record.

        Used by the executor pod when publishing to :class:`IEventStream`.
        Payload is preserved verbatim; envelope-level fields (``id``,
        ``run_id``) flow to the durable record's top-level fields.
        """
        # Surface the EventType discriminator and schema version inside the
        # payload so consumers can route without inspecting the envelope.
        durable_payload = dict(self.payload)
        durable_payload.setdefault("schema_version", self.schema_version)
        durable_payload.setdefault("server_ts_ms", self.server_ts_ms)
        return Event(
            id=self.id,
            run_id=self.run_id,
            name=self.type.value,
            payload=durable_payload,
        )


__all__ = ["TurnEvent"]
