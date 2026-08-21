"""Low-cardinality span names; high-cardinality ids stay attributes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from protocore.contracts.runtime_constants import RuntimeConstants

SPAN_NAMES = (
    "run",
    "turn",
    "step",
    "tool",
    "compact",
    "hook",
)


@dataclass(slots=True)
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "attributes": dict(self.attributes)}


def start_span(name: str, *, rc: RuntimeConstants, **attributes: Any) -> Span | None:
    if not rc.telemetry_spans_enabled:
        return None
    if name not in SPAN_NAMES:
        raise ValueError("unknown_span")
    return Span(name=name, attributes=dict(attributes))


def mark_recovery(span: Span | None, *, intent_id: str) -> Span | None:
    if span is None:
        return None
    span.attributes["recovery"] = True
    span.attributes["intent_id"] = intent_id
    return span


def is_prometheus_safe_label(key: str) -> bool:
    return key not in {"session_id", "lane_id", "operation_id", "run_id"}


__all__ = [
    "SPAN_NAMES",
    "Span",
    "is_prometheus_safe_label",
    "mark_recovery",
    "start_span",
]
