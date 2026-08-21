"""Append-only usage ledger. Failed attempt + retry is two rows."""
from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.runtime_constants import RuntimeConstants

UsageKind = str


@dataclass(slots=True)
class UsageRow:
    seq: int
    kind: UsageKind
    run_id: str
    operation_id: str | None
    input_tokens: int
    output_tokens: int
    success: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "run_id": self.run_id,
            "operation_id": self.operation_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "success": self.success,
        }

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def append_usage(
    rows: list[UsageRow],
    *,
    kind: UsageKind,
    run_id: str,
    input_tokens: int,
    output_tokens: int,
    success: bool,
    operation_id: str | None = None,
    rc: RuntimeConstants,
) -> list[UsageRow]:
    if not rc.usage_ledger_enabled:
        return list(rows)
    nxt = (rows[-1].seq + 1) if rows else 1
    row = UsageRow(
        seq=nxt,
        kind=kind,
        run_id=run_id,
        operation_id=operation_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
    )
    return [*rows, row]


def session_total(rows: list[UsageRow]) -> int:
    return sum(item.total_tokens for item in rows)


def from_seq(rows: list[UsageRow], seq: int) -> list[UsageRow]:
    return [item for item in rows if item.seq > seq]


def refuse_ledger_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("usage_ledger_disabled")


__all__ = [
    "UsageKind",
    "UsageRow",
    "append_usage",
    "from_seq",
    "refuse_ledger_when_disabled",
    "session_total",
]
