"""Intent commit + replay policy for mutating tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from protocore.contracts.runtime_constants import RuntimeConstants

ReplayPolicy = Literal["never", "safe"]
IntentStatus = Literal["open", "settled", "interrupted"]


@dataclass(slots=True)
class IntentRecord:
    operation_id: str
    tool_name: str
    tool_call_id: str
    reserved_result_ids: list[str]
    replay: ReplayPolicy
    status: IntentStatus = "open"
    result: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "reserved_result_ids": list(self.reserved_result_ids),
            "replay": self.replay,
            "status": self.status,
            "result": self.result,
        }


def replay_policy_for(tool_name: str, rc: RuntimeConstants) -> ReplayPolicy:
    never = {item.strip() for item in rc.intent_never_replay_tools.split(",") if item.strip()}
    return "never" if tool_name in never else "safe"


def commit_intent(
    *,
    tool_name: str,
    tool_call_id: str,
    rc: RuntimeConstants,
) -> IntentRecord | None:
    if not rc.intent_settlement_enabled:
        return None
    reserved = f"res_{uuid4().hex[:12]}"
    return IntentRecord(
        operation_id=f"op_{uuid4().hex[:12]}",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        reserved_result_ids=[reserved],
        replay=replay_policy_for(tool_name, rc),
    )


def settle_intent(intent: IntentRecord, *, result: str) -> IntentRecord:
    intent.status = "settled"
    intent.result = result
    return intent


def interrupt_intent(intent: IntentRecord) -> IntentRecord:
    intent.status = "interrupted"
    intent.result = "interrupted"
    return intent


def resume_open_intents(intents: list[IntentRecord]) -> list[IntentRecord]:
    """Crash mid-never → synthetic interrupted. Mid-safe stays open for replay."""
    out: list[IntentRecord] = []
    for item in intents:
        if item.status != "open":
            out.append(item)
            continue
        if item.replay == "never":
            out.append(interrupt_intent(item))
        else:
            out.append(item)
    return out


def should_dispatch(intent: IntentRecord | None) -> bool:
    if intent is None:
        return True
    return intent.status == "open"


def should_skip_never_replay(intent: IntentRecord | None) -> bool:
    return intent is not None and intent.status == "interrupted" and intent.replay == "never"


def refuse_intent_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("intent_settlement_disabled")


__all__ = [
    "IntentRecord",
    "IntentStatus",
    "ReplayPolicy",
    "commit_intent",
    "interrupt_intent",
    "refuse_intent_when_disabled",
    "replay_policy_for",
    "resume_open_intents",
    "settle_intent",
    "should_dispatch",
    "should_skip_never_replay",
]
