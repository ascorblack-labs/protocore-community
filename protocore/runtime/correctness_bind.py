"""Glue so intent, ledger, hooks, and recovery run inside query()."""
from __future__ import annotations

from typing import Any

from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.typed_hooks import HookOutcome, dispatch_hook


def persist_correctness(engine: Any) -> None:
    writer = getattr(engine, "persist_correctness", None)
    if callable(writer):
        writer(engine)


def commit_usage(
    engine: Any,
    *,
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    success: bool,
    operation_id: str | None = None,
) -> TurnEvent | None:
    if not engine.config.rc.usage_ledger_enabled:
        return None
    from protocore.runtime.usage_ledger import append_usage

    engine.usage_rows = append_usage(
        list(engine.usage_rows),
        kind=kind,
        run_id=engine.config.run_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
        operation_id=operation_id,
        rc=engine.config.rc,
    )
    persist_correctness(engine)
    if not engine.usage_rows:
        return None
    return TurnEvent(
        type=EventType.USAGE_COMMITTED,
        run_id=engine.config.run_id,
        payload=engine.usage_rows[-1].to_dict(),
    )


def fire_typed_hook(engine: Any, name: str, payload: dict[str, Any]) -> tuple[HookOutcome, TurnEvent | None]:
    from protocore.runtime.typed_hooks import HookOutcome as Outcome

    if not engine.config.rc.typed_hooks_enabled or engine.typed_hook_registry is None:
        return Outcome(decision="allow"), None
    outcome = dispatch_hook(engine.typed_hook_registry, name, payload, engine.config.rc)
    evt = TurnEvent(
        type=EventType.HOOK_FIRED,
        run_id=engine.config.run_id,
        payload={"hook": name, "decision": outcome.decision},
    )
    return outcome, evt


def mark_intent_recovery(engine: Any, intent: Any) -> list[TurnEvent]:
    from protocore.runtime.telemetry import mark_recovery, start_span

    events: list[TurnEvent] = []
    span = start_span("tool", rc=engine.config.rc, tool=getattr(intent, "tool_name", ""))
    marked = mark_recovery(span, intent_id=str(getattr(intent, "operation_id", "")))
    if marked is not None:
        engine.spans.append(marked)
        events.append(
            TurnEvent(
                type=EventType.RECOVERY_MARKED,
                run_id=engine.config.run_id,
                payload=marked.to_dict(),
            )
        )
    return events


__all__ = [
    "commit_usage",
    "fire_typed_hook",
    "mark_intent_recovery",
    "persist_correctness",
]
