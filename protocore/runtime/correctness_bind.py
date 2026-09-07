"""Glue so intent, ledger, hooks, and recovery run inside query()."""
from __future__ import annotations

from typing import Any

from protocore.contracts.middleware import (
    ILifecycleRegistry,
    LifecycleContext,
    LifecycleOutcome,
    LifecycleVerdict,
)
from protocore.contracts.types import HookEvent
from protocore.runtime.events import EventType, TurnEvent


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


def lifecycle_registry(engine: Any) -> ILifecycleRegistry | None:
    """The run's lifecycle registry, or ``None`` when the seam is not live.

    One place decides whether extensions run at all — the registry has to be
    there and the run has to have the seam switched on. Every caller asks here,
    so a coordinate dispatched from the loop and a coordinate dispatched from
    the tool dispatcher can never disagree about whether the seam exists.
    """
    registry: ILifecycleRegistry | None = getattr(engine, "lifecycle_hooks", None)
    if registry is None or not engine.config.rc.typed_hooks_enabled:
        return None
    return registry


async def fire_lifecycle(
    engine: Any, point: HookEvent, payload: dict[str, Any]
) -> tuple[LifecycleOutcome, TurnEvent | None]:
    """Dispatch one lifecycle coordinate and report what came back.

    Returns the outcome plus the ``HOOK_FIRED`` event to yield, or ``None`` for
    the event when nothing was registered — a coordinate nobody listens to
    should not fill the stream with news of its own silence.
    """
    registry = lifecycle_registry(engine)
    if registry is None:
        return LifecycleOutcome(payload=dict(payload)), None
    if not registry.registrations(point):
        return LifecycleOutcome(payload=dict(payload)), None
    outcome = await registry.dispatch(
        LifecycleContext(
            point=point,
            run_id=engine.config.run_id,
            session_id=str(getattr(engine.config, "session_id", "") or ""),
            tenant_id=str(getattr(engine.config, "tenant_id", "") or ""),
            payload=dict(payload),
        )
    )
    event_payload: dict[str, Any] = {
        "hook": point.value,
        "decision": outcome.verdict.value,
    }
    if outcome.verdict is not LifecycleVerdict.allow:
        event_payload["reason"] = outcome.reason
        event_payload["decided_by"] = outcome.decided_by
    if outcome.failures:
        event_payload["failures"] = [item.error for item in outcome.failures]
    evt = TurnEvent(
        type=EventType.HOOK_FIRED,
        run_id=engine.config.run_id,
        payload=event_payload,
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
    "fire_lifecycle",
    "lifecycle_registry",
    "mark_intent_recovery",
    "persist_correctness",
]
