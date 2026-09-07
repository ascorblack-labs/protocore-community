"""What every dispatch of a message's calls has to agree about.

Three places run the calls of one assistant message: the driver's own loop,
and the two recovery policies that dispatch what a cut-short message left
usable. They ask the same questions in the same order — has a gate held this
call, has the answer to an earlier one already ended the batch — and when each
of them wrote the questions out in its own words, the copies drifted: one
guard was tested against a flag its own walk had cleared, one release on
cancel existed in a single copy, and one repair turn was allowed to land in
the middle of a batch on two paths out of three.

So the questions live here, once, and a new entry into dispatch inherits the
answers instead of deriving them again.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    MessageRole,
    ToolCall,
)
from protocore.runtime.events import EventType, TurnEvent

#: Run one tool call, yielding everything it produces.
ToolDispatcher = Callable[[Any, ToolCall], AsyncIterator[TurnEvent]]

#: Record that a call is held at a gate, and return the wait it became.
CallParker = Callable[..., Any]


def prose_gate_just_injected(engine: Any) -> bool:
    """True iff the last thing in history is the prose gate's repair turn.

    The veto answers the vetoed call with a non-terminal error and appends a
    synthetic user turn asking for the answer in prose. A walk that carried
    on past that would put later sibling results AFTER a user message, and
    the durable transcript would then read as a batch interrupted by a
    question nobody asked. Whoever sees this stops the batch and lets the
    next round carry the correction.

    Pure: it reads the tail of history and nothing else.
    """
    history = engine.history
    if not history:
        return False
    last = history[-1]
    return (
        last.role is MessageRole.user
        and last.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
    )


async def dispatch_parking_holds(
    engine: Any,
    call: ToolCall,
    *,
    dispatch: ToolDispatcher,
    park: CallParker,
    parked: list[Any],
) -> AsyncIterator[TurnEvent]:
    """Run one call, parking it instead of finishing it if a gate holds it.

    Whether the call was held is visible as growth of ``parked`` — the same
    list the caller announces the whole message's open set from — so the
    caller needs no second channel for the answer.

    The wait is written durably BEFORE the envelope that announces it leaves
    the run, and that order is the whole point of doing it here rather than
    at the call site. A host is entitled to treat the announcement as final:
    record the pause, release the driver, stop reading. Whatever the run
    writes after that is written into a generator nobody is consuming any
    more — so a snapshot persisted after the yield is a snapshot that host
    never sees, and it comes back to a run whose durable record says it is
    waiting for nothing and refuses the very approval it was handed.
    """
    async for event in dispatch(engine, call):
        if event.type is EventType.TOOL_CALL_PENDING:
            parked.append(park(engine, call.id, event=event))
            await engine.persist_snapshot()
            yield event
            return
        yield event


async def park_deferred_hold(
    engine: Any,
    call_id: str,
    events: list[TurnEvent],
    *,
    park: CallParker,
    parked: list[Any],
) -> AsyncIterator[TurnEvent]:
    """Park one call a gate held inside a fanned-out batch, then re-emit it.

    The fan-out paths do not stream their calls: each one comes back as a
    finished list of events with the outcome beside it, so a hold registered
    mid-turn is discovered after the fact rather than at the moment the
    ``tool_call_pending`` was produced. Parking it is therefore a walk over
    that list — record the wait against the FIRST pending event, so the card
    the host draws carries what the gate said, and emit everything in the
    order the dispatcher produced it.

    A batch whose events carry no pending envelope at all is still parked, on
    the call id alone. The alternative reading — the gate held the call but
    said nothing, so nothing is waiting — is the one that finalises a run over
    a call no person ever decided.

    The wait is appended to ``parked``, the same list the caller announces the
    whole message's open set from, and it is durable before the envelope that
    announces it is emitted — for the reason spelled out on
    :func:`dispatch_parking_holds`, which a fanned-out batch shares.
    """
    pending_seen = False
    for event in events:
        if event.type is EventType.TOOL_CALL_PENDING and not pending_seen:
            parked.append(park(engine, call_id, event=event))
            await engine.persist_snapshot()
            pending_seen = True
        yield event
    if not pending_seen:
        parked.append(park(engine, call_id))
        await engine.persist_snapshot()


__all__ = [
    "dispatch_parking_holds",
    "park_deferred_hold",
    "prose_gate_just_injected",
]
