"""``LoopState`` StrEnum — :class:`QueryEngine` state machine.

2
and .

Distinct from :class:`protocore.contracts.types.RunStatus` (the persistent
PG-row mirror) and :class:`protocore.contracts.types.RunState` (the hot
Redis-Hash record): :class:`LoopState` is the **engine instance's
in-flight phase**, transitioning multiple times per persistent run.

Valid transitions enforced by :class:`InvalidStateTransitionError` — see
:func:`assert_transition`.

``AWAITING`` is the one phase that means nothing on its own. Every other state
says what the run is doing; ``AWAITING`` says only that it stopped, and the
useful part — what it stopped FOR — used to live beside the state as a latch
holding one call id. A run in ``AWAITING`` with no wait recorded beside it is
therefore not a run in a legal phase: it is a run that will never be picked up,
because nothing names the answer that would move it, and it will not finish
either. :func:`assert_awaiting_is_witnessed` refuses that state outright, at
the transition, where the caller that failed to record the wait is still on the
stack — see :mod:`protocore.contracts.interrupt` for the value it must record.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Final


class LoopState(StrEnum):
    """In-flight :class:`QueryEngine` phase."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING = "awaiting"
    COMPACTING = "compacting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Terminal states — no further transitions allowed.
TERMINAL_STATES: Final[frozenset[LoopState]] = frozenset(
    {LoopState.COMPLETED, LoopState.FAILED, LoopState.CANCELLED}
)


# Legal transitions.
_VALID_TRANSITIONS: Final[dict[LoopState, frozenset[LoopState]]] = {
    LoopState.PENDING: frozenset(
        {
            LoopState.RUNNING,
            LoopState.CANCELLED,  # cancel before start
            LoopState.FAILED,  # init failure
        }
    ),
    LoopState.RUNNING: frozenset(
        {
            LoopState.AWAITING,
            LoopState.COMPACTING,
            LoopState.COMPLETED,
            LoopState.FAILED,
            LoopState.CANCELLED,
        }
    ),
    LoopState.AWAITING: frozenset(
        {
            LoopState.RUNNING,
            LoopState.CANCELLED,
            LoopState.FAILED,
        }
    ),
    LoopState.COMPACTING: frozenset(
        {
            LoopState.RUNNING,
            LoopState.FAILED,
            LoopState.CANCELLED,
        }
    ),
    # Terminal states have no outgoing edges.
    LoopState.COMPLETED: frozenset(),
    LoopState.FAILED: frozenset(),
    LoopState.CANCELLED: frozenset(),
}


class UnwitnessedAwaitError(ValueError):
    """``AWAITING`` was entered with nothing recorded that could end the wait.

    Raised at the transition rather than discovered at the resume. By the time
    a host reads the snapshot, the call that parked without recording its
    interrupt is long gone and all that is left is a run that stopped for no
    stated reason; here the caller that skipped the record is still the one
    being refused.
    """


class InvalidStateTransitionError(ValueError):
    """Raised when a transition not in the legal table is attempted.

    Carries the offending ``(from_state, to_state)`` pair for telemetry.
    """

    def __init__(self, from_state: LoopState, to_state: LoopState) -> None:
        super().__init__(
            f"invalid LoopState transition: {from_state.value} → {to_state.value}"
        )
        self.from_state = from_state
        self.to_state = to_state


def assert_transition(from_state: LoopState, to_state: LoopState) -> None:
    """Raise :class:`InvalidStateTransitionError` if the transition is illegal."""
    legal = _VALID_TRANSITIONS.get(from_state, frozenset())
    if to_state not in legal:
        raise InvalidStateTransitionError(from_state, to_state)


def assert_awaiting_is_witnessed(to_state: LoopState, pending_interrupts: int) -> None:
    """Refuse an ``AWAITING`` that names nothing it is waiting for.

    ``pending_interrupts`` is how many open waits stand beside the state. Any
    number above zero is legal — a batch of calls parked together is one
    ``AWAITING`` carrying three interrupts, which is the whole reason the wait
    is a list and not a latch. Zero is not.
    """
    if to_state is LoopState.AWAITING and pending_interrupts <= 0:
        raise UnwitnessedAwaitError(
            "cannot enter AWAITING with no pending interrupt: the run would "
            "stop with nothing recorded that could resume it. Park the typed "
            "interrupt for the call being waited on first."
        )


def is_terminal(state: LoopState) -> bool:
    """Return ``True`` if ``state`` is a terminal phase."""
    return state in TERMINAL_STATES


__all__ = [
    "TERMINAL_STATES",
    "InvalidStateTransitionError",
    "LoopState",
    "UnwitnessedAwaitError",
    "assert_awaiting_is_witnessed",
    "assert_transition",
    "is_terminal",
]
