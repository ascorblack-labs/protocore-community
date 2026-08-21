"""``LoopState`` StrEnum — :class:`QueryEngine` state machine.

2
and .

Distinct from :class:`protocore.contracts.types.RunStatus` (the persistent
PG-row mirror) and :class:`protocore.contracts.types.RunState` (the hot
Redis-Hash record): :class:`LoopState` is the **engine instance's
in-flight phase**, transitioning multiple times per persistent run.

Valid transitions enforced by :class:`InvalidStateTransitionError` — see
:func:`assert_transition`.
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


def is_terminal(state: LoopState) -> bool:
    """Return ``True`` if ``state`` is a terminal phase."""
    return state in TERMINAL_STATES


__all__ = [
    "TERMINAL_STATES",
    "InvalidStateTransitionError",
    "LoopState",
    "assert_transition",
    "is_terminal",
]
