"""The run was told to stop, and the turn is in the middle of something.

A cancel arrives on a different task than the one driving the turn, so it can
land in any await the turn is holding: the hook round-trip before the dispatch
loop opens, a tool's own run, a snapshot persist, the gather of a parallel
batch. The loop therefore asks this policy at every seam where the next step
would be a NEW side effect, and the guarantee it is asking about is exactly
that one — a call already dispatched keeps its real result, and nothing new is
dispatched once a stop has been seen.

What "stop" then MEANS is the one decision here. Every call already in the
transcript without a result is paired with a synthetic error, so the snapshot
a resume picks up is a readable transcript rather than a set of questions
nobody answered; the run moves to its cancelled terminal, and the reason
travels with it. Written out at each seam instead, those three steps drifted:
one seam paired and another did not, so whether a cancelled run left a
renderable transcript depended on which await the cancel happened to land in.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.runtime.events import TurnEvent

#: Pair what the turn abandoned and drive the run to its cancelled terminal.
CancelTeardown = Callable[[Any], AsyncIterator[TurnEvent]]


class CancellationPolicy:
    """End a turn that has been told to stop, before it does anything else."""

    name = "cancellation"
    coordinates = frozenset({TurnCoordinate.cancel_checkpoint})

    __slots__ = ("_teardown",)

    def __init__(self, *, teardown: CancelTeardown) -> None:
        self._teardown = teardown

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if not turn.engine.stop_requested:
            return
        async for event in self._teardown(turn.engine):
            yield event
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = "stop_requested"


__all__ = ["CancellationPolicy"]
