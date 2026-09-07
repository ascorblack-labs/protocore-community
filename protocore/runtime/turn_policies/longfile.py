"""Converging a large file the model is writing in pieces.

Two decisions, one subject. A turn that ended with no progress on a file the
model is part-way through writing gets the next tool forced — append more, or
seal what is there — instead of being allowed to drift. And a run that is
about to finish while a truncation-gated file sits unsealed gets that file
sealed first, because a run whose visible outcome is half a file is a run that
failed quietly.

Both were calls sitting inline in the turn driver, at six places, each with
the same three lines of bookkeeping after it. What the decision IS lives where
it always did — the convergence module and the two drivers this policy is
handed at construction. What moved is the decision to consult it, and the six
copies of what to do with the answer.
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

#: Seal an unsealed file at a finish. Takes the run itself — more than the
#: policy's own narrowed view of it — because sealing dispatches a tool, and
#: dispatch is the loop's, not the policy's.
LongFileSeal = Callable[[Any], AsyncIterator[TurnEvent]]

#: Advance the stall clock and, on a stall, force the next tool. Yields its
#: events and then one ``bool``: whether a tool was forced.
LongFileDrive = Callable[[Any], AsyncIterator[TurnEvent | bool]]


class LongFileConvergencePolicy:
    """Force the next tool on a stalled file, and seal one left unsealed."""

    name = "longfile_convergence"
    coordinates = frozenset(
        {
            TurnCoordinate.turn_end,
            TurnCoordinate.voluntary_finish,
            TurnCoordinate.terminal_tool_finish,
        }
    )

    __slots__ = ("_drive", "_seal")

    def __init__(self, *, seal: LongFileSeal, drive: LongFileDrive) -> None:
        self._seal = seal
        self._drive = drive

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.coordinate is TurnCoordinate.turn_end:
            async for event in self._converge(turn):
                yield event
            return
        # Every other coordinate this policy sits at is a finish, and a finish
        # is where an unsealed file is sealed. It runs BEFORE the policies that
        # ask whether the turn produced anything, because a seal produces a
        # terminal tool result and they must see it.
        async for event in self._seal(turn.engine):
            yield event

    async def _converge(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        forced = False
        async for produced in self._drive(turn.engine):
            if isinstance(produced, bool):
                forced = produced
            else:
                yield produced
        if not forced:
            return
        # A forced turn at the message-budget boundary must be GRANTED its
        # slot, or the next iteration exceeds the budget and the forced stream
        # is killed by the turn cap before the model can answer it — the
        # forced-round budget charged and history mutated, with no output.
        turn.outcome.extra_turn = True
        turn.outcome.reason = "longfile_forced"
        if turn.dispatched_tools:
            # The loop rebuilds the context on its own way round from a turn
            # that dispatched tools, so asking it to restart here would skip
            # the wake events and the steering that rebuild is bracketed by.
            return
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True


__all__ = ["LongFileConvergencePolicy", "LongFileDrive", "LongFileSeal"]
