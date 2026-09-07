"""The one push towards the run's terminal tool.

A model that says "done, I created the file" and calls nothing has not
created the file. The run is told so, once, and gets one more message to do
the thing it claimed. Once, because a second telling never produced a
different answer and a run that keeps being told burns its budget on being
told.

The latch that spends the single push is turn-local state the loop can still
see; everything else about the decision is here.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.runtime.events import TurnEvent
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunPredicate,
    StateChangeEmitter,
)


class TerminalNudgePolicy:
    """Push a prose-only finish towards the terminal tool, once per turn."""

    name = "terminal_nudge"
    coordinates = frozenset({TurnCoordinate.finish_nudge})

    __slots__ = ("_append", "_required", "_state_change")

    def __init__(
        self,
        *,
        required: RunPredicate,
        append: HistoryAppender,
        state_change: StateChangeEmitter,
    ) -> None:
        self._required = required
        self._append = append
        self._state_change = state_change

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.flags.terminal_nudge_used or not self._required(turn.engine):
            return
        turn.flags.terminal_nudge_used = True
        self._append(turn.engine)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.extra_turn = True
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "terminal_tool_nudge"
        yield self._state_change(turn.engine, "terminal_tool_nudge")


__all__ = ["TerminalNudgePolicy"]
