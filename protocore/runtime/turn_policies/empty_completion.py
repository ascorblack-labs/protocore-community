"""The guard on a finish that delivered nothing, and the finish itself.

The model ended its turn with a stop and emitted no text, no tool call and no
reasoning, on a run that has neither a visible answer nor a terminal tool
result. Sealing that as completed loses the turn silently: nothing was
appended to history, so a reload shows a finished run with no answer in it.

So the run gets a bounded number of further attempts, and when that budget is
spent it fails loudly rather than reporting an empty turn as a clean answer.
A turn that already delivered an answer — or carried text or reasoning, or was
sealed by a terminal tool result — never reaches the guard at all, which is
what keeps it from competing with the floor under a short answer.

The voluntary finish itself lives here too, at every seam a run can reach it
from: the guard's whole purpose is to decide whether this finish may happen,
and the two were separated only by the accident of being written at four
different places in one function.
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
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunPredicate,
    StateChangeEmitter,
)

#: A terminal, or a completion, driven as a stream of events.
TerminalEmitter = Callable[[Any], AsyncIterator[TurnEvent]]

#: How many re-drives this run has already spent.
RedriveCount = Callable[[Any], int]


class EmptyCompletionGuardPolicy:
    """Refuse a finish that delivered nothing, until the budget runs out."""

    name = "empty_completion_guard"
    coordinates = frozenset(
        {TurnCoordinate.voluntary_finish, TurnCoordinate.terminal_tool_finish}
    )

    __slots__ = (
        "_append_redrive_nudge",
        "_charge_redrive",
        "_empty_terminal",
        "_has_final_answer",
        "_has_terminal_tool_result",
        "_redrives_spent",
        "_state_change",
        "_voluntary_completion",
    )

    def __init__(
        self,
        *,
        has_terminal_tool_result: RunPredicate,
        has_final_answer: RunPredicate,
        redrives_spent: RedriveCount,
        charge_redrive: HistoryAppender,
        append_redrive_nudge: HistoryAppender,
        empty_terminal: TerminalEmitter,
        voluntary_completion: TerminalEmitter,
        state_change: StateChangeEmitter,
    ) -> None:
        self._has_terminal_tool_result = has_terminal_tool_result
        self._has_final_answer = has_final_answer
        self._redrives_spent = redrives_spent
        self._charge_redrive = charge_redrive
        self._append_redrive_nudge = append_redrive_nudge
        self._empty_terminal = empty_terminal
        self._voluntary_completion = voluntary_completion
        self._state_change = state_change

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if turn.coordinate is TurnCoordinate.voluntary_finish and self._empty(turn):
            rc = engine.rc
            if self._redrives_spent(engine) < rc.empty_completion_guard_max_redrives:
                self._charge_redrive(engine)
                self._append_redrive_nudge(engine)
                turn.outcome.directive = TurnDirective.restart_turn
                turn.outcome.extra_turn = True
                turn.outcome.rebuild_context = True
                turn.outcome.reason = "empty_completion_redrive"
                yield self._state_change(engine, "empty_completion_redrive")
                return
            # The budget is spent and there is still no answer. Fail out loud
            # rather than sealing a silent empty completion.
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "empty_completion_exhausted"
            async for event in self._empty_terminal(engine):
                yield event
            return
        async for event in self._voluntary_completion(engine):
            yield event

    def _empty(self, turn: TurnContext) -> bool:
        engine = turn.engine
        return (
            engine.rc.empty_completion_guard_enabled
            and not turn.text_emitted
            and not turn.reasoning_emitted
            and not self._has_terminal_tool_result(engine)
            and not self._has_final_answer(engine)
        )


__all__ = ["EmptyCompletionGuardPolicy", "RedriveCount", "TerminalEmitter"]
