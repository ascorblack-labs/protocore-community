"""The bounds of a run, and the one wind-down they all lead to.

Five things can tell a run it is out of room: the cumulative tool-call budget,
the cumulative output-token budget, a run-level precondition that burnt its
attempts, the cap on assistant messages in one turn, and the wall clock. They
used to end in five different places, each with its own idea of what "stop
now" meant, and only one of them actually stopped anything.

Now they all start the same wind-down: the model is told, its tools are taken
away, and it gets a small budget of turns to write the answer it already has
the evidence for. What differs between them is only which bound was reached,
which is a fact the transcript carries.

The two that cannot be wound down end the run instead. A precondition that was
asked for and never met must not be answered around — a caller who asked for
one and did not get it has been lied to. And a turn budget spent for the
SECOND time, with the wind-down already run, is exhaustion: it routes to the
failure terminal, so nothing downstream scores a budget-exhausted run green.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from protocore.contracts.llm import MaxOutputTokensExhausted
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.contracts.types import StopReason
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.turn_policies import (
    HistoryAppender,
    RunPredicate,
)

#: Enter the wind-down for a named cause; empty when it is already running or
#: switched off, in which case the bound has to end the run itself.
WindDownEntry = Callable[..., list[TurnEvent]]

#: The turns the wind-down gets, counted from where the run actually is.
WindDownBudget = Callable[[Any, int], int]

#: Drive a terminal that is an upstream failure, or a precondition's.
TerminalEmitter = Callable[..., AsyncIterator[TurnEvent]]

#: The end-of-turn frame, on a run already in its terminal state.
MessageStopEmitter = Callable[[Any, str], TurnEvent]

#: A state-change event across a real transition.
TransitionEmitter = Callable[..., TurnEvent]

#: Say, in the run's log, that the output-token budget is spent.
OutputBudgetLogger = Callable[[Any, int, int, int], None]

#: Close the wind-down out, when it has one last thing to say.
WindDownFinalizer = Callable[[Any], TurnEvent | None]


class RunCeilingsPolicy:
    """Read the bounds of a run and start the wind-down that answers them."""

    name = "run_ceilings"
    coordinates = frozenset(
        {TurnCoordinate.turn_start, TurnCoordinate.turn_budget}
    )

    __slots__ = (
        "_deadline_reached",
        "_enter_wind_down",
        "_has_final_answer",
        "_has_terminal_tool_result",
        "_llm_terminal",
        "_log_output_budget_exhausted",
        "_message_stop",
        "_pair_orphans",
        "_precondition_exhausted",
        "_precondition_terminal",
        "_tool_call_budget_reached",
        "_transition_event",
        "_wind_down_armed",
        "_wind_down_budget",
        "_wind_down_finalize",
    )

    def __init__(
        self,
        *,
        tool_call_budget_reached: RunPredicate,
        deadline_reached: RunPredicate,
        has_terminal_tool_result: RunPredicate,
        has_final_answer: RunPredicate,
        enter_wind_down: WindDownEntry,
        wind_down_budget: WindDownBudget,
        llm_terminal: TerminalEmitter,
        precondition_exhausted: RunPredicate,
        precondition_terminal: TerminalEmitter,
        wind_down_armed: RunPredicate,
        wind_down_finalize: WindDownFinalizer,
        pair_orphans: HistoryAppender,
        transition_event: TransitionEmitter,
        message_stop: MessageStopEmitter,
        log_output_budget_exhausted: OutputBudgetLogger,
    ) -> None:
        self._tool_call_budget_reached = tool_call_budget_reached
        self._deadline_reached = deadline_reached
        self._has_terminal_tool_result = has_terminal_tool_result
        self._has_final_answer = has_final_answer
        self._enter_wind_down = enter_wind_down
        self._wind_down_budget = wind_down_budget
        self._llm_terminal = llm_terminal
        self._precondition_exhausted = precondition_exhausted
        self._precondition_terminal = precondition_terminal
        self._wind_down_armed = wind_down_armed
        self._wind_down_finalize = wind_down_finalize
        self._pair_orphans = pair_orphans
        self._transition_event = transition_event
        self._message_stop = message_stop
        self._log_output_budget_exhausted = log_output_budget_exhausted

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.coordinate is TurnCoordinate.turn_start:
            async for event in self._before_the_message(turn):
                yield event
            return
        async for event in self._at_the_turn_cap(turn):
            yield event

    # -- the bounds read before an assistant message opens -------------------

    async def _before_the_message(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        # The bound that fires first on a run that is working rather than
        # spiralling, and the one that used to do nothing: it appended a
        # paragraph of English asking the agent to wrap up, said in the same
        # breath that tools still ran, and the agent made another eighteen
        # calls. Reaching it now starts the wind-down, so the NEXT turn has no
        # tools to make a nineteenth call with.
        if self._tool_call_budget_reached(engine):
            async for event in self._wind_down(
                turn, cause=_soft_stop.CAUSE_TOOL_CALL_BUDGET
            ):
                yield event
            if turn.outcome.directive is not TurnDirective.proceed:
                return

        # A spiral that re-emits a large truncated tool call burns output
        # tokens every round and, with unbounded history, eventually trips the
        # provider's context-length ceiling. Reaching this starts the
        # wind-down; reaching it AGAIN, with the wind-down already running and
        # out of turns, ends the run before it degrades into that ceiling.
        # A budget of zero disables the guard.
        budget = engine.rc.run_max_output_tokens_budget
        spent = engine.total_usage.output_tokens
        if budget > 0 and spent > budget and not engine.is_terminal:
            self._log_output_budget_exhausted(
                engine, spent, budget, turn.flags.assistant_message_idx
            )
            async for event in self._wind_down(
                turn, cause=_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET
            ):
                yield event
            if turn.outcome.directive is not TurnDirective.proceed:
                return
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "run_output_token_budget_exhausted"
            async for event in self._llm_terminal(
                engine,
                MaxOutputTokensExhausted(
                    "run_max_output_tokens_budget exhausted "
                    f"({spent} > {budget} output tokens) — terminating before "
                    "the context-length ceiling"
                ),
                kind="run_output_token_budget_exhausted",
            ):
                yield event
            return

        # A run-level tool precondition that burnt its attempt budget ends the
        # run. Read at the top of every iteration — before another attempt is
        # spent, and before the run can reach any completion path — because a
        # caller who asked for a precondition and did not get one has been
        # lied to, and answering anyway is the failure the mechanism exists to
        # prevent. Inert for a run that carries no preconditions.
        if self._precondition_exhausted(engine) and not engine.is_terminal:
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "tool_precondition_exhausted"
            async for event in self._precondition_terminal(engine):
                yield event

    # -- the bounds read once the message is counted -------------------------

    async def _at_the_turn_cap(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        if turn.flags.assistant_message_idx > turn.turn_budget:
            async for event in self._spent_turn_budget(turn):
                yield event
            return

        # The failure the wall clock guards: the run produced a good answer
        # and was killed by an external reaper before it got round to
        # delivering it. Reaching the deadline starts the same wind-down the
        # other bounds start, so "the run was cut short" reads the same in the
        # transcript whichever bound did it. Inert when no deadline is set,
        # and the wind-down's own arming latch is durable, so a run resumed
        # near or past its deadline does not start a second one.
        if self._deadline_reached(engine):
            async for event in self._wind_down(turn, cause=_soft_stop.CAUSE_DEADLINE):
                yield event

    async def _spent_turn_budget(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        # First time here the budget being spent starts the wind-down. This
        # branch used to hold three mechanisms — a forced artifact seal, a
        # nudge towards the terminal tool, and a synthetic answer the runtime
        # submitted on the model's behalf — each with its own latch and its own
        # idea of what "finish now" meant. The wind-down subsumes all three:
        # the sealer stays on the narrowed surface while a file is open, the
        # notice is the nudge, and the answer is written by the model rather
        # than assembled out of its last words.
        async for event in self._wind_down(turn, cause=_soft_stop.CAUSE_MAX_TURNS):
            yield event
        if turn.outcome.directive is not TurnDirective.proceed:
            return

        # Second time here: the wind-down was given its turns and produced no
        # answer, or it is switched off. If a typed stream error is what
        # started it and the run is still unanswered, that error is the run's
        # outcome — surfacing it beats completing silently on a budget the
        # error is the reason we ran out of. Only if the wind-down produced
        # nothing, though: one that got the model to write its answer did the
        # job it exists for, and re-raising the upstream failure over that
        # answer would report a run that answered as a run that failed.
        turn.outcome.directive = TurnDirective.end_turn
        stored = turn.stored_stream_error
        if (
            stored is not None
            and not self._has_terminal_tool_result(engine)
            and not self._has_final_answer(engine)
        ):
            exc, kind = stored
            turn.outcome.reason = kind
            async for event in self._llm_terminal(engine, exc, kind=kind):
                yield event
            return

        # Exhaustion, not a successful completion: route to the FAILURE-class
        # terminal so nothing downstream scores a budget-exhausted run green.
        # A run that got here THROUGH the wind-down says so, because being told
        # to stop and given turns to finish in is a different fact about a run
        # than simply running out of them.
        #
        # The transition happens BEFORE the stop frame, as at every other
        # failure seam: the consumer of this stream reads the run's state while
        # handling that frame, and a transition placed after it runs only on
        # the next pull, which is the end of the stream.
        from_state = engine.state
        # Pair any tool_use the budget-spent turn appended and never got a
        # result for; without this the snapshot carries an orphan and every
        # later request forward-fills an opaque synthetic in its place.
        self._pair_orphans(engine)
        stopped = self._wind_down_armed(engine)
        finalized = self._wind_down_finalize(engine)
        if finalized is not None:
            yield finalized
        engine.transition_to(LoopState.FAILED)
        turn.outcome.reason = (
            "soft_stop_exhausted" if stopped else "max_turns_exhausted"
        )
        yield self._transition_event(
            engine, from_state, LoopState.FAILED, reason=turn.outcome.reason
        )
        yield self._message_stop(
            engine,
            _soft_stop.STOP_REASON if stopped else StopReason.max_turns.value,
        )

    async def _wind_down(
        self, turn: TurnContext, *, cause: str
    ) -> AsyncIterator[TurnEvent]:
        """Start the wind-down, or say nothing when it cannot be started."""
        events = self._enter_wind_down(turn.engine, cause=cause)
        if not events:
            return
        for event in events:
            yield event
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.turn_budget = self._wind_down_budget(
            turn.engine, turn.flags.assistant_message_idx
        )
        turn.outcome.rebuild_context = True
        turn.outcome.reason = cause
        await turn.engine.persist_snapshot()


__all__ = ["RunCeilingsPolicy", "WindDownBudget", "WindDownEntry"]
