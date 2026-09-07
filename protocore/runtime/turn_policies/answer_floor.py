"""The floor under an answer that ends a run.

A run that delegated work, produced files and then said forty characters
about them has not answered. Neither has one whose whole answer is the path
of a file the reader cannot open. Both used to complete here, quietly, as
successes — the gate that catches this at the tool seam never sees a turn
that called no tool.

So the same floor applies at this completion, and the two failures are told
apart because they cost different things. An answer that is merely too short
spends the run's single durable repair, so the test fires at most once across
both paths. An answer that is a pointer draws on a budget of its own, because
one repair turn was measured to detect that failure without fixing it. Both
are bounded, so neither can loop.

And when the floor does not take the completion, the pointer refusal is
released: a run that spent its whole budget and still hands the reader a
filing notice is the one outcome this mechanism exists to make visible, and
it is invisible everywhere else.
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

#: What makes an answer a pointer: the path it names, and the two sizes that
#: make the case — or ``None`` when the answer is not one.
PointerEvidence = tuple[str, int, int]

#: Read the pointer evidence over the answer window, before anything is added.
PointerReader = Callable[[Any], PointerEvidence | None]

#: Charge one attempt of the pointer budget; returns which attempt this was.
AttemptCharge = Callable[[Any], int]

#: Say, in the run's log, which of the two tests fired and what it measured.
RepairLogger = Callable[[Any, "PointerEvidence | None", int], None]


class AnswerFloorPolicy:
    """Refuse a completion whose answer nobody can read, once."""

    name = "answer_floor"
    coordinates = frozenset({TurnCoordinate.answer_floor})

    __slots__ = (
        "_append_repair",
        "_applies",
        "_charge_pointer",
        "_log",
        "_pointer",
        "_release_pointer",
        "_spend_short_answer_repair",
        "_state_change",
    )

    def __init__(
        self,
        *,
        applies: RunPredicate,
        pointer: PointerReader,
        charge_pointer: AttemptCharge,
        release_pointer: HistoryAppender,
        spend_short_answer_repair: HistoryAppender,
        append_repair: HistoryAppender,
        log: RepairLogger,
        state_change: StateChangeEmitter,
    ) -> None:
        self._applies = applies
        self._pointer = pointer
        self._charge_pointer = charge_pointer
        self._release_pointer = release_pointer
        self._spend_short_answer_repair = spend_short_answer_repair
        self._append_repair = append_repair
        self._log = log
        self._state_change = state_change

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        repair_text = (
            engine.prompt_text("finalize_prose_gate_repair")
            if self._applies(engine)
            else ""
        )
        if not repair_text:
            # Either the floor did not fire, or the repair it would inject is
            # empty — and an empty user turn is worse than no repair, so the
            # latch stays unspent and the attempt uncharged, exactly as when
            # the gate is off. Release the pointer refusal on the way past.
            self._release_pointer(engine)
            return
        # Read the evidence BEFORE anything is appended: the measurement is
        # taken over the answer window, and the repair turn is part of history
        # the moment it lands.
        pointer = self._pointer(engine)
        attempt = 0
        if pointer is None:
            self._spend_short_answer_repair(engine)
        else:
            attempt = self._charge_pointer(engine)
        self._append_repair(engine)
        # Persist immediately after the latch and the injection, so a crash or
        # a resume onto another process in the gap can lose neither the latch
        # (and re-fire the repair) nor the correction.
        await engine.persist_snapshot()
        self._log(engine, pointer, attempt)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.extra_turn = True
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "finalize_prose_gate_plain_stop_repair"
        yield self._state_change(engine, "finalize_prose_gate_plain_stop_repair")


__all__ = ["AnswerFloorPolicy", "PointerEvidence", "PointerReader"]
