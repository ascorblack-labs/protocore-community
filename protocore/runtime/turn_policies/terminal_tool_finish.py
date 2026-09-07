"""The run ended because the model called the tool that ends runs.

Three dispatch paths can reach this ending — the serial one, the parallel
fan-out, and the recovery path that dispatches what a truncated message left
usable — and each of them used to write the ending out again in its own
words. That is how they came to differ: one paired the calls it never got to
and the others did not, so whether a finished run left a readable transcript
depended on which path the last call happened to take.

The ending is one decision, so it is one policy. It pairs every tool_use the
turn will never answer, and then seals the run. A host that installs its own
policy under this name replaces the seal outright — which is the point of
substitution, and why the core does not also seal behind it.
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
from protocore.runtime.loop_state import LoopState


class TerminalToolFinishPolicy:
    """Pair what the turn abandoned, then seal the run it completed."""

    name = "terminal_tool_finish"
    coordinates = frozenset({TurnCoordinate.terminal_tool_finish})

    __slots__ = ("_pair_orphans",)

    def __init__(self, *, pair_orphans: Callable[[Any], None]) -> None:
        self._pair_orphans = pair_orphans

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        # The dispatch loop stopped on the terminal result, so any sibling
        # call already in the transcript has no result and never will. The
        # outbound wire repair would forward-fill an opaque placeholder so a
        # re-stream does not fail, but the durable transcript would keep the
        # orphan, and a reader of it draws a call that was never answered.
        self._pair_orphans(turn.engine)
        turn.engine.transition_to(LoopState.COMPLETED)
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = "terminal_tool_completed"
        return
        yield  # pragma: no cover - the seal has nothing to say on the wire


__all__ = ["TerminalToolFinishPolicy"]
