"""A round that came back, and whether it is work or a model repeating itself.

Two questions are asked of every usable round before anything is read off it,
and they are the same question at two scales. Did the model just say the same
thing twice inside one message — the "and then, and then, and then" a stream
falls into when it has run out of anything to add? And is it asking, again,
for a call it has already made with exactly these arguments?

Both are answered with a threshold and a consequence, and the consequence is
the sharp part: past a bound the run stops honouring what the model asked
for. Discarding a model's tool calls is an opinion about the model, with its
own configured bound, so it belongs to a policy and not to the loop that
happens to hold the round.

The stripping of the repeated tail is not here. Rewriting the buffers of a
round in flight is the loop's own bookkeeping, offered to this policy as one
probe that says what it removed.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
)
from protocore.contracts.types import ToolCall
from protocore.runtime.events import TurnEvent
from protocore.runtime.turn_policies import RunCounter

#: Split calls into the ones the run will make and the events for the rest.
IdenticalCallFilter = Callable[
    [Any, Sequence[ToolCall], int], tuple[list[ToolCall], list[TurnEvent]]
]

#: The envelope that tells the reader a repeated tail was taken off a round.
GuardEvent = Callable[[Any, str, int, int], TurnEvent]


class StreamLoopGuardPolicy:
    """Refuse a round that is repeating rather than working."""

    name = "stream_loop_guard"
    coordinates = frozenset({TurnCoordinate.stream_settled})

    __slots__ = ("_block_identical", "_guard_event", "_nudges")

    def __init__(
        self,
        *,
        nudges: RunCounter,
        block_identical: IdenticalCallFilter,
        guard_event: GuardEvent,
    ) -> None:
        self._nudges = nudges
        self._block_identical = block_identical
        self._guard_event = guard_event

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        calls: Sequence[ToolCall] = turn.pending_tool_calls
        repeated = turn.stream_repeat_guard()
        if repeated is not None:
            kind, stripped_chars = repeated
            nudge_index = self._nudges.charge(engine)
            yield self._guard_event(engine, kind, nudge_index, stripped_chars)
            if nudge_index > engine.rc.loop_guard_nudge_max:
                # Past the bound the nudge has stopped working, and what the
                # model asked for on this round is part of the loop it is in.
                # The round keeps its prose; its calls are not made.
                calls = []
        executable, blocked = self._block_identical(
            engine, calls, self._nudges.read(engine)
        )
        for event in blocked:
            yield event
        turn.outcome.tool_calls = executable


__all__ = ["StreamLoopGuardPolicy"]
