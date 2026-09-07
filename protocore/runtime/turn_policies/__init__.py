"""The turn policies, and the order the core consults them in.

Each module here holds one product decision that used to be a branch inside
the turn driver. The registry below is what makes them a set rather than a
pile: the order is declared once, in :data:`TURN_POLICY_ORDER`, and it is the
core's, not the caller's. That matters at the seams where two policies meet —
an unsealed file is sealed *before* the guard that asks whether the turn
produced an answer, because a seal produces one — and an order that came from
whichever list the host happened to build would make that a coincidence.

A registry consults, in order, every policy registered at the coordinate, and
stops at the first one that returns anything but
:attr:`~protocore.contracts.turn_policy.TurnDirective.proceed`. A policy that
has taken the turn somewhere else is not followed by one that assumes it did
not.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from protocore.contracts.turn_policy import (
    ITurnPolicy,
    TurnContext,
    TurnCoordinate,
    TurnDirective,
    TurnPolicyOutcome,
)
from protocore.runtime.events import TurnEvent

#: A state-change event that reports no transition: a policy saying something
#: happened, at a run that stayed where it was. The run goes in as itself —
#: more than the narrowed view a policy reads — because building the event
#: needs the wire identity of the round, which is the loop's to know.
StateChangeEmitter = Callable[[Any, str], "TurnEvent"]

#: Append a message to the run's history. The loop owns what the text is; a
#: policy owns the decision that the moment calls for it.
HistoryAppender = Callable[[Any], None]

#: A yes/no question about the run that the loop already knows how to answer.
RunPredicate = Callable[[Any], bool]


@dataclass(frozen=True, slots=True)
class RunCounter:
    """One of the run's own bounded counts, reachable without owning the run.

    A policy owns the decision the count feeds, not the field it lives in: the
    field stays an attribute of the run, and the policy is handed the three
    things it may do to it.
    """

    #: How much of the count is spent.
    read: Callable[[Any], int]
    #: Spend one, and say which one this was.
    charge: Callable[[Any], int]
    #: Give the whole count back.
    reset: Callable[[Any], None]

#: The order the core consults policies in, whatever order they were built in.
#: A name absent from this tuple is not a policy this core knows about, and the
#: registry says so at construction rather than at the seam.
TURN_POLICY_ORDER: Final[tuple[str, ...]] = (
    "longfile_convergence",
    "run_ceilings",
    "empty_model_turn",
    "truncated_tool_call_recovery",
    "output_cap_recovery",
    "terminal_nudge",
    "answer_floor",
    "empty_completion_guard",
    "terminal_tool_finish",
    "per_iteration_compaction",
    "stream_loop_guard",
    "provider_failure",
    "cancellation",
)


class UnknownTurnPolicyError(ValueError):
    """A policy was registered under a name the core's order does not carry."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"unknown turn policy {name!r}: add it to TURN_POLICY_ORDER, which "
            "is what fixes where it runs relative to the others"
        )
        self.name = name


class UnsupportedTurnDirectiveError(ValueError):
    """A policy answered a coordinate with a directive that seam cannot obey.

    A finish is not a place the loop can be sent back from: the tool that
    ended the run has already produced its result and the history it left is
    the run's last word. A policy that asks for a restart there is asking for
    something the loop would have to drop on the floor, and dropping it
    quietly is how a directive comes to mean nothing.
    """

    def __init__(self, coordinate: TurnCoordinate, directive: TurnDirective) -> None:
        super().__init__(
            f"turn policy answered {coordinate.value!r} with {directive.value!r}: "
            "this seam honours 'proceed' and 'end_turn' only"
        )
        self.coordinate = coordinate
        self.directive = directive


class TurnPolicyRegistry:
    """The installed policies, indexed by coordinate, ordered by the core."""

    __slots__ = ("_by_coordinate", "_policies")

    def __init__(self, policies: Iterable[ITurnPolicy]) -> None:
        ordered = sorted(policies, key=_order_index)
        self._policies: tuple[ITurnPolicy, ...] = tuple(ordered)
        by_coordinate: dict[TurnCoordinate, list[ITurnPolicy]] = {}
        for policy in self._policies:
            for coordinate in policy.coordinates:
                by_coordinate.setdefault(coordinate, []).append(policy)
        self._by_coordinate: dict[TurnCoordinate, tuple[ITurnPolicy, ...]] = {
            coordinate: tuple(entries)
            for coordinate, entries in by_coordinate.items()
        }

    @property
    def policies(self) -> tuple[ITurnPolicy, ...]:
        """Every installed policy, in the order they are consulted."""
        return self._policies

    def merged_with(self, other: TurnPolicyRegistry) -> TurnPolicyRegistry:
        """This set with ``other``'s policies substituted in by name.

        A policy of ``other`` displaces the one this set carries under the
        same name, and a name this set does not carry is added. Nothing is
        removed: substitution is how a decision is replaced, and a set that
        replaced the whole registry would take the core's own bounds with it.
        """
        by_name: dict[str, ITurnPolicy] = {
            policy.name: policy for policy in self._policies
        }
        for policy in other.policies:
            by_name[policy.name] = policy
        return TurnPolicyRegistry(by_name.values())

    def at(self, coordinate: TurnCoordinate) -> Sequence[ITurnPolicy]:
        """The policies consulted at ``coordinate``, in order."""
        return self._by_coordinate.get(coordinate, ())

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        """Consult every policy at ``turn.coordinate`` until one takes the turn.

        Each policy writes into a fresh outcome, so it can never read the one
        before it as if it were its own. What the caller reads afterwards is
        the accumulation of the consultation, not the last policy's copy: a
        directive is a statement about control flow and belongs to the single
        policy that took the turn, but ``extra_turn``, ``rebuild_context`` and
        ``turn_budget`` are GRANTS, and a policy that asks for one while
        answering ``proceed`` is asking the loop, not the policy after it. A
        reset per policy loses exactly those, and loses them silently — the
        grant is still made, the loop simply never hears it.
        """
        granted = TurnPolicyOutcome()
        for policy in self.at(turn.coordinate):
            turn.outcome = TurnPolicyOutcome()
            async for event in policy.apply(turn):
                yield event
            spoken = turn.outcome
            granted.extra_turn = granted.extra_turn or spoken.extra_turn
            granted.rebuild_context = granted.rebuild_context or spoken.rebuild_context
            if spoken.turn_budget is not None:
                granted.turn_budget = spoken.turn_budget
            if spoken.reason:
                granted.reason = spoken.reason
            if spoken.tool_calls is not None:
                granted.tool_calls = spoken.tool_calls
            granted.directive = spoken.directive
            turn.outcome = granted
            if spoken.directive is not TurnDirective.proceed:
                return


def _order_index(policy: ITurnPolicy) -> int:
    try:
        return TURN_POLICY_ORDER.index(policy.name)
    except ValueError:
        raise UnknownTurnPolicyError(policy.name) from None


__all__ = [
    "TURN_POLICY_ORDER",
    "HistoryAppender",
    "RunCounter",
    "RunPredicate",
    "StateChangeEmitter",
    "TurnPolicyRegistry",
    "UnknownTurnPolicyError",
    "UnsupportedTurnDirectiveError",
]
