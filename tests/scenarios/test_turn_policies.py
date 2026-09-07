"""The turn-policy seam, driven from outside the loop.

A policy is consulted at named coordinates, in an order the core owns, and
what it answers decides what the loop does next. These scenarios install
policy sets of their own through the public constructor seam and assert on
what a reader of the run sees — the events forwarded, the requests the model
received — so they keep working however the inside of the turn is written.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.contracts.types import TERMINAL_TOOL_METADATA_KEY
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.turn_policies import (
    TURN_POLICY_ORDER,
    TurnPolicyRegistry,
    UnknownTurnPolicyError,
    UnsupportedTurnDirectiveError,
)

from .conftest import ScenarioFactory, ScriptedTool, default_rc


class _RecordingPolicy:
    """A policy that answers ``proceed`` and remembers where it was asked."""

    def __init__(
        self,
        name: str,
        coordinates: frozenset[TurnCoordinate],
        *,
        seen: list[tuple[str, TurnCoordinate]],
    ) -> None:
        self._name = name
        self._coordinates = coordinates
        self.seen = seen

    @property
    def name(self) -> str:
        return self._name

    @property
    def coordinates(self) -> frozenset[TurnCoordinate]:
        return self._coordinates

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        self.seen.append((self._name, turn.coordinate))
        return
        yield  # pragma: no cover - the generator never yields


class _OneMoreTurnPolicy:
    """A policy that asks for exactly one more assistant message, once."""

    name = "terminal_nudge"
    coordinates = frozenset({TurnCoordinate.finish_nudge})

    def __init__(self) -> None:
        self.spent = False

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if self.spent:
            return
        self.spent = True
        turn.engine.history.append(
            _user_message("say more about what you just did"),
        )
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.extra_turn = True
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "scenario_extra_turn"
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_extra_turn"},
        )


class _EndTheTurnPolicy:
    """A policy that takes the turn over and ends it."""

    name = "answer_floor"
    coordinates = frozenset({TurnCoordinate.answer_floor})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        turn.outcome.directive = TurnDirective.end_turn
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_end_turn"},
        )


def _user_message(text: str):  # type: ignore[no-untyped-def]
    from protocore.contracts.types import Message, MessageRole, TextBlock

    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


async def test_a_turn_consults_its_policies_at_the_seams_it_passes(
    scenario: ScenarioFactory,
) -> None:
    seen: list[tuple[str, TurnCoordinate]] = []
    policy = _RecordingPolicy(
        "answer_floor", frozenset(TurnCoordinate), seen=seen
    )
    tool = ScriptedTool(tool_name="Note", content="noted")
    run = scenario(tools=[tool], turn_policies=TurnPolicyRegistry([policy]))
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Note")
    run.llm.queue_response(text="a long enough answer about the note")

    await run.run("do the thing")

    coordinates = [coordinate for _, coordinate in seen]
    assert TurnCoordinate.turn_start in coordinates
    assert TurnCoordinate.iteration_end in coordinates
    assert TurnCoordinate.turn_end in coordinates
    assert TurnCoordinate.voluntary_finish in coordinates


async def test_a_policy_that_asks_for_another_message_gets_one(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(turn_policies=TurnPolicyRegistry([_OneMoreTurnPolicy()]))
    run.llm.queue_response(text="short")
    run.llm.queue_response(text="the fuller answer")

    await run.run("do the thing")

    assert len(run.requests) == 2
    assert "scenario_extra_turn" in run.state_reasons()
    assert "the fuller answer" in "".join(run.history_texts())


async def test_a_policy_that_ends_the_turn_is_the_last_word(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(turn_policies=TurnPolicyRegistry([_EndTheTurnPolicy()]))
    run.llm.queue_response(text="an answer nobody gets to accept")

    await run.run("do the thing")

    assert "scenario_end_turn" in run.state_reasons()
    assert len(run.requests) == 1


async def test_the_core_owns_the_order_policies_are_consulted_in(
    scenario: ScenarioFactory,
) -> None:
    seen: list[tuple[str, TurnCoordinate]] = []
    late = _RecordingPolicy(
        "per_iteration_compaction",
        frozenset({TurnCoordinate.turn_start}),
        seen=seen,
    )
    early = _RecordingPolicy(
        "longfile_convergence",
        frozenset({TurnCoordinate.turn_start}),
        seen=seen,
    )
    # Built in the wrong order on purpose: the registry sorts by the core's
    # declared order, so a host cannot change what runs first by changing the
    # order it hands the policies over in.
    run = scenario(turn_policies=TurnPolicyRegistry([late, early]))
    run.llm.queue_response(text="an answer")

    await run.run("do the thing")

    assert [name for name, _ in seen][:2] == [
        "longfile_convergence",
        "per_iteration_compaction",
    ]
    assert TURN_POLICY_ORDER.index("longfile_convergence") < TURN_POLICY_ORDER.index(
        "per_iteration_compaction"
    )


def test_a_policy_the_core_does_not_know_is_refused_at_construction() -> None:
    stray = _RecordingPolicy("not_a_policy", frozenset(), seen=[])

    with pytest.raises(UnknownTurnPolicyError):
        TurnPolicyRegistry([stray])


class _GrantOnlyPolicy:
    """A policy that asks for one more message while answering ``proceed``.

    This is the shape a convergence policy has: it forced a tool of its own
    and the round it forced needs a slot, but the loop is the one that decides
    where the turn goes next, so the policy states the grant and nothing else.
    """

    name = "longfile_convergence"
    coordinates = frozenset({TurnCoordinate.turn_end})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.flags.assistant_message_idx != 1:
            return
        turn.outcome.extra_turn = True
        turn.outcome.reason = "scenario_grant"
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_grant"},
        )


class _SilentPolicy:
    """A policy at the same coordinate that has nothing to say."""

    name = "per_iteration_compaction"
    coordinates = frozenset({TurnCoordinate.turn_end})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        return
        yield  # pragma: no cover - the generator never yields


class _TurnCapPolicy:
    """The bound the grant is made against: one message unless granted more."""

    name = "run_ceilings"
    coordinates = frozenset({TurnCoordinate.turn_budget})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        if turn.flags.assistant_message_idx <= turn.turn_budget:
            return
        turn.outcome.directive = TurnDirective.end_turn
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_turn_cap"},
        )


async def test_a_grant_survives_the_policy_that_speaks_after_it(
    scenario: ScenarioFactory,
) -> None:
    """A ``proceed`` answer that asks for budget is still asking the loop.

    The grant and the silence sit at one coordinate, and the bound the grant
    is made against sits at another. If the consultation kept only the last
    policy's copy of the answer, the extra message would be dropped by the
    policy that said nothing, and the round the grant was made for would be
    killed by the cap before the model could answer it.
    """
    tool = ScriptedTool(tool_name="Note", content="noted")
    policies = TurnPolicyRegistry(
        [_GrantOnlyPolicy(), _SilentPolicy(), _TurnCapPolicy()]
    )
    run = scenario(
        tools=[tool],
        turn_policies=policies,
        rc=default_rc(max_turns_per_run=1),
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Note")
    run.llm.queue_response(text="the answer the granted round produced")

    await run.run("do the thing")

    assert "scenario_grant" in run.state_reasons()
    assert "scenario_turn_cap" not in run.state_reasons()
    assert len(run.requests) == 2
    assert "the answer the granted round produced" in "".join(run.history_texts())


class _TakeOverTheFinishPolicy:
    """A finish policy that ends the turn itself rather than letting it seal."""

    name = "terminal_tool_finish"
    coordinates = frozenset({TurnCoordinate.terminal_tool_finish})

    def __init__(self, directive: TurnDirective) -> None:
        self._directive = directive

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        turn.outcome.directive = self._directive
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_finish_taken_over"},
        )


def _terminal_tool() -> ScriptedTool:
    return ScriptedTool(
        tool_name="Finalize",
        content="the answer the terminal tool carried",
        metadata={TERMINAL_TOOL_METADATA_KEY: True},
    )


async def test_a_finish_policy_that_ends_the_turn_is_not_sealed_over(
    scenario: ScenarioFactory,
) -> None:
    """A policy that ended the finish has already said how the run ends.

    Completing the run on top of that reports a second, different ending: the
    policy's terminal events go out, and then the loop seals a state nobody
    asked it for.
    """
    tool = _terminal_tool()
    run = scenario(
        tools=[tool],
        turn_policies=TurnPolicyRegistry(
            [_TakeOverTheFinishPolicy(TurnDirective.end_turn)]
        ),
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Finalize")

    await run.run("finish it")

    assert "scenario_finish_taken_over" in run.state_reasons()
    assert run.engine.snapshot()["state"] != "completed"


async def test_a_finish_seam_refuses_a_directive_it_cannot_obey(
    scenario: ScenarioFactory,
) -> None:
    """A finish cannot be restarted, and saying so beats dropping it."""
    tool = _terminal_tool()
    run = scenario(
        tools=[tool],
        turn_policies=TurnPolicyRegistry(
            [_TakeOverTheFinishPolicy(TurnDirective.restart_turn)]
        ),
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Finalize")

    with pytest.raises(UnsupportedTurnDirectiveError):
        await run.run("finish it")


async def test_an_installed_set_replaces_a_decision_and_keeps_the_bounds(
    scenario: ScenarioFactory,
) -> None:
    """A host installs a policy; it does not thereby remove the turn cap.

    What a host has to say about a turn is a decision, and a decision is
    replaced by name. The bounds of a run — the cap on assistant messages
    above all — are the core's, and a set that took the core's place would
    leave the driver with nothing to stop a model that only ever calls tools.
    """
    seen: list[tuple[str, TurnCoordinate]] = []
    installed = _RecordingPolicy(
        "answer_floor", frozenset({TurnCoordinate.answer_floor}), seen=seen
    )
    tool = ScriptedTool(tool_name="Read")
    run = scenario(
        rc=default_rc(soft_stop_enabled=True, max_turns_per_run=2),
        tools=[tool],
        turn_policies=TurnPolicyRegistry([installed]),
    )
    for index in range(6):
        run.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}", tool_name="Read", tool_input={"v": str(index)}
        )
    run.llm.queue_response(text="the wind-down answer")

    await run.run("go")

    # The cap the host never mentioned still bit, and the run wound down
    # instead of calling tools forever.
    assert "soft_stop_notified" in run.state_reasons()
    assert len(run.requests) <= 6


class _FlagReadingPolicy:
    """A policy that reads the turn flags it is promised at two seams."""

    name = "empty_completion_guard"
    coordinates = frozenset(
        {TurnCoordinate.terminal_tool_finish, TurnCoordinate.turn_end}
    )

    def __init__(self) -> None:
        self.read: list[tuple[TurnCoordinate, bool, bool]] = []

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        self.read.append(
            (
                turn.coordinate,
                turn.flags.terminal_tool_completed,
                turn.flags.terminal_yielded,
            )
        )
        return
        yield  # pragma: no cover - the generator never yields


async def test_a_policy_reads_the_turn_flags_the_loop_writes(
    scenario: ScenarioFactory,
) -> None:
    """The flags are shared state, not a declaration.

    A field the contract names and the loop keeps as a local of its own is
    worse than no field: a policy reads a permanent False and the mistake
    looks exactly like a correct field access.
    """
    watcher = _FlagReadingPolicy()
    tool = _terminal_tool()
    run = scenario(tools=[tool], turn_policies=TurnPolicyRegistry([watcher]))
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Finalize")

    await run.run("finish it")

    finishes = [
        entry for entry in watcher.read
        if entry[0] is TurnCoordinate.terminal_tool_finish
    ]
    assert finishes, watcher.read
    assert finishes[0][1] is True


async def test_the_ending_pairs_the_call_the_turn_will_never_answer(
    scenario: ScenarioFactory,
) -> None:
    """A finished run leaves a readable transcript, not an open question.

    The model asks for the terminal tool and one more call in the same
    message. The dispatch stops on the terminal result, so the sibling's
    ``tool_use`` block is in the transcript with nothing answering it. The
    outbound wire repair would forward-fill a placeholder so a re-stream does
    not fail, but the durable record would keep the orphan, and a reader of
    it draws a call that was never answered. The ending pairs it instead.
    """
    finish = _terminal_tool()
    read = ScriptedTool(tool_name="Read", content="never reached")
    run = scenario(tools=[finish, read])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-final", "Finalize", {"v": "done"}),
            ("call-orphan", "Read", {"v": "a.txt"}),
        ]
    )

    await run.run("finish it")

    assert run.engine.state is LoopState.COMPLETED
    assert read.invocations == []
    paired = {block.tool_call_id: block for block in run.tool_results()}
    assert "call-orphan" in paired
    assert paired["call-orphan"].is_error is True


class _RefusesEveryCall:
    """A host's own opinion about a round that came back asking for tools."""

    name = "stream_loop_guard"
    coordinates = frozenset({TurnCoordinate.stream_settled})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        turn.outcome.tool_calls = []
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "scenario_calls_refused"},
        )


async def test_a_host_decides_whether_the_round_s_calls_are_honoured(
    scenario: ScenarioFactory,
) -> None:
    """Refusing what a model asked for is an opinion, so it is replaceable.

    Discarding a round's tool calls used to be written into the loop, behind
    a threshold the loop read itself. A host cannot reach a decision made
    there — it can only turn the whole mechanism off — so a run that wanted a
    different reading of "the model is looping" had nowhere to say it.
    """
    tool = ScriptedTool(tool_name="Search", content="five results")
    run = scenario(
        tools=[tool], turn_policies=TurnPolicyRegistry([_RefusesEveryCall()])
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Search")
    run.llm.queue_response(text="answering without the tool")

    await run.run("search")

    assert "scenario_calls_refused" in run.state_reasons()
    assert tool.invocations == []
    assert "answering without the tool" in "".join(run.history_texts())
