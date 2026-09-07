"""A call the model started writing and never finished.

The model opens a tool call, streams part of its arguments and then ends the
round normally. What reaches the loop is a call whose arguments were salvaged
by closing braces the model never sent. These scenarios drive that through
``run()`` and assert on what a host can see: the tool never ran, the
transcript carries an instruction to chunk, and the siblings the model DID
finish were not thrown away with it.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.interrupt import InterruptDecision, InterruptResolution
from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRoleMap
from protocore.contracts.turn_policy import TurnCoordinate, TurnDirective
from protocore.contracts.types import (
    TERMINAL_TOOL_METADATA_KEY,
    HookEvent,
)
from protocore.runtime import resume
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.turn_policies import TurnPolicyRegistry

from .conftest import ScenarioFactory, ScriptedTool, default_rc


def _call_blocks(
    calls: Sequence[tuple[str, str, dict[str, Any], bool]],
) -> list[LLMStreamEvent]:
    """One assistant round asking for ``calls``, some of them truncated."""
    import json

    events = [LLMStreamEvent(name="message_start", payload={})]
    for tool_call_id, tool_name, arguments, truncated in calls:
        events.extend(
            [
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": tool_call_id,
                        "partial_input_json": json.dumps(arguments),
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": tool_call_id,
                        "final_input": arguments,
                        "args_partial_truncated": truncated,
                    },
                ),
            ]
        )
    # ``end_turn`` is the finish the salvage happens under: a length cap is a
    # different signal with its own branch.
    events.append(
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})
    )
    return events


def _gate_next_calls(hooks: Any, count: int, *, token: str = "tok") -> None:
    for index in range(count):
        hooks.queue_action(
            HookEvent.pre_tool_use,
            HookResult(
                action=HookActionKind.ALLOW,
                modifications={
                    "requires_approval": True,
                    "approval_token": f"{token}-{index}",
                },
                reason="awaiting operator",
            ),
        )


async def test_a_truncated_call_is_taught_to_chunk_instead_of_being_run(
    scenario: ScenarioFactory,
) -> None:
    write = ScriptedTool(tool_name="Write")
    run = scenario(tools=[write])
    run.llm.queue_scripted_stream(
        _call_blocks([("call-trunc", "Write", {}, True)])
    )
    run.llm.queue_response(text="chunked instead")

    await run.run("write the file")

    assert write.invocations == []
    failed = [
        block for block in run.tool_results() if block.tool_call_id == "call-trunc"
    ]
    assert len(failed) == 1
    assert failed[0].is_error is True
    # The instruction names the call and quotes a size to aim at, in both
    # languages, so a reader of either one can act on it.
    assert "Write" in failed[0].content
    assert str(run.engine.config.rc.tool_call_max_input_chunk_bytes) in failed[0].content
    # The run was re-driven from a rebuilt context and finished normally.
    assert len(run.requests) == 2
    assert run.engine.state is LoopState.COMPLETED


async def test_the_sibling_the_model_did_finish_still_runs(
    scenario: ScenarioFactory,
) -> None:
    write = ScriptedTool(tool_name="Write")
    read = ScriptedTool(tool_name="Read", content="file body")
    run = scenario(tools=[write, read])
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Write", {}, True),
                ("call-clean", "Read", {"v": "a.txt"}, False),
            ]
        )
    )
    run.llm.queue_response(text="read it")

    await run.run("write then read")

    assert write.invocations == []
    assert read.invocations == [{"v": "a.txt"}]
    results = {block.tool_call_id: block for block in run.tool_results()}
    assert results["call-trunc"].is_error is True
    assert results["call-clean"].content == "file body"


async def test_every_gated_sibling_of_a_truncated_call_is_parked_not_just_the_first(
    scenario: ScenarioFactory,
) -> None:
    """The recovery path is a dispatch path, and it parks batches too.

    It used to stop at the first call a gate held and leave the rest of the
    message to the wire repair, so a host that approved the one call it was
    told about resumed into a run that had silently dropped the others.
    """
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=default_rc(approval_gate_web_enabled=True), tools=[deploy])
    _gate_next_calls(run.hooks, 2)
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Deploy", {}, True),
                ("call-a", "Deploy", {"v": "one"}, False),
                ("call-b", "Deploy", {"v": "two"}, False),
            ]
        )
    )

    produced = await run.run("deploy both")

    assert deploy.invocations == []
    assert run.engine.state is LoopState.AWAITING
    parked = [item.tool_call_id for item in run.engine.pending_interrupts]
    assert parked == ["call-a", "call-b"]
    # One announcement, carrying the whole open set — a host draws its cards
    # in one pass rather than learning about the second call at resume time.
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1
    assert len(announcements[0].payload["pending_interrupts"]) == 2


async def test_a_parked_batch_from_the_recovery_path_resolves_in_one_resume(
    scenario: ScenarioFactory,
) -> None:
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=default_rc(approval_gate_web_enabled=True), tools=[deploy])
    _gate_next_calls(run.hooks, 2)
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Deploy", {}, True),
                ("call-a", "Deploy", {"v": "one"}, False),
                ("call-b", "Deploy", {"v": "two"}, False),
            ]
        )
    )
    await run.run("deploy both")
    open_waits = run.engine.pending_interrupts
    stored = run.engine.snapshot()

    successor = scenario(
        rc=default_rc(approval_gate_web_enabled=True), tools=[deploy]
    )
    events = [
        evt
        async for evt in resume(
            successor.engine,
            stored,
            resolutions={
                open_waits[0].interrupt_id: InterruptResolution(
                    open_waits[0].interrupt_id, InterruptDecision.approve
                ),
                open_waits[1].interrupt_id: InterruptResolution(
                    open_waits[1].interrupt_id, InterruptDecision.deny
                ),
            },
        )
    ]

    assert successor.engine.pending_interrupts == ()
    # Exactly the approved call ran, and it ran once.
    assert deploy.invocations == [{"v": "one"}]
    assert [evt.type for evt in events].count(EventType.MESSAGE_STOP) == 1


async def test_a_terminal_sibling_does_not_seal_a_message_a_gate_is_holding(
    scenario: ScenarioFactory,
) -> None:
    """A run on its way to a decision cannot also be a finished run.

    The recovery path used to read a later sibling's terminal result while an
    earlier one sat at a gate, and then hand the driver both endings at once:
    the host was told the run was waiting, and the seal that followed moved a
    run already in AWAITING to COMPLETED — a transition the state machine does
    not have, raised out of the middle of the event stream the host was
    reading.
    """
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    finish = ScriptedTool(
        tool_name="Finalize",
        content="the answer the terminal tool carried",
        metadata={TERMINAL_TOOL_METADATA_KEY: True},
    )
    run = scenario(
        rc=default_rc(approval_gate_web_enabled=True), tools=[deploy, finish]
    )
    _gate_next_calls(run.hooks, 1)
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Deploy", {}, True),
                ("call-a", "Deploy", {"v": "one"}, False),
                ("call-b", "Finalize", {"v": "done"}, False),
            ]
        )
    )

    produced = await run.run("deploy then finish")

    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == ["call-a"]
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1
    # The gated call never ran, and the run was not sealed behind the decision
    # the host is still being asked for.
    assert deploy.invocations == []
    assert run.engine.snapshot()["state"] != "completed"


class _RecordingCancellation:
    """A host's own answer to "the run was told to stop"."""

    name = "cancellation"
    coordinates = frozenset({TurnCoordinate.cancel_checkpoint})

    def __init__(self) -> None:
        self.asked = 0
        self.acted = 0

    async def apply(self, turn: Any) -> AsyncIterator[TurnEvent]:
        self.asked += 1
        if not turn.engine.stop_requested:
            return
        self.acted += 1
        turn.outcome.directive = TurnDirective.end_turn
        turn.outcome.reason = "host_cancellation"
        yield TurnEvent(
            type=EventType.STATE_CHANGED,
            run_id="run-scenario",
            payload={"reason": "host_cancellation"},
        )


@dataclass
class _StopsItselfTool(ScriptedTool):
    """A tool whose own run is where the operator's stop lands."""

    engine: Any = None

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
        result = await super().invoke(context, arguments)
        assert self.engine is not None
        self.engine.stop()
        return result


async def test_the_recovery_walk_asks_the_installed_cancellation_policy(
    scenario: ScenarioFactory,
) -> None:
    """A stop means whatever the installed policy says it means, everywhere.

    The recovery walk used to answer a stop in its own words, straight off
    the engine flag, so a host that replaced the decision by name had it
    honoured at every other seam and ignored at this one.
    """
    cancellation = _RecordingCancellation()
    stopper = _StopsItselfTool(tool_name="Stopper", content="first")
    late = ScriptedTool(tool_name="Late", content="second")
    run = scenario(
        tools=[stopper, late],
        turn_policies=TurnPolicyRegistry([cancellation]),
    )
    stopper.engine = run.engine
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Stopper", {}, True),
                ("call-stop", "Stopper", {"v": "one"}, False),
                ("call-late", "Late", {"v": "two"}, False),
            ]
        )
    )

    await run.run("go")

    assert cancellation.acted >= 1
    assert "host_cancellation" in run.state_reasons()
    # Nothing new was dispatched after the stop was observed.
    assert late.invocations == []


async def test_a_cancel_mid_walk_releases_the_waits_the_walk_had_parked(
    scenario: ScenarioFactory,
) -> None:
    """A cancelled run waits for nothing, including its own parked calls.

    Batch parking made this reachable: a park used to end the walk at once,
    so no stop could land between parking a call and announcing it. Now the
    walk carries on, and a stop arriving mid-walk would otherwise leave the
    snapshot cancelled while it still witnessed open questions — about calls
    the teardown has just answered with synthetic errors.
    """
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    stopper = _StopsItselfTool(tool_name="Stopper", content="stopped")
    tail = ScriptedTool(tool_name="Tail", content="never")
    run = scenario(
        rc=default_rc(approval_gate_web_enabled=True),
        tools=[deploy, stopper, tail],
    )
    stopper.engine = run.engine
    _gate_next_calls(run.hooks, 1)
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Deploy", {}, True),
                ("call-gated", "Deploy", {"v": "one"}, False),
                ("call-stop", "Stopper", {"v": "two"}, False),
                ("call-tail", "Tail", {"v": "three"}, False),
            ]
        )
    )

    await run.run("deploy, then stop")

    assert run.engine.state is LoopState.CANCELLED
    assert run.engine.pending_interrupts == ()
    assert tail.invocations == []
    # The call the gate held is answered rather than left open.
    answered = {block.tool_call_id for block in run.tool_results()}
    assert "call-gated" in answered


async def test_a_vetoed_terminal_stops_the_walk_instead_of_filing_behind_it(
    scenario: ScenarioFactory,
) -> None:
    """A repair turn is a question, and results may not be filed after it.

    The gate refuses a terminal call that reported no work in prose: it
    answers the call with an error and appends a synthetic turn asking for
    the answer. Dispatching the siblings behind that would leave the durable
    transcript reading as a batch a user message interrupted — the shape the
    driver's own loop has always broken the batch to prevent.
    """
    finalize = ScriptedTool(
        tool_name="Finalize",
        content="filed",
        metadata={TERMINAL_TOOL_METADATA_KEY: True},
    )
    later = ScriptedTool(tool_name="Later", content="never")
    run = scenario(
        rc=default_rc(finalize_prose_gate_enabled=True),
        tools=[finalize, later],
        expected_terminal_tool="Finalize",
        # The host names the argument an answer would arrive in, so the core
        # can tell a background terminal from a message-carrying one.
        tool_roles=ToolRoleMap.declare(
            {}, argument_aliases={ToolArgumentSlot.answer: ["answer"]}
        ),
    )
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Later", {}, True),
                ("call-final", "Finalize", {"v": "done"}, False),
                ("call-later", "Later", {"v": "after"}, False),
            ]
        )
    )
    run.llm.queue_response(text="y" * 400)

    await run.run("finish it")

    # The veto fired, and nothing was dispatched behind the question it asked.
    # The terminal submission was withheld, and nothing was dispatched behind
    # the question the gate asked in its place.
    vetoed = [
        block for block in run.tool_results() if block.tool_call_id == "call-final"
    ]
    assert vetoed and vetoed[0].is_error is True
    assert finalize.invocations == []
    assert later.invocations == []


async def test_a_repeated_call_the_model_did_finish_is_not_skipped_with_it(
    scenario: ScenarioFactory,
) -> None:
    """Two calls that look alike are still two calls.

    A model that asks for the same tool twice with the same arguments — one
    call finished, one cut short — must have the finished one run and the
    other taught, and the walk tells them apart by the only thing that
    distinguishes them, which is their id.
    """
    note = ScriptedTool(tool_name="Note", content="noted")
    run = scenario(tools=[note])
    run.llm.queue_scripted_stream(
        _call_blocks(
            [
                ("call-trunc", "Note", {}, True),
                ("call-same", "Note", {}, False),
            ]
        )
    )
    run.llm.queue_response(text="done")

    await run.run("note it twice")

    assert note.invocations == [{}]
    results = {block.tool_call_id: block for block in run.tool_results()}
    assert results["call-trunc"].is_error is True
    assert results["call-same"].content == "noted"
