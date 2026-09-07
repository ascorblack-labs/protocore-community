"""A message whose calls are held at a gate one by one.

The pause used to be a single latch: the loop stopped at the first call a
gate held and left every call behind it undispatched and unpaired. A host
that drew its card from that pause approved one call and resumed into a run
that had quietly dropped the others. These scenarios drive a three-call
message through ``run()``, assert that every held call became its own wait,
and answer the whole set in one ``resume`` on a process that never saw the
pause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.interrupt import InterruptDecision, InterruptResolution
from protocore.contracts.types import HookEvent
from protocore.runtime import resume
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState

from .conftest import (
    DelegationTool,
    ScenarioFactory,
    ScriptedTool,
    default_rc,
)


def _gate(hooks: Any, *, token: str) -> None:
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": token},
            reason="awaiting operator",
        ),
    )


def _let_through(hooks: Any) -> None:
    hooks.queue_action(
        HookEvent.pre_tool_use, HookResult(action=HookActionKind.ALLOW)
    )


def _three_calls(run: Any) -> None:
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-1", "Deploy", {"v": "one"}),
            ("call-2", "Deploy", {"v": "two"}),
            ("call-3", "Deploy", {"v": "three"}),
        ]
    )


async def test_two_gated_calls_of_a_batch_of_three_park_two_waits(
    scenario: ScenarioFactory,
) -> None:
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=default_rc(approval_gate_web_enabled=True), tools=[deploy])
    _gate(run.hooks, token="tok-1")
    _let_through(run.hooks)
    _gate(run.hooks, token="tok-3")
    _three_calls(run)

    produced = await run.run("deploy all three")

    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == [
        "call-1",
        "call-3",
    ]
    # The call no gate held ran, and it ran once.
    assert deploy.invocations == [{"v": "two"}]
    # One announcement, carrying the whole open set.
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1
    assert [
        item["tool_call_id"] for item in announcements[0].payload["pending_interrupts"]
    ] == ["call-1", "call-3"]


async def test_the_whole_parked_batch_is_answered_by_a_process_that_never_saw_it(
    scenario: ScenarioFactory,
) -> None:
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=default_rc(approval_gate_web_enabled=True), tools=[deploy])
    _gate(run.hooks, token="tok-1")
    _let_through(run.hooks)
    _gate(run.hooks, token="tok-3")
    _three_calls(run)
    await run.run("deploy all three")
    waits = run.engine.pending_interrupts
    stored = run.engine.snapshot()

    successor_tool = ScriptedTool(tool_name="Deploy", content="deployed")
    successor = scenario(
        rc=default_rc(approval_gate_web_enabled=True), tools=[successor_tool]
    )
    events = [
        evt
        async for evt in resume(
            successor.engine,
            stored,
            resolutions={
                waits[0].interrupt_id: InterruptResolution(
                    waits[0].interrupt_id, InterruptDecision.approve
                ),
                waits[1].interrupt_id: InterruptResolution(
                    waits[1].interrupt_id, InterruptDecision.deny, reason="too broad"
                ),
            },
        )
    ]

    assert successor.engine.pending_interrupts == ()
    # Exactly the approved call ran on the second process, exactly once.
    assert successor_tool.invocations == [{"v": "one"}]
    results = {
        block.tool_call_id: block.content for block in successor.tool_results()
    }
    assert results["call-1"] == "deployed"
    assert (
        results["call-3"]
        == successor.engine.config.rc.tool_result_approval_denied_placeholder
    )
    assert [evt.type for evt in events].count(EventType.MESSAGE_STOP) == 1


async def test_a_batch_of_three_gated_calls_is_answered_three_different_ways(
    scenario: ScenarioFactory,
) -> None:
    """Approve, deny, abandon — one message, one pause, one resume."""
    deploy = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=default_rc(approval_gate_web_enabled=True), tools=[deploy])
    for index in range(3):
        _gate(run.hooks, token=f"tok-{index}")
    _three_calls(run)
    await run.run("deploy all three")
    waits = run.engine.pending_interrupts
    assert [item.tool_call_id for item in waits] == ["call-1", "call-2", "call-3"]
    assert deploy.invocations == []
    stored = run.engine.snapshot()

    successor_tool = ScriptedTool(tool_name="Deploy", content="deployed")
    successor = scenario(
        rc=default_rc(approval_gate_web_enabled=True), tools=[successor_tool]
    )
    [
        evt
        async for evt in resume(
            successor.engine,
            stored,
            resolutions={
                waits[0].interrupt_id: InterruptResolution(
                    waits[0].interrupt_id, InterruptDecision.approve
                ),
                waits[1].interrupt_id: InterruptResolution(
                    waits[1].interrupt_id, InterruptDecision.deny
                ),
                waits[2].interrupt_id: InterruptResolution(
                    waits[2].interrupt_id, InterruptDecision.abandon
                ),
            },
        )
    ]

    assert successor_tool.invocations == [{"v": "one"}]
    rc = successor.engine.config.rc
    results = {
        block.tool_call_id: block.content for block in successor.tool_results()
    }
    assert results["call-1"] == "deployed"
    assert results["call-2"] == rc.tool_result_approval_denied_placeholder
    assert results["call-3"] == rc.tool_result_approval_abandoned_placeholder


async def test_a_gate_that_races_a_delegated_group_parks_every_held_child(
    scenario: ScenarioFactory,
) -> None:
    """Two children of one message, both held, both parked.

    The group used to mark the first held child and discard the rest, so an
    operator answered one card and the second child vanished — no result, no
    wait, nothing in the transcript to say it had been asked for.
    """
    delegate = DelegationTool()
    run = scenario(
        rc=default_rc(
            approval_gate_web_enabled=True, max_concurrent_subagents=4
        ),
        tools=[delegate],
    )
    _gate(run.hooks, token="tok-d1")
    _gate(run.hooks, token="tok-d2")
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-1", "Delegate", {"v": "a"}),
            ("d-2", "Delegate", {"v": "b"}),
        ]
    )

    produced = await run.run("delegate both")

    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == [
        "d-1",
        "d-2",
    ]
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1
    assert len(announcements[0].payload["pending_interrupts"]) == 2
    assert run.tool_results() == []


@dataclass
class _ConcurrentTool(ScriptedTool):
    """A tool the run may fan out with its siblings."""

    is_concurrent_safe: bool = True


async def test_a_gate_that_races_a_parallel_batch_parks_every_held_call(
    scenario: ScenarioFactory,
) -> None:
    """Two reads of one fan-out, both held, both parked.

    The eligibility snapshot is taken before the batch opens, so a gate that
    appears between the snapshot and the dispatch races it: the calls fan out
    and are then held one by one. The fan-out used to mark the first held
    call and discard the outcomes behind it — the same "first one only" the
    serial path was cured of, in the one path no scenario watched.
    """
    read = _ConcurrentTool(tool_name="Read", content="a file")
    run = scenario(
        rc=default_rc(approval_gate_web_enabled=True), tools=[read]
    )
    _gate(run.hooks, token="tok-r1")
    _gate(run.hooks, token="tok-r2")
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-r1", "Read", {"v": "a"}),
            ("call-r2", "Read", {"v": "b"}),
        ]
    )

    produced = await run.run("read both")

    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == [
        "call-r1",
        "call-r2",
    ]
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1
    assert len(announcements[0].payload["pending_interrupts"]) == 2
    # Neither held call left a result behind it.
    assert run.tool_results() == []
