"""Cancellation: what a stop that actually stops looks like from outside.

``stop()`` arrives on a different task than the one iterating ``run()`` —
that is the whole point of it, and it is why these scenarios drive the run
in a task and cancel it from the test's own. What is asserted is what an
operator can check afterwards: the tool did not run, the model was not
asked again, and the cancellation is in the snapshot the next process will
read.
"""
from __future__ import annotations

import asyncio

from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState

from .conftest import DelegationTool, ScenarioFactory, ScriptedTool, default_rc


async def test_a_run_cancelled_before_it_opens_never_reaches_the_model(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.llm.queue_response(text="never asked for")
    run.engine.stop()

    produced = await run.run("go")

    assert run.requests == ()
    assert run.engine.state is LoopState.CANCELLED
    assert any(
        evt.type is EventType.MESSAGE_STOP
        and evt.payload.get("stop_reason") == StopReason.cancelled.value
        for evt in produced
    )


async def test_a_cancel_during_a_tool_call_interrupts_the_call_in_flight(
    scenario: ScenarioFactory,
) -> None:
    """The operator's stop lands while the tool is awaiting, and it lands now."""
    tool = ScriptedTool(tool_name="Slow", delay_seconds=30.0)
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Slow", tool_input={}
    )
    run.llm.queue_response(text="never reached")

    await asyncio.wait_for(
        run.run_and_stop_when(tool.started.is_set), timeout=5.0
    )

    assert tool.invocations == [{}]
    assert len(run.requests) == 1


async def test_a_cancel_during_a_tool_call_does_not_ask_the_model_again(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Slow", delay_seconds=30.0)
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Slow", tool_input={}
    )
    run.llm.queue_response(text="a second answer nobody asked for")

    await asyncio.wait_for(
        run.run_and_stop_when(tool.started.is_set), timeout=5.0
    )

    assert len(run.requests) == 1
    assert not any("a second answer" in text for text in run.history_texts())


async def test_a_cancel_during_delegation_does_not_leave_the_child_answering(
    scenario: ScenarioFactory,
) -> None:
    """A stopped leader does not accept work from a subagent it abandoned.

    The child is a whole nested run; the leader has no handle on it, so the
    only thing core can promise is that a cancelled leader neither records
    the child's result nor asks the model what to do with it.
    """
    tool = DelegationTool(delay_seconds=30.0)
    run = scenario(rc=default_rc(max_concurrent_subagents=4), tools=[tool])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-1", "Delegate", {"v": "a", "delay": 30.0}),
            ("d-2", "Delegate", {"v": "b", "delay": 30.0}),
        ]
    )
    run.llm.queue_response(text="never reached")

    await asyncio.wait_for(
        run.run_and_stop_when(tool.started.is_set), timeout=5.0
    )

    assert len(run.requests) == 1
    assert [block.content for block in run.tool_results()] == []


async def test_the_cancellation_is_in_the_snapshot_the_next_process_reads(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()

    assert run.engine.snapshot()["stop_requested"] is False
    run.engine.stop()
    assert run.engine.snapshot()["stop_requested"] is True


async def test_a_cancelled_run_stays_cancelled_across_a_cold_start(
    scenario: ScenarioFactory,
) -> None:
    """A cancel landing moments before the process died is not lost with it."""
    first = scenario()
    first.engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    first.engine.stop()

    second = scenario()
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    assert second.engine.stop_requested is True


async def test_the_resumed_turn_ends_cancelled_instead_of_spending_budget(
    scenario: ScenarioFactory,
) -> None:
    first = scenario()
    first.engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    first.engine.stop()

    second = scenario()
    second.llm.queue_response(text="work the operator called off")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    await second.run(None)

    assert second.requests == ()
    assert second.engine.state is LoopState.CANCELLED


async def test_a_snapshot_taken_before_the_stop_never_un_cancels_a_live_run(
    scenario: ScenarioFactory,
) -> None:
    """The flag is one-way, so an older snapshot cannot revive cancelled work."""
    first = scenario()
    older = first.engine.snapshot()

    second = scenario()
    second.engine.stop()
    await second.engine.resume_from_snapshot(older)

    assert second.engine.stop_requested is True


class _StoppingTool(ScriptedTool):
    """A tool that asks the run to stop, then returns its result normally.

    The gap this opens is the one an operator's stop lands in for real: the
    call in flight completed, and the stop is observed between it and the next
    call of the same assistant message. Scripting it from inside the tool is
    the only way to hold that gap open deterministically — a stop from another
    task cancels the await instead of landing after it.
    """

    engine: object = None

    async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
        result = await super().invoke(context, arguments)
        assert self.engine is not None
        self.engine.stop()  # type: ignore[attr-defined]
        return result


async def test_a_stop_between_two_calls_of_one_message_leaves_the_second_undone(
    scenario: ScenarioFactory,
) -> None:
    """No NEW side effect once a stop has been seen, mid-message included.

    The guards before the dispatch loop cannot see a stop that arrives while
    the loop is already running. Without a checkpoint between two calls the
    loop would go on to dispatch the second one — a side effect performed
    after the run was cancelled — and leave its call unanswered in the
    transcript.
    """
    first = _StoppingTool(tool_name="Slow", content="first done")
    second = ScriptedTool(tool_name="After", content="second done")
    run = scenario(tools=[first, second])
    first.engine = run.engine
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-1", "Slow", {"v": "a"}),
            ("call-2", "After", {"v": "b"}),
        ]
    )

    produced = await run.run("do both")

    assert first.invocations == [{"v": "a"}]
    assert second.invocations == []
    assert run.engine.state is LoopState.CANCELLED
    assert any(
        evt.type is EventType.MESSAGE_STOP
        and evt.payload.get("stop_reason") == StopReason.cancelled.value
        for evt in produced
    )
    # The call the loop never dispatched is paired anyway, so the snapshot a
    # resume reads is a transcript rather than an unanswered question.
    assert {block.tool_call_id for block in run.tool_results()} == {
        "call-1",
        "call-2",
    }
