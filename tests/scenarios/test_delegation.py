"""Delegation: a turn that hands work to subagents, and what comes back.

A delegation tool is one that spawns a whole nested run, so the loop treats
it differently from an ordinary call: several of them in one assistant
message may overlap, the cap on that overlap is a runtime constant, and one
child failing is not the group failing. All of that is asserted here from
the outside — the invocation record of the scripted tool, the order of the
results in history, and the request the model gets next.
"""
from __future__ import annotations

import logging
import time

import pytest

from .conftest import DelegationTool, ScenarioFactory, default_rc


async def test_a_delegated_call_returns_its_child_result_to_the_model(
    scenario: ScenarioFactory,
) -> None:
    tool = DelegationTool(content="child")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="d-1", tool_name="Delegate", tool_input={"v": "task-a"}
    )
    run.llm.queue_response(text="collected")

    await run.run("delegate it")

    assert tool.invocations == [{"v": "task-a"}]
    assert [block.content for block in run.tool_results()] == ["child:task-a"]
    assert any("child:task-a" in text for text in run.request_texts(1))


async def test_two_delegated_calls_in_one_message_overlap(
    scenario: ScenarioFactory,
) -> None:
    """Both children are in flight at once, so the group costs one child's wait."""
    tool = DelegationTool()
    run = scenario(rc=default_rc(max_concurrent_subagents=4), tools=[tool])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-1", "Delegate", {"v": "a", "delay": 0.15}),
            ("d-2", "Delegate", {"v": "b", "delay": 0.15}),
        ]
    )
    run.llm.queue_response(text="both back")

    started = time.monotonic()
    await run.run()
    elapsed = time.monotonic() - started

    assert len(tool.invocations) == 2
    assert elapsed < 0.29


async def test_a_cap_of_one_makes_the_group_serial(
    scenario: ScenarioFactory,
) -> None:
    tool = DelegationTool()
    run = scenario(rc=default_rc(max_concurrent_subagents=1), tools=[tool])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-1", "Delegate", {"v": "a", "delay": 0.08}),
            ("d-2", "Delegate", {"v": "b", "delay": 0.08}),
        ]
    )
    run.llm.queue_response(text="both back")

    started = time.monotonic()
    await run.run()
    elapsed = time.monotonic() - started

    assert len(tool.invocations) == 2
    assert elapsed >= 0.16


async def test_the_results_keep_the_order_the_model_asked_in(
    scenario: ScenarioFactory,
) -> None:
    """The slow child was asked for first, so its result is first."""
    tool = DelegationTool()
    run = scenario(rc=default_rc(max_concurrent_subagents=4), tools=[tool])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-slow", "Delegate", {"v": "slow", "delay": 0.12}),
            ("d-fast", "Delegate", {"v": "fast"}),
        ]
    )
    run.llm.queue_response(text="both back")

    await run.run()

    assert [block.tool_call_id for block in run.tool_results()] == ["d-slow", "d-fast"]


async def test_one_failing_child_does_not_take_its_sibling_down(
    scenario: ScenarioFactory,
) -> None:
    tool = DelegationTool()
    run = scenario(rc=default_rc(max_concurrent_subagents=4), tools=[tool])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-bad", "Delegate", {"v": "bad", "fail": True}),
            ("d-good", "Delegate", {"v": "good"}),
        ]
    )
    run.llm.queue_response(text="one of two")

    await run.run()

    results = {block.tool_call_id: block for block in run.tool_results()}
    assert results["d-bad"].is_error is True
    assert results["d-good"].is_error is False
    assert results["d-good"].content == "ok:good"


async def test_a_fan_out_states_how_many_tree_slots_it_charged(
    scenario: ScenarioFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three children, three charges against the one tree budget.

    The budget is an in-process object and only a run that PAUSES writes its
    count into a snapshot, so a fan-out that starts and finishes in one
    process left no record of the bound it operated under at all — which put
    a reader in the position of inferring the reservation from how long the
    group took. It says so instead: one line per charged slot, naming the
    group it belongs to and the capacity it was charged against.
    """
    tool = DelegationTool()
    run = scenario(
        rc=default_rc(
            max_concurrent_subagents=4,
            max_concurrent_subagents_per_tree=8,
        ),
        tools=[tool],
    )
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-1", "Delegate", {"v": "a"}),
            ("d-2", "Delegate", {"v": "b"}),
            ("d-3", "Delegate", {"v": "c"}),
        ]
    )
    run.llm.queue_response(text="three back")

    with caplog.at_level(logging.WARNING):
        await run.run("fan out")

    charged = [
        record.getMessage()
        for record in caplog.records
        if "query.subagent_tree_budget.charged" in record.getMessage()
    ]
    # One line per child, all of them naming the same fan-out group.
    assert len(charged) == 3
    assert {line.split("group=")[1].split()[0] for line in charged} == {"d-1"}
    assert all("capacity=8" in line for line in charged)
    assert all("unlimited=False" in line for line in charged)
    # Each line names the slots held at the moment it was charged, and every
    # order in the group appears exactly once — so N children is N charges
    # against one budget, read rather than timed.
    assert sorted(line.split("order=")[1].split()[0] for line in charged) == [
        "0",
        "1",
        "2",
    ]
    assert all(int(line.split("charged=")[1].split()[0]) >= 1 for line in charged)


async def test_an_unlimited_tree_says_it_is_unlimited(
    scenario: ScenarioFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The zero sentinel is a real setting, so the evidence names it."""
    tool = DelegationTool()
    run = scenario(
        rc=default_rc(
            max_concurrent_subagents=4,
            max_concurrent_subagents_per_tree=0,
        ),
        tools=[tool],
    )
    run.llm.queue_multi_tool_call_response(
        tool_calls=[("d-1", "Delegate", {"v": "a"}), ("d-2", "Delegate", {"v": "b"})]
    )
    run.llm.queue_response(text="both back")

    with caplog.at_level(logging.WARNING):
        await run.run("fan out")

    charged = [
        record.getMessage()
        for record in caplog.records
        if "query.subagent_tree_budget.charged" in record.getMessage()
    ]
    assert len(charged) == 2
    assert all("capacity=0" in line and "unlimited=True" in line for line in charged)
