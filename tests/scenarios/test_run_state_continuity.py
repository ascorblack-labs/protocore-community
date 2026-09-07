"""What a run has already spent is still spent after the process dies.

The allowances a run carries — the tree's cumulative work ledger above all —
are the reason a snapshot is not just a transcript. A run picked up by a second
process with a fresh budget would delegate again, spend again, and nothing
downstream could tell that apart from a run that had never spent anything: the
loop "die, get picked up, spend the cap again" has no upper bound.

Everything here is driven through the entry points a host calls and asserted on
what the second process can see from outside the engine.
"""
from __future__ import annotations

from .conftest import DelegationTool, ScenarioFactory, default_rc


async def test_a_second_process_does_not_get_a_fresh_tree_budget(
    scenario: ScenarioFactory,
) -> None:
    """The child runs the first process started are still counted after it dies."""
    tool = DelegationTool(content="child")
    first = scenario(
        rc=default_rc(model_context_window=32_000, max_subagent_runs_per_tree=1),
        tools=[tool],
    )
    first.llm.queue_tool_call_response(
        tool_call_id="d-1", tool_name="Delegate", tool_input={"v": "a"}
    )
    first.llm.queue_response(text="one child done")
    await first.run("delegate once")

    assert tool.invocations == [{"v": "a"}]

    second = scenario(
        rc=default_rc(model_context_window=32_000, max_subagent_runs_per_tree=1),
        tools=[tool],
    )
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    second.llm.queue_tool_call_response(
        tool_call_id="d-2", tool_name="Delegate", tool_input={"v": "b"}
    )
    second.llm.queue_response(text="refused and finished")
    await second.run("delegate again")

    # The cap was spent before the restart, so the second call never reaches
    # the tool: the refusal is what the model is shown instead of a child.
    assert tool.invocations == [{"v": "a"}]
    assert any(
        "budget" in text.lower() or "exhaust" in text.lower()
        for text in second.request_texts(1)
    )


async def test_a_resumed_run_that_has_budget_left_still_delegates(
    scenario: ScenarioFactory,
) -> None:
    """The ledger bounds the tree; it does not refuse a tree under its cap."""
    tool = DelegationTool(content="child")
    first = scenario(
        rc=default_rc(model_context_window=32_000, max_subagent_runs_per_tree=4),
        tools=[tool],
    )
    first.llm.queue_tool_call_response(
        tool_call_id="d-1", tool_name="Delegate", tool_input={"v": "a"}
    )
    first.llm.queue_response(text="one child done")
    await first.run("delegate once")

    second = scenario(
        rc=default_rc(model_context_window=32_000, max_subagent_runs_per_tree=4),
        tools=[tool],
    )
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    second.llm.queue_tool_call_response(
        tool_call_id="d-2", tool_name="Delegate", tool_input={"v": "b"}
    )
    second.llm.queue_response(text="second child done")
    await second.run("delegate again")

    assert tool.invocations == [{"v": "a"}, {"v": "b"}]


async def test_the_run_state_a_snapshot_states_is_the_one_it_comes_back_with(
    scenario: ScenarioFactory,
) -> None:
    """The round trip itself: what goes out under one name comes back under it."""
    first = scenario(rc=default_rc(model_context_window=32_000))
    first.llm.queue_response(text="answered")
    await first.run("a question")

    payload = first.engine.snapshot()

    second = scenario(rc=default_rc(model_context_window=32_000))
    await second.engine.resume_from_snapshot(payload)

    assert second.engine.snapshot()["run_scoped_state"] == payload["run_scoped_state"]


async def test_a_run_state_is_shared_with_the_calls_the_run_makes(
    scenario: ScenarioFactory,
) -> None:
    """One object, not a copy: what a tool call sees is the run's own state."""
    tool = DelegationTool(content="child")
    run = scenario(rc=default_rc(model_context_window=32_000), tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="d-1", tool_name="Delegate", tool_input={"v": "a"}
    )
    run.llm.queue_response(text="done")
    await run.run("delegate it")

    seen = tool.contexts[0].run_state
    assert seen is run.engine.run_state
    assert seen is not None
    assert seen.run_work_ledger is not None
