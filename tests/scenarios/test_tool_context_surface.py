"""What the loop actually hands a tool, seen from inside the tool.

A host writes its tools against the invocation context, so the context is a
published surface even though nothing in it is a route or a schema. Two things
about it are easy to state and easy to break without noticing: it carries the
run and nothing about whoever owns the run, and the run state it carries is THE
run's, shared by reference with every other call in the same run.

Driven through the entry points a host calls, and asserted on the contexts the
tool was given rather than on anything read off the engine.
"""
from __future__ import annotations

from protocore.contracts.tools import ToolContext

from .conftest import ScenarioFactory, ScriptedTool


def _fields() -> set[str]:
    return set(ToolContext.model_fields)


async def test_a_tool_is_handed_the_run_and_nothing_about_who_owns_it(
    scenario: ScenarioFactory,
) -> None:
    """The context names the run; everything else is the host's to carry.

    A field here that only a host can fill is a field every host has to know
    about, including the ones that have no such notion at all — so the account
    a scope belongs to, the workspace it addresses and anything else of that
    kind travels in the host's own compartment of the run state instead.
    """
    tool = ScriptedTool(content="noted")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Note", tool_input={"v": "a"}
    )
    run.llm.queue_response(text="done")

    await run.run("note something")

    assert _fields() == {
        "tenant_id",
        "run_id",
        "session_id",
        "work_scope",
        "evidence",
        "run_state",
        "metadata",
    }
    context = tool.contexts[0]
    assert (context.tenant_id, context.run_id, context.session_id) == (
        "tenant-scenario",
        "run-scenario",
        "sess-scenario",
    )


async def test_two_calls_in_one_run_share_one_run_state(
    scenario: ScenarioFactory,
) -> None:
    """By reference, not by value: a streak written by one call is the run's.

    A copy would be a second run's worth of allowances that nothing reconciles
    with the first, and every per-run cap would then be per-call.
    """
    tool = ScriptedTool(content="noted")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Note", tool_input={"v": "a"}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-2", tool_name="Note", tool_input={"v": "b"}
    )
    run.llm.queue_response(text="done")

    await run.run("note twice")

    first, second = tool.contexts[0], tool.contexts[1]
    assert first.run_state is not None
    assert first.run_state is second.run_state


async def test_the_call_the_model_asked_for_is_named_on_the_bag(
    scenario: ScenarioFactory,
) -> None:
    """The dispatcher stamps the call id, so a result can address its call."""
    tool = ScriptedTool(content="noted")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Note", tool_input={"v": "a"}
    )
    run.llm.queue_response(text="done")

    await run.run("note something")

    assert tool.contexts[0].metadata["tool_call_id"] == "c-1"


async def test_a_call_carries_the_run_trees_identity_for_what_it_observes(
    scenario: ScenarioFactory,
) -> None:
    """Provenance is one value, and it names this run rather than a text.

    A tool the deployment has not bound as an evidence producer carries the
    identity and no binding — which is exactly why the two are one value: the
    pair says "this call may state what it saw" only when both halves are
    there.
    """
    tool = ScriptedTool(content="noted")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Note", tool_input={"v": "a"}
    )
    run.llm.queue_response(text="done")

    await run.run("note something")

    evidence = tool.contexts[0].evidence
    assert evidence is not None
    assert evidence.origin is not None
    assert evidence.origin.run_id == "run-scenario"
    assert evidence.origin.root_run_id == "run-scenario"
    assert evidence.producer_binding is None
