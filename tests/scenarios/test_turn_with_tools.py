"""A turn that answers, and a turn that uses tools, seen from outside.

These scenarios drive ``QueryEngine.run()`` — the entry the host's executor
uses — and assert on the requests the provider received, the events the
caller iterated, and the history that survives. Nothing here names an
internal of the loop, so a rewrite of the loop that keeps its promises keeps
these green.
"""
from __future__ import annotations

import pytest

from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool, default_rc


async def test_a_plain_answer_reaches_the_reader_and_the_history(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.llm.queue_response(text="the answer")

    await run.run("what is it")

    assert EventType.MESSAGE_START in run.event_types()
    assert EventType.MESSAGE_STOP in run.event_types()
    assert "the answer" in "".join(run.history_texts())


async def test_the_model_is_shown_the_question_it_must_answer(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.llm.queue_response(text="fine")

    await run.run("a very specific question")

    assert any("a very specific question" in text for text in run.request_texts(0))


async def test_the_registered_tool_is_on_the_surface_the_model_sees(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Note")
    run = scenario(tools=[tool])
    run.llm.queue_response(text="no tool needed")

    await run.run()

    assert "Note" in run.advertised_tool_names(0)


async def test_a_tool_call_runs_once_and_its_result_goes_back_to_the_model(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Note", content="noted-42")
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run()

    assert tool.invocations == [{"v": "x"}]
    assert [block.content for block in run.tool_results()] == ["noted-42"]
    assert any("noted-42" in text for text in run.request_texts(1))


async def test_the_tool_result_is_announced_to_the_reader(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note")])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_response(text="done")

    produced = await run.run()

    assert EventType.TOOL_RESULT in [evt.type for evt in produced]


async def test_a_tool_that_raises_is_reported_to_the_model_not_to_the_process(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Note", raises=RuntimeError("disk on fire"))
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_response(text="recovered")

    await run.run()

    results = run.tool_results()
    assert results and results[0].is_error is True
    assert len(run.requests) >= 2


async def test_a_call_to_a_tool_that_is_not_registered_ends_as_an_error_result(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note")])
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Absent", tool_input={}
    )
    run.llm.queue_response(text="never mind")

    await run.run()

    results = run.tool_results()
    assert results and results[0].is_error is True
    assert results[0].tool_call_id == "call-1"


async def test_two_tool_calls_in_one_message_keep_the_order_the_model_asked_for(
    scenario: ScenarioFactory,
) -> None:
    first = ScriptedTool(tool_name="Slow", content="slow-out", delay_seconds=0.02)
    second = ScriptedTool(tool_name="Fast", content="fast-out")
    run = scenario(tools=[first, second])
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-slow", "Slow", {"v": "1"}),
            ("call-fast", "Fast", {"v": "2"}),
        ]
    )
    run.llm.queue_response(text="both done")

    await run.run()

    assert [block.tool_call_id for block in run.tool_results()] == [
        "call-slow",
        "call-fast",
    ]


async def test_a_continuation_run_drives_the_history_it_was_handed(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="earlier ask")])
    )
    run.llm.queue_response(text="late answer")

    await run.run(None)

    assert any("earlier ask" in text for text in run.request_texts(0))
    assert "late answer" in "".join(run.history_texts())


async def test_a_continuation_with_no_history_is_refused_rather_than_invented(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()

    with pytest.raises(ValueError):
        await run.run(None)

    assert run.requests == ()


async def test_a_continuation_that_does_not_end_on_a_question_is_refused(
    scenario: ScenarioFactory,
) -> None:
    run = scenario()
    run.engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="said")])
    )

    with pytest.raises(ValueError):
        await run.run(None)

    assert run.requests == ()


async def test_a_second_turn_on_a_rearmed_engine_carries_the_first(
    scenario: ScenarioFactory,
) -> None:
    """A conversation is turns on one engine, and turn two sees turn one."""
    run = scenario(rc=default_rc(model_context_window=32_000))
    run.llm.queue_response(text="first answer")
    run.llm.queue_response(text="second answer")

    await run.run("first ask")
    run.engine.rearm()
    await run.run("second ask")

    assert any("first ask" in text for text in run.request_texts(1))
    assert any("first answer" in text for text in run.request_texts(1))
