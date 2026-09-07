"""A declared tool sequence survives the pause in the middle of it.

A run may be handed an ordered list of tools it MUST call before it answers.
When one of those tools is the one that asks a person a question, the sequence
straddles a pause: the asking tool runs, the run parks, the process that
parked it may be gone by the time the answer arrives, and the entry that comes
AFTER the question has to be forced on the first turn of the resumed run.

That is the whole scenario below, driven through public entry points only —
``QueryEngine.run`` to park, ``snapshot``/``resume_from_snapshot`` to cross the
process boundary, and ``run(answer)`` to continue. It is asserted on what
reaches the provider, ``LLMRequest.extra['forced_tool_choice']``, because that
is what a forced choice IS; the counters behind it are private.

The failure it pins: the answer to a question never passes the dispatcher, so
nothing folded it into precondition progress. The entry naming the asking tool
stayed outstanding forever, and the turn that should have forced the next tool
forced the already-answered question instead — or, when the asking tool is no
longer on that turn's surface, forced nothing at all and the run just stopped.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolPrecondition,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime import LoopState
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)
from protocore.tools.ask_user import ASK_USER_TOOL_NAME, AskUserTool

AFTER = "SearchDocs"


class _AfterTool(Tool):
    """The step the run owes AFTER the question is answered."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return AFTER

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=AFTER,
            description="looks something up",
            parameters=ToolParameterSchema(properties={"query": {"type": "string"}}),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(tool_call_id="", content="found it", is_error=False)


class _Runtime:
    def __init__(self) -> None:
        self.llm = InMemoryLLMProvider()
        self.tools = InMemoryToolRegistry()
        self.events = InMemoryEventStream()
        self.hooks = InMemoryHookManager()
        self.skills = InMemorySkillStore()
        self.blobs = InMemoryBlobStore()

    def engine(self) -> QueryEngine:
        return QueryEngine(
            config=QueryEngineConfig(
                run_id="run-precond-pause",
                tenant_id="tenant-1",
                account_id="tenant-1",
                session_id="sess-1",
                model_name="scripted-model",
                rc=LoopConstants(model_context_window=4_096),
                tool_preconditions=(
                    ToolPrecondition(tool=ASK_USER_TOOL_NAME, calls=1),
                    ToolPrecondition(tool=AFTER, calls=1),
                ),
            ),
            llm_provider=self.llm,
            tool_registry=self.tools,
            event_stream=self.events,
            hook_manager=self.hooks,
            skill_store=self.skills,
            blob_store=self.blobs,
        )


def _forced(runtime: _Runtime) -> list[str | None]:
    return [request.extra.get("forced_tool_choice") for request in runtime.llm.calls]


def _take_the_answer(engine: QueryEngine, tool_call_id: str) -> None:
    """Release the parked wait, the way the layer that collects an answer does."""
    engine.clear_pending_approval(tool_call_id)
    if engine.state is LoopState.AWAITING:
        engine.transition_to(LoopState.RUNNING)


def _answer(tool_call_id: str) -> Message:
    """The answer, shaped the way the collecting layer hands it over."""
    return Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(
                tool_call_id=tool_call_id,
                content='{"answers": [{"selected": ["the second one"]}]}',
                is_error=False,
            )
        ],
    )


@pytest.mark.asyncio
async def test_the_entry_after_an_answered_question_is_forced_on_a_cold_resume() -> None:
    runtime = _Runtime()
    after = _AfterTool()
    runtime.tools.register(AskUserTool())
    runtime.tools.register(after)
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_ask",
        tool_name=ASK_USER_TOOL_NAME,
        tool_input={
            "questions": [
                {"question": "which one?", "options": [{"label": "the second one"}]}
            ]
        },
    )

    parked = runtime.engine()
    async for _ in parked.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="do the thing")])
    ):
        pass

    assert parked.state is LoopState.AWAITING, "the question must park the run"
    assert _forced(runtime) == [ASK_USER_TOOL_NAME], (
        "the first entry names the asking tool, so it is what the first turn forces"
    )
    stored = parked.snapshot()

    # A different process entirely: nothing is shared but the snapshot.
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_after", tool_name=AFTER, tool_input={"query": "q"}
    )
    runtime.llm.queue_response(text="here is the answer")
    successor = runtime.engine()
    await successor.resume_from_snapshot(stored)
    _take_the_answer(successor, "toolu_ask")

    async for _ in successor.run(_answer("toolu_ask")):
        pass

    assert _forced(runtime)[1] == AFTER, (
        "the answer satisfies the question's entry, so the resumed run's first "
        "turn must force the entry that comes after it"
    )
    assert len(after.calls) == 1
    assert _forced(runtime)[2:] == [None], (
        "with every entry satisfied the run forces nothing again"
    )
    assert successor.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_an_answer_to_a_call_that_is_not_the_current_entry_is_not_progress() -> None:
    """Only the outstanding entry's own tool advances it.

    The fold reads the call's name back out of the transcript rather than
    trusting the caller, so a result carrying some other call's id leaves the
    sequence exactly where it was.
    """
    runtime = _Runtime()
    after = _AfterTool()
    runtime.tools.register(AskUserTool())
    runtime.tools.register(after)
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_ask",
        tool_name=ASK_USER_TOOL_NAME,
        tool_input={
            "questions": [
                {"question": "which one?", "options": [{"label": "the second one"}]}
            ]
        },
    )
    parked = runtime.engine()
    async for _ in parked.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="do the thing")])
    ):
        pass
    stored = parked.snapshot()

    runtime.llm.queue_response(text="nothing to do")
    successor = runtime.engine()
    await successor.resume_from_snapshot(stored)
    _take_the_answer(successor, "toolu_ask")
    async for _ in successor.run(_answer("toolu_nobody_called_this")):
        pass

    assert _forced(runtime)[1] == ASK_USER_TOOL_NAME, (
        "an unrecognised call id is not the answer to the parked question, so "
        "the question's entry is still the outstanding one"
    )
