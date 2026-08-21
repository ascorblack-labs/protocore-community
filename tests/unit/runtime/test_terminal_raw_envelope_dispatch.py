"""Tests for generic truncation recovery of raw tool-call envelopes.

When an output cap cuts tool arguments mid-stream, the parser may surface a
raw ``{"__raw__": "<partial json>"}`` envelope. Terminal and non-terminal
calls must both use the same safe recovery path: the partial call is not
dispatched and the model is asked to emit complete arguments.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

TERMINAL_TOOL = "final_answer"
OTHER_TOOL = "lookup"


def _build_engine(
    *,
    rc: RuntimeConstants,
    llm: object,
    expected_terminal_tool: str | None = TERMINAL_TOOL,
) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-rawenv",
            tenant_id="tenant-test",
            session_id="sess-rawenv",
            model_name="test-model",
            rc=rc,
            expected_terminal_tool=expected_terminal_tool,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


class _RecordingTool(Tool):
    """Tool that records whether an incomplete call reached dispatch."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="record invocation",
            parameters=ToolParameterSchema(
                properties={"message": {"type": "string"}},
            ),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.invocations.append(dict(arguments))
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "toolu_raw")),
            content="invoked",
            is_error=False,
        )


class _RawEnvelopeLLM:
    """Always emits a tool call cut off before its arguments close."""

    def __init__(self, *, tool_name: str = TERMINAL_TOOL) -> None:
        self._tool_name = tool_name
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "toolu_raw", "tool_name": self._tool_name},
        )
        yield LLMStreamEvent(
            name="tool_use_input_delta",
            payload={
                "tool_call_id": "toolu_raw",
                "partial_input_json": '{"message": "m", "refs": ',
            },
        )
        yield LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": "toolu_raw",
                "final_input": {"__raw__": '{"message": "m", "refs": '},
                "truncated_by_output_cap": True,
            },
        )
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "max_tokens"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def _run_raw_envelope_case(
    *, tool_name: str, expected_terminal_tool: str | None
) -> tuple[_RawEnvelopeLLM, _RecordingTool, list[TurnEvent]]:
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_output_recovery_rounds=1,
    )
    llm = _RawEnvelopeLLM(tool_name=tool_name)
    tool = _RecordingTool(tool_name)
    engine = _build_engine(
        rc=rc,
        llm=llm,
        expected_terminal_tool=expected_terminal_tool,
    )
    engine.tools.register(tool)  # type: ignore[attr-defined]

    events: list[TurnEvent] = []
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    async for event in engine.run(user_msg):
        events.append(event)
    return llm, tool, events


def _resume_events(events: list[TurnEvent]) -> list[TurnEvent]:
    return [
        event
        for event in events
        if event.type is EventType.STATE_CHANGED
        and event.payload.get("reason") == "tool_call_truncation_recovery"
    ]


@pytest.mark.asyncio
async def test_terminal_raw_envelope_uses_generic_truncation_resume() -> None:
    llm, tool, events = await _run_raw_envelope_case(
        tool_name=TERMINAL_TOOL,
        expected_terminal_tool=TERMINAL_TOOL,
    )

    assert _resume_events(events)
    assert tool.invocations == []
    assert len(llm.calls) >= 2


@pytest.mark.asyncio
async def test_non_terminal_raw_envelope_uses_generic_truncation_resume() -> None:
    llm, tool, events = await _run_raw_envelope_case(
        tool_name=OTHER_TOOL,
        expected_terminal_tool=TERMINAL_TOOL,
    )

    assert _resume_events(events)
    assert tool.invocations == []
    assert len(llm.calls) >= 2
