"""The salvage-job loop dispatches synthetic Write/AppendFile calls without a
pre-dispatch cancel checkpoint, so a user cancel landing in the
truncation-recovery turn's await gaps can mutate the workspace AFTER
``engine.stop_requested`` flips.

The non-truncated dispatch path (query.py:~1869) re-checks
``engine.stop_requested`` BEFORE every dispatch (added explicitly so a
cancel landing in prior await gaps cannot cause a side effect after
cancellation). The salvage loop did NOT carry that guard, so the
synthetic recovery write still landed on disk post-cancel. The fix adds
the same ``if engine.stop_requested`` checkpoint at the top of each
salvage iteration and routes to the shared cancel teardown
(``_emit_dispatch_cancel_teardown``) without dispatching.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

WRITE = "Write"
APPEND = "AppendFile"
FINALIZE = "FinalizeFile"
READ = "Read"
TARGET = "/workspace/big.py"


def _build_engine(*, rc: RuntimeConstants, llm: object) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-f12-08",
            tenant_id="tenant-test",
            session_id="sess-f12-08",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


class _ByteReportingFileTool(Tool):
    """Write/AppendFile/FinalizeFile that returns PROD-faithful JSON. Read
    returns a minimal ack. Records every invocation so the test can
    assert the synthetic salvage write was (or was not) dispatched."""

    def __init__(self, name: str, files: dict[str, str], *, terminal: bool = False) -> None:
        self._name = name
        self._files = files
        self._terminal = terminal
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        props: dict[str, Any] = {"path": {"type": "string"}}
        required = ["path"]
        if self._name in (WRITE, APPEND):
            props["content"] = {"type": "string"}
            required.append("content")
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} a file",
            parameters=ToolParameterSchema(properties=props, required=required),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        path = str(arguments.get("path", TARGET))
        if self._name == WRITE:
            content = str(arguments.get("content", ""))
            self._files[path] = content
            payload = {"path": path, "bytes_written": len(content.encode())}
        elif self._name == APPEND:
            content = str(arguments.get("content", ""))
            self._files[path] = self._files.get(path, "") + content
            total = len(self._files[path].encode())
            payload = {
                "path": path,
                "bytes_appended": len(content.encode()),
                "bytes_total": total,
            }
        elif self._name == FINALIZE:
            payload = {
                "path": path,
                "bytes_total": len(self._files.get(path, "").encode()),
            }
        else:
            payload = {"ok": True}
        return ToolResult(
            tool_call_id="tc", content=json.dumps(payload), is_error=False
        )


def _ev_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    final_input: dict[str, Any],
    truncated_by_output_cap: bool = False,
) -> list[LLMStreamEvent]:
    stop_reason = "length" if truncated_by_output_cap else "tool_use"
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": tool_call_id,
                "final_input": final_input,
                "truncated_by_output_cap": truncated_by_output_cap,
            },
        ),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": stop_reason}),
    ]


class _CancelAfterTruncatedWriteLLM:
    """Yields a single truncated Write (so the engine enters the salvage
    loop), then requests ``engine.stop()`` BEFORE the next stream opens.
    Subsequent calls yield a benign empty stream so the outer loop can
    observe ``engine.stop_requested`` and route to CANCELLED.
    """

    def __init__(self, engine: QueryEngine) -> None:
        self._engine = engine
        self._first = True
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if self._first:
            self._first = False
            partial = "def big():\n" + ("    x = 1\n" * 100)
            for ev in _ev_tool_call(
                tool_call_id="t1",
                tool_name=WRITE,
                final_input={"path": TARGET, "content": partial},
                truncated_by_output_cap=True,
            ):
                # Request stop BEFORE yielding the terminal ``message_stop``
                # (the engine's per-delta stop check fires on this yield,
                # breaking the stream). The race window is then:
                # ``tool_use_stop`` (with ``truncated_by_output_cap``) was
                # already processed → ``stream_result.tool_calls`` carries
                # the truncated call → the engine enters the recovery
                # branch → the salvage loop's pre-dispatch checkpoint
                # sees ``stop_requested`` and routes to CANCELLED.
                # Calling ``stop()`` AFTER the for loop is unsafe: the
                # engine breaks on the finish delta from ``message_stop``
                # and the iterator's ``aclose()`` raises ``GeneratorExit``
                # inside the stub, preventing any post-loop code from
                # running. Setting the flag inside the loop, BEFORE the
                # last yield, is the only way the recovery branch sees it.
                if ev.name == "message_stop":
                    self._engine.stop()
                yield ev
            return
        # Any subsequent stream — yield an empty-but-clean message so the
        # outer loop's stop check at the top of the no-tool end_turn branch
        # routes to CANCELLED.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="content_block_start", payload={"kind": "text"}
        )
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": "ignored"}
        )
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": "end_turn"}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise RuntimeError("unused")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def _run(engine: QueryEngine) -> list[Any]:
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="write a big file")]
    )
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
        if len(events) > 200:
            break
    return events


@pytest.mark.asyncio
async def test_salvage_loop_skips_dispatch_after_stop_requested() -> None:
    """A ``stop_requested`` landing during the
    truncation-recovery turn (between the recovery persist and the salvage
    loop) MUST route to the CANCELLED teardown WITHOUT dispatching the
    synthetic salvage write. The workspace must be UNCHANGED (the partial
    body the parser recovered never lands on disk).

    FAILS on pre-fix code: the salvage loop had no pre-dispatch
    ``stop_requested`` checkpoint, so the synthetic Write ran, mutating
    the workspace AFTER cancellation.
    """
    rc = RuntimeConstants(
        model_context_window=8_192,
        longfile_convergence_enabled=True,
        longfile_stall_turns=2,
        longfile_expected_floor_bytes=4096,
        longfile_max_forced_appends=8,
        longfile_max_forced_finalizes=2,
        max_output_recovery_rounds=3,
        max_turns_per_run=10,
    )
    files: dict[str, str] = {}
    tools = [
        _ByteReportingFileTool(WRITE, files),
        _ByteReportingFileTool(APPEND, files),
        _ByteReportingFileTool(FINALIZE, files),
        _ByteReportingFileTool(READ, files),
    ]
    engine = _build_engine(rc=rc, llm=None)  # type: ignore[arg-type]
    engine.llm = _CancelAfterTruncatedWriteLLM(engine)  # type: ignore[assignment]
    for tool in tools:
        engine.tools.register(tool)  # type: ignore[attr-defined]

    events = await _run(engine)

    # The run terminates in CANCELLED (not COMPLETED, not FAILED).
    assert engine.is_terminal
    assert engine.state is LoopState.CANCELLED, (
        f"expected CANCELLED, got {engine.state}"
    )
    # The synthetic salvage Write NEVER dispatched — the recovery teardown
    # routed to CANCELLED before the salvage loop iterated.
    write_tool = next(t for t in tools if t.name == WRITE)
    assert write_tool.invocations == [], (
        "synthetic salvage write was dispatched AFTER engine.stop_requested; "
        "the recovery loop is missing the pre-dispatch cancel checkpoint"
    )
    # The workspace is unchanged (the partial body the parser recovered
    # never landed on disk).
    assert files == {}, f"workspace mutated post-cancel: {files!r}"
    # The events emit a cancelled-stop.
    stops = [e for e in events if e.type.value == "message_stop"]
    assert stops, "expected a terminal MESSAGE_STOP"
    assert stops[-1].payload.get("stop_reason") in (
        "cancelled",
        "stop_requested",
    )
