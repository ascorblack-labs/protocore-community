"""Runtime-driven chunk-recovery for a truncated mutation call, under ANY
``finish_reason``.

Root cause: a ``Write`` whose ``content`` was cut at the output cap arrived as
``{"path": "novye_lyudi.html"}`` with ``content`` MISSING under
``finish_reason="tool_use"``. The mid-tool-call recovery branch only fired on
``finish_reason="length"``, so the content-less call fell through to dispatch →
Pydantic ``Field required`` → a 59-call spiral.

The recovery trigger is generalised from ``finish_reason == "length"`` to the
provider-agnostic ``truncated_by_output_cap`` signal (any finish_reason); the
content-less mutation is NEVER dispatched. The runtime drives the model into
the explicit chunk protocol with a structured recovery message naming the PATH
+ the per-call chunk budget + ``Write(header) -> AppendFile(chunks) ->
FinalizeFile``.
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
TARGET = "novye_lyudi.html"


def _build_engine(*, rc: RuntimeConstants, llm: object) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-chunk",
            tenant_id="tenant-test",
            session_id="sess-chunk",
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


class _FileTool(Tool):
    """In-memory Write/AppendFile/FinalizeFile that records the resulting file
    content — the CI stand-in for the workspace. ``content`` is REQUIRED on
    Write/AppendFile, so a content-less Write reaching dispatch would raise."""

    def __init__(self, name: str, files: dict[str, str], *, requires_content: bool) -> None:
        self._name = name
        self._files = files
        self._requires_content = requires_content
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        props: dict[str, Any] = {"path": {"type": "string"}}
        required = ["path"]
        if self._requires_content:
            props["content"] = {"type": "string"}
            required.append("content")
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} a file",
            parameters=ToolParameterSchema(properties=props, required=required),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        path = str(arguments.get("path", ""))
        if self._name == WRITE:
            self._files[path] = str(arguments["content"])
        elif self._name == APPEND:
            self._files[path] = self._files.get(path, "") + str(arguments["content"])
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "tc")),
            content="ok",
            is_error=False,
        )


def _ev_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    final_input: dict[str, Any],
    truncated_by_output_cap: bool = False,
    stop_reason: str = "tool_use",
) -> list[LLMStreamEvent]:
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


class _ScriptedLLM:
    """Replays a list of per-call event-lists."""

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._idx = 0
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = min(self._idx, len(self._scripts) - 1)
        self._idx += 1
        for ev in self._scripts[idx]:
            yield ev

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


async def _run(engine: QueryEngine) -> list[TurnEvent]:
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="make html")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_truncated_write_under_tool_use_is_not_dispatched() -> None:
    """C2 — the prod symptom: a content-less Write under finish_reason=tool_use
    flagged truncated_by_output_cap must NOT be dispatched (no Field required)."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    scripts = [
        # 1) The truncated content-less Write (cap hit under tool_use).
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name=WRITE,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        # 2) Model recovers with a complete small Write, then ends.
        _ev_tool_call(
            tool_call_id="tc2",
            tool_name=WRITE,
            final_input={"path": TARGET, "content": "<html>done</html>"},
            stop_reason="tool_use",
        ),
        # 3) end_turn.
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "done"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # No Write invocation lacks content — the content-less one was intercepted.
    assert all("content" in inv for inv in write.invocations), (
        "C2: a content-less Write must NEVER reach dispatch"
    )
    # The recovered (complete) Write DID dispatch and produced the file.
    assert files.get(TARGET) == "<html>done</html>"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_recovery_message_names_path_and_chunk_protocol() -> None:
    """C3 — the recovery context must name the PATH + the chunk protocol
    (Write -> AppendFile -> FinalizeFile) + the per-call chunk budget."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    scripts = [
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name=WRITE,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "stop"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    events = await _run(engine)

    # The recovery instruction must reach the next LLM call's history.
    second_call = llm.calls[1] if len(llm.calls) > 1 else None
    assert second_call is not None, "expected a re-stream after the truncated Write"
    history_text = "\n".join(
        block.text
        for msg in second_call.messages
        for block in msg.content_blocks
        if hasattr(block, "text") and getattr(block, "text", None)
    )
    assert TARGET in history_text, "C3: recovery context must name the truncated PATH"
    assert APPEND in history_text, "C3: recovery context must name AppendFile"
    assert FINALIZE in history_text, "C3: recovery context must name FinalizeFile"

    recovery_evts = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and "truncation" in str(e.payload.get("reason", ""))
    ]
    assert recovery_evts, "C3: a truncation-recovery state_changed must fire"


@pytest.mark.asyncio
async def test_single_write_truncates_then_chunked_completes_file() -> None:
    """C6 — the stand's '0/4 single Write -> 4/4 chunked' in CI form:
    DETECT -> NOT-dispatch -> chunk-recovery drives a COMPLETE file via
    Write(header) + AppendFile(chunk) + FinalizeFile -> loop bounded."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    scripts = [
        # 1) Single big Write truncates (content cut at the cap).
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name=WRITE,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        # 2) Model writes the header chunk.
        _ev_tool_call(
            tool_call_id="tc2",
            tool_name=WRITE,
            final_input={"path": TARGET, "content": "<!doctype html><html>"},
            stop_reason="tool_use",
        ),
        # 3) Model appends the body chunk.
        _ev_tool_call(
            tool_call_id="tc3",
            tool_name=APPEND,
            final_input={"path": TARGET, "content": "<body>hello</body></html>"},
            stop_reason="tool_use",
        ),
        # 4) Model finalizes.
        _ev_tool_call(
            tool_call_id="tc4",
            tool_name=FINALIZE,
            final_input={"path": TARGET},
            stop_reason="tool_use",
        ),
        # 5) end_turn.
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "complete"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # The file is COMPLETE (doctype ... </html>) and the content-less Write
    # never corrupted it.
    assert TARGET in files
    assert files[TARGET].startswith("<!doctype html>")
    assert files[TARGET].endswith("</html>")
    # The content-less Write (tc1) was intercepted, not dispatched: every Write
    # invocation carries content.
    assert all("content" in inv for inv in write.invocations)
    # Loop bounded — completed cleanly, did not spiral to max_turns.
    assert engine.state is LoopState.COMPLETED
    assert len(llm.calls) <= 6


@pytest.mark.asyncio
async def test_truncation_recovery_is_loop_bounded() -> None:
    """C2/C4 — a model that keeps re-emitting the SAME truncated Write must
    terminate (bounded), not spiral. With max_output_recovery_rounds budget
    the run goes terminal FAILED rather than hitting max_turns_per_run."""
    rc = RuntimeConstants(
        model_context_window=8_192,
        max_output_recovery_rounds=2,
        max_turns_per_run=50,
    )
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    # Every call returns the same truncated content-less Write.
    truncated = _ev_tool_call(
        tool_call_id="tc",
        tool_name=WRITE,
        final_input={"path": TARGET},
        truncated_by_output_cap=True,
        stop_reason="tool_use",
    )
    llm = _ScriptedLLM([truncated])  # single script, replayed forever
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # Bounded: never dispatched the broken Write, and terminated well under the
    # 50-turn cap (the recovery budget is 2).
    assert write.invocations == []
    assert engine.state is LoopState.FAILED
    assert len(llm.calls) <= rc.max_output_recovery_rounds + 3


@pytest.mark.asyncio
async def test_non_content_truncation_uses_generic_resume_not_chunk_protocol() -> None:
    """A non-content truncated call (no ``path`` + missing ``content`` shape)
    must get the GENERIC resume message, NOT the file-chunk
    Write->AppendFile->FinalizeFile protocol (which would be irrelevant)."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    files: dict[str, str] = {}
    # A generic tool with a required ``query`` field (not a file mutation).
    search = _FileTool("Search", files, requires_content=False)

    class _SearchTool(Tool):
        def __init__(self) -> None:
            self.invocations: list[dict[str, Any]] = []

        @property
        def name(self) -> str:
            return "Search"

        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="Search",
                description="Search",
                parameters=ToolParameterSchema(
                    properties={"query": {"type": "string"}},
                    required=["query"],
                ),
            )

        async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
            self.invocations.append(dict(arguments))
            return ToolResult(tool_call_id="s", content="ok", is_error=False)

    del search
    scripts = [
        # A truncated Search call (no path, no content) flagged truncated.
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name="Search",
            final_input={},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "stop"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(_SearchTool())  # type: ignore[attr-defined]

    await _run(engine)

    second_call = llm.calls[1] if len(llm.calls) > 1 else None
    assert second_call is not None
    history_text = "\n".join(
        block.text
        for msg in second_call.messages
        for block in msg.content_blocks
        if hasattr(block, "text") and getattr(block, "text", None)
    )
    # Generic resume mentions the tool name but NOT the file-chunk protocol.
    assert "Search" in history_text
    assert APPEND not in history_text, (
        "a non-content truncation must NOT get the AppendFile chunk message"
    )
    assert FINALIZE not in history_text


@pytest.mark.asyncio
async def test_path_only_non_content_tool_uses_generic_resume() -> None:
    """A truncated call to a tool that carries a ``path`` but is NOT a content
    writer (e.g. Read) must get the GENERIC resume, NOT the file-chunk protocol.
    The decision is schema-based: Read declares no ``content`` parameter."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)

    class _ReadTool(Tool):
        @property
        def name(self) -> str:
            return "Read"

        @property
        def definition(self) -> ToolDefinition:
            # Read declares ``path`` (+ ``file_path`` alias-ish) but NO ``content``.
            return ToolDefinition(
                name="Read",
                description="Read a file",
                parameters=ToolParameterSchema(
                    properties={"path": {"type": "string"}},
                    required=["path"],
                ),
            )

        async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(tool_call_id="r", content="data", is_error=False)

    scripts = [
        # A truncated Read flagged truncated, path present, content absent.
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name="Read",
            final_input={"path": "doc.md"},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "ok"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(_ReadTool())  # type: ignore[attr-defined]

    await _run(engine)

    second_call = llm.calls[1] if len(llm.calls) > 1 else None
    assert second_call is not None
    history_text = "\n".join(
        block.text
        for msg in second_call.messages
        for block in msg.content_blocks
        if hasattr(block, "text") and getattr(block, "text", None)
    )
    assert "Read" in history_text
    assert APPEND not in history_text, (
        "a path-only NON-content tool (Read) must NOT get the "
        "file-chunk AppendFile message"
    )
    assert FINALIZE not in history_text


class _DynamicContentTool(Tool):
    """A dynamic/tenant tool that REQUIRES a ``content`` field but is NOT a
    chunkable file mutation (e.g. ``post_comment(content=...)``). Optionally
    opts into chunk-recovery via ``chunkable_content_mutation``."""

    def __init__(self, *, name: str, chunkable: bool | None) -> None:
        self._name = name
        self._chunkable = chunkable
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} (dynamic content tool)",
            parameters=ToolParameterSchema(
                properties={
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                },
                required=["target", "content"],
                chunkable_content_mutation=self._chunkable,
            ),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        return ToolResult(tool_call_id="d", content="ok", is_error=False)


def _generic_resume_scripts(*, tool_name: str) -> list[list[LLMStreamEvent]]:
    return [
        # A truncated call: ``target`` present, required ``content`` cut.
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name=tool_name,
            final_input={"target": "thread-7"},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "ok"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]


def _history_text(request: object) -> str:
    return "\n".join(
        block.text
        for msg in getattr(request, "messages", [])
        for block in msg.content_blocks
        if hasattr(block, "text") and getattr(block, "text", None)
    )


@pytest.mark.asyncio
async def test_dynamic_content_required_tool_uses_generic_resume_not_chunk() -> None:
    """A dynamic/tenant tool whose required ``content`` was cut, but which is
    NEITHER on the built-in allowlist NOR explicitly flagged, must get the
    GENERIC tool-call resume — NOT the Write->AppendFile->Finalize file-chunk
    protocol (telling it to AppendFile a non-existent file is wrong).
    The old shape-only predicate (``content`` declared) wrongly matched it."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    tool = _DynamicContentTool(name="PostComment", chunkable=None)
    llm = _ScriptedLLM(_generic_resume_scripts(tool_name="PostComment"))
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert len(llm.calls) > 1
    history_text = _history_text(llm.calls[1])
    assert "PostComment" in history_text
    assert APPEND not in history_text, (
        ": a dynamic content-required tool (not allowlisted/flagged) must"
        "NOT be routed into the AppendFile chunk protocol"
    )
    assert FINALIZE not in history_text


@pytest.mark.asyncio
async def test_dynamic_content_tool_opt_in_flag_gets_chunk_protocol() -> None:
    """The explicit ``chunkable_content_mutation=True`` opt-in routes a
    per-tenant content-mutation tool INTO chunk-recovery (so the flag is a
    real, universal opt-in, not a no-op)."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    tool = _DynamicContentTool(name="TenantDoc", chunkable=True)
    llm = _ScriptedLLM(_generic_resume_scripts(tool_name="TenantDoc"))
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert len(llm.calls) > 1
    history_text = _history_text(llm.calls[1])
    # The flagged tool gets the file-chunk protocol naming AppendFile/Finalize.
    assert APPEND in history_text and FINALIZE in history_text, (
        ": an explicitly flagged content-mutation tool must get the"
        "Write->AppendFile->FinalizeFile chunk protocol"
    )


@pytest.mark.asyncio
async def test_repeat_truncation_before_any_write_keeps_write_header() -> None:
    """When a Write to the SAME path truncates AGAIN before any chunk has
    SUCCESSFULLY been written, the recovery message must STILL instruct
    ``Write(header)`` (NOT 'continue with AppendFile' — there is no file to
    append to). It must also LOWER the header budget vs the first prompt."""
    rc = RuntimeConstants(
        model_context_window=8_192,
        max_output_recovery_rounds=4,
        write_chunk_token_budget=1600,
        truncation_chunk_recovery_repeat_budget_divisor=2,
        truncation_chunk_recovery_min_chunk_token_budget=100,
    )
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    truncated_write = _ev_tool_call(
        tool_call_id="tc-trunc",
        tool_name=WRITE,
        final_input={"path": TARGET},  # content cut — never dispatches
        truncated_by_output_cap=True,
        stop_reason="tool_use",
    )
    scripts = [
        # 1) first truncation, 2) SECOND truncation (still no successful write),
        # 3) end_turn.
        truncated_write,
        truncated_write,
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "stop"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # No successful write landed → the path must NOT be marked mid-chunked.
    assert TARGET not in engine._mid_chunked_write_paths
    # The SECOND recovery prompt (history of the 3rd LLM call) must STILL tell
    # the model to Write the header and must NOT carry the AppendFile-continue
    # note.
    assert len(llm.calls) >= 3
    second_recovery = _history_text(llm.calls[2])
    assert "Write(" in second_recovery, (
        ": a repeat truncation before any write must still instruct Write(header)"
    )
    assert "ALREADY started chunking" not in second_recovery, (
        ": must NOT tell the model to AppendFile a non-existent file"
    )
    # The header budget was lowered on the repeat: 1600 -> 800 (divisor 2).
    assert "800" in second_recovery, (
        ": the repeat header budget must be lowered (1600 // 2 = 800)"
    )


@pytest.mark.asyncio
async def test_truncation_after_successful_write_steers_to_appendfile() -> None:
    """Once a chunk has ACTUALLY been written (a successful Write/AppendFile),
    a later truncation of the SAME path DOES get the stronger 'you already
    started chunking; continue with AppendFile' directive."""
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=4)
    files: dict[str, str] = {}
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    scripts = [
        # 1) Truncated Write (content cut) → recovery says Write(header).
        _ev_tool_call(
            tool_call_id="tc1",
            tool_name=WRITE,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        # 2) Model writes a real header chunk SUCCESSFULLY (marks mid-chunked).
        _ev_tool_call(
            tool_call_id="tc2",
            tool_name=WRITE,
            final_input={"path": TARGET, "content": "<!doctype html><html>"},
            stop_reason="tool_use",
        ),
        # 3) A follow-up AppendFile truncates (content cut) → recovery must now
        #    steer to AppendFile (the file exists).
        _ev_tool_call(
            tool_call_id="tc3",
            tool_name=APPEND,
            final_input={"path": TARGET},
            truncated_by_output_cap=True,
            stop_reason="tool_use",
        ),
        # 4) end_turn.
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "stop"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # The successful header Write marked the path mid-chunked.
    assert TARGET in engine._mid_chunked_write_paths
    assert files.get(TARGET) == "<!doctype html><html>"
    # The recovery prompt AFTER the successful write (history of the 4th LLM
    # call) carries the AppendFile-continue directive.
    assert len(llm.calls) >= 4
    third_recovery = _history_text(llm.calls[3])
    assert "ALREADY started chunking" in third_recovery, (
        ": after a successful chunk write, a repeat truncation must steer"
        "to AppendFile"
    )


def _ev_multi_tool_call(
    *,
    calls: list[dict[str, Any]],
    stop_reason: str = "length",
) -> list[LLMStreamEvent]:
    """Build ONE assistant turn that emits MULTIPLE tool_use blocks.

    Each ``calls`` entry is ``{"tool_call_id", "tool_name", "final_input",
    "truncated_by_output_cap"?}``. Models stream complete sibling calls BEFORE
    the cap cuts the final one, so order matters: list the complete call(s)
    first, the truncated tail call last.
    """
    events: list[LLMStreamEvent] = [LLMStreamEvent(name="message_start", payload={})]
    for call in calls:
        events.append(
            LLMStreamEvent(
                name="tool_use_start",
                payload={
                    "tool_call_id": call["tool_call_id"],
                    "tool_name": call["tool_name"],
                },
            )
        )
        events.append(
            LLMStreamEvent(
                name="tool_use_stop",
                payload={
                    "tool_call_id": call["tool_call_id"],
                    "final_input": call["final_input"],
                    "truncated_by_output_cap": call.get(
                        "truncated_by_output_cap", False
                    ),
                },
            )
        )
    events.append(
        LLMStreamEvent(name="message_stop", payload={"stop_reason": stop_reason})
    )
    return events


class _ReadTool(Tool):
    """A complete (non-mutating) read that records invocations + returns data."""

    def __init__(self) -> None:
        self.invocations: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Read"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Read",
            description="Read a file",
            parameters=ToolParameterSchema(
                properties={"path": {"type": "string"}},
                required=["path"],
            ),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "r")),
            content="FILE-CONTENTS-OK",
            is_error=False,
        )


@pytest.mark.asyncio
async def test_complete_sibling_call_dispatched_when_one_call_truncates() -> None:
    """When an assistant turn emits a COMPLETE call (Read) alongside a
    truncated Write cut at the output cap, the complete sibling must be
    DISPATCHED (its work is not lost), and its ``tool_use`` must carry a REAL
    tool_result on the next request — NOT a forward-filled ``is_error`` pairing
    placeholder synthesised by ``_repair_outbound_tool_pairing``.

    Before the fix the truncation branch appended the complete call's
    ``tool_use`` to history but never dispatched it, so the next outbound
    request forward-filled the opaque pairing placeholder and the legitimately
    completed Read was dropped.
    """
    rc = RuntimeConstants(model_context_window=8_192, max_output_recovery_rounds=3)
    files: dict[str, str] = {}
    read = _ReadTool()
    write = _FileTool(WRITE, files, requires_content=True)
    append = _FileTool(APPEND, files, requires_content=True)
    finalize = _FileTool(FINALIZE, files, requires_content=False)

    placeholder = rc.tool_result_pairing_repair_placeholder

    scripts = [
        # 1) One assistant turn: complete Read(doc.md) THEN truncated Write
        #    (content cut at the output cap). The cap ends the stream, so the
        #    complete Read precedes the truncated tail call.
        _ev_multi_tool_call(
            calls=[
                {
                    "tool_call_id": "read1",
                    "tool_name": "Read",
                    "final_input": {"path": "doc.md"},
                },
                {
                    "tool_call_id": "write1",
                    "tool_name": WRITE,
                    "final_input": {"path": TARGET},  # content cut
                    "truncated_by_output_cap": True,
                },
            ],
            stop_reason="length",
        ),
        # 2) Model recovers + ends.
        [
            LLMStreamEvent(name="message_start", payload={}),
            LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
            LLMStreamEvent(name="content_block_delta", payload={"text": "done"}),
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
        ],
    ]
    llm = _ScriptedLLM(scripts)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(read)  # type: ignore[attr-defined]
    engine.tools.register(write)  # type: ignore[attr-defined]
    engine.tools.register(append)  # type: ignore[attr-defined]
    engine.tools.register(finalize)  # type: ignore[attr-defined]

    await _run(engine)

    # The COMPLETE Read was dispatched — its work is not lost.
    assert read.invocations, (
        "the complete sibling Read must be dispatched, not dropped"
    )
    assert read.invocations[0].get("path") == "doc.md"

    # The next outbound request must carry a REAL tool_result for the Read
    # tool_use — NOT the forward-filled pairing placeholder. The placeholder
    # would mean ``_repair_outbound_tool_pairing`` had to synthesise an
    # is_error result for an orphaned tool_use (the exact bug symptom).
    assert len(llm.calls) > 1, "expected a re-stream after the truncated turn"
    second_request = llm.calls[1]
    read_result_texts: list[str] = []
    for msg in second_request.messages:
        if msg.role is not MessageRole.tool:
            continue
        for block in msg.content_blocks:
            if getattr(block, "tool_call_id", None) == "read1":
                read_result_texts.append(str(getattr(block, "content", "")))
    assert read_result_texts, "the Read tool_use must have a paired result"
    assert all(placeholder not in txt for txt in read_result_texts), (
        "the Read result must be the REAL dispatch result, not the "
        "forward-filled pairing placeholder"
    )
    assert any("FILE-CONTENTS-OK" in txt for txt in read_result_texts), (
        "the Read tool_use must be paired with its REAL dispatch result"
    )


class _OutputBudgetSpiralLLM:
    """Every call emits a big text turn (lots of output tokens) then loops with
    a tool call so the run never voluntarily ends — modelling the prod
    runaway-output spiral that grew history toward the context-length ceiling."""

    def __init__(self, *, output_tokens_per_call: int) -> None:
        self._otpc = output_tokens_per_call
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "x"})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        # Report large output usage so total_usage.output_tokens grows fast.
        yield LLMStreamEvent(
            name="usage",
            payload={"input_tokens": 100, "output_tokens": self._otpc},
        )
        # A tool call so the turn is non-terminal and the loop continues.
        yield LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "tcx", "tool_name": APPEND},
        )
        yield LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": "tcx",
                "final_input": {"path": TARGET, "content": "more"},
            },
        )
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "tool_use"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_run_output_token_budget_terminates_runaway() -> None:
    """C4 — a runaway-output spiral is bounded by the cumulative output-token
    budget (terminates FAILED) BEFORE it can reach the context-length ceiling,
    independently of max_turns_per_run."""
    rc = RuntimeConstants(
        model_context_window=8_192,
        max_turns_per_run=500,  # high turn cap — the TOKEN budget must bound it
        run_max_output_tokens_budget=10_000,
    )
    files: dict[str, str] = {}
    append = _FileTool(APPEND, files, requires_content=True)
    llm = _OutputBudgetSpiralLLM(output_tokens_per_call=4_000)
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(append)  # type: ignore[attr-defined]

    events = await _run(engine)

    assert engine.state is LoopState.FAILED
    # 4000 output tokens/call, budget 10000 → trips within a handful of calls,
    # far under the 500-turn cap.
    assert len(llm.calls) <= 6
    budget_evts = [
        e
        for e in events
        if "run_output_token_budget" in str(e.payload.get("reason", ""))
        or "run_output_token_budget" in str(e.payload.get("kind", ""))
    ]
    assert budget_evts, "C4: expected a run_output_token_budget terminal signal"
