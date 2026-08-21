"""Salvage recovery ``Write`` / ``AppendFile`` dispatch from
:func:`_salvage_truncated_write_to_disk` MUST preserve the
``_longfile_last_mutation_truncated`` flag.

Without the fix, :func:`longfile_convergence.observe_tool_result` clears the
flag on any successful byte-adding write (it reads the landing as a clean
write). But a salvage RECOVERS a truncated write: the file is mid-content with
more to come, so its tail is semantically still truncated. Without the
preservation, a salvaged partial that is already ``>=
longfile_expected_floor_bytes`` (the realistic ~10-16 KB-recovered shape)
would read as ``plausibly_complete`` → the next model-idle stall forces a
PREMATURE ``FinalizeFile``, sealing the half-written file. The pre-dispatch
persist captures the truncated flag before the in-``_dispatch_tool`` persist, and
``keep_truncated_tail=True`` is threaded through ``_dispatch_tool`` into
``observe_tool_result`` so the post-``observe`` clear is suppressed for this
synthetic-recovery call.
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
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.longfile_convergence import (
    observe_tool_result,
    plausibly_complete,
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


def _rc(**overrides: object) -> RuntimeConstants:
    base: dict[str, object] = {
        "model_context_window": 4_096,
        "longfile_convergence_enabled": True,
        "longfile_stall_turns": 2,
        "longfile_max_forced_appends": 8,
        "longfile_max_forced_finalizes": 2,
        "longfile_plateau_delta_fraction": 0.25,
        "longfile_plateau_min_mutations": 2,
        "longfile_expected_floor_bytes": 4096,
        "longfile_min_finalize_fraction": 1.0,
        "longfile_tail_anchor_chars": 200,
    }
    base.update(overrides)
    return RuntimeConstants(**base)


def _write_call(path: str, content: str = "x") -> ToolCall:
    return ToolCall(id="tc", name="Write", arguments={"path": path, "content": content})


def _append_call(path: str, content: str = "x") -> ToolCall:
    return ToolCall(
        id="tc", name="AppendFile", arguments={"path": path, "content": content}
    )


def _write_result(bytes_written: int, path: str = TARGET) -> str:
    return json.dumps({"path": path, "bytes_written": bytes_written})


def _append_result(
    bytes_appended: int,
    bytes_total: int,
    *,
    path: str = TARGET,
    line_count_total: int | None = None,
) -> str:
    payload: dict[str, object] = {
        "path": path,
        "bytes_appended": bytes_appended,
        "bytes_total": bytes_total,
    }
    if line_count_total is not None:
        payload["line_count_total"] = line_count_total
    return json.dumps(payload)


# ── §observe_tool_result keep_truncated parameter ──────────────────────────


def test_observe_tool_result_keep_truncated_preserves_truncated_tail_flag(
    engine_factory,
) -> None:
    """A successful Write with ``keep_truncated=True``
    (synthetic-recovery salvage) MUST NOT clear
    ``_longfile_last_mutation_truncated``. The flag was set BEFORE the dispatch
    by the salvage caller, and the salvage IS still mid-content. Clearing it
    would let ``plausibly_complete`` flip to True on a half-file that already
    meets the expected-floor.

    FAILS on pre-fix code: ``observe_tool_result`` unconditionally cleared the
    flag on any successful byte-adding write, with no caller control to keep
    it set.
    """
    engine = engine_factory(rc=_rc())
    # Pre-set the truncated-tail flag (this is exactly what
    # ``_salvage_truncated_write_to_disk`` does before dispatching).
    engine._longfile_last_mutation_truncated = True
    # Bind the active path + size ABOVE the expected floor (the realistic
    # ~10-16 KB-recovered shape handled by the re-assert.
    engine._longfile_active_path = TARGET
    engine._longfile_active_file_bytes = 8_192  # > 4096 expected_floor

    # A successful Write with keep_truncated=True (the salvage shape).
    observe_tool_result(
        engine,
        _write_call(TARGET, content="recovered prefix..."),
        _write_result(bytes_written=8_192),
        is_error=False,
        keep_truncated=True,
    )

    # The truncated-tail flag is preserved — the file is still mid-content.
    assert engine._longfile_last_mutation_truncated is True, (
        "observe_tool_result cleared the truncated-tail flag even though "
        "keep_truncated=True; the half-file is now plausibly_complete=True "
        "and the next stall will force a PREMATURE FinalizeFile"
    )
    # And the higher-level gate agrees: the file is NOT plausibly complete
    # even though its size is well above the expected floor.
    assert plausibly_complete(engine) is False, (
        "plausibly_complete flipped to True on a still-truncated half-file; "
        "the driver will force FinalizeFile on the next stall — the exact "
        "outcome this guard was added to prevent"
    )


def test_observe_tool_result_default_clears_truncated_tail_flag(
    engine_factory,
) -> None:
    """The default path (a normal, non-recovery
    successful Write/AppendFile) MUST still clear the truncated-tail flag.
    Otherwise a later GENUINE clean append would keep the file classified as
    mid-content forever, and the finalize path would be unreachable (the
    anti-wedge guarantee the comment on ``keep_truncated`` cites).
    """
    engine = engine_factory(rc=_rc())
    engine._longfile_last_mutation_truncated = True
    engine._longfile_active_path = TARGET
    engine._longfile_active_file_bytes = 8_192

    # Default ``keep_truncated=False`` — a normal successful Write.
    observe_tool_result(
        engine,
        _write_call(TARGET, content="clean rewrite"),
        _write_result(bytes_written=8_192),
        is_error=False,
    )

    # The flag is cleared — the file is now a clean rewrite.
    assert engine._longfile_last_mutation_truncated is False, (
        "observe_tool_result did not clear the truncated-tail flag on a "
        "default-path successful Write; the anti-wedge guarantee is broken "
        "and the finalize path is permanently unreachable"
    )
    # And the higher-level gate now agrees.
    assert plausibly_complete(engine) is True, (
        "a clean above-floor file should be plausibly_complete=True so the "
        "driver can force FinalizeFile at the next plateau"
    )


def test_observe_tool_result_keep_truncated_on_append_preserves_flag(
    engine_factory,
) -> None:
    """The same preservation holds for
    ``AppendFile`` (the salvage name is the model's own declared op, so a
    salvaged ``AppendFile`` is a real case). A keep_truncated=True AppendFile
    must NOT clear the flag, but it MUST still advance the byte-adding
    counter and append the delta to the plateau read.
    """
    engine = engine_factory(rc=_rc())
    engine._longfile_last_mutation_truncated = True
    engine._longfile_active_path = TARGET
    engine._longfile_active_file_bytes = 6_000

    observe_tool_result(
        engine,
        _append_call(TARGET, content="recovered..."),
        _append_result(bytes_appended=512, bytes_total=6_512),
        is_error=False,
        keep_truncated=True,
    )

    # Flag preserved.
    assert engine._longfile_last_mutation_truncated is True
    # But the byte tracking DID advance (the delta landed, the running size
    # updated, the plateau saw a new delta) — the file is growing despite
    # being still mid-content.
    assert engine._longfile_active_file_bytes == 6_512
    assert engine._longfile_mutation_deltas == [512]
    # Stall clock reset.
    assert engine._turns_since_last_byte_adding_mutation == 0


def test_observe_tool_result_disabled_driver_is_noop(
    engine_factory,
) -> None:
    """A disabled driver must not touch the flag
    on a keep_truncated=True Write. (A pre-existing property of
    ``observe_tool_result``; locked in to ensure the new param doesn't
    accidentally arm the driver when the RC kill-switch is off.)
    """
    engine = engine_factory(
        rc=_rc(longfile_convergence_enabled=False)
    )
    engine._longfile_last_mutation_truncated = True

    observe_tool_result(
        engine,
        _write_call(TARGET),
        _write_result(bytes_written=8_192),
        is_error=False,
        keep_truncated=True,
    )

    # Driver disabled → no touch. The pre-set value persists.
    assert engine._longfile_last_mutation_truncated is True


# ── §plausibly_complete gate (the load-bearing consumer) ───────────────────


def test_plausibly_complete_false_when_truncated_tail_flag_set_and_above_floor(
    engine_factory,
) -> None:
    """The gate that decides append-vs-finalize
    must stay False when the truncated-tail flag is set, even if the file
    already meets the expected-floor. This is the exact "half-file at floor"
    shape that the bug would have allowed to flip to True.
    """
    engine = engine_factory(rc=_rc())
    engine._longfile_active_path = TARGET
    engine._longfile_active_file_bytes = 4_096  # == expected_floor
    # Flag NOT cleared by the salvage (the fix's preserve path).
    engine._longfile_last_mutation_truncated = True

    # The gate is False — the file is still mid-content, no premature seal.
    assert plausibly_complete(engine) is False
    # Sanity: with the flag cleared, the same size IS plausibly complete.
    engine._longfile_last_mutation_truncated = False
    assert plausibly_complete(engine) is True


# ── §salvage flow end-to-end (flag persists through the synthetic dispatch) ─


class _ByteReportingFileTool(Tool):
    """Write/AppendFile/FinalizeFile that records every invocation. The test
    only needs Write to land bytes (the salvage's first chunk)."""

    def __init__(self, name: str) -> None:
        self._name = name
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

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.invocations.append(dict(arguments))
        path = str(arguments.get("path", TARGET))
        if self._name == WRITE:
            content = str(arguments.get("content", ""))
            payload = {"path": path, "bytes_written": len(content.encode())}
        elif self._name == APPEND:
            content = str(arguments.get("content", ""))
            payload = {
                "path": path,
                "bytes_appended": len(content.encode()),
                "bytes_total": 12345,
            }
        else:
            payload = {"path": path, "bytes_total": 0}
        return ToolResult(
            tool_call_id="tc", content=json.dumps(payload), is_error=False
        )


def _build_engine(llm: object) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-f12-09",
            tenant_id="tenant-test",
            session_id="sess-f12-09",
            model_name="qwen3.6-35b-a3b",
            rc=_rc(),
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
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


class _TruncatedWriteThenIdleLLM:
    """Yields a single truncated Write (so the engine enters the salvage
    branch), then a benign stream so the outer loop exits cleanly. The test
    inspects the engine's snapshot state AFTER the run completes to assert
    the truncated-tail flag was preserved through the synthetic-recovery
    dispatch."""

    def __init__(self) -> None:
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
                yield ev
            return
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta", payload={"text": "ignored"}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={"kind": "text"})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise RuntimeError("unused")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_salvage_dispatch_preserves_truncated_tail_flag_through_run() -> None:
    """End-to-end salvage: a truncated Write's
    recovered partial lands via the synthetic-recovery dispatch, and the
    truncated-tail flag MUST remain True through the run. A pod kill between
    the in-``_dispatch_tool`` persist and a hypothetical post-dispatch
    re-assert is safe because the flag is set BEFORE the dispatch (and the
    in-dispatch persist captures it). A subsequent GENUINE clean append
    later in the run would clear it (anti-wedge) — but the test inspects
    the state BEFORE such a clean append happens.

    FAILS on pre-fix code: ``observe_tool_result`` cleared the flag on the
    salvage's successful Write, so a snapshot read after the dispatch
    returns ``_longfile_last_mutation_truncated=False``. The next stall
    then forces a PREMATURE FinalizeFile.
    """
    engine = _build_engine(llm=None)  # type: ignore[arg-type]
    engine.llm = _TruncatedWriteThenIdleLLM()  # type: ignore[assignment]
    tools: list[_ByteReportingFileTool] = []
    for name in (WRITE, APPEND, FINALIZE, READ):
        tool = _ByteReportingFileTool(name)
        tools.append(tool)
        engine.tools.register(tool)  # type: ignore[attr-defined]

    # The user prompt — the engine will:
    #   1) stream a truncated Write (so the engine enters the salvage path)
    #   2) the salvage function dispatches a synthetic Write to land bytes
    #   3) re-open the LLM; the second stream yields benign end_turn
    #   4) the run completes
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="write big.py")]
    )
    async for _evt in engine.run(user_msg):
        pass

    # The run completed normally.
    assert engine.state is LoopState.COMPLETED, (
        f"expected COMPLETED, got {engine.state}"
    )
    # The salvage's synthetic Write was dispatched.
    write_tool = next(t for t in tools if t.name == WRITE)
    assert write_tool.invocations, "salvage should have dispatched a synthetic Write"
    # ── The fix: the truncated-tail flag is preserved on the engine. ──
    # Without ``keep_truncated_tail=True`` (and the pre-set of the flag before
    # dispatch), the salvage's successful Write would have cleared the flag.
    assert engine._longfile_last_mutation_truncated is True, (
        "truncated-tail flag was cleared by the salvage's synthetic-recovery "
        "Write; a cross-pod resume between persists would now evaluate "
        "plausibly_complete=True on a half-file and force FinalizeFile"
    )
    # The file size DID land (the salvage wrote bytes).
    assert engine._longfile_active_file_bytes > 0
    # And the higher-level gate is False — the half-file is NOT complete.
    assert plausibly_complete(engine) is False
