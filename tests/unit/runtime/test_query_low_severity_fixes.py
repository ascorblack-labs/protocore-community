"""Regression tests for low-severity fixes in ``protocore.runtime.query``.

Covers three defects in the same query-main-loop area:

 * Pre-terminal self-verify at each terminal-completion injection site
   (single-element batch, parallel batch, serial) breaks the dispatch loop
   WITHOUT bumping ``max_messages``. When the terminal tool completes on the
   final allowed message, the next iteration exhausts the budget and the run
   exits FAILED(max_turns) despite the already-submitted terminal answer.

 * Pairing synthesis (``_synthesize_missing_tool_results``) runs on
   cancel/LLM-error/hook-deny/compaction teardowns but NOT on the two
   DURABLE-EXIT paths it should also cover:
   (a) terminal-tool-completed COMPLETED exits — sibling tool_use blocks left
   in history after the dispatch loop breaks on the terminal result;
   (b) the max-turns FAILED exit — the run carries an unpaired tool_use
   when the turn budget exhausts.
   The outbound wire repair prevents provider 400s but the durable
   ``engine.history`` / ``session_messages`` mirror stays unpaired, so chat
   reducers / dashboards render a tool call with no result.

 * Once the deadline early-finalize latch fires, the terminal-only guard
   (``_terminal_only_blocks``) rejects every non-``expected_terminal_tool``
   dispatch. The longfile convergence driver AND the voluntary-seal helper
   force FinalizeFile/AppendFile — neither of which is the expected terminal
   tool — so a tenant combining ``agent_max_seconds>0`` +
   ``expected_terminal_tool`` + ``terminal_tool_nudge_enabled`` +
   ``longfile_convergence_enabled`` sees a forced seal short-circuited into
   a ``terminal_only`` is_error result: forced budget charged, one-shot seal
   latches consumed, file left unsealed.

The tests are end-to-end scripts that drive a real :class:`QueryEngine`
through the in-memory adapter set.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _maybe_drive_longfile_convergence,
    _maybe_seal_longfile_at_voluntary_finish,
    _synthesize_missing_tool_results,
    _terminal_only_enforced,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool

TERMINAL_TOOL = "final_answer"
READ_TOOL = "Read"


# ---------------------------------------------------------------------------
# Shared test scaffolding — mirrors test_resilience_recovery_loop.py
# ---------------------------------------------------------------------------
def _build_engine(
    *,
    rc: RuntimeConstants,
    llm: object,
    expected_terminal_tool: str | None = TERMINAL_TOOL,
    pre_terminal_self_verify_trigger: object | None = None,
) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-f1a",
            tenant_id="tenant-f1a",
            session_id="sess-f1a",
            model_name="qwen3.6-35b-a3b",
            #  — prose-gate at DEFAULT: ``final_answer`` here is a
            # MESSAGE-CARRYING terminal (schema declares ``message``) ⟹ the
            # schema-conditioned gate auto-exempts it (no RC override needed).
            rc=rc,
            expected_terminal_tool=expected_terminal_tool,
            pre_terminal_self_verify_trigger=pre_terminal_self_verify_trigger,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


class _TerminalTool(Tool):
    """Terminal tool that records its invocations and returns terminal metadata."""

    def __init__(self) -> None:
        self.invoked_with: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return TERMINAL_TOOL

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=TERMINAL_TOOL,
            description="submit terminal answer",
            parameters=ToolParameterSchema(
                properties={
                    "message": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
            ),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.invoked_with.append(dict(arguments))
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "toolu_term")),
            content="submitted",
            is_error=False,
            metadata={TERMINAL_TOOL_METADATA_KEY: True},
        )


class _ReadTool(Tool):
    """Read-only tool registered so the test surface matches a real tenant."""

    def __init__(self) -> None:
        self.invoked_with: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return READ_TOOL

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=READ_TOOL,
            description="read a file",
            parameters=ToolParameterSchema(
                properties={"path": {"type": "string"}},
            ),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.invoked_with.append(dict(arguments))
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "toolu_read")),
            content="<file contents>",
            is_error=False,
        )


def _terminal_metadata_in_history(engine: QueryEngine) -> bool:
    return any(
        getattr(b, "metadata", {}).get(TERMINAL_TOOL_METADATA_KEY) is True
        for m in engine.history
        for b in m.content_blocks
    )


def _orphan_tool_use_ids(engine: QueryEngine) -> set[str]:
    """Return the set of ``ToolUseBlock.tool_call_id`` values that have NO
 matching ``ToolResultBlock`` in history — the unpaired orphans left behind
 by missing pairing synthesis.
 """
    paired: set[str] = set()
    for m in engine.history:
        for b in m.content_blocks:
            if isinstance(b, ToolResultBlock):
                paired.add(b.tool_call_id)
    orphans: set[str] = set()
    for m in engine.history:
        for b in m.content_blocks:
            if isinstance(b, ToolUseBlock):
                if b.tool_call_id not in paired:
                    orphans.add(b.tool_call_id)
    return orphans


def _terminal_then_end_turn_llm():
    """LLM that calls the terminal tool on the FIRST stream then end-turn."""

    class _Llm:
        def __init__(self) -> None:
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self.calls.append(request)
            if len(self.calls) == 1:
                # First (and only) turn: call the terminal tool.
                yield LLMStreamEvent(name="message_start", payload={})
                yield LLMStreamEvent(
                    name="tool_use_start",
                    payload={
                        "tool_call_id": "toolu_term",
                        "tool_name": TERMINAL_TOOL,
                    },
                )
                yield LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": "toolu_term",
                        "partial_input_json": '{"message": "42", "refs": []}',
                    },
                )
                yield LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": "toolu_term",
                        "final_input": {"message": "42", "refs": []},
                    },
                )
                yield LLMStreamEvent(
                    name="message_stop",
                    payload={"stop_reason": StopReason.tool_use.value},
                )
                return
            # Subsequent: plain end-turn.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.end_turn.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            from protocore.contracts.llm import LLMResponse

            return LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[]),
                stop_reason=StopReason.end_turn,
            )

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    return _Llm()


# ===========================================================================
# Pre-terminal self-verify bumps max_messages
# ===========================================================================
class _TerminalThenRepairedAnswerLLM:
    """LLM that calls the terminal tool on turn 1 (so the self-verify gate
    fires), then a plain text end-turn on the corrective re-driven turn.

    Tests the self-verify AT THE BUDGET BOUNDARY: the terminal tool completes
    on the FINAL allowed message (``max_turns_per_run=1`` → turn 1 IS the
    budget). The self-verify gate injects the corrective user turn before
    finalising and re-drives the outer loop. Without the ``max_messages`` bump,
    the corrective turn is turn 2 and ``assistant_message_idx(2) >
    max_messages(1)`` immediately trips the max-turns exhaustion path: the
    corrective stream never runs, only ONE LLM call fires, and the run exits
    FAILED(max_turns) despite the already-submitted terminal answer. With the
    fix the bump grants turn 2 a fresh slot, the corrective end-turn runs (a
    SECOND LLM call), and the run COMPLETES.
    """

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if len(self.calls) == 1:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_term", "tool_name": TERMINAL_TOOL},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": "toolu_term",
                    "partial_input_json": '{"message": "first answer", "refs": []}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={
                    "tool_call_id": "toolu_term",
                    "final_input": {"message": "first answer", "refs": []},
                },
            )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )
            return
        # Corrective re-driven turn: end-turn, no further tool calls.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(
            name="content_block_delta",
            payload={"text": "corrected answer"},
        )
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": StopReason.end_turn.value},
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _self_verify_trigger(_engine: object) -> str:
    """Generic corrective trigger — opaque to core."""
    return "Your previous answer cited an unobserved ref. Re-issue the terminal answer."


@pytest.mark.asyncio
async def test_self_verify_injection_grants_fresh_max_messages_slot() -> None:
    """When the self-verify gate fires on the final allowed message, the
    corrective turn must be granted a fresh ``max_messages`` slot. Without the
    fix, the next outer iteration exhausts the budget and the run exits
    FAILED(max_turns) despite the terminal answer already being in history.

    Driving AT THE BUDGET BOUNDARY: ``max_turns_per_run=1`` so the terminal
    tool completes on the ONLY in-budget message (turn 1). The gate injects
    the corrective user turn before finalising; the corrective turn is
    therefore turn 2 — out of budget unless the bump grants it a slot.

    Discriminating assertion: with the bump removed the unfixed code yields
    FAILED with exactly ONE LLM call (the corrective stream never runs); with
    the fix it yields COMPLETED with TWO LLM calls (the corrective end-turn
    runs). ``max_turns_per_run=2`` would make this NON-pinning.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        pre_terminal_self_verify_enabled=True,
        pre_terminal_self_verify_max_extra_turns=1,
    )
    llm = _TerminalThenRepairedAnswerLLM()
    engine = _build_engine(
        rc=rc, llm=llm, pre_terminal_self_verify_trigger=_self_verify_trigger
    )
    engine.tools.register(_TerminalTool())  # type: ignore[arg-type]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # LLM was called twice: terminal-tool turn + the corrective re-driven turn.
    assert len(llm.calls) == 2
    # The terminal result landed in history (the model's first submission).
    assert _terminal_metadata_in_history(engine)
    # The corrective user turn was injected (the trigger returned a string).
    assert any(
        m.role is MessageRole.user
        and any(
            "unobserved ref" in b.text
            for b in m.content_blocks
            if isinstance(b, TextBlock)
        )
        for m in engine.history
    )
    # The terminal nudge is suppressed
    # because the terminal result IS in history — the run must complete via
    # the normal end-turn path on the corrective turn.
    assert engine.state is LoopState.COMPLETED
    # No FAILED(max_turns) state-change fired (the pre-fix symptom).
    failed = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("to") == LoopState.FAILED.value
    ]
    assert failed == []


@pytest.mark.asyncio
async def test_self_verify_disabled_remains_bit_identical_no_bump() -> None:
    """When the self-verify gate is DISABLED (the default) the run completes
    without a corrective turn: terminal-tool submission → COMPLETED, no extra
    LLM stream. The bump is conditional on the gate actually firing.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_turns_per_run=2,
        # pre_terminal_self_verify_enabled defaults to False; the trigger
        # would not fire even if it were enabled.
    )
    llm = _terminal_then_end_turn_llm()
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(_TerminalTool())  # type: ignore[arg-type]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    async for _evt in engine.run(user_msg):
        pass

    # LLM called exactly once: the terminal-tool turn. No corrective re-drive.
    assert len(llm.calls) == 1
    assert _terminal_metadata_in_history(engine)
    assert engine.state is LoopState.COMPLETED


# ===========================================================================
# Pairing synthesis at orphan paths
# ===========================================================================
@pytest.mark.asyncio
async def test_terminal_completed_synthesises_orphan_sibling_results() -> None:
    """When the model emits ``[terminal_tool, Read]`` in one assistant turn,
    the dispatch loop walks the tools in order and BREAKS on the terminal
    result. The sibling ``Read`` tool_use is left in ``engine.history``
    without a paired ``ToolResultBlock``. The outbound wire repair prevents
    the next provider 400 but the durable snapshot carries the orphan.

    The fix: call ``_synthesize_missing_tool_results`` at the
    terminal-tool-completed exit so the durable history is pairing-valid for
    session consumers.
    """
    rc = RuntimeConstants(model_context_window=4_096)
    terminal_call_id = "toolu_term"
    read_call_id = "toolu_read"

    class _Llm:
        def __init__(self) -> None:
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self.calls.append(request)
            # First (and only) turn: TWO tool_use blocks in one assistant
            # message — terminal_tool FIRST so the dispatch loop walks it
            # and breaks; Read SECOND, undispatched.
            yield LLMStreamEvent(name="message_start", payload={})
            # terminal
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": terminal_call_id, "tool_name": TERMINAL_TOOL},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": terminal_call_id,
                    "partial_input_json": '{"message": "42", "refs": []}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={
                    "tool_call_id": terminal_call_id,
                    "final_input": {"message": "42", "refs": []},
                },
            )
            # sibling Read
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": read_call_id, "tool_name": READ_TOOL},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": read_call_id,
                    "partial_input_json": '{"path": "/tmp/x"}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={
                    "tool_call_id": read_call_id,
                    "final_input": {"path": "/tmp/x"},
                },
            )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            from protocore.contracts.llm import LLMResponse

            return LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[]),
                stop_reason=StopReason.end_turn,
            )

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    llm = _Llm()
    engine = _build_engine(rc=rc, llm=llm)
    terminal = _TerminalTool()
    read = _ReadTool()
    engine.tools.register(terminal)  # type: ignore[arg-type]
    engine.tools.register(read)  # type: ignore[arg-type]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    async for _evt in engine.run(user_msg):
        pass

    # Terminal tool was dispatched (FIRST in LLM order); the Read sibling
    # was NOT dispatched (the loop broke on the terminal result).
    assert terminal.invoked_with == [{"message": "42", "refs": []}]
    assert read.invoked_with == []
    # Run COMPLETED via the terminal path.
    assert _terminal_metadata_in_history(engine)
    assert engine.state is LoopState.COMPLETED
    # The durable history is PAIRING-VALID: every tool_use has a paired
    # tool_result. The pre-fix symptom was an orphan Read tool_use left in
    # history (the wire repair would have forward-filled an opaque synthetic,
    # but the in-memory ``engine.history`` mirror that session consumers render
    # stayed unpaired).
    assert _orphan_tool_use_ids(engine) == set()


@pytest.mark.asyncio
async def test_max_turns_failed_exit_e2e_synthesises_orphan() -> None:
    """END-TO-END drive of the max-turns FAILED exit (``query.py`` ~line 882).
 When the turn budget exhausts with an unpaired tool_use still in
 ``engine.history`` (a resume-from-partial-batch orphan), the FAILED exit
 must call ``_synthesize_missing_tool_results`` so the PERSISTED snapshot is
 pairing-valid for session consumers.

 Drive: a pre-existing assistant ``ToolUseBlock`` (orphan) is seeded in
 ``engine.history`` (modelling a cross-pod resume that rehydrated a partial
 batch). ``max_turns_per_run=1`` and NO ``expected_terminal_tool`` — the
 model emits ONE in-budget non-terminal ``Read`` turn, then the next outer
 iteration exhausts the budget and routes to the FAILED exit. The outbound
 wire repair keeps the in-flight requests pairing-valid on the wire, but
 ``engine.history`` itself carries the orphan UNTIL the synthesis call pairs
 it.
 """
    # Wind-down off: it would grant the run more turns before the FAILED exit,
    # and the seam under test is the exit's orphan synthesis, not how many turns
    # precede it.
    rc = RuntimeConstants(
        model_context_window=4_096, max_turns_per_run=1, soft_stop_enabled=False
    )

    class _OneReadThenNothingLLM:
        def __init__(self) -> None:
            self.calls: list[LLMRequest] = []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            self.calls.append(request)
            # Single in-budget turn: one non-terminal Read (dispatched +
            # paired), so the run does NOT complete and the next outer
            # iteration trips the max-turns FAILED exit.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_live_read", "tool_name": READ_TOOL},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": "toolu_live_read",
                    "partial_input_json": '{"path": "/tmp/x"}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={
                    "tool_call_id": "toolu_live_read",
                    "final_input": {"path": "/tmp/x"},
                },
            )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            from protocore.contracts.llm import LLMResponse

            return LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[]),
                stop_reason=StopReason.end_turn,
            )

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    llm = _OneReadThenNothingLLM()
    # No expected_terminal_tool — the deadline / nudge / guaranteed-terminal
    # early-finalize paths stay inert, so the budget-exhaustion routes
    # straight to the FAILED exit (the seam under test).
    engine = _build_engine(rc=rc, llm=llm, expected_terminal_tool=None)
    engine.tools.register(_ReadTool())  # type: ignore[arg-type]
    # Seed the resume-from-partial-batch orphan: an assistant tool_use with
    # NO paired tool_result, exactly as a cross-pod rehydrate of a partial
    # batch would leave it.
    orphan_id = "toolu_resumed_orphan"
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id=orphan_id,
                    name=READ_TOOL,
                    arguments_json='{"path": "/resumed"}',
                )
            ],
        )
    )
    # Sanity — the orphan is present BEFORE the run drives the FAILED exit.
    assert orphan_id in _orphan_tool_use_ids(engine)

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # The run hit the max-turns FAILED exit (NOT a clean completion).
    assert engine.state is LoopState.FAILED
    failed = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "max_turns_exhausted"
    ]
    assert failed, "expected the max-turns FAILED exit to fire"
    # The persisted history is PAIRING-VALID: the resumed orphan was
    # synthesised a paired is_error tool_result by the FAILED-exit synthesis.
    # Pre-fix this orphan survived in ``engine.history`` (the wire repair only
    # fixes the outbound copy).
    assert _orphan_tool_use_ids(engine) == set()
    # The synthetic result carries the configured interrupted placeholder.
    synth = [
        b
        for m in engine.history
        if m.role is MessageRole.tool
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock) and b.tool_call_id == orphan_id
    ]
    assert len(synth) == 1
    assert synth[0].is_error is True
    assert synth[0].content == rc.tool_result_interrupted_placeholder


def test_synthesise_helper_contract() -> None:
    """Direct contract test of the load-bearing primitive
    ``_synthesize_missing_tool_results`` the exit-path fixes call. NOT a
    seam-pinning test (the seams are pinned end-to-end above); this asserts
    the helper's invariants the exit sites rely on: one synthetic is_error
    tool_result per orphan, inserted IMMEDIATELY after the orphan's assistant
    message (the assistant->tool adjacency the wire requires), and full
    idempotency on a second call.
    """
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_orphan_1",
                    name="Read",
                    arguments_json='{"path": "/x"}',
                ),
                ToolUseBlock(
                    tool_call_id="toolu_orphan_2",
                    name="Read",
                    arguments_json='{"path": "/y"}',
                ),
            ],
        ),
    ]
    inserted = _synthesize_missing_tool_results(
        history, error_content="<interrupted>"
    )
    assert inserted == 2
    # The new messages were inserted IMMEDIATELY after the orphan's
    # assistant message — adjacency invariant preserved.
    assert history[0].role is MessageRole.user
    assert history[1].role is MessageRole.assistant
    # tool_role messages, one per orphan id.
    tool_results = [m for m in history if m.role is MessageRole.tool]
    assert len(tool_results) == 2
    assert {m.content_blocks[0].tool_call_id for m in tool_results} == {
        "toolu_orphan_1",
        "toolu_orphan_2",
    }
    for m in tool_results:
        assert m.content_blocks[0].is_error is True
        assert m.content_blocks[0].content == "<interrupted>"
    # Idempotent — a second call is a no-op.
    assert _synthesize_missing_tool_results(history, error_content="x") == 0


# ===========================================================================
# Longfile forced seals skip under terminal-only enforcement
# ===========================================================================
def _longfile_engine() -> QueryEngine:
    """Engine wired with the longfile driver and a stub ``FinalizeFile`` /
    ``AppendFile`` tool surface, plus a deadline that has already latched
    the terminal-only enforcement.

    Mirrors ``test_terminal_finalize.py::_build_terminal_engine``.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        agent_max_seconds=30.0,
        agent_deadline_finalize_slack_seconds=5.0,
        terminal_tool_nudge_enabled=True,
        longfile_convergence_enabled=True,
    )
    return _build_engine(rc=rc, llm=_terminal_then_end_turn_llm())


@pytest.mark.asyncio
async def test_voluntary_seal_skips_under_terminal_only_enforcement() -> None:
    """Once the deadline early-finalize latch fires, the terminal-only guard
    rejects every non-``expected_terminal_tool`` dispatch with a
    ``terminal_only`` is_error. The synthetic voluntary-seal helper that tries
    to dispatch FinalizeFile for the truncation-gated file would short-circuit
    through that guard: forced budget charged, one-shot
    ``_longfile_voluntary_seal_used`` latch consumed, the synthetic assistant
    tool_use left in history, file left unsealed — every seal-related side
    effect fires except the seal itself.

    The fix: ``_maybe_seal_longfile_at_voluntary_finish`` checks
    ``_terminal_only_enforced(engine)`` and skips cleanly (no budget charge,
    no latch consumption, no synthetic tool_use) so the deadline path can drive
    the model to its expected terminal tool and complete.
    """
    engine = _longfile_engine()
    # Force the truncation-gated + active-path state so
    # ``terminal_seal_required`` returns True and the helper would, absent the
    # terminal-only guard, proceed all the way to the FinalizeFile dispatch.
    engine._longfile_active_path = "/tmp/f1a_05_artifact.txt"
    engine._longfile_truncated_paths.add("/tmp/f1a_05_artifact.txt")
    engine._longfile_finalized = False
    # CRITICAL — seed byte progress ABOVE the empty-finalize floor
    # (``longfile_expected_floor_bytes(4096) * longfile_min_finalize_fraction(1.0)``
    # = 4096) so ``finalize_permitted`` → True and therefore
    # ``terminal_seal_required`` → True. WITHOUT this the helper returns at the
    # ``terminal_seal_required`` gate BEFORE ever reaching the guard, and the
    # test would pass whether or not the guard exists (non-discriminating).
    # 8192 is comfortably past the 4096 floor.
    engine._longfile_active_file_bytes = 8192
    # Start the wind-down — the durable, cross-pod state the guard reads.
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)
    engine._terminal_only_active = True
    # Register FinalizeFile on the tool surface.
    finalize = MockTool(tool_name="FinalizeFile", response_content="sealed")
    engine.tools.register(finalize)  # type: ignore[arg-type]
    # PRECONDITION — the helper would otherwise dispatch: the seal IS required
    # (so control reaches the terminal-only guard, not the earlier early-returns)
    # and the terminal-only latch IS in force (so the guard fires). If either of
    # these is False the test is not exercising the guard.
    import protocore.runtime.longfile_convergence as _lf

    assert _lf.terminal_seal_required(engine) is True
    assert _terminal_only_enforced(engine) is True
    # Latches / budget must be untouched by the helper after the fix.
    pre_voluntary_used = engine._longfile_voluntary_seal_used
    pre_forced_budget = engine._longfile_forced_finalizes
    pre_history_len = len(engine.history)

    events = [evt async for evt in _maybe_seal_longfile_at_voluntary_finish(engine)]

    # No events yielded (the helper short-circuited before dispatching).
    assert events == []
    # The one-shot seal latch is UNCHANGED — no latches consumed. (Pre-fix the
    # helper would have set this to True before the dispatch.)
    assert engine._longfile_voluntary_seal_used == pre_voluntary_used
    assert engine._longfile_voluntary_seal_used is False
    # The forced-finalize budget is UNCHANGED — no budget charged. (Pre-fix
    # ``commit_forced_finalize`` would have incremented it.)
    assert engine._longfile_forced_finalizes == pre_forced_budget
    # The durable history is UNCHANGED — no synthetic assistant tool_use
    # appended that would have been left dangling by the rejected dispatch.
    assert len(engine.history) == pre_history_len
    # The FinalizeFile tool was NEVER called (the deadline is in force;
    # the model is supposed to call the expected terminal tool now). Pre-fix
    # the helper dispatches FinalizeFile, which the terminal-only guard then
    # rejects — the wasted dispatch the fix prevents.
    assert finalize.calls == []


@pytest.mark.asyncio
async def test_longfile_convergence_skips_under_terminal_only_enforcement() -> None:
    """The longfile convergence driver (``_maybe_drive_longfile_convergence``)
    forces the next assistant stream's ``tool_choice`` to AppendFile or
    FinalizeFile. Once the deadline latch fires, the next iteration's
    ``_dispatch_tool`` would short-circuit with a ``terminal_only`` is_error
    — forced budget charged, INCOMPLETE continue message appended, snapshot
    persisted, and a wasted LLM turn burns a stream read.

    The fix: bail BEFORE the budget/continue/persist side effects so the
    deadline path can drive the model to its expected terminal tool. The
    final bool sentinel must be ``False`` (no forced action) so the caller
    does NOT ``continue`` the outer loop.
    """
    engine = _longfile_engine()
    engine._longfile_active_path = "/tmp/f1a_05_convergence.txt"
    engine._longfile_truncated_paths.add("/tmp/f1a_05_convergence.txt")
    # CRITICAL — seed the state so ``decide_next_forced_tool`` returns a NON-None
    # forced tool (the terminal-seal branch): file comfortably past the
    # empty-finalize floor (8192 >= 4096) so ``finalize_permitted`` is True,
    # the forced-append budget is already spent
    # (``_longfile_forced_appends >= longfile_max_forced_appends(8)``), and the
    # tail is mid-content (``_longfile_last_mutation_truncated`` → NOT
    # plausibly_complete) so the model "keeps truncating" branch fires. WITHOUT
    # this, ``decide_next_forced_tool`` returns None in the setup and the helper
    # yields ``[False]`` with zero side effects whether or not the guard exists
    # (non-discriminating). Finalize budget is fresh (0 < 2) so a seal is
    # permitted.
    engine._longfile_active_file_bytes = 8192
    engine._longfile_forced_appends = engine.config.rc.longfile_max_forced_appends
    engine._longfile_last_mutation_truncated = True
    # Start the wind-down.
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)
    engine._terminal_only_active = True
    # PRECONDITION — absent the terminal-only guard the driver WOULD force a
    # tool here. (Asserted with the terminal-only latch cleared so we probe the
    # pure decision; the latch is real for the actual call below.)
    import protocore.runtime.longfile_convergence as _lf

    assert _lf.decide_next_forced_tool(engine) == "FinalizeFile"
    assert _terminal_only_enforced(engine) is True

    pre_history_len = len(engine.history)
    pre_forced_budget = engine._longfile_forced_finalizes
    pre_force_flag = getattr(engine, "_longfile_force_next_tool", None)

    items: list[object] = []
    async for item in _maybe_drive_longfile_convergence(engine):
        items.append(item)

    # The helper yielded no events and a single final ``False`` sentinel —
    # the caller sees no forced action and does NOT continue the outer loop.
    # (Pre-fix: a forced "FinalizeFile" → events + a final ``True`` sentinel.)
    bool_sentinels = [x for x in items if isinstance(x, bool)]
    event_yields = [x for x in items if not isinstance(x, bool)]
    assert bool_sentinels == [False]
    assert event_yields == []
    # No INCOMPLETE continue message appended (the side effect that would
    # have polluted the prompt on the deadline-final turn). Pre-fix the forced
    # FinalizeFile path appends a user continue message.
    assert len(engine.history) == pre_history_len
    # No forced-finalize budget charged. Pre-fix ``commit_forced_finalize``
    # would have incremented it.
    assert engine._longfile_forced_finalizes == pre_forced_budget
    # No ``force_next_tool`` set (the next stream runs with the model's
    # natural tool_choice, not the forced terminal-blocked choice). Pre-fix
    # ``set_force_next_tool(engine, "FinalizeFile")`` would have set it.
    assert getattr(engine, "_longfile_force_next_tool", None) == pre_force_flag
