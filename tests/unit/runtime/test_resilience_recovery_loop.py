"""Loop-level recovery: the post-tool empty response, and what counts as an answer.

Drives ``engine.run`` end-to-end with scripted LLM mocks. Two things are held
here. The post-tool empty-response nudge injects an API-valid
assistant('(empty)') + user(nudge) pair and re-streams once, and is
bit-identical when off. And the predicates that decide whether a run has
answered, or has written a file, draw only from THIS run's turns — a prior run
of the session is seeded into history verbatim, and its fluent answer is exactly
what would make an unanswered run look answered.
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
    PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY,
    SESSION_HISTORY_SEED_METADATA_KEY,
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
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _history_has_file_write_result,
    _terminal_tool_nudge_required,
    run_has_final_answer,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

TERMINAL_TOOL = "final_answer"

_PRIOR_RUN_ANSWER = (
    "The migration completed successfully and all 14 checks passed."
)


def _build_engine(
    *,
    rc: RuntimeConstants,
    llm: object,
    expected_terminal_tool: str | None = TERMINAL_TOOL,
    register_terminal: bool = True,
) -> QueryEngine:
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-recovery",
            tenant_id="tenant-test",
            session_id="sess-recovery",
            model_name="test-model",
            #  — prose-gate left at DEFAULT: ``final_answer`` is
            # a MESSAGE-CARRYING terminal (its schema declares ``message``), so
            # the schema-conditioned gate exempts it automatically. The
            # guaranteed-terminal / post-tool-empty resilience paths here are
            # therefore unaffected; the gate has its own coverage in
            # test_finalize_terminal_gate.py.
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
    if register_terminal:
        engine.tools.register(_RecordingTerminalTool())  # type: ignore[attr-defined]
    return engine


class _RecordingTerminalTool(Tool):
    """Terminal tool that records the message it was called with."""

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


class _PlainTextEndTurnLLM:
    """Emits a normal text turn that ends WITHOUT calling the terminal tool."""

    def __init__(self, *, text: str = "Here is my analysis: the value is 42.") -> None:
        self._text = text
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": self._text})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _terminal_metadata_in_history(engine: QueryEngine) -> bool:
    return any(
        getattr(b, "metadata", {}).get(TERMINAL_TOOL_METADATA_KEY) is True
        for m in engine.history
        for b in m.content_blocks
    )


# ---------------------------------------------------------------------------
# Helpers / predicate
# ---------------------------------------------------------------------------






def test_partial_attempts_are_not_answer_candidates() -> None:
    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    engine.history.extend(
        [
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="complete answer")],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="INCOMPLETE PREFIX")],
                metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
            ),
        ]
    )

    assert run_has_final_answer(engine) is True

    engine.history.pop(0)
    assert run_has_final_answer(engine) is False


def test_only_strict_true_marks_a_partial_attempt() -> None:
    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="ordinary answer")],
            metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: "true"},
        )
    )

    assert run_has_final_answer(engine) is True












def _seed_prior_run(engine: QueryEngine, *, question: str, answer: str) -> None:
    """Prepend a prior run's Q+A exactly as cross-run history seeding does."""
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=question)],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=answer)],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )








def testrun_has_final_answer_ignores_prior_run_seeded_turns() -> None:
    """The same run boundary applies to the empty-completion guard's
    precondition: a seeded prior answer must not mask an unanswered run."""
    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    _seed_prior_run(
        engine,
        question="What is the invoice total?",
        answer=_PRIOR_RUN_ANSWER,
    )
    assert run_has_final_answer(engine) is False
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="this run's own words")],
        )
    )
    assert run_has_final_answer(engine) is True


def _seed_prior_run_terminal_answer(engine: QueryEngine, *, answer: str) -> None:
    """Prepend a prior run that ANSWERED through the terminal tool.

    The successful terminal ``tool_use``/``tool_result`` pair is exactly what
    cross-run history seeding carries forward from a run that finalised
    properly — the ordinary shape of any session whose earlier turn worked.
    """
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_prior_term",
                    name=TERMINAL_TOOL,
                    arguments_json=json.dumps({"message": answer}),
                )
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="toolu_prior_term",
                    content="submitted",
                    is_error=False,
                    metadata={TERMINAL_TOOL_METADATA_KEY: True},
                )
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )




def test_prior_run_terminal_result_leaves_the_finalisation_check_armed() -> None:
    """The "already answered" gate reads the run boundary.

    A prior run's seeded terminal result must leave it firing for a current run
    that has answered nothing — the seeded turn is the run before this one
    finishing properly, which is the ordinary shape of any continued session.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    _seed_prior_run_terminal_answer(engine, answer=_PRIOR_RUN_ANSWER)

    assert _terminal_tool_nudge_required(engine) is True
    assert run_has_final_answer(engine) is False


def test_this_runs_own_terminal_result_still_disarms_the_finalisation_check() -> None:
    """The complement: the boundary must not cost the check its real job."""
    rc = RuntimeConstants(
        model_context_window=4_096,
        terminal_tool_nudge_enabled=True,
    )
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    _seed_prior_run_terminal_answer(engine, answer=_PRIOR_RUN_ANSWER)
    for message in engine.history:
        message.metadata.pop(SESSION_HISTORY_SEED_METADATA_KEY, None)

    assert _terminal_tool_nudge_required(engine) is False


def test_write_first_steer_survives_a_prior_runs_seeded_write() -> None:
    """The write-first half of the terminal-tool nudge asks whether THIS run
    produced its file deliverable. A prior run's seeded ``Write`` must not
    answer it: the run that has written nothing is exactly the one the steer
    is for."""
    rc = RuntimeConstants(model_context_window=4_096)
    engine = _build_engine(rc=rc, llm=_PlainTextEndTurnLLM())
    write_tool = rc.terminal_tool_nudge_file_write_tool_names[0]
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_prior_write",
                    name=write_tool,
                    arguments_json=json.dumps({"path": "report.md", "content": "prior"}),
                )
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="toolu_prior_write", content="wrote", is_error=False
                )
            ],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )
    assert _history_has_file_write_result(engine) is False

    # The same pair, produced by THIS run, does answer it.
    for message in engine.history:
        message.metadata.pop(SESSION_HISTORY_SEED_METADATA_KEY, None)
    assert _history_has_file_write_result(engine) is True












# ---------------------------------------------------------------------------
# Post-tool empty-response nudge
# ---------------------------------------------------------------------------


class _ReadTool(Tool):
    """A trivial non-terminal tool the model calls before going empty."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read",
            description="read a record",
            parameters=ToolParameterSchema(properties={"path": {"type": "string"}}),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=str(arguments.get("__tool_call_id__", "toolu_read")),
            content="record-contents",
            is_error=False,
        )


class _ToolThenEmptyThenAnswerLLM:
    """Turn 1: call ``read``. Turn 2: FULLY empty (no text/tools/reasoning).
    Turn 3 (after the nudge): submit the terminal tool."""

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = len(self.calls) - 1
        if idx == 0:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_read", "tool_name": "read"},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": "toolu_read",
                    "partial_input_json": '{"path": "/x"}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "toolu_read", "final_input": {"path": "/x"}},
            )
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
            )
            return
        if idx == 1:
            # FULLY empty turn — no content blocks, no reasoning.
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
            )
            return
        # Turn 3 — answer.
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "toolu_term", "tool_name": TERMINAL_TOOL},
        )
        yield LLMStreamEvent(
            name="tool_use_input_delta",
            payload={
                "tool_call_id": "toolu_term",
                "partial_input_json": '{"message": "done"}',
            },
        )
        yield LLMStreamEvent(
            name="tool_use_stop",
            payload={"tool_call_id": "toolu_term", "final_input": {"message": "done"}},
        )
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_post_tool_empty_nudge_recovers_when_enabled() -> None:
    """A fully-empty turn right after a tool result triggers the API-valid
    assistant('(empty)') + user(nudge) injection + a re-stream that answers."""
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=3,
        resilience_post_tool_empty_nudge_enabled=True,
        post_tool_empty_nudge_user_text="PROCESS_AND_CONTINUE",
        post_tool_empty_nudge_assistant_text="(empty)",
    )
    llm = _ToolThenEmptyThenAnswerLLM()
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(_ReadTool())  # type: ignore[attr-defined]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # The nudge fired (state-change marker).
    nudges = [
        e
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "post_tool_empty_nudge"
    ]
    assert len(nudges) == 1
    # API-valid pair appended: an assistant '(empty)' then a user nudge.
    user_nudges = [
        m
        for m in engine.history
        if m.role is MessageRole.user and "PROCESS_AND_CONTINUE" in m.text
    ]
    assert len(user_nudges) == 1
    # Wire invariant: the message right BEFORE the nudge user-turn is an
    # assistant turn (never tool->user).
    idx = engine.history.index(user_nudges[0])
    assert engine.history[idx - 1].role is MessageRole.assistant
    # The model then answered on the re-stream.
    assert _terminal_metadata_in_history(engine)
    assert engine.state is LoopState.COMPLETED
    # 3 streams: tool-call, empty, answer.
    assert len(llm.calls) == 3


class _ToolThenAlwaysEmptyLLM:
    """Turn 1: call ``read``. EVERY subsequent turn: FULLY empty.

    Used for the combined regression: the post-tool empty nudge fires
    (injecting the synthetic ``(empty)`` assistant turn), the model never
    recovers, the empty-nudge budget exhausts, and the loop falls through to
    the guaranteed-terminal backstop. The only assistant TextBlock in history
    is then the synthetic ``(empty)`` recovery scaffolding.
    """

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        idx = len(self.calls) - 1
        if idx == 0:
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": "toolu_read", "tool_name": "read"},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": "toolu_read",
                    "partial_input_json": '{"path": "/x"}',
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": "toolu_read", "final_input": {"path": "/x"}},
            )
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
            )
            return
        # Every later turn: fully empty (no text / tools / reasoning).
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        from protocore.contracts.llm import LLMResponse

        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)








@pytest.mark.asyncio
async def test_post_tool_empty_nudge_off_is_bit_identical() -> None:
    """RC OFF → the fully-empty turn falls through to the normal no-tool
    end-turn (the run COMPLETES on the empty turn with no nudge)."""
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=3,
        resilience_post_tool_empty_nudge_enabled=False,
        # Isolate the post-tool-empty-nudge off-state: the separate
        # empty-completion guard also recovers a bare-empty turn, so disable it
        # here to keep this test's "bit-identical to baseline" claim exact.
        empty_completion_guard_enabled=False,
    )
    llm = _ToolThenEmptyThenAnswerLLM()
    engine = _build_engine(rc=rc, llm=llm)
    engine.tools.register(_ReadTool())  # type: ignore[attr-defined]

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="q")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert [
        e
        for e in events
        if e.payload.get("reason") == "post_tool_empty_nudge"
    ] == []
    # The empty turn ended the run (2 streams: tool-call, empty). No 3rd
    # answering stream because the nudge never fired.
    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED


