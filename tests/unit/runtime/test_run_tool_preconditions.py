"""Run-level tool preconditions, driven through the REAL ``QueryEngine.run`` loop.

A run may carry an ordered list of tools it MUST call before the agent is free
to answer. The mechanism is the provider's native ``tool_choice``, threaded as
``LLMRequest.extra['forced_tool_choice']`` — so these tests assert on what
reaches the wire per stream, plus the terminal state of the run, rather than on
the internal counters.

The properties under test:

* a run with NO preconditions is untouched — nothing forced, and the long-file
  convergence hint still behaves exactly as it did;
* ``[{A, 2}]`` forces A until TWO successful calls land, then forces nothing;
* an ERROR result does not advance progress;
* an entry whose tool keeps failing exhausts its attempt budget and FAILS the
  run with a reason naming the tool and the last error;
* ``[{A,1}, {B,1}, {A,1}]`` is a SEQUENCE, not a set — A is forced again;
* the long-file convergence hint is deferred, never consumed, while
  preconditions run, and is still forced afterwards.
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
    ToolPrecondition,
    ToolResult,
)
from protocore.runtime import longfile_convergence as lfc
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

SEARCH = "SearchDocs"
LOCATE = "RouteLocate"
APPEND = "AppendFile"


class _ScriptedTool(Tool):
    """A tool whose first ``failing_calls`` invocations return an error result.

    ``failing_calls=None`` means every call fails, which is how the attempt-cap
    test models a backend that can never succeed.
    """

    def __init__(self, name: str, *, failing_calls: int | None = 0) -> None:
        self._name = name
        self._failing_calls = failing_calls
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"{self._name} does a thing",
            parameters=ToolParameterSchema(
                properties={"query": {"type": "string"}}, required=[]
            ),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        fails = (
            self._failing_calls is None or len(self.calls) <= self._failing_calls
        )
        if fails:
            return ToolResult(
                tool_call_id="tc",
                content=f"{self._name} backend is unreachable",
                is_error=True,
            )
        return ToolResult(
            tool_call_id="tc",
            content=json.dumps({"ok": True, "tool": self._name}),
            is_error=False,
        )


def _ev_tool_call(*, tool_call_id: str, tool_name: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={"tool_call_id": tool_call_id, "final_input": {"query": "q"}},
        ),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "tool_use"}),
    ]


def _ev_prose(text: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": text}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"}),
    ]


class _ScriptedLLM:
    """Replays one scripted stream per call, recording each outbound request.

    ``hint_at_call`` records the long-file convergence hint as it stood when
    each request was assembled, which is how the "deferred, not consumed" test
    observes the hint mid-run rather than only at the end.
    """

    def __init__(self, scripts: list[list[LLMStreamEvent]]) -> None:
        self._scripts = scripts
        self._idx = 0
        self.calls: list[LLMRequest] = []
        self.hint_at_call: list[str | None] = []
        self.engine: QueryEngine | None = None

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        if self.engine is not None:
            self.hint_at_call.append(lfc.peek_force_next_tool(self.engine))
        idx = min(self._idx, len(self._scripts) - 1)
        self._idx += 1
        for ev in self._scripts[idx]:
            yield ev

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _build_engine(
    *,
    llm: _ScriptedLLM,
    rc: RuntimeConstants,
    preconditions: tuple[ToolPrecondition, ...] = (),
) -> QueryEngine:
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-precond",
            tenant_id="tenant-test",
            session_id="sess-precond",
            model_name="qwen3.6-35b-a3b",
            rc=rc,
            tool_preconditions=preconditions,
        ),
        llm_provider=llm,  # type: ignore[arg-type]
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    llm.engine = engine
    return engine


def _rc(**overrides: Any) -> RuntimeConstants:
    return RuntimeConstants(model_context_window=8_192, **overrides)


async def _run(engine: QueryEngine) -> list[Any]:
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="answer my question")]
    )
    return [evt async for evt in engine.run(user_msg)]


def _forced(llm: _ScriptedLLM) -> list[str | None]:
    return [req.extra.get("forced_tool_choice") for req in llm.calls]


@pytest.mark.asyncio
async def test_no_preconditions_forces_nothing_and_leaves_the_longfile_hint_alone(
) -> None:
    """The default (empty) list must be indistinguishable from before.

    Nothing is forced on the model's own account, the convergence hint is still
    consumed exactly once by the stream that carries it, and the run completes.
    """
    llm = _ScriptedLLM(
        [
            _ev_tool_call(tool_call_id="t1", tool_name=SEARCH),
            _ev_prose("here is the answer"),
        ]
    )
    engine = _build_engine(llm=llm, rc=_rc())
    engine.tools.register(_ScriptedTool(SEARCH))  # type: ignore[attr-defined]
    engine.tools.register(_ScriptedTool(APPEND))  # type: ignore[attr-defined]
    lfc.set_force_next_tool(engine, "AppendFile")

    await _run(engine)

    # The ONLY forced choice is the convergence hint's, on the first stream —
    # the precondition mechanism contributed nothing.
    assert _forced(llm)[0] == APPEND
    assert all(f is None for f in _forced(llm)[1:])
    assert lfc.peek_force_next_tool(engine) is None, (
        "the convergence hint must still be consumed exactly once when the run "
        "carries no preconditions"
    )
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_entry_asking_for_two_calls_forces_until_both_land() -> None:
    """``[{A, 2}]`` forces A on every stream until TWO successful calls land,
    then forces nothing for the rest of the run."""
    llm = _ScriptedLLM(
        [
            _ev_tool_call(tool_call_id="t1", tool_name=SEARCH),
            _ev_tool_call(tool_call_id="t2", tool_name=SEARCH),
            _ev_prose("now I can answer"),
        ]
    )
    engine = _build_engine(
        llm=llm,
        rc=_rc(),
        preconditions=(ToolPrecondition(tool=SEARCH, calls=2),),
    )
    tool = _ScriptedTool(SEARCH)
    engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert _forced(llm) == [SEARCH, SEARCH, None], (
        "A must be forced for each of its two required calls and never again"
    )
    assert len(tool.calls) == 2
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_error_result_does_not_advance_progress() -> None:
    """A tool that ERRORED did not run: the entry stays outstanding and A is
    forced again, only clearing once a SUCCESSFUL call lands."""
    llm = _ScriptedLLM(
        [
            _ev_tool_call(tool_call_id="t1", tool_name=SEARCH),
            _ev_tool_call(tool_call_id="t2", tool_name=SEARCH),
            _ev_prose("done"),
        ]
    )
    engine = _build_engine(
        llm=llm,
        rc=_rc(run_tool_precondition_max_attempts=3),
        preconditions=(ToolPrecondition(tool=SEARCH, calls=1),),
    )
    # The first call errors, the second succeeds.
    tool = _ScriptedTool(SEARCH, failing_calls=1)
    engine.tools.register(tool)  # type: ignore[attr-defined]

    await _run(engine)

    assert _forced(llm) == [SEARCH, SEARCH, None], (
        "the errored call must not satisfy the entry, so A is forced again; the "
        "successful second call must satisfy it, so A is never forced a third time"
    )
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_attempt_cap_fails_the_run_naming_tool_and_last_error() -> None:
    """A tool that can never succeed exhausts the entry's attempt budget and
    FAILS the run — it never quietly answers without the promised call."""
    llm = _ScriptedLLM([_ev_tool_call(tool_call_id="t1", tool_name=SEARCH)])
    engine = _build_engine(
        llm=llm,
        rc=_rc(run_tool_precondition_max_attempts=2),
        preconditions=(ToolPrecondition(tool=SEARCH, calls=1),),
    )
    engine.tools.register(_ScriptedTool(SEARCH, failing_calls=None))  # type: ignore[attr-defined]

    events = await _run(engine)

    assert _forced(llm) == [SEARCH, SEARCH], (
        "exactly the budgeted number of forced attempts, then no more"
    )
    assert engine.state is LoopState.FAILED
    errors = [
        evt
        for evt in events
        if evt.type is EventType.ERROR
        and evt.payload.get("kind") == "tool_precondition_unsatisfied"
    ]
    assert errors, "the run must fail with a stated tool-precondition reason"
    message = errors[-1].payload["message"]
    assert SEARCH in message, "the reason must name the tool"
    assert "backend is unreachable" in message, "the reason must carry the last error"


@pytest.mark.asyncio
async def test_duplicate_entries_are_a_sequence_not_a_set() -> None:
    """``[{A,1}, {B,1}, {A,1}]`` forces A, then B, then A AGAIN — repeating a
    tool is meaningful, so an already-called tool does not satisfy a later
    entry."""
    llm = _ScriptedLLM(
        [
            _ev_tool_call(tool_call_id="t1", tool_name=SEARCH),
            _ev_tool_call(tool_call_id="t2", tool_name=LOCATE),
            _ev_tool_call(tool_call_id="t3", tool_name=SEARCH),
            _ev_prose("answer"),
        ]
    )
    engine = _build_engine(
        llm=llm,
        rc=_rc(),
        preconditions=(
            ToolPrecondition(tool=SEARCH, calls=1),
            ToolPrecondition(tool=LOCATE, calls=1),
            ToolPrecondition(tool=SEARCH, calls=1),
        ),
    )
    search = _ScriptedTool(SEARCH)
    engine.tools.register(search)  # type: ignore[attr-defined]
    engine.tools.register(_ScriptedTool(LOCATE))  # type: ignore[attr-defined]

    await _run(engine)

    assert _forced(llm) == [SEARCH, LOCATE, SEARCH, None]
    assert len(search.calls) == 2, "the repeated entry must demand its own call"
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_longfile_hint_survives_a_run_that_had_preconditions() -> None:
    """The convergence hint is a different concern: preconditions take the
    forced slot first, but the hint is DEFERRED, not consumed, and is still
    forced once the preconditions are done."""
    llm = _ScriptedLLM(
        [
            _ev_tool_call(tool_call_id="t1", tool_name=SEARCH),
            _ev_tool_call(tool_call_id="t2", tool_name=APPEND),
            _ev_prose("sealed and answered"),
        ]
    )
    engine = _build_engine(
        llm=llm,
        rc=_rc(),
        preconditions=(ToolPrecondition(tool=SEARCH, calls=1),),
    )
    engine.tools.register(_ScriptedTool(SEARCH))  # type: ignore[attr-defined]
    engine.tools.register(_ScriptedTool(APPEND))  # type: ignore[attr-defined]
    lfc.set_force_next_tool(engine, "AppendFile")

    await _run(engine)

    assert _forced(llm) == [SEARCH, APPEND, None], (
        "the precondition owns the forced slot first; the hint is forced after"
    )
    # ``hint_at_call`` is sampled as each stream opens, i.e. just after its
    # request was assembled. Still pending on the precondition stream ⇒ forcing
    # the precondition did not burn it; gone on the next ⇒ that stream is the
    # one that consumed it, exactly once.
    assert llm.hint_at_call[0] == APPEND
    assert llm.hint_at_call[1] is None
    assert engine.state is LoopState.COMPLETED


def test_config_rejects_more_entries_than_the_runtime_constant_allows() -> None:
    """An over-long list is refused rather than truncated: a caller who asked
    for a precondition and did not get one has been lied to."""
    rc = _rc(run_tool_precondition_max_entries=2)
    with pytest.raises(ValueError, match="run_tool_precondition_max_entries"):
        QueryEngineConfig(
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            rc=rc,
            tool_preconditions=tuple(
                ToolPrecondition(tool=f"Tool{i}") for i in range(3)
            ),
        )


def test_config_rejects_calls_above_the_runtime_constant() -> None:
    rc = _rc(run_tool_precondition_max_calls=2)
    with pytest.raises(ValueError, match="run_tool_precondition_max_calls"):
        QueryEngineConfig(
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            rc=rc,
            tool_preconditions=(ToolPrecondition(tool=SEARCH, calls=3),),
        )


@pytest.mark.asyncio
async def test_progress_survives_snapshot_and_resume() -> None:
    """A run re-driven on another pod must neither re-force a satisfied entry
    nor be handed a fresh attempt budget for one that keeps failing."""
    preconditions = (
        ToolPrecondition(tool=SEARCH, calls=1),
        ToolPrecondition(tool=LOCATE, calls=2),
    )
    source = _build_engine(
        llm=_ScriptedLLM([_ev_prose("x")]), rc=_rc(), preconditions=preconditions
    )
    source._tool_precondition_index = 1
    source._tool_precondition_calls = 1
    source._tool_precondition_attempts = 2
    source._tool_precondition_last_error = "locator timed out"

    resumed = _build_engine(
        llm=_ScriptedLLM([_ev_prose("x")]), rc=_rc(), preconditions=preconditions
    )
    await resumed.resume_from_snapshot(source.snapshot())

    assert resumed._tool_precondition_index == 1
    assert resumed._tool_precondition_calls == 1
    assert resumed._tool_precondition_attempts == 2
    assert resumed._tool_precondition_last_error == "locator timed out"


def test_precondition_requires_at_least_one_call() -> None:
    """``calls`` has a floor of 1 — a zero-call precondition is not a
    precondition."""
    with pytest.raises(ValueError):
        ToolPrecondition(tool=SEARCH, calls=0)
