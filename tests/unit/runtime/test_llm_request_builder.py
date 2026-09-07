"""Every provider call is assembled by one builder.

Four call paths reach a provider: the action stream, the deep loop's plan
call, the deep loop's prompted-JSON plan fallback, and the Tier-2 compaction
summariser. Each used to construct its own request, so they disagreed on how
the model was resolved (live override vs. frozen config), on how a forced tool
was spelled (``forced_tool_choice`` vs. a wire-shaped ``tool_choice``) and on
whether a temperature was stated at all.

These tests pin the assembled request on all four paths — the shape each one
had before the builder existed, so the consolidation is provably observable-
identical — and then pin the three properties the builder adds: one model
resolution, one forced-choice key, one temperature policy.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.llm import LLMRequest, LLMStreamEvent
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.context.compaction import (
    CompactionState,
    run_tier2_summarisation,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.loop_strategies import PLAN_TOOL_NAME
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool

MODEL = "test-model-a"
OVERRIDE_MODEL = "test-model-b"


def _build_engine(
    *,
    run_mode: str,
    llm: Any,
    rc: LoopConstants | None = None,
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in ("Read", "Write"):
        registry.register(MockTool(tool_name=name, description=f"{name} tool"))
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-builder",
            tenant_id="tenant-builder",
            session_id="sess-builder",
            model_name=MODEL,
            rc=rc or LoopConstants(model_context_window=8_192),
            run_mode=run_mode,
            thinking_enabled=(run_mode == "deep"),
            reasoning_effort="low",
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _scripted_action_llm() -> InMemoryLLMProvider:
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    return llm


async def _drive(engine: QueryEngine) -> None:
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="do the thing")]
    )
    [evt async for evt in engine.run(initial)]


class _StubClassified:
    """Duck-typed stand-in for a host adapter's classified-error verdict.

    Core reads ``.should_fallback`` off the attached object via ``getattr``
    (``loop_strategies._is_fallback_worthy``); the import boundary forbids
    importing a host classifier, so the duck shape is mirrored here.
    """

    def __init__(self, *, should_fallback: bool) -> None:
        self.should_fallback = should_fallback


def _fallback_worthy_error() -> Exception:
    from protocore.contracts.llm import LLMProviderError

    exc = LLMProviderError("forced tool rejected (HTTP 400)")
    object.__setattr__(exc, "classified", _StubClassified(should_fallback=True))
    return exc


def _plan_json_stream(plan_json: str) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": plan_json}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


# ---------------------------------------------------------------------------
# Characterisation — the four paths' assembled requests
# ---------------------------------------------------------------------------


async def test_action_stream_request_shape() -> None:
    llm = _scripted_action_llm()
    engine = _build_engine(run_mode="direct", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature == LLMRequest.model_fields["temperature"].default
    assert set(request.extra) == {
        "cache_breakpoints",
        "enable_thinking",
        "reasoning_effort",
    }
    assert request.extra["enable_thinking"] is False
    assert request.extra["reasoning_effort"] == "low"
    assert [t.name for t in request.tools] == ["Read", "Write"]
    obs = request.observability
    assert obs is not None
    assert obs.tenant_id == "tenant-builder"
    assert obs.run_id == "run-builder"
    assert obs.session_id == "sess-builder"
    assert obs.call_purpose == "run"
    assert obs.call_category == "agent_call"


async def test_plan_request_shape() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature == LLMRequest.model_fields["temperature"].default
    assert [t.name for t in request.tools] == [PLAN_TOOL_NAME]
    assert request.extra["enable_thinking"] is True
    assert request.extra["reasoning_effort"] == "low"
    obs = request.observability
    assert obs is not None
    assert obs.call_purpose == "deep_plan"
    assert obs.call_category == "planning"


async def test_plan_fallback_request_shape() -> None:
    base = InMemoryLLMProvider()
    base.queue_response(text="done", stop_reason=StopReason.end_turn)
    plan_json = '{"plan": ["a"], "next_tool": "Write", "task_complete": true}'
    captured: dict[str, LLMRequest] = {}

    class _RejectThenFallback:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: LLMRequest) -> Any:
            self._call += 1
            if self._call == 1:
                yield LLMStreamEvent(name="message_start", payload={})
                raise _fallback_worthy_error()
            if self._call == 2:
                captured["fallback"] = request
                for evt in _plan_json_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_RejectThenFallback())
    await _drive(engine)

    request = captured["fallback"]
    assert request.model == MODEL
    assert request.temperature == LLMRequest.model_fields["temperature"].default
    assert list(request.tools) == []
    assert request.extra == {"response_format": {"type": "json_object"}}
    obs = request.observability
    assert obs is not None
    assert obs.call_purpose == "deep_plan_fallback"
    assert obs.call_category == "planning"


async def test_compaction_summariser_request_shape() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        compaction_keep_recent_turns=1,
    )
    llm = InMemoryLLMProvider()
    llm.queue_response(text='{"summary": "they greeted each other"}')
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello there " * 20)]),
        Message(
            role=MessageRole.assistant, content_blocks=[TextBlock(text="hi back " * 20)]
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    await run_tier2_summarisation(
        history=history,
        compaction_llm=llm,
        state=CompactionState(),
        rc=rc,
        model_name=MODEL,
    )

    request = llm.calls[0]
    assert request.model == MODEL
    assert request.temperature == rc.compaction_summary_temperature
    assert list(request.tools) == []
    assert request.max_tokens == rc.compaction_summary_max_output_tokens
    assert request.extra == {}


# ---------------------------------------------------------------------------
# The properties the single builder guarantees
# ---------------------------------------------------------------------------


async def test_plan_call_states_its_forced_tool_under_the_shared_key() -> None:
    """One spelling of a forced choice across every path.

    The plan call used to state its forced tool in the wire shape under
    ``extra['tool_choice']`` while the action stream used the bare-name
    ``extra['forced_tool_choice']``. Both reach a provider as the same
    single-tool choice, but only one of them can be read by a single reader.
    """
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    await _drive(engine)

    request = llm.calls[0]
    assert request.extra["forced_tool_choice"] == PLAN_TOOL_NAME
    assert "tool_choice" not in request.extra


async def test_live_model_override_reaches_the_plan_call() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["a"], "next_tool": "Write", "task_complete": True},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    engine = _build_engine(run_mode="deep", llm=llm)
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    await _drive(engine)

    assert [call.model for call in llm.calls] == [OVERRIDE_MODEL] * len(llm.calls)


async def test_live_model_override_reaches_the_plan_fallback() -> None:
    base = InMemoryLLMProvider()
    base.queue_response(text="done", stop_reason=StopReason.end_turn)
    plan_json = '{"plan": ["a"], "next_tool": "Write", "task_complete": true}'
    captured: dict[str, LLMRequest] = {}

    class _RejectThenFallback:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: LLMRequest) -> Any:
            self._call += 1
            if self._call == 1:
                yield LLMStreamEvent(name="message_start", payload={})
                raise _fallback_worthy_error()
            if self._call == 2:
                captured["fallback"] = request
                for evt in _plan_json_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_RejectThenFallback())
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    await _drive(engine)

    assert captured["fallback"].model == OVERRIDE_MODEL


async def test_live_model_override_reaches_the_compaction_summariser() -> None:
    """The summariser call is resolved the same way as every other call.

    It is issued against a separately injected provider, so nothing else in
    the run states which model it names; a live override that skipped it split
    one agent turn across two models with no event saying so.
    """
    from protocore.runtime.context.compaction import CompactionAttempt
    from protocore.runtime.query import _run_compaction

    llm = _scripted_action_llm()
    engine = _build_engine(run_mode="direct", llm=llm)
    engine.apply_live_controls(model_name=OVERRIDE_MODEL)
    engine.transition_to(LoopState.RUNNING)

    seen: dict[str, Any] = {}

    async def _record(**kwargs: Any) -> CompactionAttempt:
        seen.update(kwargs)
        return CompactionAttempt()

    engine.context_manager.run_compaction = _record  # type: ignore[method-assign]
    [evt async for evt in _run_compaction(engine)]

    assert seen["model_name"] == OVERRIDE_MODEL
