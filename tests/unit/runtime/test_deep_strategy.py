"""Deep (SGR) loop strategy.

Deep mode runs the stand-validated SGR step BEFORE the shared action loop:

  1. a forced ``plan`` tool call (native, ``tool_choice=plan``, ``enable_thinking``
     on + ``reasoning_effort`` bounding CoT) — the lean schema
     ``{plan, next_tool, task_complete}`` (+ optional ``reasoning_summary``);
  2. the parsed plan is emitted as exactly one ``REASONING_STEP`` event and
     recorded into history as a planning turn;
  3. the existing shared ``_stream_one_assistant_message`` loop then drives the
     real action (here a forced ``Write``), so dispatch / pairing / repair /
     loop-detection are NOT duplicated.

Direct mode is unchanged: no plan call, no ``REASONING_STEP``.

Stand parity: the forced native ``plan`` tool RESPECTS the ``next_tool`` enum
(unlike ``guided_json`` which let the model invent ``create_html_file``).
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
)
from protocore.runtime.events.types import EventType
from protocore.runtime.loop_strategies import (
    PLAN_TOOL_NAME,
    DeepStrategy,
    DirectStrategy,
    plan_tool_schema,
    select_strategy,
)
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


def _build_engine(
    *,
    run_mode: str,
    llm: InMemoryLLMProvider,
    rc: RuntimeConstants | None = None,
    system_prompt_sections: tuple[str, ...] = (),
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
        registry.register(MockTool(tool_name=name, description=f"{name} tool"))
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-deep",
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="qwen3.6-35b-a3b",
            rc=rc or RuntimeConstants(model_context_window=8_192),
            run_mode=run_mode,
            thinking_enabled=(run_mode == "deep"),
            reasoning_effort="low",
            system_prompt_sections=system_prompt_sections,
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


# ---------------------------------------------------------------------------
# Plan tool schema (§A.5) — lifted verbatim from the stand
# ---------------------------------------------------------------------------


def test_plan_tool_schema_lean_shape() -> None:
    schema = plan_tool_schema(["Read", "Write"], include_summary=False)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == PLAN_TOOL_NAME
    params = schema["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["required"] == ["plan", "next_tool", "task_complete"]
    assert params["properties"]["next_tool"]["enum"] == ["Read", "Write"]
    # Lean variant carries NO prose reasoning fields.
    assert "reasoning_summary" not in params["properties"]
    assert "situation" not in params["properties"]
    assert "rationale" not in params["properties"]


def test_plan_tool_schema_summary_variant() -> None:
    schema = plan_tool_schema(["Write"], include_summary=True)
    params = schema["function"]["parameters"]
    assert "reasoning_summary" in params["properties"]
    assert params["properties"]["reasoning_summary"]["maxLength"] == 280
    assert params["required"][0] == "reasoning_summary"


# ---------------------------------------------------------------------------
# Strategy selection — single branch point on run_mode
# ---------------------------------------------------------------------------


def test_select_strategy_branches_on_run_mode() -> None:
    assert isinstance(select_strategy("direct"), DirectStrategy)
    assert isinstance(select_strategy("deep"), DeepStrategy)


# ---------------------------------------------------------------------------
# Deep turn — one REASONING_STEP then action via the shared loop
# ---------------------------------------------------------------------------


async def test_deep_mode_emits_one_reasoning_step_then_acts() -> None:
    llm = InMemoryLLMProvider()
    # call#1 — the forced plan tool-call (lean schema).
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={
            "plan": ["create index.html", "write the landing markup"],
            "next_tool": "Write",
            "task_complete": False,
        },
    )
    # call#2 — the action the shared loop drives (Write the landing).
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    # call#3 — the model finishes after the tool result.
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши лендинг для кофейни")],
    )
    events = [evt async for evt in engine.run(initial)]

    reasoning_steps = [e for e in events if e.type is EventType.REASONING_STEP]
    assert len(reasoning_steps) == 1, "Deep mode emits exactly one REASONING_STEP"
    payload = reasoning_steps[0].payload
    assert payload["plan"] == ["create index.html", "write the landing markup"]
    assert payload["next_tool"] == "Write"
    assert payload["task_complete"] is False

    # The plan call's request was a FORCED single-tool call (tool_choice=plan),
    # carried via extra so the host adapter forces it natively, and
    # exposed only the plan tool.
    plan_call = llm.calls[0]
    assert plan_call.extra.get("tool_choice") == {
        "type": "function",
        "function": {"name": PLAN_TOOL_NAME},
    }
    assert plan_call.extra.get("enable_thinking") is True
    assert plan_call.extra.get("reasoning_effort") == "low"
    assert [t.name for t in plan_call.tools] == [PLAN_TOOL_NAME]

    # The shared loop then dispatched the real Write tool (action phase reuses
    # the full surface, NOT the plan-only surface).
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    dispatched = {e.payload.get("tool_name") for e in tool_starts}
    assert "Write" in dispatched


async def test_direct_mode_emits_no_reasoning_step() -> None:
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="direct", llm=llm)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="write a landing")],
    )
    events = [evt async for evt in engine.run(initial)]

    assert not [e for e in events if e.type is EventType.REASONING_STEP]
    # No forced plan tool call — the very first call is the auto action turn.
    assert llm.calls[0].extra.get("tool_choice") is None
    # Direct still threads the thinking axis (off) into extra.
    assert llm.calls[0].extra.get("enable_thinking") is False


def _plan_ack_results(engine: QueryEngine) -> list[str]:
    """The synthetic plan-ack tool_result bodies recorded into history."""

    out: list[str] = []
    for message in engine.history:
        if message.role is not MessageRole.tool:
            continue
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock) and "Plan recorded" in block.content:
                out.append(block.content)
    return out


async def test_deep_plan_ack_carries_source_discipline_and_no_dump_constraints() -> None:
    """The Deep synthetic plan-ack tool_result MUST carry the source-delegation
    + 'reference the file, do NOT paste its body' constraints, so a future
    plan-ack edit cannot silently delete the fix while leaving the suite green.
    Deep-only seam (``_append_plan_turn``)."""

    llm = InMemoryLLMProvider()
    # next_tool="Write" matches the InMemory surface so the plan parses (the
    # ack content is independent of which next_tool the plan picked).
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={
            "plan": ["gather sources", "write the article"],
            "next_tool": "Write",
            "task_complete": False,
        },
    )
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "article.md", "content": "# Article"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши статью и приведи реальные источники")],
    )
    _ = [evt async for evt in engine.run(initial)]

    acks = _plan_ack_results(engine)
    assert len(acks) == 1, "Deep records exactly one synthetic plan-ack"
    ack = acks[0]
    # Source discipline + GENERIC delegate-for-sources (H3) — no hardcoded
    # tool/subagent/client name; source-gathering goes to a configured subagent.
    assert "delegate source-gathering to a configured subagent" in ack
    assert "see your subagent catalog" in ack
    assert "never write sources" in ack
    # Reference-don't-paste-body (H4).
    assert "do NOT paste a file's full body" in ack


async def test_direct_mode_injects_no_plan_ack() -> None:
    """The plan-ack constraint is Deep-only — a Direct run records NO synthetic
    plan-ack tool_result (no forced plan turn), so the H3/H4 plan-ack text never
    appears on the Direct path."""

    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="direct", llm=llm)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="write a landing")],
    )
    _ = [evt async for evt in engine.run(initial)]

    assert _plan_ack_results(engine) == []


# ---------------------------------------------------------------------------
# Plan-call wire correctness
# ---------------------------------------------------------------------------


async def test_plan_request_includes_system_prompt_sections() -> None:
    """The deep plan call must see the always-on scaffolding/persona.

    Regression: ``_fetch_plan`` shipped raw ``context.messages``, bypassing
    ``_prepend_system_sections``. The plan call shapes the action turn, so it
    must carry the same system prefix the action stream does — otherwise
    persona/anti-leak rules are absent for the first deep call.
    """
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["do it"], "next_tool": "Write", "task_complete": False},
    )
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    sentinel = "ALWAYS-ON SCAFFOLDING SENTINEL"
    engine = _build_engine(
        run_mode="deep", llm=llm, system_prompt_sections=(sentinel,)
    )
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши лендинг")],
    )
    [evt async for evt in engine.run(initial)]

    plan_request = llm.calls[0]
    system_msgs = [
        m for m in plan_request.messages if m.role is MessageRole.system
    ]
    assert system_msgs, "plan call must carry a system message"
    assert any(
        sentinel in b.text
        for m in system_msgs
        for b in m.content_blocks
        if isinstance(b, TextBlock)
    ), "plan call system prefix must include the scaffolding section"


async def test_plan_tool_request_carries_additional_properties_false() -> None:
    """§A.5 ``additionalProperties: false`` must survive onto the live request.

    The standalone ``plan_tool_schema`` helper had it, but the schema actually
    placed on ``LLMRequest.tools`` dropped it (``ToolParameterSchema`` had no
    such field). The strict flag now rides on the request so a strict-capable
    adapter forbids model-invented fields.
    """
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["do it"], "next_tool": "Write", "task_complete": False},
    )
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm)
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    [evt async for evt in engine.run(initial)]

    plan_request = llm.calls[0]
    assert [t.name for t in plan_request.tools] == [PLAN_TOOL_NAME]
    assert plan_request.tools[0].parameters.additional_properties is False


async def test_deep_mode_degrades_when_provider_ignores_forced_plan() -> None:
    """A non-plan tool echoed by a misbehaving provider must NOT become a plan.

    If ``tool_choice=plan`` is ignored and the model returns some OTHER tool,
    ``_fetch_plan`` must reject it (no plan args) so NO bogus ``REASONING_STEP``
    is emitted; the shared loop still drives the action turn.
    """
    llm = InMemoryLLMProvider()
    # call#1 — provider ignores the forced plan and returns Write directly.
    llm.queue_tool_call_response(
        tool_call_id="toolu_rogue",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    # call#2 — the shared action loop's own turn.
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm)
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    assert not [e for e in events if e.type is EventType.REASONING_STEP], (
        "a non-plan tool must not be recorded as an SGR plan"
    )
    # The loop still acted (degraded to direct), so the run is not wedged.
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert {e.payload.get("tool_name") for e in tool_starts} & {"Write"}


async def test_deep_mode_rejects_malformed_plan_args() -> None:
    """Malformed plan args (next_tool not in surface) must not emit a step.

    The §A.5 validator rejects a plan whose ``next_tool`` is not a live tool,
    rather than emitting a ``REASONING_STEP`` with an invalid ``next_tool``.
    """
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={
            "plan": ["do it"],
            "next_tool": "NotARealTool",  # not in the surface
            "task_complete": False,
        },
    )
    llm.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm)
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    assert not [e for e in events if e.type is EventType.REASONING_STEP]


# ---------------------------------------------------------------------------
# Plan validation canonicalization + usage + errors
# ---------------------------------------------------------------------------


def test_validated_plan_args_drops_extra_keys() -> None:
    """Canonicalize to the §A.5 key set — extra provider keys never persist.

    Even when a provider ignores ``additionalProperties: false`` and echoes
    junk fields, only ``plan``/``next_tool``/``task_complete`` (+ summary) reach
    history/REASONING_STEP.
    """
    out = DeepStrategy._validated_plan_args(
        {
            "plan": ["a", "b"],
            "next_tool": "Write",
            "task_complete": False,
            "evil_extra": {"injected": True},
            "another": "drop me",
        },
        ["Read", "Write"],
        include_summary=False,
    )
    assert out == {"plan": ["a", "b"], "next_tool": "Write", "task_complete": False}


def test_validated_plan_args_requires_summary_when_enabled() -> None:
    """§A.5 summary variant marks ``reasoning_summary`` required → reject if absent."""
    base = {"plan": ["a"], "next_tool": "Write", "task_complete": True}
    # Missing summary in summary mode → rejected.
    assert (
        DeepStrategy._validated_plan_args(base, ["Write"], include_summary=True) is None
    )
    # Present + bounded → kept.
    ok = DeepStrategy._validated_plan_args(
        {**base, "reasoning_summary": "short"}, ["Write"], include_summary=True
    )
    assert ok is not None and ok["reasoning_summary"] == "short"
    # Over-long summary → rejected.
    assert (
        DeepStrategy._validated_plan_args(
            {**base, "reasoning_summary": "x" * 9999},
            ["Write"],
            include_summary=True,
        )
        is None
    )


def test_record_plan_usage_accumulates_into_engine_total() -> None:
    """The plan call's tokens fold into ``engine.total_usage`` (no deep undercount)."""
    engine = _build_engine(run_mode="deep", llm=InMemoryLLMProvider())
    before_in = engine.total_usage.input_tokens
    before_out = engine.total_usage.output_tokens
    DeepStrategy._record_plan_usage(
        engine,
        {"input_tokens": 123, "output_tokens": 45, "cache_read_input_tokens": 7},
    )
    assert engine.total_usage.input_tokens == before_in + 123
    assert engine.total_usage.output_tokens == before_out + 45
    assert engine.total_usage.cache_read_tokens >= 7


async def test_deep_plan_call_usage_is_accounted_end_to_end() -> None:
    """A usage delta on the plan call is folded into the engine total.

    Uses a stub provider whose FIRST stream (the plan call) emits a usage
    envelope; the second/third are the normal scripted action + finish.
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    plan_stream: list[LLMStreamEvent] = [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "toolu_plan", "tool_name": PLAN_TOOL_NAME},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": "toolu_plan",
                "final_input": {
                    "plan": ["do it"],
                    "next_tool": "Write",
                    "task_complete": False,
                },
            },
        ),
        LLMStreamEvent(
            name="usage",
            payload={"input_tokens": 200, "output_tokens": 30},
        ),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        ),
    ]

    class _PlanUsageProvider:
        def __init__(self) -> None:
            self._first = True

        async def stream_with_tools(self, request: object):  # type: ignore[no-untyped-def]
            if self._first:
                self._first = False
                for evt in plan_stream:
                    yield evt
                return
            async for evt in base.stream_with_tools(request):  # type: ignore[arg-type]
                yield evt

    engine = _build_engine(run_mode="deep", llm=_PlanUsageProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    # The plan step still emitted its REASONING_STEP and the plan usage landed.
    assert [e for e in events if e.type is EventType.REASONING_STEP]
    assert engine.total_usage.input_tokens >= 200


async def test_deep_plan_call_error_degrades_to_shared_loop() -> None:
    """A typed LLM error on the plan call must NOT escape ``prepare_turn``.

    The deep turn degrades to the shared action loop (which still acts) rather
    than leaking the exception type before the turn starts.
    """
    from protocore.contracts.llm import LLMProviderError, LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    class _PlanErrorProvider:
        def __init__(self) -> None:
            self._first = True

        async def stream_with_tools(self, request: object):  # type: ignore[no-untyped-def]
            if self._first:
                self._first = False
                yield LLMStreamEvent(name="message_start", payload={})
                raise LLMProviderError("plan upstream exploded")
            async for evt in base.stream_with_tools(request):  # type: ignore[arg-type]
                yield evt

    engine = _build_engine(run_mode="deep", llm=_PlanErrorProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    # Must not raise out of the run; degrades to the action loop.
    events = [evt async for evt in engine.run(initial)]

    assert not [e for e in events if e.type is EventType.REASONING_STEP]
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert {e.payload.get("tool_name") for e in tool_starts} & {"Write"}


# ---------------------------------------------------------------------------
# Run-modes Fix #1 — provider-robust plan fallback (DeepSeek-class strict 400)
# ---------------------------------------------------------------------------


class _StubClassified:
    """Duck-typed stand-in for the adapter's ``ClassifiedError`` verdict.

    The host OpenAI-compatible adapter attaches a ``classified`` object to
    every raised ``LLMError`` and core reads ``.should_fallback`` via
    ``getattr`` (see ``loop_strategies._is_fallback_worthy``). Core tests cannot
    import the host classifier (import boundary), so we mirror the exact
    duck shape core relies on.
    """

    def __init__(self, *, should_fallback: bool) -> None:
        self.should_fallback = should_fallback


def _provider_error_with_verdict(*, should_fallback: bool) -> Exception:
    from protocore.contracts.llm import LLMProviderError

    exc = LLMProviderError("deepseek forced-tool/json_schema rejected (HTTP 400)")
    # Mirror ``error_classifier._normalise_error``: attach the verdict object.
    object.__setattr__(exc, "classified", _StubClassified(should_fallback=should_fallback))
    return exc


def _plan_json_text_stream(plan_json: str) -> list[Any]:
    """A text-only stream (no tool calls) emitting ``plan_json`` as the answer.

    Models the no-forced-tool prompted-JSON fallback response: the plan arrives
    as ordinary assistant text, which ``_fetch_plan_fallback`` accumulates and
    parses with ``json_utils``.
    """
    from protocore.contracts.llm import LLMStreamEvent

    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(
            name="content_block_delta", payload={"text": plan_json, "kind": "text"}
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


def test_parse_plan_text_recovers_object_from_prose_and_fences() -> None:
    """``_parse_plan_text`` reuses json_utils — strips think, pulls embedded json."""
    from protocore.runtime.loop_strategies import _parse_plan_text

    # Bare object.
    assert _parse_plan_text('{"plan": ["a"], "next_tool": "Write"}') == {
        "plan": ["a"],
        "next_tool": "Write",
    }
    # <think> preamble + fenced object + trailing prose.
    messy = (
        "<think>deepseek reasons here</think>\n"
        "Here is the plan:\n```json\n"
        '{"plan": ["step"], "next_tool": "Write", "task_complete": false}\n'
        "```\nDone."
    )
    assert _parse_plan_text(messy) == {
        "plan": ["step"],
        "next_tool": "Write",
        "task_complete": False,
    }
    # No JSON at all → None.
    assert _parse_plan_text("no json here") is None
    assert _parse_plan_text("") is None


async def test_deep_mode_fallback_elicits_plan_when_forced_tool_rejected() -> None:
    """DeepSeek-class strict-400 on the forced Plan → prompted-JSON fallback.

    The forced ``Plan`` call raises a FALLBACK-WORTHY ``LLMProviderError``
    (``classified.should_fallback=True``); ``_fetch_plan`` must re-elicit the
    plan WITHOUT a forced tool, parse the prompted JSON, and emit exactly one
    ``REASONING_STEP`` — no silent degrade to Direct. The fallback request must
    carry NO tools + ``response_format={"type":"json_object"}``.
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    # The shared action loop's turns (after the plan is recorded).
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    plan_json = (
        '{"plan": ["create index.html", "write markup"], '
        '"next_tool": "Write", "task_complete": false}'
    )

    captured: dict[str, Any] = {}

    class _RejectThenFallbackProvider:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: Any):  # type: ignore[no-untyped-def]
            self._call += 1
            if self._call == 1:
                # Forced Plan call → DeepSeek-class strict-schema rejection.
                yield LLMStreamEvent(name="message_start", payload={})
                raise _provider_error_with_verdict(should_fallback=True)
            if self._call == 2:
                # The no-forced-tool fallback. Capture its shape.
                captured["fallback_request"] = request
                for evt in _plan_json_text_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_RejectThenFallbackProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши лендинг для кофейни")],
    )
    events = [evt async for evt in engine.run(initial)]

    reasoning_steps = [e for e in events if e.type is EventType.REASONING_STEP]
    assert len(reasoning_steps) == 1, (
        "the fallback must still emit exactly one REASONING_STEP (no silent degrade)"
    )
    payload = reasoning_steps[0].payload
    assert payload["plan"] == ["create index.html", "write markup"]
    assert payload["next_tool"] == "Write"
    assert payload["task_complete"] is False

    fb = captured["fallback_request"]
    # No forced tool, no tools at all on the degrade path.
    assert list(fb.tools) == []
    assert fb.extra.get("tool_choice") is None
    # First rung uses json_object response_format (DeepSeek accepts this).
    assert fb.extra.get("response_format") == {"type": "json_object"}
    # The prompted instruction names the json shape + the enum + the literal
    # "json" token (required by DeepSeek's json_object mode), and tells the
    # model not to call a tool.
    last_user = [m for m in fb.messages if m.role is MessageRole.user][-1]
    instruction = " ".join(
        b.text for b in last_user.content_blocks if isinstance(b, TextBlock)
    )
    assert "json" in instruction.lower()
    assert "next_tool" in instruction
    assert "Write" in instruction

    # The shared loop still drove the real action afterwards.
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert "Write" in {e.payload.get("tool_name") for e in tool_starts}


async def test_deep_mode_no_fallback_on_non_fallback_worthy_error() -> None:
    """A NON-fallback-worthy plan error degrades silently (no fallback fired).

    A stall/timeout-class error (``classified.should_fallback=False``, or no
    verdict at all) must NOT trigger the prompted-JSON fallback — the deep turn
    degrades to the shared loop exactly as before, and only ONE plan call is
    made (no second elicitation).
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    plan_calls = {"count": 0}

    class _NonFallbackErrorProvider:
        def __init__(self) -> None:
            self._first = True

        async def stream_with_tools(self, request: Any):  # type: ignore[no-untyped-def]
            # A forced-Plan call is the one carrying tool_choice=Plan.
            if request.extra.get("tool_choice") is not None or (
                request.tools and request.tools[0].name == PLAN_TOOL_NAME
            ):
                plan_calls["count"] += 1
            if self._first:
                self._first = False
                yield LLMStreamEvent(name="message_start", payload={})
                raise _provider_error_with_verdict(should_fallback=False)
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_NonFallbackErrorProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    assert not [e for e in events if e.type is EventType.REASONING_STEP], (
        "a non-fallback-worthy error must not trigger the prompted-JSON fallback"
    )
    assert plan_calls["count"] == 1, "no second plan elicitation on a non-fallback error"
    # The shared loop still acted.
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert {e.payload.get("tool_name") for e in tool_starts} & {"Write"}


async def test_deep_mode_fallback_steps_down_to_plain_when_json_object_rejected() -> None:
    """If the ``json_object`` rung is ALSO rejected, step down to the plain rung.

    Mirrors ``complete_structured``'s json_schema→json_object→plain ladder: a
    provider that rejects ``response_format={"type":"json_object"}`` with a
    fallback-worthy 400 still gets a plan via the final plain rung.
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    plan_json = '{"plan": ["do it"], "next_tool": "Write", "task_complete": false}'
    rungs: list[Any] = []

    class _PlainOnlyProvider:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: Any):  # type: ignore[no-untyped-def]
            self._call += 1
            if self._call == 1:
                # Forced Plan → strict-schema rejection.
                yield LLMStreamEvent(name="message_start", payload={})
                raise _provider_error_with_verdict(should_fallback=True)
            if self._call in (2, 3):
                rungs.append(request.extra.get("response_format"))
                if request.extra.get("response_format") is not None:
                    # json_object rung also rejected (fallback-worthy).
                    yield LLMStreamEvent(name="message_start", payload={})
                    raise _provider_error_with_verdict(should_fallback=True)
                # Plain rung → succeeds.
                for evt in _plan_json_text_stream(plan_json):
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_PlainOnlyProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    # Ladder visited json_object first, then plain.
    assert rungs == [{"type": "json_object"}, None]
    reasoning_steps = [e for e in events if e.type is EventType.REASONING_STEP]
    assert len(reasoning_steps) == 1
    assert reasoning_steps[0].payload["next_tool"] == "Write"


async def test_deep_mode_fallback_accepted_but_invalid_plan_rejects_no_stepdown() -> None:
    """An ACCEPTED rung with an INVALID plan is REJECTED — no step-down, no step.

    The step-down ladder is REJECTION-only. When the provider ACCEPTS the
    ``json_object`` rung (no LLMError) but the returned text yields an invalid
    plan (``next_tool`` not in the surface), ``_fetch_plan_fallback`` must
    ``return None`` — exactly as the forced-tool path returns None on
    ``_validated_plan_args`` failure — rather than trying the looser ``plain``
    rung (which could accept a DIFFERENT hallucinated plan). Asserts: NO plain
    rung request is made, and NO ``REASONING_STEP`` is emitted (degrade).
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "x"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    # Accepted json_object response, but the plan is INVALID (bad next_tool).
    bad_plan_json = (
        '{"plan": ["do it"], "next_tool": "NotARealTool", "task_complete": false}'
    )
    fallback_rungs: list[Any] = []

    class _AcceptInvalidProvider:
        def __init__(self) -> None:
            self._call = 0

        async def stream_with_tools(self, request: Any):  # type: ignore[no-untyped-def]
            self._call += 1
            if self._call == 1:
                # Forced Plan → strict-schema rejection (fallback-worthy).
                yield LLMStreamEvent(name="message_start", payload={})
                raise _provider_error_with_verdict(should_fallback=True)
            # Any subsequent call is a fallback rung — record its response_format.
            if request.extra.get("response_format") is not None or list(request.tools) == []:
                # Only count genuine fallback elicitations (no tools, prompted).
                if not request.tools:
                    fallback_rungs.append(request.extra.get("response_format"))
                    # The provider ACCEPTS this rung but returns an invalid plan.
                    for evt in _plan_json_text_stream(bad_plan_json):
                        yield evt
                    return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_AcceptInvalidProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    events = [evt async for evt in engine.run(initial)]

    # Only the FIRST (json_object) rung was attempted — the accepted-but-invalid
    # plan stopped the ladder; the looser plain rung was NEVER requested.
    assert fallback_rungs == [{"type": "json_object"}], (
        "an accepted-but-invalid rung must NOT step down to the plain rung"
    )
    assert not [e for e in events if e.type is EventType.REASONING_STEP], (
        "an invalid plan must be rejected (no REASONING_STEP), like the forced path"
    )
    # The deep turn degraded to the shared loop, which still acted.
    tool_starts = [e for e in events if e.type is EventType.TOOL_USE_START]
    assert {e.payload.get("tool_name") for e in tool_starts} & {"Write"}


# ---------------------------------------------------------------------------
# Fix #4 — planning turn carries NON-EMPTY reasoning_content (deepseek contract)
# ---------------------------------------------------------------------------


def _plan_assistant_turn(engine: QueryEngine) -> Message:
    """The assistant (Plan tool_use) message ``_append_plan_turn`` recorded.

    ``_append_plan_turn`` appends an assistant turn (the Plan ToolUse) then a
    paired tool turn (the ack), so the assistant message is the penultimate
    history entry.
    """
    assert len(engine.history) >= 2
    msg = engine.history[-2]
    assert msg.role is MessageRole.assistant
    return msg


def test_append_plan_turn_uses_captured_plan_reasoning() -> None:
    """When the plan stream produced CoT, the planning turn echoes it verbatim.

    ``_fetch_plan``/``_fetch_plan_fallback`` stash the captured reasoning on the
    engine; ``_append_plan_turn`` must put it on the recorded assistant Message's
    ``reasoning_content`` (mirroring the main stream) so deepseek thinking-mode
    sees it on the ACTION-turn echo. The transient is consumed-and-cleared.
    """
    from protocore.runtime.loop_strategies import _DEEP_PLAN_REASONING_ATTR

    engine = _build_engine(run_mode="deep", llm=InMemoryLLMProvider())
    setattr(engine, _DEEP_PLAN_REASONING_ATTR, "the model's plan-call chain of thought")
    plan_args = {"plan": ["do it"], "next_tool": "Write", "task_complete": False}

    DeepStrategy()._append_plan_turn(engine, plan_args)

    msg = _plan_assistant_turn(engine)
    assert msg.reasoning_content == "the model's plan-call chain of thought"
    # Consumed-and-cleared: the transient must not leak to a later turn.
    assert not hasattr(engine, _DEEP_PLAN_REASONING_ATTR)


def test_append_plan_turn_falls_back_to_reasoning_summary() -> None:
    """No captured CoT (e.g. json_object fallback) → use the plan reasoning_summary."""
    from protocore.runtime.loop_strategies import _DEEP_PLAN_REASONING_ATTR

    engine = _build_engine(run_mode="deep", llm=InMemoryLLMProvider())
    # Empty captured reasoning (the json_object rung suppresses native CoT).
    setattr(engine, _DEEP_PLAN_REASONING_ATTR, "")
    plan_args = {
        "plan": ["do it"],
        "next_tool": "Write",
        "task_complete": False,
        "reasoning_summary": "Write the landing markup first.",
    }

    DeepStrategy()._append_plan_turn(engine, plan_args)

    msg = _plan_assistant_turn(engine)
    assert msg.reasoning_content == "Write the landing markup first."


def test_append_plan_turn_placeholder_when_no_reasoning_or_summary() -> None:
    """No captured CoT AND no summary → a minimal NON-EMPTY placeholder.

    deepseek's thinking-mode contract needs ``reasoning_content`` PRESENT +
    non-empty (not verbatim). ``_append_plan_turn`` must NEVER leave it None /
    empty when a plan turn is recorded; the placeholder names the planned tool.
    """
    engine = _build_engine(run_mode="deep", llm=InMemoryLLMProvider())
    # No transient attr at all (forced path that emitted no thinking deltas and
    # the lean schema carries no summary).
    plan_args = {"plan": ["do it"], "next_tool": "Write", "task_complete": False}

    DeepStrategy()._append_plan_turn(engine, plan_args)

    msg = _plan_assistant_turn(engine)
    assert isinstance(msg.reasoning_content, str)
    assert msg.reasoning_content.strip(), "reasoning_content must never be empty"
    # The placeholder references the planned next_tool (bilingual EN+RU).
    assert "Write" in msg.reasoning_content


async def test_deep_mode_plan_turn_reasoning_content_set_end_to_end() -> None:
    """End-to-end: the recorded planning turn carries non-empty reasoning_content.

    Drives a full deep run where the plan stream emits a ``thinking`` delta; the
    planning-turn assistant Message in history must carry that reasoning so the
    action turn's echoed context satisfies deepseek's thinking-mode contract.
    """
    from protocore.contracts.llm import LLMStreamEvent

    base = InMemoryLLMProvider()
    base.queue_tool_call_response(
        tool_call_id="toolu_write",
        tool_name="Write",
        tool_input={"path": "index.html", "content": "<html></html>"},
    )
    base.queue_response(text="done", stop_reason=StopReason.end_turn)

    plan_stream: list[Any] = [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="content_block_delta",
            payload={"text": "I should write the landing first.", "kind": "thinking"},
        ),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "toolu_plan", "tool_name": PLAN_TOOL_NAME},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={
                "tool_call_id": "toolu_plan",
                "final_input": {
                    "plan": ["write landing"],
                    "next_tool": "Write",
                    "task_complete": False,
                },
            },
        ),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        ),
    ]

    class _PlanWithCoTProvider:
        def __init__(self) -> None:
            self._first = True

        async def stream_with_tools(self, request: Any):  # type: ignore[no-untyped-def]
            if self._first:
                self._first = False
                for evt in plan_stream:
                    yield evt
                return
            async for evt in base.stream_with_tools(request):
                yield evt

    engine = _build_engine(run_mode="deep", llm=_PlanWithCoTProvider())  # type: ignore[arg-type]
    initial = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="напиши лендинг")]
    )
    [evt async for evt in engine.run(initial)]

    # Find the recorded planning-turn assistant message (the one whose tool_use
    # is the Plan tool) and assert it carries the captured reasoning.
    plan_turns = [
        m
        for m in engine.history
        if m.role is MessageRole.assistant
        and any(
            getattr(b, "name", None) == PLAN_TOOL_NAME for b in m.content_blocks
        )
    ]
    assert len(plan_turns) == 1
    assert plan_turns[0].reasoning_content == "I should write the landing first."
