"""#1/#4 — distinct wire ``turn_id`` per assistant-message round, asserted at the
generator / event-stream level (not just on the engine helper).

The chat per-round grouping contract (``AssistantGroup`` /
``coalesceConsecutiveAssistants``) keys turns by ``turn_id`` and expects a DISTINCT
id per LLM round, with ``block_idx`` unique WITHIN a turn. The Deep ``REASONING_STEP``
is emitted BEFORE the first ``message_start`` and the reducer MERGES the plan
placeholder into that first ``message_start`` by matching ``turn_id`` — so they must
share one id. Pre-loop terminals (compaction-failed / hook-denied / stop-before-start)
keep the LEGACY (unsuffixed) id because they run before the first round opens.
"""

from __future__ import annotations

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.events import EventType, TurnEvent
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


def _build_engine(
    *,
    run_mode: str = "direct",
    llm: InMemoryLLMProvider,
    hook_manager: InMemoryHookManager | None = None,
    rc: RuntimeConstants | None = None,
    run_id: str = "rw",
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
        registry.register(MockTool(tool_name=name, description=f"{name} tool"))
    return QueryEngine(
        config=QueryEngineConfig(
            run_id=run_id,
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="qwen3.6-35b-a3b",
            rc=rc or RuntimeConstants(model_context_window=8_192),
            run_mode=run_mode,
            thinking_enabled=(run_mode == "deep"),
            reasoning_effort="low",
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=hook_manager or InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _turn_id(evt: TurnEvent) -> str | None:
    return evt.payload.get("turn_id") if evt.payload else None


async def test_direct_two_rounds_have_distinct_round_ids() -> None:
    """Direct run with a tool round + a final round: EVERY frame of round 1 shares
    ``turn-<run>-<turn_count>-1`` and EVERY frame of round 2 shares ``...-2``,
    with no frame orphaned from its round."""
    llm = InMemoryLLMProvider()
    # round 1 — a tool call (Write).
    llm.queue_tool_call_response(
        tool_call_id="toolu_w",
        tool_name="Write",
        tool_input={"path": "a.txt", "content": "hi"},
    )
    # round 2 — final prose.
    llm.queue_response(text="all done", stop_reason=StopReason.end_turn)

    engine = _build_engine(llm=llm, run_id="rw")
    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="write a.txt")])
        )
    ]

    # turn_count incremented once in run() → "1"; round seq makes it -1 / -2.
    starts = [e for e in events if e.type is EventType.MESSAGE_START]
    assert len(starts) == 2, "two assistant-message rounds → two message_start frames"
    id_round1 = _turn_id(starts[0])
    id_round2 = _turn_id(starts[1])
    assert id_round1 == "turn-rw-1-1"
    assert id_round2 == "turn-rw-1-2"
    assert id_round1 != id_round2

    # Partition every turn_id-bearing frame by round and assert each carries the
    # round's id — no orphaned frame. Round boundary = the i-th message_start.
    round1_types = {
        EventType.MESSAGE_START,
        EventType.CONTENT_BLOCK_START,
        EventType.CONTENT_BLOCK_DELTA,
        EventType.CONTENT_BLOCK_STOP,
        EventType.TOOL_USE_START,
        EventType.TOOL_USE_INPUT_DELTA,
        EventType.TOOL_USE_STOP,
        EventType.TOOL_RESULT,
        EventType.MESSAGE_STOP,
    }
    # Walk events; track the current round id from the most recent message_start.
    current_id: str | None = None
    seen_ids: set[str] = set()
    for evt in events:
        if evt.type is EventType.MESSAGE_START:
            current_id = _turn_id(evt)
        if evt.type in round1_types and _turn_id(evt) is not None:
            seen_ids.add(_turn_id(evt))  # type: ignore[arg-type]
            # Every framed event must match the round it is inside.
            assert _turn_id(evt) == current_id, (
                f"frame {evt.type} carried {_turn_id(evt)} but round is {current_id}"
            )
    assert seen_ids == {"turn-rw-1-1", "turn-rw-1-2"}


async def test_block_idx_resets_per_round() -> None:
    """``block_idx`` restarts at 0 within each round's turn (the
    'block_idx unique within a turn' frontend contract)."""
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_w",
        tool_name="Write",
        tool_input={"path": "a.txt", "content": "hi"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(llm=llm, run_id="rb")
    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
        )
    ]

    # Group content_block_start block_idx by round id.
    per_round: dict[str, list[int]] = {}
    for evt in events:
        if evt.type is EventType.CONTENT_BLOCK_START:
            tid = _turn_id(evt)
            assert tid is not None
            per_round.setdefault(tid, []).append(evt.payload["block_idx"])

    # Each round that emitted any content block starts its block indices at 0.
    assert per_round, "expected at least one content_block_start"
    for tid, idxs in per_round.items():
        assert idxs[0] == 0, f"round {tid} block_idx should restart at 0, got {idxs}"


async def test_deep_reasoning_step_shares_id_with_first_message_start() -> None:
    """Deep: the ``REASONING_STEP`` (emitted before message_start) and the FIRST
    ``message_start`` carry the SAME turn_id — the reducer plan-merge requirement."""
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="toolu_plan",
        tool_name=PLAN_TOOL_NAME,
        tool_input={"plan": ["do it"], "next_tool": "Write", "task_complete": False},
    )
    llm.queue_tool_call_response(
        tool_call_id="toolu_w",
        tool_name="Write",
        tool_input={"path": "a.txt", "content": "x"},
    )
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)

    engine = _build_engine(run_mode="deep", llm=llm, run_id="rd")
    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="напиши файл")])
        )
    ]

    reasoning = [e for e in events if e.type is EventType.REASONING_STEP]
    starts = [e for e in events if e.type is EventType.MESSAGE_START]
    assert len(reasoning) == 1
    assert starts, "deep run still streams an action message"
    reasoning_id = _turn_id(reasoning[0])
    first_start_id = _turn_id(starts[0])
    assert reasoning_id == "turn-rd-1-1"
    assert reasoning_id == first_start_id, (
        "the SGR plan placeholder turn_id must equal round 1's message_start id "
        "so the chat reducer merges the plan into the streamed turn"
    )


async def test_hook_denied_terminal_keeps_legacy_id() -> None:
    """A UserPromptSubmit-denied run terminates BEFORE the first round opens, so
    its terminal frames keep the LEGACY (unsuffixed) ``turn-<run>-<turn_count>``
    id — no suffixed orphan with a preceding message_start."""
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.user_prompt_submit,
        HookResult(action=HookActionKind.DENY, reason="blocked"),
    )

    llm = InMemoryLLMProvider()
    engine = _build_engine(llm=llm, hook_manager=hooks, run_id="rh")
    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        )
    ]

    # No message_start at all (denied before the first round opened).
    assert not [e for e in events if e.type is EventType.MESSAGE_START]
    # The terminal message_stop carries the legacy unsuffixed id.
    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops, "hook-denied still emits a terminal message_stop"
    for stop in stops:
        assert _turn_id(stop) == "turn-rh-1", (
            "pre-loop terminal must keep the legacy turn-<run>-<turn_count> id"
        )
