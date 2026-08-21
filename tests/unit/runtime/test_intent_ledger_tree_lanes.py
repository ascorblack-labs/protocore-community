"""Intent, ledger, fork/clone, lanes, hooks, telemetry — shipped paths."""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.intent import (
    refuse_intent_when_disabled,
    replay_policy_for,
    resume_open_intents,
)
from protocore.runtime.lanes import (
    acquire_lane,
    create_lane,
    ensure_main,
    mark_diverged,
    refuse_lanes_when_disabled,
    release_lane,
    reviewer_blocks_main,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import query
from protocore.runtime.session_tree import clone_session, fork_session, refuse_tree_when_disabled
from protocore.runtime.telemetry import is_prometheus_safe_label, mark_recovery, start_span
from protocore.runtime.typed_hooks import (
    PUBLISHED_HOOKS,
    HookOutcome,
    HookRegistry,
    dispatch_hook,
    refuse_hooks_when_disabled,
)
from protocore.runtime.usage_ledger import append_usage, from_seq, refuse_ledger_when_disabled, session_total
from protocore.tests_support.adapters import InMemoryLLMProvider

from ._tool_fixtures import MockTool


def _on(**overrides: object) -> RuntimeConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "intent_settlement_enabled": True,
        "usage_ledger_enabled": True,
        "session_tree_enabled": True,
        "lanes_enabled": True,
        "typed_hooks_enabled": True,
        "telemetry_spans_enabled": True,
    }
    values.update(overrides)
    return RuntimeConstants(**values)  # type: ignore[arg-type]


def test_flags_off_refuse() -> None:
    with pytest.raises(ValueError, match="intent_settlement_disabled"):
        refuse_intent_when_disabled(False)
    with pytest.raises(ValueError, match="usage_ledger_disabled"):
        refuse_ledger_when_disabled(False)
    with pytest.raises(ValueError, match="session_tree_disabled"):
        refuse_tree_when_disabled(False)
    with pytest.raises(ValueError, match="lanes_disabled"):
        refuse_lanes_when_disabled(False)
    with pytest.raises(ValueError, match="typed_hooks_disabled"):
        refuse_hooks_when_disabled(False)


def test_replay_policy_write_never_read_safe() -> None:
    rc = _on()
    assert replay_policy_for("Write", rc) == "never"
    assert replay_policy_for("Read", rc) == "safe"


@pytest.mark.asyncio
async def test_query_write_intent_then_crash_does_not_rewrite(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    writes: list[dict[str, object]] = []

    async def on_write(args: dict[str, object]) -> None:
        writes.append(args)

    in_memory_runtime["tools"].register(
        MockTool(tool_name="Write", description="write", on_invoke=on_write, response_content="ok")
    )
    llm.queue_tool_call_response(
        tool_call_id="w1",
        tool_name="Write",
        tool_input={"path": "f.txt", "content": "once"},
    )
    llm.queue_response(text="done")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="write")]))
    events = [evt async for evt in query(engine)]
    assert any(evt.type == EventType.INTENT_COMMITTED for evt in events)
    assert len(writes) == 1
    assert engine.open_intents
    # Crash after intent, before a second dispatch of the same call.
    engine.open_intents[-1].status = "open"
    engine.open_intents = resume_open_intents(engine.open_intents)
    assert engine.open_intents[-1].status == "interrupted"
    llm.queue_tool_call_response(
        tool_call_id="w1",
        tool_name="Write",
        tool_input={"path": "f.txt", "content": "twice"},
    )
    llm.queue_response(text="recovered")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    more = [evt async for evt in query(engine)]
    assert len(writes) == 1
    assert any(
        evt.type == EventType.TOOL_RESULT and evt.payload.get("content") == "interrupted"
        for evt in more
    )


@pytest.mark.asyncio
async def test_query_read_safe_replays(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    reads: list[dict[str, object]] = []

    async def on_read(args: dict[str, object]) -> None:
        reads.append(args)

    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", description="read", on_invoke=on_read, response_content="hi")
    )
    llm.queue_tool_call_response(tool_call_id="r1", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="read")]))
    await _drain(query(engine))
    llm.queue_tool_call_response(tool_call_id="r2", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok2")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    await _drain(query(engine))
    assert len(reads) == 2


async def _drain(agen) -> list:
    return [item async for item in agen]


def test_ledger_fail_retry_compact_sum() -> None:
    rc = _on()
    rows: list[Any] = []
    rows = append_usage(rows, kind="inference", run_id="r", input_tokens=10, output_tokens=1, success=False, rc=rc)
    rows = append_usage(rows, kind="retry", run_id="r", input_tokens=10, output_tokens=2, success=True, rc=rc)
    rows = append_usage(rows, kind="compaction", run_id="r", input_tokens=3, output_tokens=0, success=True, rc=rc)
    rows = append_usage(rows, kind="inference", run_id="r", input_tokens=4, output_tokens=5, success=True, rc=rc)
    assert [item.seq for item in rows] == [1, 2, 3, 4]
    assert session_total(rows) == 10 + 1 + 10 + 2 + 3 + 4 + 5
    assert [item.seq for item in from_seq(rows, 2)] == [3, 4]
    assert append_usage([], kind="inference", run_id="r", input_tokens=1, output_tokens=1, success=True, rc=RuntimeConstants()) == []


def test_fork_clone_do_not_mutate_source() -> None:
    rc = _on()
    history = ["u0", "a0", "u1", "a1"]
    forked = fork_session(history, upto_index=1, parent_session_id="s", rc=rc)
    assert history == ["u0", "a0", "u1", "a1"]
    assert forked.history == ["u0", "a0"]
    assert forked.parent_session_id == "s"
    assert forked.audit["parent_session_id"] == "s"
    assert forked.session_id != "s"
    cloned = clone_session(history, settled=True, parent_session_id="s", rc=rc)
    assert cloned.history == history
    with pytest.raises(ValueError, match="clone_requires_settled"):
        clone_session(history, settled=False, parent_session_id="s", rc=rc)
    with pytest.raises(ValueError, match="session_tree_disabled"):
        fork_session(history, upto_index=0, parent_session_id="s", rc=RuntimeConstants())


def test_lanes_reviewer_after_diverge_does_not_block_main() -> None:
    rc = _on()
    lanes = ensure_main([])
    lanes = create_lane(lanes, lane_id="reviewer", cursor=0, model="m", toolset=("Read",), rc=rc)
    lanes = acquire_lane(lanes, "reviewer", "pod-b")
    lanes = mark_diverged(lanes, "reviewer")
    assert not reviewer_blocks_main(lanes)
    lanes = acquire_lane(lanes, "main", "pod-a")
    assert lanes[0].locked_by == "pod-a"
    lanes = release_lane(lanes, "reviewer", "pod-b")
    assert next(item.locked_by for item in lanes if item.lane_id == "reviewer") is None
    with pytest.raises(ValueError, match="lanes_disabled"):
        create_lane([], lane_id="x", cursor=0, model="m", toolset=(), rc=RuntimeConstants())


def test_typed_hooks_and_telemetry() -> None:
    rc = _on()
    registry = HookRegistry()
    assert "before_tool" in PUBLISHED_HOOKS
    registry.register("before_tool", lambda payload: HookOutcome(decision="require_approval", approval_token="tok"))
    out = dispatch_hook(registry, "before_tool", {"tool_name": "Write"}, rc)
    assert out.decision == "require_approval"
    off = dispatch_hook(registry, "before_tool", {}, RuntimeConstants())
    assert off.decision == "allow"
    span = start_span("tool", rc=rc, tool="Write")
    assert span is not None
    marked = mark_recovery(span, intent_id="op_1")
    assert marked is not None and marked.attributes["recovery"] is True
    assert marked.attributes["intent_id"] == "op_1"
    assert start_span("tool", rc=RuntimeConstants()) is None
    assert not is_prometheus_safe_label("session_id")
    assert is_prometheus_safe_label("tenant")


@pytest.mark.asyncio
async def test_query_before_tool_approval_pauses(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    registry = HookRegistry()
    registry.register(
        "before_tool",
        lambda payload: HookOutcome(decision="require_approval", approval_token="tok-h"),
    )
    engine.typed_hook_registry = registry
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", description="w"))
    llm.queue_tool_call_response(tool_call_id="w2", tool_name="Write", tool_input={"path": "x"})
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="w")]))
    events = [evt async for evt in query(engine)]
    assert any(
        evt.type == EventType.TOOL_CALL_PENDING and evt.payload.get("requires_approval")
        for evt in events
    )
    assert engine._pending_approval_tool_call_id == "w2"
    assert engine.state is LoopState.AWAITING


@pytest.mark.asyncio
async def test_query_inference_fail_then_retry_two_ledger_rows(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    from protocore.contracts.llm import LLMTimeoutError

    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    calls = {"n": 0}
    orig = llm.stream_with_tools

    async def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMTimeoutError("timeout")
        async for item in orig(request):
            yield item

    llm.stream_with_tools = flaky  # type: ignore[method-assign]
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]))
    events = [evt async for evt in query(engine)]
    kinds = [evt.payload.get("kind") for evt in events if evt.type == EventType.USAGE_COMMITTED]
    assert "inference" in kinds
    assert "retry" in kinds
    assert [item.seq for item in engine.usage_rows] == [1, 2]
    assert session_total(engine.usage_rows) == sum(item.total_tokens for item in engine.usage_rows)


@pytest.mark.asyncio
async def test_query_resume_marks_recovery_span(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    rc = _on()
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", description="w"))
    llm.queue_tool_call_response(tool_call_id="w1", tool_name="Write", tool_input={"path": "f"})
    llm.queue_response(text="done")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="w")]))
    await _drain(query(engine))
    engine.open_intents[-1].status = "open"
    engine.open_intents = resume_open_intents(engine.open_intents)
    llm.queue_tool_call_response(tool_call_id="w1", tool_name="Write", tool_input={"path": "f"})
    llm.queue_response(text="recovered")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    more = [evt async for evt in query(engine)]
    assert any(evt.type == EventType.RECOVERY_MARKED for evt in more)
    assert any(getattr(span, "attributes", {}).get("recovery") is True for span in engine.spans)


@pytest.mark.asyncio
async def test_query_fires_published_hooks(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    rc = _on()
    engine = engine_factory(rc=rc)
    seen: list[str] = []
    registry = HookRegistry()
    for name in PUBLISHED_HOOKS:
        registry.register(name, lambda payload, hooked=name: _record_hook(seen, hooked))
    engine.typed_hook_registry = registry
    in_memory_runtime["tools"].register(MockTool(tool_name="Read", description="r"))
    llm.queue_tool_call_response(tool_call_id="r1", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="r")]))
    events = [evt async for evt in query(engine)]
    fired = {evt.payload.get("hook") for evt in events if evt.type == EventType.HOOK_FIRED}
    assert "before_run" in fired
    assert "transform_context" in fired
    assert "before_tool" in fired
    assert "after_tool" in fired


def _record_hook(seen: list[str], name: str) -> HookOutcome:
    seen.append(name)
    return HookOutcome(decision="allow")
