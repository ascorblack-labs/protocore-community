"""Parallel-safe tool dispatch tests.

Pins the contract of the parallel-dispatch branch in
:func:`protocore.runtime.query._stream_one_assistant_message`:

* Mixed ``[read, read, write, read]`` batch — read tools (which set
  ``is_concurrent_safe=True`` AND ``is_destructive=False``) run under
  ``asyncio.gather``; the destructive write stays serial; history order
  matches the LLM-requested tool-call order regardless of gather
  completion order.

* All-serial batch (single non-concurrent-safe tool) preserves the
  pre-Wave-10 behaviour exactly (no parallel branch).

* ``_is_parallel_safe_tool`` predicate matches the documented contract
  (registry miss → False, missing ClassVar → False, destructive → False).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookResult, HookSpec
from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_CIRCUIT_BREAKER,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    TERMINAL_TOOL_METADATA_KEY,
    TERMINAL_TOOL_STATUS_COMPLETED,
    TERMINAL_TOOL_STATUS_METADATA_KEY,
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
)
from protocore.contracts.verification import EvidenceProducerBinding, EvidenceRecord
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _apply_deferred_tool_history,
    _drain_dispatch_tool_deferred,
    _hook_matchers_could_match_tool,
    _is_parallel_safe_tool,
    _pre_tool_use_match_predicate,
)
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)

from ._tool_fixtures import MockTool

# ---------------------------------------------------------------------------
# _is_parallel_safe_tool predicate
# ---------------------------------------------------------------------------


def test_is_parallel_safe_returns_false_for_unknown_tool(
    engine_factory,
) -> None:
    """A tool name not in the registry → conservative serial path."""
    engine = engine_factory()
    call = ToolCall(id="t-1", name="NotRegistered", arguments={})

    assert _is_parallel_safe_tool(engine, call) is False


def test_is_parallel_safe_returns_false_when_concurrent_safe_unset(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``is_concurrent_safe`` defaults to ``False`` on bare MockTool → serial.

    Mirrors the pre-Wave-10 invariant: any tool that has not explicitly
    opted in to concurrent dispatch stays serial. ``MockTool`` doesn't
    set the ClassVar so ``getattr(tool, "is_concurrent_safe", False)``
    returns ``False``.
    """
    engine = engine_factory()
    tool = MockTool(tool_name="MyTool", description="mock")
    in_memory_runtime["tools"].register(tool)
    call = ToolCall(id="t-1", name="MyTool", arguments={})

    assert _is_parallel_safe_tool(engine, call) is False


def test_is_parallel_safe_returns_false_for_destructive_tool(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``is_destructive=True`` blocks parallel dispatch even if concurrent-safe.

    Defensive ordering — destructive ops MUST serialise so causal order
    with sibling reads is preserved as the LLM emitted them.
    """
    engine = engine_factory()

    class _DestructiveTool(MockTool):
        is_concurrent_safe = True
        is_destructive = True

    tool = _DestructiveTool(tool_name="Destruct", description="destructive")
    in_memory_runtime["tools"].register(tool)
    call = ToolCall(id="t-1", name="Destruct", arguments={})

    assert _is_parallel_safe_tool(engine, call) is False


def test_is_parallel_safe_returns_true_for_concurrent_safe_non_destructive(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``is_concurrent_safe AND not is_destructive`` → parallel-eligible."""
    engine = engine_factory()

    class _ReadOnlyTool(MockTool):
        is_concurrent_safe = True
        is_destructive = False

    tool = _ReadOnlyTool(tool_name="Read", description="ro")
    in_memory_runtime["tools"].register(tool)
    call = ToolCall(id="t-1", name="Read", arguments={})

    assert _is_parallel_safe_tool(engine, call) is True


# ---------------------------------------------------------------------------
# Deferred-dispatch helpers (history mutation lives outside the gather)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_dispatch_tool_deferred_does_not_mutate_history(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``_drain_dispatch_tool_deferred`` returns events + outcome, no history append.

    The history mutation is the responsibility of the parallel-dispatch
    orchestrator (which calls ``_apply_deferred_tool_history`` in the
    LLM-requested order after the gather completes). The helper itself
    must NOT mutate ``engine.history`` — otherwise multiple concurrent
    drains would race on append order.
    """
    engine = engine_factory()

    class _ReadOnlyTool(MockTool):
        is_concurrent_safe = True
        is_destructive = False

    tool = _ReadOnlyTool(
        tool_name="Read", description="ro", response_content="payload"
    )
    in_memory_runtime["tools"].register(tool)
    call = ToolCall(id="t-1", name="Read", arguments={"path": "x.md"})

    history_before = list(engine.history)
    events, outcome = await _drain_dispatch_tool_deferred(engine, call)

    # No history mutation — caller will handle it via
    # ``_apply_deferred_tool_history`` in the canonical request order.
    assert engine.history == history_before
    assert outcome is not None
    assert outcome.success is True
    assert outcome.content == "payload"
    # The dispatcher's per-tool events (hook_fired/tool_result/etc.)
    # were captured in the buffer.
    assert any(evt.type is EventType.TOOL_RESULT for evt in events)


def test_apply_deferred_tool_history_appends_in_caller_order(
    engine_factory,
) -> None:
    """``_apply_deferred_tool_history`` mutates history with the supplied call.

    The caller controls the order — the helper just appends one
    ``ToolResultBlock`` per call. Used by the parallel orchestrator to
    serialise the deferred batch in the LLM-requested order regardless
    of gather completion order.
    """
    engine = engine_factory()
    call_a = ToolCall(id="a", name="Read", arguments={})
    call_b = ToolCall(id="b", name="Read", arguments={})
    outcome_a = DispatchOutcome(
        tool_call=call_a,
        success=True,
        content="A",
        is_error=False,
        approval_required=False,
    )
    outcome_b = DispatchOutcome(
        tool_call=call_b,
        success=True,
        content="B",
        is_error=False,
        approval_required=False,
    )

    _apply_deferred_tool_history(engine, call_a, outcome_a)
    _apply_deferred_tool_history(engine, call_b, outcome_b)

    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["a", "b"]
    assert [tr.content for tr in tool_results] == ["A", "B"]


# ---------------------------------------------------------------------------
# parallel branch must thread the effective (floor-merged) policy
# ---------------------------------------------------------------------------


class _RecordingDispatcher(ToolDispatcher):
    """Test double for :class:`ToolDispatcher` that captures ``visibility_policy``.

    Inherits from :class:`ToolDispatcher` so
    :func:`_ensure_tool_dispatcher` returns our cached instance instead
    of replacing it with a fresh one. Overrides :meth:`dispatch` to
    record the policy the helper threads in, then yields a single
    synthetic ``TOOL_RESULT`` event + a success :class:`DispatchOutcome`
    so the helper returns cleanly.

    ``dispatch`` restates the parent's full parameter list and carries no
    suppression, so the type checker really does compare the two signatures.
    Drift then surfaces where it is cheapest to fix — as a named override
    error naming both signatures at check time — rather than as a
    ``TypeError`` the first time a caller passes an argument this double
    never learned about. Absorbing the parameter list into ``**kwargs``
    would let the double keep compiling through any such change, which is
    silence rather than compatibility: a double whose purpose is to pin a
    contract has to break when the contract moves under it.
    """

    def __init__(self) -> None:
        # We must not invoke the parent ``__init__`` because the engine's
        # registry / hooks aren't available here. Use ``object.__setattr__``
        # to skip the dataclass-style initialiser.
        object.__setattr__(self, "captured_visibility_policy", None)
        object.__setattr__(self, "captured_visibility_policy_id", None)

    async def dispatch(
        self,
        *,
        tool_call: ToolCall,
        ctx: ToolContext,
        visibility_policy: ToolVisibilityPolicy,
        timeout_seconds: int,
        subagent_whitelist: Iterable[str] | None = None,
        preapproved_tool_call_id: str | None = None,
        admit_evidence: Callable[[tuple[EvidenceRecord, ...], EvidenceProducerBinding], None]
        | None = None,
    ) -> AsyncIterator[TurnEvent | DispatchOutcome]:
        object.__setattr__(self, "captured_visibility_policy", visibility_policy)
        object.__setattr__(self, "captured_visibility_policy_id", id(visibility_policy))
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=ctx.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": True,
                "content_blocks": [{"type": "text", "text": "ok"}],
            },
        )
        yield DispatchOutcome(
            tool_call=tool_call,
            success=True,
            content="ok",
            is_error=False,
            approval_required=False,
        )


@pytest.mark.asyncio
async def test_drain_dispatch_tool_deferred_uses_effective_tool_policy(
    engine_factory,
) -> None:
    """parallel branch threads the EFFECTIVE policy, not the raw config.

    The serial ``_dispatch_tool`` and the per-turn surface use
    :attr:`QueryEngine.effective_tool_policy` (which merges the RC
    ``tool_surface_forced_pins`` floor into ``forced_pinned`` plus the live
    progressive-discovery pins into ``pinned``). The parallel branch must
    use the SAME policy so advertise/dispatch parity holds: a tenant policy
    whose ``visible`` whitelist omits ``Read`` but whose floor is the
    default RC floor (which includes ``Read``) advertises ``Read`` on the
    surface yet, when batched ``>=2`` in a turn, the parallel gate would
    compute ``allowed=visible|pinned|forced_pinned`` from the raw policy
    (forced_pinned empty) → ``Read`` denied, while the SAME ``Read``
    alone (serial) succeeds.

    Pin the contract by injecting a recording dispatcher and asserting the
    captured ``visibility_policy`` carries the floor-merged
    ``forced_pinned`` (i.e. is the effective policy, NOT the raw config
    policy).
    """
    from dataclasses import replace as _dc_replace

    from protocore.contracts.tool_registry import ToolVisibilityPolicy
    from protocore.runtime.query import _drain_dispatch_tool_deferred

    # Config: visible whitelist omits Read; raw forced_pinned empty.
    # RC default floor includes Read → effective_tool_policy MUST include
    # Read in forced_pinned.
    raw_policy = ToolVisibilityPolicy(
        visible={"Write", "Edit"},  # NOT Read
        blocked=set(),
        pinned=set(),
        forced_pinned=frozenset(),  # raw config has NO floor
    )
    rc = RuntimeConstants(
        model_context_window=4_096,
        tool_surface_forced_pins=("Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep"),
    )
    engine = engine_factory(rc=rc)
    # Override the config to install the per-tenant ``visible`` policy.
    engine.config = _dc_replace(
        engine.config,
        tool_visibility_policy=raw_policy,
    )

    # Sanity-check: ``effective_tool_policy`` carries the floor-merged
    # forced_pinned, while the raw config policy has it empty.
    effective = engine.effective_tool_policy
    assert "Read" in effective.forced_pinned  # floor merged in
    assert engine.config.tool_visibility_policy.forced_pinned == frozenset()  # raw is empty

    # Replace the cached dispatcher with our recording double so we capture
    # the policy threaded into the parallel branch.
    engine._tool_dispatcher = _RecordingDispatcher()  # type: ignore[attr-defined]

    call = ToolCall(id="t-r", name="Read", arguments={"path": "x.md"})
    await _drain_dispatch_tool_deferred(engine, call)

    captured = engine._tool_dispatcher.captured_visibility_policy
    assert captured is not None
    # The parallel branch MUST thread the EFFECTIVE policy, not the raw
    # config policy. Without the captured policy would be the raw
    # one (forced_pinned empty) and ``Read`` would be denied on the gate.
    assert "Read" in captured.forced_pinned, (
        f"parallel branch must thread effective policy; got forced_pinned="
        f"{sorted(captured.forced_pinned)}"
    )


# ---------------------------------------------------------------------------
# Mixed-batch dispatch through the full query() loop
# ---------------------------------------------------------------------------


class _RecordingReadTool(MockTool):
    """Read-only tool that records dispatch-start times to assert overlap."""

    is_concurrent_safe = True
    is_destructive = False

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        # Sleep so concurrent dispatches actually overlap on the event
        # loop (asyncio.gather would otherwise complete instantly under
        # the test harness and we could not distinguish parallel from
        # serial by wall-clock).
        await asyncio.sleep(0.1)
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=False,
        )


class _ParallelTerminalReadTool(_RecordingReadTool):
    """Read-only fixture that marks its successful result terminal."""

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        await asyncio.sleep(0.1)
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=False,
            metadata={
                TERMINAL_TOOL_METADATA_KEY: True,
                TERMINAL_TOOL_STATUS_METADATA_KEY: TERMINAL_TOOL_STATUS_COMPLETED,
            },
        )


class _RecordingWriteTool(MockTool):
    """Destructive write tool — MUST stay serial in the dispatch loop."""

    is_concurrent_safe = False
    is_destructive = True

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        await asyncio.sleep(0.1)
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=False,
        )


def _queue_multi_tool_stream(
    llm: Any,
    *,
    tool_calls: list[tuple[str, str, dict[str, Any]]],
) -> None:
    """Queue one assistant turn that emits multiple tool_use blocks.

    ``tool_calls`` is a list of ``(tool_call_id, tool_name, args)``.
    Mirrors the existing single-tool queue helper but emits N
    ``tool_use_*`` blocks in one ``message_start`` → ``message_stop``
    envelope.
    """
    import json as _json

    stream: list[LLMStreamEvent] = [
        LLMStreamEvent(name="message_start", payload={}),
    ]
    for tool_call_id, tool_name, args in tool_calls:
        args_json = _json.dumps(args)
        stream.extend(
            [
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": tool_call_id,
                        "partial_input_json": args_json,
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": tool_call_id,
                        "final_input": args,
                    },
                ),
            ]
        )
    stream.append(
        LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": StopReason.tool_use.value},
        )
    )
    llm._scripted_streams.append(stream)


@pytest.mark.asyncio
async def test_mixed_batch_history_preserves_llm_request_order(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``[read1, read2, write, read3]`` → history matches request order.

 Critical invariant: even though ``read1`` and ``read2`` run
 concurrently under ``asyncio.gather`` (and could complete in either
 order), the ``ToolResultBlock`` entries in ``engine.history`` MUST
 appear in the original LLM-requested order so the next LLM call
 sees a stable causal chain.

 The write between the reads forces three sub-batches:

 * parallel(read1, read2)
 * serial(write)
 * single(read3)

 The single-element trailing read goes through the serial fast-path
 (no asyncio.gather overhead) but still lands AFTER the write in
 history.
 """
    engine = engine_factory()

    read_tool = _RecordingReadTool(
        tool_name="Read",
        description="read",
        response_content="READ_OUTPUT",
    )
    write_tool = _RecordingWriteTool(
        tool_name="Write",
        description="write",
        response_content="WRITE_OUTPUT",
    )
    in_memory_runtime["tools"].register(read_tool)
    in_memory_runtime["tools"].register(write_tool)

    # Turn 1: assistant emits four tool calls in order [r1, r2, w, r3].
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
            ("call-w", "Write", {"path": "out", "content": "data"}),
            ("call-r3", "Read", {"path": "c"}),
        ],
    )
    # Turn 2: assistant emits final text → end_turn.
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _evt in engine.run(user_msg):
        pass

    # All four tools were invoked exactly once.
    assert len(read_tool.calls) == 3
    assert len(write_tool.calls) == 1

    # History must contain four ToolResultBlock entries in the original
    # LLM-requested order — r1, r2, w, r3.
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == [
        "call-r1",
        "call-r2",
        "call-w",
        "call-r3",
    ]
    assert [tr.content for tr in tool_results] == [
        "READ_OUTPUT",
        "READ_OUTPUT",
        "WRITE_OUTPUT",
        "READ_OUTPUT",
    ]


@pytest.mark.asyncio
async def test_two_concurrent_reads_actually_run_in_parallel(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Two ``[read, read]`` calls dispatch under ``asyncio.gather``.

    Both reads sleep 100ms. Serial dispatch would take ~200ms; parallel
    dispatch should complete in ~100ms. Use a generous upper bound (180
    ms) so test machine noise does not produce flakes — the bound is
    still well below the 200ms serial floor.
    """
    engine = engine_factory()
    read_tool = _RecordingReadTool(
        tool_name="Read",
        description="read",
        response_content="OK",
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    start = asyncio.get_event_loop().time()
    async for _evt in engine.run(user_msg):
        pass
    elapsed = asyncio.get_event_loop().time() - start

    # Parallel: ~100ms; serial: ~200ms. Bound at 180ms — still safely
    # below the serial floor on commodity CI hardware.
    assert elapsed < 0.18, (
        f"two concurrent reads should run in parallel; took {elapsed:.3f}s "
        f"(serial floor ≈ 0.20s)"
    )
    assert len(read_tool.calls) == 2


@pytest.mark.asyncio
async def test_parallel_terminal_result_preserves_metadata_and_stops_loop(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A terminal result from a deferred parallel batch completes the run."""
    engine = engine_factory()
    read_tool = _RecordingReadTool(
        tool_name="Read",
        description="read",
        response_content="READ_OK",
    )
    answer_tool = _ParallelTerminalReadTool(
        tool_name="AnswerRead",
        description="terminal read-only answer",
        response_content="ANSWER_OK",
    )
    in_memory_runtime["tools"].register(read_tool)
    in_memory_runtime["tools"].register(answer_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-answer", "AnswerRead", {"path": "answer"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="should not be called")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    events = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert len(in_memory_runtime["llm"].calls) == 1
    assert engine.state is LoopState.COMPLETED

    answer_result_evt = next(
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT
        and evt.payload["tool_call_id"] == "call-answer"
    )
    assert answer_result_evt.payload["metadata"][TERMINAL_TOOL_METADATA_KEY] is True

    tool_results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [block.tool_call_id for block in tool_results] == ["call-r1", "call-answer"]
    assert tool_results[-1].metadata[TERMINAL_TOOL_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_single_call_uses_serial_fastpath(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A single tool_use turn falls through the serial dispatcher fast-path.

 The parallel branch only fans out under
 ``asyncio.gather`` when the parallel-eligible batch has ≥2 calls.
 A single call uses the unchanged serial path so behaviour is
 identical to the pre-Wave-10 contract.
 """
    engine = engine_factory()
    read_tool = _RecordingReadTool(
        tool_name="Read", description="read", response_content="ONE"
    )
    in_memory_runtime["tools"].register(read_tool)

    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-only",
        tool_name="Read",
        tool_input={"path": "x"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _evt in engine.run(user_msg):
        pass

    assert len(read_tool.calls) == 1
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "call-only"


# ---------------------------------------------------------------------------
# Hook approval semantics preserved in parallel dispatch
# ---------------------------------------------------------------------------


def test_hook_matchers_predicate_empty_matchers_could_match() -> None:
    """Empty matcher dict ⇒ matches every tool name ⇒ predicate True."""
    assert _hook_matchers_could_match_tool({}, "Read") is True
    assert _hook_matchers_could_match_tool({}, "Bash") is True


def test_hook_matchers_predicate_scalar_eq_filters_tool_name() -> None:
    """``{"tool_name": "Bash"}`` matches only "Bash"."""
    matchers = {"tool_name": "Bash"}
    assert _hook_matchers_could_match_tool(matchers, "Bash") is True
    assert _hook_matchers_could_match_tool(matchers, "Read") is False


def test_hook_matchers_predicate_in_filter_on_tool_name() -> None:
    """``{"tool_name": {"$in": ["Bash"]}}`` matches only the listed names."""
    matchers = {"tool_name": {"$in": ["Bash", "Write"]}}
    assert _hook_matchers_could_match_tool(matchers, "Bash") is True
    assert _hook_matchers_could_match_tool(matchers, "Write") is True
    assert _hook_matchers_could_match_tool(matchers, "Read") is False


def test_hook_matchers_predicate_non_tool_name_field_is_conservative() -> None:
    """Matcher on a non-``tool_name`` field returns True (conservative).

    We cannot evaluate ``tool_input.path`` matchers statically without
    invoking the dispatcher — keeping the parallel-safe set strictly
    smaller in ambiguity is the right side to err on for the approval
    contract.
    """
    matchers = {"tool_input.path": {"$regex": "/etc/.*"}}
    assert _hook_matchers_could_match_tool(matchers, "Read") is True
    assert _hook_matchers_could_match_tool(matchers, "Bash") is True


@pytest.mark.asyncio
async def test_pre_tool_use_predicate_no_hooks_returns_false_for_all_tools(
    engine_factory,
) -> None:
    """No registered hooks → predicate returns False for every tool name."""
    engine = engine_factory()
    predicate = await _pre_tool_use_match_predicate(engine)
    assert predicate is not None
    assert predicate("Read") is False
    assert predicate("Bash") is False


@pytest.mark.asyncio
async def test_pre_tool_use_predicate_bash_hook_does_not_block_read(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Bash-scoped PreToolUse hook → Read still parallel-safe; Bash not.

    Mirrors the current production hook surface: bundled Bash safety
    hooks must NOT prevent Read/Grep/PCM reads from running in
    parallel.
    """
    engine = engine_factory()
    spec = HookSpec(
        id="hook-bash-only",
        tenant_id=engine.config.tenant_id,
        event=HookEvent.pre_tool_use,
        executor="http",
        matchers={"tool_name": {"$in": ["Bash"]}},
    )
    await in_memory_runtime["hooks"].register(spec)
    predicate = await _pre_tool_use_match_predicate(engine)
    assert predicate is not None
    assert predicate("Read") is False
    assert predicate("Bash") is True


@pytest.mark.asyncio
async def test_pre_tool_use_predicate_open_hook_blocks_every_tool(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A hook with empty matchers blocks every tool from parallel dispatch."""
    engine = engine_factory()
    spec = HookSpec(
        id="hook-open",
        tenant_id=engine.config.tenant_id,
        event=HookEvent.pre_tool_use,
        executor="prompt",
        matchers={},
    )
    await in_memory_runtime["hooks"].register(spec)
    predicate = await _pre_tool_use_match_predicate(engine)
    assert predicate is not None
    assert predicate("Read") is True
    assert predicate("Bash") is True


def test_is_parallel_safe_with_hook_predicate_steers_to_serial(
    engine_factory,
    in_memory_runtime,
) -> None:
    """When the hook predicate says a tool name could match → not parallel-safe."""
    engine = engine_factory()

    class _ReadOnlyTool(MockTool):
        is_concurrent_safe = True
        is_destructive = False

    in_memory_runtime["tools"].register(
        _ReadOnlyTool(tool_name="Read", description="ro")
    )
    call = ToolCall(id="t-1", name="Read", arguments={})

    # Predicate says "yes, this could match a hook" → serial path.
    assert _is_parallel_safe_tool(engine, call, lambda _name: True) is False
    # Predicate says "no" → parallel-safe (existing predicate clauses still hold).
    assert _is_parallel_safe_tool(engine, call, lambda _name: False) is True


class _StatelessApprovalHookManager:
    """Hook manager that says ``requires_approval=True`` for EVERY PreToolUse.

    Stateless on purpose: the parallel orchestrator's pre-turn predicate
    runs ``list()`` to learn the scope of registered hooks; if the
    predicate decides the tool is hook-matchable the dispatch path is
    forced serial. This fixture also reports ONE matching HookSpec via
    ``list()`` so :func:`_pre_tool_use_match_predicate` builds a
    "always True" predicate.
    """

    def __init__(self, tenant_id: str, tool_name: str) -> None:
        self._tenant_id = tenant_id
        self._tool_name = tool_name
        self.invocations: list[tuple[HookEvent, dict[str, Any], str]] = []

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        self.invocations.append((event, payload, tenant_id))
        if event is HookEvent.pre_tool_use:
            return HookResult(
                action=HookActionKind.ALLOW,
                modifications={
                    "requires_approval": True,
                    "approval_token": "tok-amend",
                },
                reason="amend regression test",
            )
        return HookResult(action=HookActionKind.ALLOW)

    async def register(self, spec: HookSpec) -> None:  # pragma: no cover
        return

    async def unregister(
        self, hook_id: str, tenant_id: str
    ) -> None:  # pragma: no cover
        return

    async def list(
        self,
        tenant_id: str,
        *,
        event: HookEvent | None = None,
    ) -> list[HookSpec]:
        if event is not None and event is not HookEvent.pre_tool_use:
            return []
        return [
            HookSpec(
                id="hook-amend-stateless",
                tenant_id=self._tenant_id,
                event=HookEvent.pre_tool_use,
                executor="http",
                matchers={"tool_name": {"$in": [self._tool_name]}},
            )
        ]


class _TargetedPostHookManager:
    """PostToolUse hook manager that modifies one target tool result."""

    def __init__(self, target_call_id: str, replacement: str) -> None:
        self._target_call_id = target_call_id
        self._replacement = replacement
        self.invocations: list[tuple[HookEvent, dict[str, Any], str]] = []

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        self.invocations.append((event, payload, tenant_id))
        if (
            event is HookEvent.post_tool_use
            and payload.get("tool_call_id") == self._target_call_id
        ):
            return HookResult(
                action=HookActionKind.MODIFY,
                modifications={"tool_output": self._replacement},
            )
        return HookResult(action=HookActionKind.ALLOW)

    async def register(self, spec: HookSpec) -> None:  # pragma: no cover
        return

    async def unregister(
        self, hook_id: str, tenant_id: str
    ) -> None:  # pragma: no cover
        return

    async def list(
        self,
        tenant_id: str,
        *,
        event: HookEvent | None = None,
    ) -> list[HookSpec]:
        # No PreToolUse hooks: this fixture must not steer the tool serial.
        return []


@pytest.mark.asyncio
async def test_hook_gated_tool_uses_web_mode_downgrade_in_parallel_batch(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Two adjacent concurrent-safe tools with a PreToolUse hook + web mode.

 With ``approval_gate_web_enabled=False`` (production default) the
 pre-turn hook predicate steers the tools BACK onto the serial
 dispatch path so the existing ``_dispatch_tool`` web-mode downgrade
 re-runs the dispatcher with ``preapproved=True`` and the tools
 actually execute. NO ``TOOL_CALL_PENDING`` event reaches the
 consumer; the engine completes the turn normally.
 """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    assert engine.config.rc.approval_gate_web_enabled is False  # sanity
    read_tool = _RecordingReadTool(
        tool_name="Read", description="read", response_content="OK"
    )
    in_memory_runtime["tools"].register(read_tool)
    # Swap the engine's hook manager for the stateless approval one.
    engine.hooks = _StatelessApprovalHookManager(
        tenant_id=engine.config.tenant_id, tool_name="Read"
    )

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # No pending event surfaced — the serial path's downgrade
    # transparently re-dispatched both calls.
    assert EventType.TOOL_CALL_PENDING not in [e.type for e in events]
    # Both reads actually executed.
    assert len(read_tool.calls) == 2
    # History has BOTH tool results in LLM-requested order.
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-r1", "call-r2"]
    # Engine terminates COMPLETED, not AWAITING.
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_hook_gated_tool_emits_single_pending_event_when_kill_switch_on(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``approval_gate_web_enabled=True`` keeps the single-pending flow.

 With the kill switch ON the serial path's TOOL_CALL_PENDING envelope
 reaches the outer loop on the FIRST hook-matched call; the engine
 transitions to AWAITING and the rest of the batch never executes. The
 predicate steered the calls onto the serial path so only ONE pending
 event surfaces (not one per parallel batch outcome).
 """
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            approval_gate_web_enabled=True,
        ),
    )
    read_tool = _RecordingReadTool(
        tool_name="Read", description="read", response_content="WAIT"
    )
    in_memory_runtime["tools"].register(read_tool)
    engine.hooks = _StatelessApprovalHookManager(
        tenant_id=engine.config.tenant_id, tool_name="Read"
    )

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
        ],
    )

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    pending_events: list[Any] = []
    async for evt in engine.run(user_msg):
        if evt.type is EventType.TOOL_CALL_PENDING:
            pending_events.append(evt)

    # Exactly one pending event surfaces — the serial path stops on the
    # first hook-required approval.
    assert len(pending_events) == 1
    # The pending event is for the FIRST tool call (LLM-requested order).
    assert pending_events[0].payload.get("tool_call_id") == "call-r1"
    # Engine snapshot pins the first call as pending approval.
    assert engine.state is LoopState.AWAITING
    # No tool executed (gate paused before invoke).
    assert read_tool.calls == []


# ---------------------------------------------------------------------------
# Helper-bag state follows LLM order in parallel dispatch
# ---------------------------------------------------------------------------


def _make_helpers_with_rc(rc: RuntimeConstants) -> dict[str, Any]:
    """Build a helper bag with the RC wired so the dispatcher reads caps."""
    return {"rc": rc, "run_metadata": {}}


class _ScriptedReadTool(MockTool):
    """Read tool whose per-call outcome is controlled by a queue.

    ``outcomes`` is a list of (is_error, content) tuples consumed in
    call order. Each invoke pops the head; if empty, returns a benign
    success. Parallel dispatch order is intentionally unpredictable;
    this fixture lets a test pin which call gets which outcome by
    arguments dict instead of arrival order.
    """

    is_concurrent_safe = True
    is_destructive = False

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        delay = float(arguments.get("delay", 0.0) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        path = str(arguments.get("path", ""))
        is_error = bool(arguments.get("error", False))
        return ToolResult(
            tool_call_id="",
            content=f"out:{path}",
            is_error=is_error,
        )


@pytest.mark.asyncio
async def test_helper_bag_streak_state_follows_llm_order_success_then_error(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``[success, error]`` batch → final streak state matches LLM order.

 The dispatcher tracks a consecutive-error streak on the helper bag;
 serial order is ``streak.success_reset -> streak.error_inc == count=1``.
 Under parallel dispatch the two calls race on the shared helper bag, so
 the final state depended on gather completion order before the fix. With
 snapshot/restore + LLM-order replay the final state is deterministic: a
 single error after a success → ``count=1`` on the error's signature.
 """
    rc = RuntimeConstants(model_context_window=4_096)
    engine = engine_factory(rc=rc)
    engine._helpers = _make_helpers_with_rc(rc)  # type: ignore[attr-defined]

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-ok", "Read", {"path": "ok"}),
            ("call-err", "Read", {"path": "err", "error": True}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _evt in engine.run(user_msg):
        pass

    # After the LLM-order replay: success reset cleared the streak,
    # then the error incremented to count=1.
    state = engine._helpers.get("tool_dispatch.consecutive_error_state")  # type: ignore[attr-defined]
    assert state is not None
    assert state["tool_name"] == "Read"
    assert state["count"] == 1


@pytest.mark.asyncio
async def test_helper_bag_streak_state_follows_llm_order_error_then_success(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``[error, success]`` batch → success in LLM order clears the streak.

 After replay in LLM order: error increments → count=1, then success
 in LLM order resets the streak to empty. Without the fix the final
 state could carry the streak forward if the error completed last.
 """
    rc = RuntimeConstants(model_context_window=4_096)
    engine = engine_factory(rc=rc)
    engine._helpers = _make_helpers_with_rc(rc)  # type: ignore[attr-defined]

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-err", "Read", {"path": "err", "error": True}),
            ("call-ok", "Read", {"path": "ok"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _evt in engine.run(user_msg):
        pass

    # After replay: success in LLM order cleared the streak entirely.
    state = engine._helpers.get("tool_dispatch.consecutive_error_state")  # type: ignore[attr-defined]
    assert state is None, (
        f"success in LLM order must reset the streak; got {state!r}"
    )


@pytest.mark.asyncio
async def test_helper_bag_state_restored_after_parallel_dispatch_succeeds(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Parallel batch's helper-bag mutations are discarded then replayed.

 The intermediate dispatcher-side mutations during the gather MUST NOT
 leak into the per-run helper bag; only the LLM-order replay's final
 state may. This test pins a pre-gather streak value and verifies that
 after a 2-tool error batch the final state reflects ONLY the replay's
 increments (the pre-gather state is the baseline; replay added two
 increments in order).
 """
    rc = RuntimeConstants(model_context_window=4_096)
    engine = engine_factory(rc=rc)
    helpers = _make_helpers_with_rc(rc)
    # Pre-populate a streak from a prior turn (different signature so
    # the new errors don't compound).
    helpers["tool_dispatch.consecutive_error_state"] = {
        "tool_name": "PriorTool",
        "signature": "prior-sig",
        "count": 2,
    }
    engine._helpers = helpers  # type: ignore[attr-defined]

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-e1", "Read", {"path": "x", "error": True}),
            ("call-e2", "Read", {"path": "x", "error": True}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _evt in engine.run(user_msg):
        pass

    # Replay first error: switches tool from PriorTool to Read → count=1.
    # Replay second error: same (tool, signature) → count=2.
    state = helpers["tool_dispatch.consecutive_error_state"]
    assert state["tool_name"] == "Read"
    assert state["count"] == 2


@pytest.mark.asyncio
async def test_helper_bag_replay_uses_original_error_at_cap_boundary(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Parallel replay records the raw error signature, not cap-rewritten text."""

    rc = RuntimeConstants(
        model_context_window=4_096,
        tool_dispatch_consecutive_error_cap=2,
    )
    engine = engine_factory(rc=rc)
    helpers = _make_helpers_with_rc(rc)
    engine._helpers = helpers  # type: ignore[attr-defined]

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-e1", "Read", {"path": "x", "error": True, "delay": 0.02}),
            ("call-e2", "Read", {"path": "x", "error": True}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    state = helpers["tool_dispatch.consecutive_error_state"]
    assert state["tool_name"] == "Read"
    assert state["count"] == 2
    assert state["signature"] == ToolDispatcher._error_signature(
        DispatchErrorKind.execution,
        "out:x",
        "Read",
    )

    tool_result_events = [evt for evt in events if evt.type is EventType.TOOL_RESULT]
    event_texts = [
        evt.payload["content_blocks"][0]["text"] for evt in tool_result_events
    ]
    assert event_texts[0] == "out:x"
    assert event_texts[1].startswith(
        "This tool+error combination repeated 2 consecutive times."
    )
    assert event_texts[1].endswith("out:x")

    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-e1", "call-e2"]
    assert tool_results[0].content == event_texts[0]
    assert tool_results[1].content == event_texts[1]


def _arm_terminal_only_at_the_model_call(engine: Any, provider: Any) -> None:
    """Arm the terminal-only latch where the nudge arms it: inside the turn.

    ``run()`` lowers the latch at the turn boundary, so a test that sets it
    before the turn opens is describing the PREVIOUS turn and the dispatcher
    never sees it. In the loop the nudge fires between assistant messages and
    the message after it dispatches its tools under the latch; arming at the
    model call puts the dispatcher in exactly that state, without scripting a
    whole extra assistant message whose events the streak assertions would then
    have to step around.
    """
    inner = provider.stream_with_tools

    async def _armed(request: Any) -> AsyncIterator[LLMStreamEvent]:
        engine._terminal_only_active = True
        async for event in inner(request):
            yield event

    provider.stream_with_tools = _armed


@pytest.mark.asyncio
async def test_terminal_only_blocked_parallel_reads_do_not_pollute_error_streak(
    engine_factory,
    in_memory_runtime,
) -> None:
    """a terminal-only-blocked parallel read must NOT advance the
    consecutive-error streak (serial parity).

    When the terminal-only finalisation latch is active, a non-terminal tool
    dispatch is short-circuited with a synthetic blocked outcome. The SERIAL
    path (``_dispatch_tool``) appends the error tool_result and returns
    WITHOUT touching the durable consecutive-error streak. The parallel path,
    however, returned a ``success=False`` synthetic outcome and then ran the
    unconditional ``_replay_dispatcher_helper_state``, whose error branch
    INCREMENTED the streak — so two adjacent parallel-safe reads blocked by
    the latch polluted the streak in parallel mode but not serial mode,
    breaking serial/parallel equivalence (a later genuine error could trip
    the cap prematurely).

    With the fix, two blocked parallel reads leave the streak untouched, just
    like the serial path.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        # ``expected_terminal_tool`` resolves ``pcm_answer`` as the single
        # permitted terminal tool; the deadline-finalize latch makes the
        # terminal-only guard enforceable.
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    helpers = _make_helpers_with_rc(rc)
    engine._helpers = helpers  # type: ignore[attr-defined]
    # Arm the terminal-only latch as the deadline/contract-repair nudge would
    # — inside the turn — so the blocks fire on the parallel reads.
    _arm_terminal_only_at_the_model_call(engine, in_memory_runtime["llm"])
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    # Two adjacent parallel-safe reads (NOT the terminal tool) — both blocked.
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-r1", "Read", {"path": "a"}),
            ("call-r2", "Read", {"path": "b"}),
        ],
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Both reads were blocked (never executed) — terminal-only mode.
    assert read_tool.calls == []
    blocked_results = [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT
        and evt.payload.get("error", {}).get("kind") == "terminal_only"
    ]
    assert len(blocked_results) == 2

    # Both blocked tool_results landed in history in LLM-requested order.
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-r1", "call-r2"]
    assert all(tr.is_error for tr in tool_results)

    # The crux: the consecutive-error streak must NOT have been advanced by
    # the blocked synthetics — matching the serial no-streak-mutation path.
    state = helpers.get("tool_dispatch.consecutive_error_state")
    assert state is None, (
        f"terminal-only blocked parallel reads must not advance the error "
        f"streak; got {state!r}"
    )

    # The hard circuit breaker must ALSO stay untouched: a terminal-only
    # finalize-gate veto routed through the parallel
    # ``_apply_deferred_tool_history`` is dispatched with
    # ``track_circuit_breaker=False``. A regression that drops that argument at
    # the terminal-only branch (``query.py`` ``_terminal_only_blocks`` parallel
    # path) would advance the breaker here and trip it mid-gate. Assert no
    # breaker mutation at the REAL loop level (the direct-call unit test cannot
    # catch that regression).
    assert engine._circuit_breaker_streak is None  # type: ignore[attr-defined]
    assert "Read" not in engine._circuit_broken_tools  # type: ignore[attr-defined]
    assert "Read" not in engine.effective_tool_policy.blocked
    breaker_nudges = [
        msg
        for msg in engine.history
        if msg.role is MessageRole.user
        and msg.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
    ]
    assert breaker_nudges == [], (
        "a terminal-only finalize-gate veto must NOT inject the circuit-breaker "
        f"corrective turn; got {breaker_nudges!r}"
    )


@pytest.mark.asyncio
async def test_terminal_only_blocked_serial_read_does_not_pollute_error_streak(
    engine_factory,
    in_memory_runtime,
) -> None:
    """baseline — the SERIAL terminal-only block leaves the streak
    untouched (the behaviour the parallel path must match).

    A single read blocked by the terminal-only latch goes through the serial
    ``_dispatch_tool`` short-circuit, which never calls the consecutive-error
    cap. This pins the reference behaviour the parallel fix mirrors.
    """
    rc = RuntimeConstants(
        model_context_window=4_096,
        max_turns_per_run=1,
        terminal_tool_nudge_enabled=True,
    )
    engine = engine_factory(rc=rc, expected_terminal_tool="pcm_answer")
    helpers = _make_helpers_with_rc(rc)
    engine._helpers = helpers  # type: ignore[attr-defined]
    _arm_terminal_only_at_the_model_call(engine, in_memory_runtime["llm"])
    _soft_stop.enter(engine, cause_name=_soft_stop.CAUSE_DEADLINE)

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    # A single read uses the serial fast-path (no parallel batch).
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-solo",
        tool_name="Read",
        tool_input={"path": "x"},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert read_tool.calls == []
    blocked_results = [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT
        and evt.payload.get("error", {}).get("kind") == "terminal_only"
    ]
    assert len(blocked_results) == 1
    state = helpers.get("tool_dispatch.consecutive_error_state")
    assert state is None


@pytest.mark.asyncio
async def test_parallel_replay_preserves_post_tool_use_modified_error_output(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Replay updates helper state without undoing PostToolUse redaction."""

    rc = RuntimeConstants(model_context_window=4_096)
    engine = engine_factory(rc=rc)
    helpers = _make_helpers_with_rc(rc)
    engine._helpers = helpers  # type: ignore[attr-defined]
    engine.hooks = _TargetedPostHookManager("call-e1", "[redacted]")

    read_tool = _ScriptedReadTool(
        tool_name="Read", description="read", response_content="ok"
    )
    in_memory_runtime["tools"].register(read_tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-e1", "Read", {"path": "secret", "error": True}),
            ("call-e2", "Read", {"path": "ok"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[Any] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    tool_result_events = {
        evt.payload["tool_call_id"]: evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT
    }
    assert tool_result_events["call-e1"].payload["content_blocks"][0]["text"] == (
        "[redacted]"
    )

    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-e1", "call-e2"]
    assert tool_results[0].content == "[redacted]"
    assert tool_results[0].is_error is True
