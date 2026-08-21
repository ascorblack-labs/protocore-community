"""Concurrent (parallel) subagent delegation dispatch tests.

Pins the delegation fan-out branch in
:func:`protocore.runtime.query._stream_one_assistant_message`:

* Adjacent delegation-eligible ``Agent`` calls emitted in one assistant turn
  run concurrently under a bounded semaphore (real wall-clock overlap proof).
* ``max_concurrent_subagents`` bounds concurrency so an over-cap group runs in
  waves.
* History/result ORDER matches the LLM-requested order regardless of child
  completion order.
* Gate off (or cap resolving to 1) ⇒ the exact serial path (timing = sum).
* One child failing does not cancel siblings (error isolation).
* ``_is_delegation_parallel_safe`` predicate contract.

Delegation tools are deliberately NOT ``is_concurrent_safe`` (each spawns a full
nested run); the fake below sets ``is_parallel_delegation=True`` — the generic
flag core keys on to identify the delegation tool without hardcoding "Agent".
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import (
    SUBAGENT_TREE_PERMIT_METADATA_KEY,
    ToolContext,
    ToolInvocationError,
)
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime.query import (
    _dispatch_subagent_under_semaphore,
    _dispatch_tool,
    _is_delegation_parallel_safe,
    _synthesize_delegation_error_result,
)
from protocore.runtime.subagent_budget import SubagentTreeBudget, SubagentTreePermit
from protocore.runtime.tool_dispatch import (
    HELPER_SUBAGENT_TREE_BUDGET_KEY,
    HELPER_SUBAGENT_TREE_PERMIT_KEY,
)

from ._tool_fixtures import MockTool
from .test_query_parallel_safe_tools import _queue_multi_tool_stream


class _ScriptedDelegationTool(MockTool):
    """Fake delegation tool: NOT concurrent-safe, but delegation-fan-out eligible.

    Per-call behaviour is driven by the ``arguments`` dict so a test can pin
    which call sleeps how long / errors / raises regardless of dispatch arrival
    order.
    """

    is_concurrent_safe = False
    is_parallel_delegation = True

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        delay = float(arguments.get("delay", 0.0) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        if arguments.get("raise"):
            raise ToolInvocationError(f"boom:{arguments.get('path')}")
        path = str(arguments.get("path", ""))
        return ToolResult(
            tool_call_id="",
            content=f"out:{path}",
            is_error=bool(arguments.get("error", False)),
        )


class _RecordingDelegationTool(_ScriptedDelegationTool):
    """Delegation fake that records each invocation's own execution window.

    ``windows`` holds one ``{"path", "started", "finished"}`` per call, timed on
    the running loop's clock. Two children ran concurrently iff their windows
    intersect — the direct observation, independent of how long the turn took
    in total.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.windows: list[dict[str, Any]] = []

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        started = asyncio.get_running_loop().time()
        try:
            return await super().invoke(context, arguments)
        finally:
            self.windows.append(
                {
                    "path": str(arguments.get("path", "")),
                    "started": started,
                    "finished": asyncio.get_running_loop().time(),
                }
            )


def _register_delegation_tool(runtime: dict[str, Any]) -> _ScriptedDelegationTool:
    tool = _ScriptedDelegationTool(tool_name="Agent", description="delegate")
    runtime["tools"].register(tool)
    return tool


def _tool_result_ids(engine: Any) -> list[str]:
    return [
        block.tool_call_id
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


# ---------------------------------------------------------------------------
# _is_delegation_parallel_safe predicate
# ---------------------------------------------------------------------------


def test_delegation_predicate_false_for_unknown_tool(engine_factory) -> None:
    engine = engine_factory()
    call = ToolCall(id="t-1", name="NotRegistered", arguments={})
    assert _is_delegation_parallel_safe(engine, call) is False


def test_delegation_predicate_false_without_flag(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Plain", description="x"))
    call = ToolCall(id="t-1", name="Plain", arguments={})
    assert _is_delegation_parallel_safe(engine, call) is False


def test_delegation_predicate_true_for_flagged_tool(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    _register_delegation_tool(in_memory_runtime)
    call = ToolCall(id="t-1", name="Agent", arguments={})
    assert _is_delegation_parallel_safe(engine, call) is True


def test_delegation_predicate_false_when_gate_off(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, parallel_subagents_enabled=False)
    )
    _register_delegation_tool(in_memory_runtime)
    call = ToolCall(id="t-1", name="Agent", arguments={})
    assert _is_delegation_parallel_safe(engine, call) is False


def test_delegation_predicate_hook_steers_serial(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    _register_delegation_tool(in_memory_runtime)
    call = ToolCall(id="t-1", name="Agent", arguments={})
    assert _is_delegation_parallel_safe(engine, call, lambda _n: True) is False
    assert _is_delegation_parallel_safe(engine, call, lambda _n: False) is True


# ---------------------------------------------------------------------------
# Real overlap proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_three_subagents_run_concurrently(
    engine_factory, in_memory_runtime
) -> None:
    """Three delegation calls each sleeping 0.1s ⇒ total ≈ 0.1s (not 0.3s)."""
    engine = engine_factory()  # default cap=4
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.1}),
            ("call-b", "Agent", {"path": "b", "delay": 0.1}),
            ("call-c", "Agent", {"path": "c", "delay": 0.1}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    start = asyncio.get_event_loop().time()
    async for _evt in engine.run(user_msg):
        pass
    elapsed = asyncio.get_event_loop().time() - start

    assert len(tool.calls) == 3
    # Parallel: ~0.1s; serial floor ≈ 0.3s. Bound well below the serial floor.
    assert elapsed < 0.22, f"three subagents should overlap; took {elapsed:.3f}s"
    assert _tool_result_ids(engine) == ["call-a", "call-b", "call-c"]


@pytest.mark.asyncio
async def test_children_execution_windows_actually_overlap(
    engine_factory, in_memory_runtime
) -> None:
    """Children's [start, end] windows overlap pairwise — not merely "fast enough".

    Total elapsed time is an INDIRECT instrument: a dispatcher that serialised
    the children but happened to shorten them would still pass an elapsed-time
    bound. This records each child's own execution window and asserts genuine
    pairwise intersection, which is the same test applied to recorded call
    windows in production telemetry, and it cannot be satisfied by anything
    other than real concurrency.
    """
    engine = engine_factory()  # default cap=4
    tool = _RecordingDelegationTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.1}),
            ("call-b", "Agent", {"path": "b", "delay": 0.1}),
            ("call-c", "Agent", {"path": "c", "delay": 0.1}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    assert len(tool.windows) == 3
    overlapping = [
        (left["path"], right["path"])
        for i, left in enumerate(tool.windows)
        for right in tool.windows[i + 1 :]
        if left["started"] < right["finished"] and right["started"] < left["finished"]
    ]
    # Three children, all three pairs must intersect: with cap=4 none of them
    # waits on the semaphore, so all three windows are open at the same instant.
    assert len(overlapping) == 3, (
        f"expected all 3 pairs to overlap, got {overlapping}; windows={tool.windows}"
    )


@pytest.mark.asyncio
async def test_children_windows_stay_disjoint_when_gate_is_off(
    engine_factory, in_memory_runtime
) -> None:
    """The overlap instrument reads ZERO on the serial path — it can fail.

    Guards the test above from being vacuously true. Same tool, same recording,
    same assertion shape, gate off: every window must be disjoint. Without this
    a recording bug that reported identical timestamps for every child would let
    the overlap proof pass while proving nothing.
    """
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, parallel_subagents_enabled=False
        )
    )
    tool = _RecordingDelegationTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.05}),
            ("call-b", "Agent", {"path": "b", "delay": 0.05}),
            ("call-c", "Agent", {"path": "c", "delay": 0.05}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    assert len(tool.windows) == 3
    overlapping = [
        (left["path"], right["path"])
        for i, left in enumerate(tool.windows)
        for right in tool.windows[i + 1 :]
        if left["started"] < right["finished"] and right["started"] < left["finished"]
    ]
    assert overlapping == [], f"serial dispatch must not overlap; got {overlapping}"


@pytest.mark.asyncio
async def test_cap_two_runs_four_subagents_in_waves(
    engine_factory, in_memory_runtime
) -> None:
    """cap=2 with 4 calls each sleeping 0.1s ⇒ two waves ≈ 0.2s (not 0.1, not 0.4)."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, max_concurrent_subagents=2)
    )
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.1}),
            ("call-b", "Agent", {"path": "b", "delay": 0.1}),
            ("call-c", "Agent", {"path": "c", "delay": 0.1}),
            ("call-d", "Agent", {"path": "d", "delay": 0.1}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    start = asyncio.get_event_loop().time()
    async for _evt in engine.run(user_msg):
        pass
    elapsed = asyncio.get_event_loop().time() - start

    assert len(tool.calls) == 4
    # Two waves of two ⇒ ~0.2s. Above the full-parallel floor (~0.1) and below
    # the fully-serial ceiling (~0.4).
    assert 0.18 <= elapsed < 0.36, f"cap=2 should run in two waves; took {elapsed:.3f}s"
    assert _tool_result_ids(engine) == ["call-a", "call-b", "call-c", "call-d"]


# ---------------------------------------------------------------------------
# History order preserved regardless of completion order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_order_preserved_when_second_finishes_first(
    engine_factory, in_memory_runtime
) -> None:
    """B (fast) finishes before A (slow) ⇒ results still appended A, B, C."""
    engine = engine_factory()
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.15}),
            ("call-b", "Agent", {"path": "b", "delay": 0.01}),
            ("call-c", "Agent", {"path": "c", "delay": 0.08}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    assert len(tool.calls) == 3
    # Completion order was B, C, A — but history MUST be LLM-requested order.
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-a", "call-b", "call-c"]
    assert [tr.content for tr in tool_results] == ["out:a", "out:b", "out:c"]


# ---------------------------------------------------------------------------
# Gate off / cap=1 ⇒ serial path (timing = sum)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_off_dispatches_serially(engine_factory, in_memory_runtime) -> None:
    """parallel_subagents_enabled=False ⇒ three 0.1s calls take ≈ 0.3s (serial)."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, parallel_subagents_enabled=False
        )
    )
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.1}),
            ("call-b", "Agent", {"path": "b", "delay": 0.1}),
            ("call-c", "Agent", {"path": "c", "delay": 0.1}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    start = asyncio.get_event_loop().time()
    async for _evt in engine.run(user_msg):
        pass
    elapsed = asyncio.get_event_loop().time() - start

    assert len(tool.calls) == 3
    # Serial: ~0.3s. Prove NO gather overlap (would be ~0.1s).
    assert elapsed >= 0.27, f"gate off must dispatch serially; took {elapsed:.3f}s"
    assert _tool_result_ids(engine) == ["call-a", "call-b", "call-c"]


@pytest.mark.asyncio
async def test_cap_one_dispatches_serially(engine_factory, in_memory_runtime) -> None:
    """max_concurrent_subagents=1 ⇒ two 0.1s calls take ≈ 0.2s (exact serial path)."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, max_concurrent_subagents=1)
    )
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.1}),
            ("call-b", "Agent", {"path": "b", "delay": 0.1}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    start = asyncio.get_event_loop().time()
    async for _evt in engine.run(user_msg):
        pass
    elapsed = asyncio.get_event_loop().time() - start

    assert len(tool.calls) == 2
    assert elapsed >= 0.18, f"cap=1 must serialise; took {elapsed:.3f}s"
    assert _tool_result_ids(engine) == ["call-a", "call-b"]


@pytest.mark.asyncio
async def test_single_delegation_call_uses_serial_fastpath(
    engine_factory, in_memory_runtime
) -> None:
    """A lone delegation call falls through to the serial single-call path."""
    engine = engine_factory()
    tool = _register_delegation_tool(in_memory_runtime)

    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-only",
        tool_name="Agent",
        tool_input={"path": "x"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    assert len(tool.calls) == 1
    assert _tool_result_ids(engine) == ["call-only"]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_child_does_not_cancel_siblings(
    engine_factory, in_memory_runtime
) -> None:
    """Middle child errors (aborted) ⇒ siblings still return; all 3 results land."""
    engine = engine_factory()
    tool = _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.02}),
            ("call-b", "Agent", {"path": "b", "raise": True}),
            ("call-c", "Agent", {"path": "c", "delay": 0.02}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    assert len(tool.calls) == 3
    tool_results = [
        block
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [tr.tool_call_id for tr in tool_results] == ["call-a", "call-b", "call-c"]
    by_id = {tr.tool_call_id: tr for tr in tool_results}
    assert by_id["call-a"].is_error is False
    assert by_id["call-c"].is_error is False
    assert by_id["call-b"].is_error is True


# ---------------------------------------------------------------------------
# Unit: defensive gather helpers
# ---------------------------------------------------------------------------


def test_synthesize_delegation_error_result_shape(engine_factory) -> None:
    engine = engine_factory()
    call = ToolCall(id="c-x", name="Agent", arguments={})
    events, outcome = _synthesize_delegation_error_result(
        engine, call, RuntimeError("kaput")
    )
    assert outcome.is_error is True
    assert outcome.success is False
    assert "RuntimeError" in outcome.content
    assert len(events) == 1
    assert events[0].payload["tool_call_id"] == "c-x"
    assert events[0].payload["success"] is False


@pytest.mark.asyncio
async def test_dispatch_under_semaphore_bounds_concurrency(
    engine_factory, in_memory_runtime
) -> None:
    """The semaphore wrapper serialises acquisition beyond its width."""
    engine = engine_factory()
    _register_delegation_tool(in_memory_runtime)
    semaphore = asyncio.Semaphore(1)
    budget = SubagentTreeBudget(0)  # unlimited: isolate the per-group width bound

    call_a = ToolCall(id="c-a", name="Agent", arguments={"path": "a", "delay": 0.1})
    call_b = ToolCall(id="c-b", name="Agent", arguments={"path": "b", "delay": 0.1})

    start = asyncio.get_event_loop().time()
    await asyncio.gather(
        _dispatch_subagent_under_semaphore(engine, call_a, semaphore, budget),
        _dispatch_subagent_under_semaphore(engine, call_b, semaphore, budget),
    )
    elapsed = asyncio.get_event_loop().time() - start
    # width-1 semaphore ⇒ the two 0.1s drains serialise to ~0.2s.
    assert elapsed >= 0.18, f"width-1 semaphore must serialise; took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Tree-budget wiring: mint into helpers + thread the permit to the child
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_acquires_tree_slot_and_threads_permit(
    engine_factory, in_memory_runtime
) -> None:
    """The dispatch site draws a tree slot and stamps the child's permit on ctx."""
    engine = engine_factory()

    seen: dict[str, Any] = {}

    class _RecordingTool(_ScriptedDelegationTool):
        async def invoke(
            self, context: ToolContext, arguments: dict[str, Any]
        ) -> Any:
            # The permit is present in the child's dispatch metadata, and its
            # slot is HELD for the duration of the child's own run.
            seen["permit"] = context.metadata.get(SUBAGENT_TREE_PERMIT_METADATA_KEY)
            seen["held_during_run"] = not budget.unlimited and _slot_is_held(budget)
            return await super().invoke(context, arguments)

    tool = _RecordingTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    budget = SubagentTreeBudget(1)
    semaphore = asyncio.Semaphore(1)
    call = ToolCall(id="c-a", name="Agent", arguments={"path": "a"})
    await _dispatch_subagent_under_semaphore(engine, call, semaphore, budget)

    assert isinstance(seen["permit"], SubagentTreePermit)
    assert seen["held_during_run"] is True
    # Released in the dispatcher's finally ⇒ the single slot is free again.
    fresh = await asyncio.wait_for(budget.acquire(), timeout=1.0)
    await fresh.release()


@pytest.mark.asyncio
async def test_budget_minted_into_helper_bag_and_reused(
    engine_factory, in_memory_runtime
) -> None:
    """The leader mints the shared budget into ``engine._helpers`` on fan-out."""
    engine = engine_factory()
    setattr(engine, "_helpers", {})  # noqa: B010 — simulate the executor's bag
    _register_delegation_tool(in_memory_runtime)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a", "delay": 0.02}),
            ("call-b", "Agent", {"path": "b", "delay": 0.02}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    budget = engine._helpers.get(HELPER_SUBAGENT_TREE_BUDGET_KEY)
    assert isinstance(budget, SubagentTreeBudget)


def _slot_is_held(budget: SubagentTreeBudget) -> bool:
    """True when the cap-1 budget currently has its single slot checked out."""
    return budget._semaphore is not None and budget._semaphore.locked()


@pytest.mark.asyncio
async def test_serial_single_delegation_releases_tree_slot(
    engine_factory, in_memory_runtime
) -> None:
    """A SINGLE delegation call releases the run's tree slot while the child runs.

    Regression for the deadlock hole: a run holding a tree slot that makes one
    (serial-path) delegation call must release its slot around the child join —
    otherwise a parallel fan-out beneath the serial hop wedges the tree at the
    cap. cap=1, this engine holds the only slot; during the child's run the slot
    must be FREE, and reacquired once the child completes.
    """
    engine = engine_factory()
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()  # this run holds the single slot (1/1)
    setattr(  # noqa: B010 — simulate a child engine that holds a tree permit
        engine,
        "_helpers",
        {
            HELPER_SUBAGENT_TREE_BUDGET_KEY: budget,
            HELPER_SUBAGENT_TREE_PERMIT_KEY: permit,
        },
    )

    observed: dict[str, Any] = {}

    class _SlotObservingTool(_ScriptedDelegationTool):
        async def invoke(
            self, context: ToolContext, arguments: dict[str, Any]
        ) -> Any:
            observed["slot_free_during_child"] = not budget._semaphore.locked()
            return await super().invoke(context, arguments)

    tool = _SlotObservingTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-solo", "Agent", {"path": "solo"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    # Exactly one delegation call ⇒ the serial path (not the ≥2 gather).
    assert len(tool.calls) == 1
    assert observed["slot_free_during_child"] is True
    # Reacquired at the choke point ⇒ the run holds its slot again.
    assert budget._semaphore.locked() is True


@pytest.mark.asyncio
async def test_dispatch_tool_releases_tree_slot_for_delegation_child(
    engine_factory, in_memory_runtime
) -> None:
    """The choke point in _dispatch_tool releases the slot around ANY delegation.

    Every serial-style delegation join funnels through _dispatch_tool — the
    single-call serial path AND both truncation-recovery sibling loops
    (query.py's output-cap and finish_reason=='stop' recovery). Driving
    _dispatch_tool DIRECTLY (decoupled from whichever loop selected the call)
    proves the release/reacquire covers all of them: the truncation-recovery
    loops dispatch a non-truncated sibling Task call through this exact function,
    so they inherit the release for free.
    """
    engine = engine_factory()
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()  # this run holds the only slot
    setattr(  # noqa: B010
        engine,
        "_helpers",
        {
            HELPER_SUBAGENT_TREE_BUDGET_KEY: budget,
            HELPER_SUBAGENT_TREE_PERMIT_KEY: permit,
        },
    )

    observed: dict[str, Any] = {}

    class _SlotObservingTool(_ScriptedDelegationTool):
        async def invoke(
            self, context: ToolContext, arguments: dict[str, Any]
        ) -> Any:
            observed["slot_free_during_child"] = not budget._semaphore.locked()
            return await super().invoke(context, arguments)

    tool = _SlotObservingTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    call = ToolCall(id="c-direct", name="Agent", arguments={"path": "direct"})
    async for _evt in _dispatch_tool(engine, call):
        pass

    assert observed["slot_free_during_child"] is True
    assert budget._semaphore.locked() is True  # reacquired after the child join


@pytest.mark.asyncio
async def test_dispatch_tool_keeps_tree_slot_for_non_delegation_tool(
    engine_factory, in_memory_runtime
) -> None:
    """A non-delegation tool through _dispatch_tool does LOCAL work ⇒ keeps the slot."""
    engine = engine_factory()
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()
    setattr(  # noqa: B010
        engine,
        "_helpers",
        {
            HELPER_SUBAGENT_TREE_BUDGET_KEY: budget,
            HELPER_SUBAGENT_TREE_PERMIT_KEY: permit,
        },
    )

    observed: dict[str, Any] = {}

    class _PlainSlotObservingTool(MockTool):
        async def invoke(
            self, context: ToolContext, arguments: dict[str, Any]
        ) -> Any:
            observed["slot_held_during_tool"] = budget._semaphore.locked()
            return await super().invoke(context, arguments)

    tool = _PlainSlotObservingTool(tool_name="Read", description="read")
    in_memory_runtime["tools"].register(tool)

    call = ToolCall(id="c-read", name="Read", arguments={"path": "x"})
    async for _evt in _dispatch_tool(engine, call):
        pass

    # Non-delegation ⇒ no release; the run keeps its slot the whole time.
    assert observed["slot_held_during_tool"] is True
    assert budget._semaphore.locked() is True


def test_delegation_slot_release_is_centralized_in_dispatch_tool() -> None:
    """Single-choke-point guard: the release/reacquire lives ONLY in _dispatch_tool.

    Regression guard for the deadlock class: every serial-style delegation await
    goes through _dispatch_tool, so the tree-slot release/reacquire must live
    there (not re-scattered across call sites, which is how the two
    truncation-recovery loops went uncovered). Asserts the choke point exists and
    the removed per-call-site serial wrapper has not crept back.
    """
    import importlib
    import inspect
    import pathlib

    # ``protocore.runtime`` rebinds ``query`` as a package attribute, so both
    # ``from … import query`` and ``import …query`` resolve a function, not the
    # module. Pull the real module object out of ``sys.modules`` via importlib.
    query_mod = importlib.import_module("protocore.runtime.query")

    dispatch_src = inspect.getsource(query_mod._dispatch_tool)
    assert "dispatch_tree_permit" in dispatch_src
    assert "release_while_waiting" in dispatch_src
    assert "reacquire" in dispatch_src

    # The old per-call-site serial wrapper was removed in favour of the choke
    # point — its local name must not reappear anywhere in the module.
    module_src = pathlib.Path(inspect.getsourcefile(query_mod)).read_text()
    assert "serial_tree_permit" not in module_src
