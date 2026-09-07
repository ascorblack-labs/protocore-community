"""The tree's cumulative child-run counter is incremented by the loop itself.

``RunWorkLedger`` has always been able to refuse a delegation once the tree has
started ``max_subagent_runs_per_tree`` child runs, and the pre-dispatch gate has
always asked it. What was missing was the counting half: nothing in the loop
ever reserved a slot, so the gate compared 0 against the cap forever and the
bound never fired. Any layer above the loop that did its own reserving bounded
only the delegation path it happened to own.

The charge now sits at the two places a child run actually starts — the
concurrent fan-out and the serial dispatch — one per child, and nowhere else.
One per CHILD, not one per call: a delegation tool whose arguments carry a batch
starts one child run per element, and the tool is asked how many rather than the
loop guessing one.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.middleware import (
    LifecycleDecision,
    LifecycleVerdict,
    RegistrationKind,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    HookEvent,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from protocore.hooks import HookManager
from protocore.runtime.query import _tool_is_delegation
from protocore.runtime.run_work_budget import RunWorkLedger
from protocore.runtime.subagent_budget import SubagentTreeBudget

from .test_query_parallel_safe_tools import _queue_multi_tool_stream
from .test_query_parallel_subagents import (
    _register_delegation_tool,
    _ScriptedDelegationTool,
)


def _rc(**overrides: Any) -> LoopConstants:
    return LoopConstants(model_context_window=4_096, **overrides)


def _ledger_of(engine: Any) -> RunWorkLedger:
    ledger = engine.run_state.run_work_ledger
    assert isinstance(ledger, RunWorkLedger)
    return ledger


async def _drive(engine: Any) -> None:
    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass


async def test_a_fan_out_of_three_charges_exactly_three(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a"}),
            ("call-b", "Agent", {"path": "b"}),
            ("call-c", "Agent", {"path": "c"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 3


async def test_a_serial_delegation_charges_exactly_one(
    engine_factory, in_memory_runtime
) -> None:
    """One call takes the serial route; it must be charged once, not twice."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    tool = _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-solo", "Agent", {"path": "solo"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert len(tool.calls) == 1
    assert _ledger_of(engine).child_runs_started == 1


async def test_two_serial_turns_accumulate(engine_factory, in_memory_runtime) -> None:
    """Wave after wave is exactly the shape an instantaneous cap cannot see."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-1", "Agent", {"path": "1"})],
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-2", "Agent", {"path": "2"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 2


async def test_delegation_past_the_cap_is_refused(
    engine_factory, in_memory_runtime
) -> None:
    """cap=2 with three calls in one turn: two run, the third is refused."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=2))
    tool = _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a"}),
            ("call-b", "Agent", {"path": "b"}),
            ("call-c", "Agent", {"path": "c"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert len(tool.calls) == 2
    assert _ledger_of(engine).child_runs_started == 2
    refusals = [
        block
        for message in engine.history
        for block in message.content_blocks
        if getattr(block, "is_error", False)
        and "subagent_run_budget_exhausted" in (getattr(block, "content", "") or "")
    ]
    assert len(refusals) == 1


async def test_a_non_delegation_tool_is_never_charged(
    engine_factory, in_memory_runtime
) -> None:
    from ._tool_fixtures import MockTool

    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", description="read a file")
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-r", "Read", {"path": "a"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 0


async def test_the_delegation_predicate_has_no_side_effects(
    engine_factory, in_memory_runtime
) -> None:
    """``_tool_is_delegation`` is consulted several times per dispatch.

    It is a pure classification question, asked on every tool call rather than
    every child run, so a charge inside it would debit the tree several times
    over for one child.
    """
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    ledger = RunWorkLedger(max_child_runs=10, max_tokens=0)
    engine.run_state.run_work_ledger = ledger
    _register_delegation_tool(in_memory_runtime)
    tool_call = ToolCall(id="call-x", name="Agent", arguments={"path": "x"})

    for _ in range(25):
        assert _tool_is_delegation(engine, tool_call) is True

    assert ledger.child_runs_started == 0


class _BatchDelegationTool(_ScriptedDelegationTool):
    """A delegation tool whose one call starts one child run PER TASK.

    The shape core cannot infer and must be told: the whole point of the
    ``child_run_count`` accessor is that how the arguments map to child runs
    lives in the tool's own schema.
    """

    @staticmethod
    def child_run_count(arguments: Any) -> int:
        tasks = arguments.get("tasks")
        return len(tasks) if isinstance(tasks, list) else 1


def _register_batch_tool(runtime: dict[str, Any]) -> _BatchDelegationTool:
    tool = _BatchDelegationTool(tool_name="Agent", description="delegate")
    runtime["tools"].register(tool)
    return tool


async def test_a_batch_of_three_is_charged_three(
    engine_factory, in_memory_runtime
) -> None:
    """One call, three child runs, three slots — not one."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_batch_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-batch", "Agent", {"tasks": ["a", "b", "c"]})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 3


async def test_the_cap_bounds_child_runs_not_delegation_calls(
    engine_factory, in_memory_runtime
) -> None:
    """A batch the tree cannot pay for in full never dispatches at all.

    With four slots left a call asking for ten is refused whole. Admitting a
    prefix would need whoever knows the batch's shape to slice it, and a grant
    handed out and thrown away is how the cap becomes a suggestion.
    """
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=4))
    tool = _register_batch_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-big", "Agent", {"tasks": list("abcdefghij")})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert tool.calls == []
    assert _ledger_of(engine).child_runs_started == 0
    refusals = [
        block
        for message in engine.history
        for block in message.content_blocks
        if getattr(block, "is_error", False)
        and "subagent_run_budget_short" in (getattr(block, "content", "") or "")
    ]
    assert len(refusals) == 1
    assert "at most 4" in refusals[0].content


async def test_a_registry_that_declares_no_count_still_reads_as_one(
    engine_factory, in_memory_runtime
) -> None:
    """The accessor is optional; a tool without it costs one slot per call."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-solo", "Agent", {"tasks": ["a", "b", "c"]})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 1


async def test_a_cancelled_child_leaves_no_tree_slot_held(
    engine_factory, in_memory_runtime
) -> None:
    """The parent's release is idempotent, so a cancelled subtree frees its slot.

    The parent releases the tree slot before the join and a cancelled child
    never reacquires it; the accounting therefore rests entirely on ``release``
    being a no-op on an already-released permit. That is an invariant worth a
    test rather than a comment.
    """
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()
    assert budget.in_use == 1

    await permit.release_while_waiting()
    assert budget.in_use == 0
    # The child was cancelled mid-gather: its ``finally`` reacquire never ran.
    await permit.release()

    assert budget.in_use == 0
    fresh = await asyncio.wait_for(budget.acquire(), timeout=1.0)
    assert budget.in_use == 1
    await fresh.release()


async def test_a_child_that_holds_a_permit_still_charges_once(
    engine_factory, in_memory_runtime
) -> None:
    """A run dispatched under the tree budget charges its own children normally."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()
    engine.run_state.subagent_tree_budget = budget
    engine.run_state.subagent_tree_permit = permit
    in_memory_runtime["tools"].register(
        _ScriptedDelegationTool(tool_name="Agent", description="delegate")
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-solo", "Agent", {"path": "solo"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger_of(engine).child_runs_started == 1


@pytest.mark.parametrize("cap", [0])
async def test_the_unlimited_sentinel_counts_without_refusing(
    engine_factory, in_memory_runtime, cap: int
) -> None:
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=cap))
    tool = _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[
            ("call-a", "Agent", {"path": "a"}),
            ("call-b", "Agent", {"path": "b"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert len(tool.calls) == 2
    assert _ledger_of(engine).child_runs_started == 2
    assert _ledger_of(engine).delegation_refusal_reason() == ""


async def test_an_approval_gated_delegation_is_charged_once(
    engine_factory, in_memory_runtime
) -> None:
    """Parked for approval, then approved: one child run, one slot.

    The call reaches the dispatch twice — once to be parked, once when the
    approval lands — and starts exactly one child run. Charging on both passes
    would debit the tree twice for work it did once, which is the same
    over-count from the other direction as charging a batch as one.
    """
    from protocore.contracts.hooks import HookActionKind, HookResult
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory(
        rc=_rc(max_subagent_runs_per_tree=10, approval_gate_web_enabled=True)
    )
    tool = _register_delegation_tool(in_memory_runtime)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-gated", "Agent", {"path": "gated"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)
    assert tool.calls == []

    async for _evt in resume_approved_tool(
        engine, ToolCall(id="call-gated", name="Agent", arguments={"path": "gated"})
    ):
        pass

    assert len(tool.calls) == 1
    assert _ledger_of(engine).child_runs_started == 1


async def test_a_typed_hook_veto_leaves_the_ledger_untouched(
    engine_factory, in_memory_runtime
) -> None:
    """Parked at the ``pre_tool_use`` coordinate: nothing started, nothing charged.

    The charge sits after every seam that can still stop the call, so a
    delegation waiting on an operator has not yet spent a slot; if the operator
    never approves it, the tree keeps the budget.
    """

    engine = engine_factory(
        rc=_rc(
            max_subagent_runs_per_tree=10,
            typed_hooks_enabled=True,
            intent_settlement_enabled=True,
        )
    )
    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleDecision(
            verdict=LifecycleVerdict.require_approval, approval_token="tok-h"
        ),
        owner="test",
    )
    engine.lifecycle_hooks = registry
    tool = _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-veto", "Agent", {"path": "v"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert tool.calls == []
    assert _ledger_of(engine).child_runs_started == 0


async def test_a_sibling_that_spends_the_budget_mid_dispatch_is_refused_at_the_charge(
    engine_factory, in_memory_runtime
) -> None:
    """The gate and the serial charge are not one act, so the budget can move.

    The serial path gates the call, then releases its tree permit and runs the
    typed ``before_tool`` seam — both of which suspend — and only then charges.
    A sibling spending the budget in that gap leaves the charge with less than
    the call was gated for, and the charge is the last place that can still
    stop the dispatch. Here the hook plays the sibling.

    Nothing is charged for the refused call, so the slots the sibling took are
    the only ones spent, and the leader is told which budget refused it.
    """

    engine = engine_factory(
        rc=_rc(
            max_subagent_runs_per_tree=2,
            typed_hooks_enabled=True,
            intent_settlement_enabled=True,
        )
    )
    ledger_seen: list[RunWorkLedger] = []

    def _sibling_spends_it(_ctx: Any) -> None:
        # Runs after the gate said yes and before the charge is taken.
        ledger = _ledger_of(engine)
        if not ledger_seen:
            ledger_seen.append(ledger)
            ledger.reserve_child_runs(2, call_id="call-sibling")

    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.observe,
        _sibling_spends_it,
        owner="test",
    )
    engine.lifecycle_hooks = registry
    tool = _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-loser", "Agent", {"path": "loser"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert tool.calls == []
    # Only the sibling's two slots are spent: the refused call pays nothing.
    assert _ledger_of(engine).child_runs_started == 2
    # The refusal reached the leader as an ordinary failed tool result, and the
    # history stays pairing-valid — one result for the one call that was made.
    results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [block.tool_call_id for block in results] == ["call-loser"]
    assert results[0].is_error is True
    assert "Delegation budget exhausted" in results[0].content
