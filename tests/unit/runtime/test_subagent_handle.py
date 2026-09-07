"""One child run, one reservation — whichever way the call was made.

The tree's cumulative ledger counts child runs STARTED and never refunds one,
so a call charged twice permanently shrinks the tree's budget by work that was
never done. Every route to a child run has to charge exactly once: the serial
dispatch, the concurrent fan-out, a call parked at an approval and dispatched
again when the approval lands, and — the new one — a background spawn, which is
the route that could most easily have introduced a second charge, because it
reaches the ledger and then keeps running with the parent's turn already over.

This is the gate on all of that, and it belongs after the handle tests rather
than beside them: the reservation is only interesting once there is more than
one way to start a child.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.background import AgentRef, WorkSpec
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    SubagentTask,
    TextBlock,
    ToolCall,
    ToolResult,
)
from protocore.runtime.query import _dispatch_tool, _resolve_run_work_ledger
from protocore.runtime.run_work_budget import RunWorkLedger
from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryWorkPool,
)
from tests._fixtures.delegation import DelegationContract

from ._tool_fixtures import MockTool
from .test_query_parallel_safe_tools import _queue_multi_tool_stream
from .test_query_parallel_subagents import _register_delegation_tool


def _rc(**overrides: Any) -> LoopConstants:
    return LoopConstants(model_context_window=4_096, **overrides)


def _ledger(engine: Any) -> RunWorkLedger:
    return _resolve_run_work_ledger(engine)


async def _drive(engine: Any) -> None:
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    ):
        pass


class _BackgroundSpawnTool(DelegationContract, MockTool):
    """A delegation tool whose calls are all background spawns."""

    is_concurrent_safe = False

    @staticmethod
    def is_background_call(arguments: Any) -> bool:
        return True

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(tool_call_id="", content="started task-1", is_error=False)


# ── the handle ──────────────────────────────────────────────────────────────


async def test_a_launched_child_is_addressable_before_it_answers() -> None:
    pool = InMemoryWorkPool()
    dispatch = InMemoryAgentDispatch(pool)

    handle = await dispatch.dispatch(
        SubagentTask(
            subagent_id="researcher",
            parent_run_id="run-1",
            task_prompt="go",
            background=True,
        )
    )

    identity = handle.identity()
    assert identity.id
    assert identity.kind == "agent"
    assert identity.agent == AgentRef(name="researcher")
    assert pool.get(identity.id) is identity


async def test_the_pool_holds_the_child_under_the_same_id() -> None:
    pool = InMemoryWorkPool()
    dispatch = InMemoryAgentDispatch(pool)
    handle = await dispatch.dispatch(
        SubagentTask(subagent_id="reviewer", parent_run_id="run-1", task_prompt="go")
    )

    await handle.wait()

    record = pool.get(handle.identity().id)
    assert record is not None and record.terminal


async def test_a_command_and_a_child_share_one_address_space() -> None:
    """The point of the merged pool: one panel, not two."""
    pool = InMemoryWorkPool()
    dispatch = InMemoryAgentDispatch(pool)

    async def _start(task_id: str) -> Any:
        from protocore.tests_support.adapters import InMemoryWorkHandle

        async def _body() -> str:
            return "built"

        return InMemoryWorkHandle(task_id=task_id, pool=pool, run=_body())

    await pool.launch(WorkSpec(session_id="run-1", label="build"), _start)
    await dispatch.dispatch(
        SubagentTask(subagent_id="reviewer", parent_run_id="run-1", task_prompt="go")
    )

    kinds = sorted(record.kind for record in pool.list("run-1"))
    assert kinds == ["agent", "command"]


# ── exactly one reservation, whichever route ────────────────────────────────


async def test_a_serial_delegation_is_charged_once(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_delegation_tool(in_memory_runtime)
    _queue_multi_tool_stream(
        in_memory_runtime["llm"], tool_calls=[("call-1", "Agent", {"path": "a"})]
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger(engine).child_runs_started == 1


async def test_a_background_spawn_is_charged_once(
    engine_factory, in_memory_runtime
) -> None:
    """The route added last, and the one that could most easily double-charge."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    in_memory_runtime["tools"].register(
        _BackgroundSpawnTool(tool_name="Agent", description="delegate")
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-bg", "Agent", {"background": True})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger(engine).child_runs_started == 1


async def test_the_same_call_reaching_dispatch_twice_is_charged_once(
    engine_factory, in_memory_runtime
) -> None:
    """A call parked at an approval and dispatched again is the same work."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    _register_delegation_tool(in_memory_runtime)
    call = ToolCall(id="call-approved", name="Agent", arguments={"path": "a"})

    for _ in range(2):
        async for _event in _dispatch_tool(engine, call):
            pass

    assert _ledger(engine).child_runs_started == 1


async def test_a_background_spawn_does_not_charge_again_on_a_later_turn(
    engine_factory, in_memory_runtime
) -> None:
    """The child outlives the turn; the charge does not follow it."""
    engine = engine_factory(rc=_rc(max_subagent_runs_per_tree=10))
    in_memory_runtime["tools"].register(
        _BackgroundSpawnTool(tool_name="Agent", description="delegate")
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-bg", "Agent", {"background": True})],
    )
    _queue_multi_tool_stream(
        in_memory_runtime["llm"],
        tool_calls=[("call-bg2", "Agent", {"background": True})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    await _drive(engine)

    assert _ledger(engine).child_runs_started == 2
