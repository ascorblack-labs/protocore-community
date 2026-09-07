"""A delegation that does not wait, and a wake that says what happened.

Delegation used to have exactly one shape: the caller blocked on the child's
whole nested run, holding its turn and its slot in the tree budget for the
duration. That is the right shape when the answer is needed now and the wrong
one when it is not, and there was no way to say which.

A background delegation says the other one. The call returns as soon as its
children are launched, the parent's turn ends without them, and the outcome
comes back through the same wake the pool delivers a finished command with —
which is why the wake line had to start naming what actually happened.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.background import AgentRef
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock, ToolCall, ToolResult
from protocore.runtime.events import EventType
from protocore.runtime.query import _dispatch_tool, _drain_dispatch_tool_deferred
from protocore.runtime.query import _query as query
from protocore.runtime.subagent_budget import SubagentTreeBudget
from protocore.tests_support.adapters import InMemoryLLMProvider
from tests._fixtures.delegation import DelegationContract

from ._tool_fixtures import MockTool
from .fake_background_pool import FakeBackgroundPool


def _rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)


class _SpawnTool(DelegationContract, MockTool):
    """A delegation tool that launches into the pool and may or may not wait.

    Stands in for the host's dispatch tool: on a background call it hands back
    the pool's id and returns, and on a foreground one it waits for the child.
    """

    is_concurrent_safe = False

    def __init__(self, pool: FakeBackgroundPool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pool = pool
        self.launched: list[str] = []
        #: Tree-budget occupancy observed from inside the call, which is the
        #: only place the release-around-the-join is visible.
        self.slots_in_use: list[int] = []

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> ToolResult:
        budget = context.run_state.subagent_tree_budget
        if budget is not None:
            self.slots_in_use.append(budget.in_use)
        record = self.pool.start(
            context.session_id,
            owner_scope=context.work_scope,
            notify_on_finish=bool(arguments.get("notify", True)),
            kind="agent",
            label=str(arguments.get("label", "")),
            agent=AgentRef(name=str(arguments.get("agent", "researcher"))),
        )
        self.launched.append(record.id)
        if not self.is_background_call(arguments):
            self.pool.finish(record.id)
            return ToolResult(tool_call_id="", content="child answered", is_error=False)
        return ToolResult(
            tool_call_id="",
            content=f"started {record.id}, hard timeout 900s",
            is_error=False,
        )


def _register(runtime: dict[str, Any], pool: FakeBackgroundPool) -> _SpawnTool:
    tool = _SpawnTool(pool, tool_name="Agent", description="delegate")
    runtime["tools"].register(tool)
    return tool


# ── the call that does not wait ─────────────────────────────────────────────


async def test_a_background_call_returns_an_id_and_leaves_the_child_running(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    tool = _register(in_memory_runtime, pool)

    _events, outcome = await _drain(engine, "c-1", {"background": True})

    assert outcome is not None
    assert not outcome.is_error
    assert tool.launched[0] in outcome.content
    running = pool.get(tool.launched[0])
    assert running is not None and running.status == "running"


async def _drain(engine: Any, call_id: str, arguments: dict[str, Any]) -> Any:
    return await _drain_dispatch_tool_deferred(
        engine, ToolCall(id=call_id, name="Agent", arguments=arguments)
    )


async def _dispatch_serially(
    engine: Any, call_id: str, arguments: dict[str, Any]
) -> None:
    """The serial path, which is where a run releases its slot around a join."""
    async for _ in _dispatch_tool(
        engine, ToolCall(id=call_id, name="Agent", arguments=arguments)
    ):
        pass


async def test_a_background_spawn_holds_no_slot_on_its_children_behalf(
    engine_factory, in_memory_runtime
) -> None:
    """The release exists for a join, and a background spawn has no join.

    A foreground delegation blocks its caller on the child's whole nested run,
    so the caller hands its slot back for the duration: a permit holder blocked
    on a descendant is what wedges the tree at its cap. A background spawn
    returns as soon as the children are launched and the parent goes straight
    back to local work — and a run doing local work is precisely who the budget
    means a permit to be held by. What the parent must not do is hold a slot
    ON BEHALF of a child it is not waiting for, and it does not: the slot it
    keeps is its own, given back when its own run ends.
    """
    engine = engine_factory(rc=_rc())
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    tool = _register(in_memory_runtime, pool)
    budget = SubagentTreeBudget(2)
    permit = await budget.acquire()
    engine.run_state.subagent_tree_budget = budget
    engine.run_state.subagent_tree_permit = permit

    await _dispatch_serially(engine, "c-bg", {"background": True})
    await _dispatch_serially(engine, "c-fg", {})

    assert tool.slots_in_use == [1, 0]
    assert budget.in_use == 1


# ── the wake ────────────────────────────────────────────────────────────────


async def test_the_wake_names_the_status_the_outcome_and_the_duration(
    engine_factory, in_memory_runtime
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="noted")
    engine = engine_factory(rc=_rc())
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    record = pool.start(
        engine.config.session_id,
        notify_on_finish=True,
        kind="agent",
        label="survey the tree",
        agent=AgentRef(name="researcher"),
    )
    pool.finish(record.id, "failed", exit_code=None, error="ran out of turns",
                duration_seconds=12.0)
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )

    events = [evt async for evt in query(engine)]

    assert [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]
    line = next(
        msg.text for msg in engine.history if "background tasks finished" in msg.text
    )
    assert record.id in line
    assert "failed" in line
    assert "survey the tree" in line
    assert "agent researcher" in line
    assert "ran out of turns" in line
    assert "12.0s" in line


async def test_only_the_root_run_is_woken(engine_factory, in_memory_runtime) -> None:
    """A child woken by its own work would spend its turns reporting to nobody."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="noted")
    engine = engine_factory(
        rc=_rc(),
        parent_run_id="run-parent",
        subagent_id="researcher",
        root_run_id="run-parent",
    )
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    record = pool.start(engine.config.session_id, notify_on_finish=True)
    pool.finish(record.id)
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )

    events = [evt async for evt in query(engine)]

    assert not [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]
    assert not any("background tasks finished" in msg.text for msg in engine.history)
    assert pool.drained == []
