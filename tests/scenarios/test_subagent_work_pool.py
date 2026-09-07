"""Delegation from the outside: a child with an address, and one that is waited on.

Everything here is driven through the public entry points — the engine's own
run, the pool the host injects, and the contracts a host implements — because
the shapes under test are exactly the ones a host has to supply. A subagent is
a unit of work in the session's pool, told apart from a background command only
by its kind, and these are the properties that follow from that.
"""
from __future__ import annotations

import asyncio
import builtins
from typing import Any

import pytest

from protocore.contracts.agent_dispatch import IAgentDispatch, SubagentHandle
from protocore.contracts.background import AgentRef, TaskRecord, WorkSpec
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import SubagentDef, SubagentResult, SubagentTask
from protocore.runtime.child_capabilities import (
    ParentCapabilities,
    narrow_child_capabilities,
)
from protocore.runtime.events import EventType
from protocore.runtime.run_work_budget import RunWorkLedger
from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryWorkHandle,
    InMemoryWorkPool,
)

from .conftest import ScenarioFactory, default_rc


def _definition(**overrides: Any) -> SubagentDef:
    values: dict[str, Any] = {
        "id": "researcher",
        "tenant_id": "tenant-scenario",
        "name": "Researcher",
        "description": "reads things",
        "system_prompt": "research",
    }
    values.update(overrides)
    return SubagentDef(**values)


def _task(**overrides: Any) -> SubagentTask:
    values: dict[str, Any] = {
        "subagent_id": "researcher",
        "parent_run_id": "run-scenario",
        "task_prompt": "survey the tree",
    }
    values.update(overrides)
    return SubagentTask(**values)


# ── foreground: the caller waits on the handle ──────────────────────────────


async def test_a_foreground_delegation_is_a_wait_on_the_handle() -> None:
    """The old blocking dispatch is now something the caller writes."""
    dispatch: IAgentDispatch = InMemoryAgentDispatch()
    handle: SubagentHandle = await dispatch.dispatch(_task())

    result = await handle.wait()

    assert isinstance(result, SubagentResult)
    assert result.subagent_id == "researcher"
    assert handle.identity().terminal


# ── background: an id now, an answer later ──────────────────────────────────


async def test_a_background_spawn_answers_with_an_id_while_the_child_runs() -> None:
    pool = InMemoryWorkPool()
    released = asyncio.Event()

    async def _start(task_id: str) -> Any:
        async def _body() -> str:
            await released.wait()
            return "surveyed"

        return InMemoryWorkHandle(task_id=task_id, pool=pool, run=_body())

    record = await pool.launch(
        WorkSpec(
            session_id="sess-scenario",
            kind="agent",
            label="survey",
            notify_on_finish=True,
            agent=AgentRef(name="researcher"),
        ),
        _start,
    )

    assert record.id
    assert record.status == "running"

    released.set()
    handle = pool.handle(record.id)
    assert handle is not None
    assert await handle.wait() == "surveyed"
    assert pool.get(record.id) is not None


async def test_a_finished_child_wakes_the_run_with_more_than_an_id(
    scenario: ScenarioFactory,
) -> None:
    pool = _FinishedAgentPool()
    run = scenario(rc=default_rc(background_tasks_enabled=True), background_pool=pool)
    run.llm.queue_response(text="noted")

    events = await run.run("go")

    assert [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]
    line = next(
        msg.text
        for msg in run.engine.history
        if "background tasks finished" in getattr(msg, "text", "")
    )
    assert "agent researcher" in line
    assert "survey" in line
    assert "succeeded" in line
    assert "2.0s" in line


class _FinishedAgentPool:
    """A pool holding one child run that finished before this run resumed."""

    def __init__(self) -> None:
        self.record = TaskRecord(
            id="task-1",
            session_id="sess-scenario",
            kind="agent",
            status="succeeded",
            label="survey",
            started_at=0.0,
            finished_at=2.0,
            exit_code=0,
            notify_on_finish=True,
            agent=AgentRef(name="researcher"),
        )
        self._drained = False
        self.stopped_sessions: list[tuple[str, float]] = []

    def mark_session_attached(self, session_id: str) -> None:
        return None

    async def ensure_session_attached(self, session_id: str) -> bool:
        return True

    def list(self, session_id: str) -> builtins.list[TaskRecord]:
        return [self.record] if session_id == self.record.session_id else []

    def get(self, task_id: str) -> TaskRecord | None:
        return self.record if task_id == self.record.id else None

    async def refresh(self, task_id: str) -> object:
        return self.get(task_id)

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        if self._drained or session_id != self.record.session_id:
            return []
        self._drained = True
        return [self.record.id]

    async def stop_session(
        self, session_id: str, grace_seconds: float = 0.0
    ) -> builtins.list[TaskRecord]:
        self.stopped_sessions.append((session_id, grace_seconds))
        return []


# ── stopping a parent stops what it owns ────────────────────────────────────


async def test_stopping_a_scope_stops_the_children_under_it() -> None:
    """Retire order, from the outside: the scope goes, its work goes with it."""
    pool = InMemoryWorkPool()

    async def _start(task_id: str) -> Any:
        async def _body() -> str:
            await asyncio.sleep(10)
            return ""

        return InMemoryWorkHandle(task_id=task_id, pool=pool, run=_body())

    parent_task = await pool.launch(WorkSpec(session_id="run-parent"), _start)
    first = await pool.launch(
        WorkSpec(session_id="run-child", kind="agent"), _start
    )
    second = await pool.launch(
        WorkSpec(session_id="run-child", kind="agent"), _start
    )

    stopped = await pool.stop_session("run-child", 0.0)

    assert {record.id for record in stopped} == {first.id, second.id}
    survivor = pool.get(parent_task.id)
    assert survivor is not None and survivor.status == "running"
    await pool.stop_session("run-parent")


# ── the tree budget survives a cold resume ──────────────────────────────────


async def test_a_tree_that_spent_its_budget_is_still_spent_after_a_resume(
    scenario: ScenarioFactory,
) -> None:
    """The bound is on the tree's whole life, so a fresh process inherits it."""
    run = scenario(rc=default_rc(max_subagent_runs_per_tree=2))
    ledger = RunWorkLedger(max_child_runs=2, max_tokens=0)
    ledger.reserve_child_runs(2, call_id="call-a")
    run.engine.run_state.run_work_ledger = ledger

    payload = run.engine.run_state.to_snapshot()

    resumed = scenario(rc=default_rc(max_subagent_runs_per_tree=2))
    resumed.engine.run_state.apply_snapshot(payload)
    restored = resumed.engine.run_state.run_work_ledger
    assert restored is not None

    assert restored.child_runs_started == 2
    assert restored.delegation_refusal_reason(1)
    assert restored.remaining_child_runs == 0


# ── a child gets less, never more ───────────────────────────────────────────


def test_a_child_never_reaches_what_its_parent_does_not_have() -> None:
    roles = ToolRoleMap.declare(
        {
            "Read": [ToolRole.reads_path],
            "Delegate": [ToolRole.delegates_work, ToolRole.never_delegated],
        }
    )

    child = narrow_child_capabilities(
        ParentCapabilities(tools=frozenset({"Read", "Delegate"}), permission_mode="b"),
        _definition(
            tool_whitelist=["Read", "Delegate", "Bash"], permission_mode="a"
        ),
        roles=roles,
        permission_mode_order=("a", "b"),
    )

    assert child.tools == frozenset({"Read"})
    assert child.permission_mode == "b"
    assert child.depth == 1


# ── a definition carries how its child is driven ────────────────────────────


def test_a_definition_names_its_own_model_turn_cap_and_ceiling() -> None:
    """Per-agent, so a cheap classifier is not driven like a researcher.

    Declarations, not enforcement: nothing in the loop builds a child run, so
    the host that does is the one that has to read them. What is pinned here is
    that they survive a round trip intact and that the defaults say "inherit"
    rather than picking a value of their own — a default model name or a turn
    cap of one would be the core deciding for every host.
    """
    definition = _definition(model="small-model", max_turns=2, timeout_seconds=30.0)
    restored = SubagentDef.model_validate(definition.model_dump())

    assert restored.model == "small-model"
    assert restored.max_turns == 2
    assert restored.timeout_seconds == pytest.approx(30.0)

    silent = _definition()
    assert (silent.model, silent.max_turns, silent.timeout_seconds) == ("", 0, None)
    assert silent.permission_mode == ""


async def test_one_call_may_override_the_ceiling_the_definition_states() -> None:
    task = _task(timeout_seconds=5.0, expected_seconds=1.0, notify_on_finish=True)
    pool = InMemoryWorkPool()
    dispatch = InMemoryAgentDispatch(pool)

    handle = await dispatch.dispatch(task)

    identity = handle.identity()
    assert identity.timeout_seconds == pytest.approx(5.0)
    assert identity.expected_seconds == pytest.approx(1.0)
    assert identity.notify_on_finish
