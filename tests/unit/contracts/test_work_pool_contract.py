"""One pool holds both kinds of work, and names it before it starts.

A background shell command and a delegated child run used to be two unrelated
mechanisms: the command was a record with an id, a status and a stop; the child
run was a function call that blocked its caller and had no address at all. These
tests hold the contract to the merged shape — a record of either ``kind``, an id
that exists before anything is spawned, a handle that waits and stops, and a
session-wide stop for the work a run owes its own session before it ends.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.agent_dispatch import IAgentDispatch, SubagentHandle
from protocore.contracts.background import (
    AgentRef,
    BackgroundTaskView,
    IWorkPool,
    TaskRecord,
    WorkHandle,
    WorkSpec,
    describe_finished_task,
)
from protocore.contracts.types import SubagentDef, SubagentResult, SubagentTask
from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryWorkHandle,
    InMemoryWorkPool,
)


def _spec(**overrides: Any) -> WorkSpec:
    values: dict[str, Any] = {"session_id": "s-1"}
    values.update(overrides)
    return WorkSpec(**values)


# ── the record ──────────────────────────────────────────────────────────────


def test_a_record_is_of_one_of_two_kinds() -> None:
    command = TaskRecord.minted("t-1", _spec(kind="command", label="build"))
    agent = TaskRecord.minted(
        "t-2", _spec(kind="agent", agent=AgentRef(name="researcher"))
    )

    assert command.kind == "command"
    assert agent.kind == "agent"
    assert agent.agent is not None and agent.agent.name == "researcher"


def test_a_record_reads_as_the_view_the_loop_asks_for() -> None:
    record = TaskRecord.minted("t-1", _spec())

    assert isinstance(record, BackgroundTaskView)


def test_a_fresh_record_has_no_duration_and_a_settled_one_does() -> None:
    record = TaskRecord.minted("t-1", _spec())
    assert record.duration_seconds is None
    assert not record.terminal

    record.started_at, record.finished_at = 10.0, 12.5
    record.status = "succeeded"

    assert record.duration_seconds == pytest.approx(2.5)
    assert record.terminal


def test_a_wake_line_names_more_than_an_id_and_a_status() -> None:
    """The line an agent is woken by has to be enough to decide on."""
    record = TaskRecord.minted("t-1", _spec(kind="command", label="build docs"))
    record.status, record.exit_code = "failed", 2
    record.error = "no such target"
    record.started_at, record.finished_at = 0.0, 4.0

    line = describe_finished_task(record)

    assert "t-1" in line
    assert "failed" in line
    assert "build docs" in line
    assert "no such target" in line
    assert "4.0s" in line


def test_a_finished_child_run_is_described_by_its_agent() -> None:
    record = TaskRecord.minted(
        "t-9", _spec(kind="agent", agent=AgentRef(name="reviewer"))
    )
    record.status, record.exit_code = "succeeded", 0
    record.started_at, record.finished_at = 0.0, 1.0

    assert "agent reviewer" in describe_finished_task(record)


# ── launch ──────────────────────────────────────────────────────────────────


async def test_the_id_exists_before_the_work_does() -> None:
    """``start`` is handed the id, so nothing is ever running unnamed."""
    pool = InMemoryWorkPool()
    seen: list[str] = []
    released = asyncio.Event()

    async def _start(task_id: str) -> WorkHandle[Any]:
        seen.append(task_id)

        async def _body() -> str:
            await released.wait()
            return "done"

        return InMemoryWorkHandle(task_id=task_id, pool=pool, run=_body())

    record = await pool.launch(_spec(label="slow"), _start)

    assert seen == [record.id]
    assert pool.get(record.id) is not None
    assert record.status == "running"
    released.set()
    handle = pool.handle(record.id)
    assert handle is not None
    assert await handle.wait() == "done"


async def test_a_launch_that_never_starts_still_leaves_a_named_record() -> None:
    """The record is minted first, so a failing start is still visible work."""
    pool = InMemoryWorkPool()

    async def _start(task_id: str) -> WorkHandle[Any]:
        raise RuntimeError("spawn refused")

    with pytest.raises(RuntimeError):
        await pool.launch(_spec(), _start)

    assert [record.id for record in pool.list("s-1")] == ["task-1"]


async def test_the_pool_is_a_work_pool_by_shape() -> None:
    assert isinstance(InMemoryWorkPool(), IWorkPool)


# ── the handle ──────────────────────────────────────────────────────────────


async def _launch(pool: InMemoryWorkPool, body: Any, **spec: Any) -> TaskRecord:
    async def _start(task_id: str) -> WorkHandle[Any]:
        return InMemoryWorkHandle(task_id=task_id, pool=pool, run=body)

    return await pool.launch(_spec(**spec), _start)


async def test_a_handle_names_its_record_without_waiting() -> None:
    pool = InMemoryWorkPool()

    async def _body() -> str:
        await asyncio.sleep(10)
        return ""

    record = await _launch(pool, _body(), label="long")
    handle = pool.handle(record.id)
    assert handle is not None

    assert handle.identity().id == record.id
    assert handle.identity().label == "long"
    await handle.stop()


async def test_stopping_settles_the_record() -> None:
    pool = InMemoryWorkPool()

    async def _body() -> str:
        await asyncio.sleep(10)
        return ""

    record = await _launch(pool, _body())
    pool.advance(3.0)
    handle = pool.handle(record.id)
    assert handle is not None

    stopped = await handle.stop()

    assert stopped.status == "stopped"
    assert stopped.duration_seconds == pytest.approx(3.0)


async def test_a_failure_is_carried_on_the_record() -> None:
    pool = InMemoryWorkPool()

    async def _body() -> str:
        raise RuntimeError("child blew up")

    record = await _launch(pool, _body())
    handle = pool.handle(record.id)
    assert handle is not None
    with pytest.raises(RuntimeError):
        await handle.wait()

    assert pool.get(record.id) is not None
    settled = pool.get(record.id)
    assert settled is not None
    assert settled.status == "failed"
    assert settled.error == "child blew up"


# ── session-wide stop and terminal notification ─────────────────────────────


async def test_stopping_a_session_stops_everything_still_running() -> None:
    pool = InMemoryWorkPool()

    async def _body() -> str:
        await asyncio.sleep(10)
        return ""

    first = await _launch(pool, _body())
    second = await _launch(pool, _body(), kind="agent")
    other = await _launch(pool, _body(), session_id="s-2")

    stopped = await pool.stop_session("s-1", 0.0)

    assert {record.id for record in stopped} == {first.id, second.id}
    third = pool.get(other.id)
    assert third is not None and third.status == "running"
    await pool.stop_session("s-2")


async def test_a_scope_stops_the_work_it_owns_inside_a_session_it_shares() -> None:
    """Where the two questions come apart, and why the record has to say.

    A delegated run shares its parent's session — same workspace, same wake
    delivery — and owns only the work it started itself. Keyed on the session
    alone, its stop would match either everything (its parent's commands with
    it) or nothing at all; keyed on the owner, it matches exactly its own.
    """
    pool = InMemoryWorkPool()

    async def _body() -> str:
        await asyncio.sleep(10)
        return ""

    parents = await _launch(pool, _body())
    childs = await _launch(pool, _body(), kind="agent", owner_scope="run-child")

    stopped = await pool.stop_session("run-child", 0.0)

    assert [record.id for record in stopped] == [childs.id]
    survivor = pool.get(parents.id)
    assert survivor is not None and survivor.status == "running"

    # And the session still reaches what its children left behind: a child
    # that died without retiring is exactly the case a session sweep is for.
    swept = await pool.stop_session("s-1", 0.0)
    assert [record.id for record in swept] == [parents.id]


async def test_a_subscriber_is_told_when_work_settles() -> None:
    pool = InMemoryWorkPool()
    settled: list[TaskRecord] = []
    cancel = pool.subscribe(settled.append)

    async def _body() -> str:
        return "ok"

    record = await _launch(pool, _body())
    handle = pool.handle(record.id)
    assert handle is not None
    await handle.wait()
    await asyncio.sleep(0)

    assert [item.id for item in settled] == [record.id]
    cancel()


# ── dispatch answers with a handle ──────────────────────────────────────────


def _definition(**overrides: Any) -> SubagentDef:
    values: dict[str, Any] = {
        "id": "researcher",
        "tenant_id": "t-1",
        "name": "Researcher",
        "description": "reads things",
        "system_prompt": "research",
    }
    values.update(overrides)
    return SubagentDef(**values)


async def test_dispatch_hands_back_a_handle_not_a_result() -> None:
    dispatch: IAgentDispatch = InMemoryAgentDispatch()
    handle: SubagentHandle = await dispatch.dispatch(
        SubagentTask(subagent_id="researcher", parent_run_id="r-1", task_prompt="go")
    )

    identity = handle.identity()
    assert identity.kind == "agent"
    assert identity.agent is not None and identity.agent.name == "researcher"

    result = await handle.wait()
    assert isinstance(result, SubagentResult)
    assert result.success


async def test_a_dispatched_child_can_be_stopped_through_its_handle() -> None:
    dispatch = InMemoryAgentDispatch()
    handle = await dispatch.dispatch(
        SubagentTask(subagent_id="researcher", parent_run_id="r-1", task_prompt="go")
    )

    record = await handle.stop()

    assert record.terminal


# ── the fields that drive a child ───────────────────────────────────────────


def test_a_definition_states_how_its_child_is_driven() -> None:
    definition = _definition(
        model="a-model",
        max_turns=4,
        timeout_seconds=90.0,
        permission_mode="strict",
        background=True,
    )

    assert definition.model == "a-model"
    assert definition.max_turns == 4
    assert definition.timeout_seconds == pytest.approx(90.0)
    assert definition.permission_mode == "strict"
    assert definition.background


def test_a_definition_that_says_none_of_it_inherits() -> None:
    definition = _definition()

    assert definition.model == ""
    assert definition.max_turns == 0
    assert definition.timeout_seconds is None
    assert definition.permission_mode == ""
    assert not definition.background


def test_one_call_states_how_it_is_collected() -> None:
    task = SubagentTask(
        subagent_id="researcher",
        parent_run_id="r-1",
        task_prompt="go",
        background=True,
        notify_on_finish=True,
        expected_seconds=30.0,
        timeout_seconds=120.0,
    )

    assert task.background
    assert task.notify_on_finish
    assert task.expected_seconds == pytest.approx(30.0)
    assert task.timeout_seconds == pytest.approx(120.0)
