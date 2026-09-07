"""A child run stops its own background work before it reports back.

Nobody was doing this. The pool has never known anything about the run tree, so
a child run that started a background command and then finished simply walked
away from it: the pool went on holding a scope no live run was watching, nothing
above knew to look, and the process behind the record outlived the whole tree
that started it.

The obligation belongs to the run that owns the scope, and only to that run — a
child sharing its parent's scope is sharing it with whatever comes next, and
stopping that on the way out would kill its parent's commands.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResult,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _drain_dispatch_tool_deferred
from protocore.tests_support.adapters import InMemoryLLMProvider
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool
from .fake_background_pool import FakeBackgroundPool


def _rc(**overrides: Any) -> LoopConstants:
    values: dict[str, Any] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)


def _child(engine_factory: Any, **overrides: Any) -> Any:
    return engine_factory(
        rc=overrides.pop("rc", _rc()),
        run_id="run-child",
        parent_run_id="run-parent",
        root_run_id="run-parent",
        subagent_id="researcher",
        **overrides,
    )


async def _drive(engine: Any) -> None:
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    ):
        pass


async def test_a_child_stops_its_own_scope_when_its_run_ends(
    engine_factory, in_memory_runtime
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="child answer")
    engine = _child(engine_factory, work_session_id="run-child-work")
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    # Filed the way the loop actually files it: under the session it shares
    # with its parent, owned by the scope that will retire it.
    own = pool.start(engine.config.session_id, owner_scope="run-child-work")

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED
    assert pool.stopped_sessions == [("run-child-work", 3.0)]
    stopped = pool.get(own.id)
    assert stopped is not None and stopped.status == "stopped"


async def test_the_grace_comes_from_the_constants(
    engine_factory, in_memory_runtime
) -> None:
    """The child's own time budget pays for its teardown, so it is stated."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="child answer")
    engine = _child(
        engine_factory,
        rc=_rc(child_run_retire_grace_seconds=0.5),
        work_session_id="run-child-work",
    )
    pool = FakeBackgroundPool()
    engine.background_pool = pool

    await _drive(engine)

    assert pool.stopped_sessions == [("run-child-work", 0.5)]


async def test_a_child_sharing_its_parents_scope_stops_nothing(
    engine_factory, in_memory_runtime
) -> None:
    """It owns nothing to retire, and its parent's commands are not its to kill."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="child answer")
    engine = _child(engine_factory)
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    shared = pool.start(engine.config.session_id)

    await _drive(engine)

    assert pool.stopped_sessions == []
    still_running = pool.get(shared.id)
    assert still_running is not None and still_running.status == "running"


async def test_a_root_run_retires_nothing(engine_factory, in_memory_runtime) -> None:
    """The session outlives the root run; its commands are meant to."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="answer")
    engine = engine_factory(rc=_rc())
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    pool.start(engine.config.session_id)

    await _drive(engine)

    assert pool.stopped_sessions == []


async def test_a_pool_that_fails_to_retire_does_not_fail_the_run(
    engine_factory, in_memory_runtime
) -> None:
    """Cleanup that goes wrong is not the run going wrong, at the last step."""

    class _RefusingPool(FakeBackgroundPool):
        async def stop_session(self, scope: str, grace_seconds: float = 0.0) -> Any:
            raise RuntimeError("pool is gone")

    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="child answer")
    engine = _child(engine_factory, work_session_id="run-child-work")
    engine.background_pool = _RefusingPool()

    await _drive(engine)

    assert engine.state is LoopState.COMPLETED


async def test_a_run_still_in_flight_retires_nothing(
    engine_factory, in_memory_runtime
) -> None:
    """Retirement is the end of the run, not the end of a turn.

    An engine that drove a turn and is not terminal — a run parked awaiting an
    approval, or one re-armed for another turn — still owns its work.
    """
    engine = _child(engine_factory, work_session_id="run-child-work")
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    pool.start(engine.config.session_id, owner_scope="run-child-work")

    await engine.retire_own_background_work()

    assert engine.state is LoopState.PENDING
    assert pool.stopped_sessions == []


async def test_a_pool_that_holds_no_scopes_is_simply_not_asked(
    engine_factory,
) -> None:
    """A run wired without a pool has nothing to retire and must not crash."""
    engine = _child(engine_factory, work_session_id="run-child-work")
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.COMPLETED)

    await engine.retire_own_background_work()

    assert engine.background_pool is None


assert CONVENTIONAL_TOOL_ROLES is not None


class _LaunchingTool(MockTool):
    """A tool that files background work exactly the way the loop tells it to."""

    is_concurrent_safe = False

    def __init__(self, pool: FakeBackgroundPool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pool = pool
        self.contexts: list[Any] = []
        self.launched: list[str] = []

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> ToolResult:
        self.contexts.append(context)
        record = self.pool.start(context.session_id, owner_scope=context.work_scope)
        self.launched.append(record.id)
        return ToolResult(tool_call_id="", content=f"started {record.id}")


async def test_the_scope_a_run_retires_is_the_scope_its_launches_are_filed_under(
    engine_factory, in_memory_runtime
) -> None:
    """The two halves of the obligation have to name the same thing.

    A run that stops one scope while its tools file work under another stops
    nothing: the leak stays open, and every test of it still passes, because
    the double is handed whichever scope the test chose. So the scope travels
    on the invocation context, and this drives a real call to pin that the
    scope a tool files under is the scope the run then spends its grace on.
    """
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    engine = _child(engine_factory, work_session_id="run-child-work")
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    tool = _LaunchingTool(pool, tool_name="Runner", description="launch")
    in_memory_runtime["tools"].register(tool)

    await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Runner", arguments={})
    )

    context = tool.contexts[0]
    assert context.session_id == engine.config.session_id
    assert context.work_scope == "run-child-work"

    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.COMPLETED)
    await engine.retire_own_background_work()

    reaped = pool.get(tool.launched[0])
    assert reaped is not None and reaped.status == "stopped"


async def test_a_root_runs_launches_stay_the_sessions(
    engine_factory, in_memory_runtime
) -> None:
    """Its work outlives it on purpose, so it files under no narrower scope."""
    engine = engine_factory(rc=_rc())
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    tool = _LaunchingTool(pool, tool_name="Runner", description="launch")
    in_memory_runtime["tools"].register(tool)

    await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Runner", arguments={})
    )

    assert tool.contexts[0].work_scope == ""
    filed = pool.get(tool.launched[0])
    assert filed is not None and filed.owner_scope == ""


async def test_a_drive_cancelled_inside_the_retire_still_left_its_pickup_point(
    engine_factory, in_memory_runtime
) -> None:
    """The snapshot is the obligation that cannot be skipped, so it goes first.

    A teardown that cancels the driver a second time lands wherever the drive
    happens to be awaiting. With the retire ahead of the persist, that await
    was the grace — and the run then ended with nothing written for anyone to
    pick up.
    """

    class _SlowPool(FakeBackgroundPool):
        async def stop_session(self, scope: str, grace_seconds: float = 0.0) -> Any:
            await asyncio.Event().wait()

    engine = _child(engine_factory, work_session_id="run-child-work")
    engine.background_pool = _SlowPool()
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.COMPLETED)
    written: list[Any] = []

    class _Recorder:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def emit(self, event: Any) -> None:
            if event.name == "state_snapshot":
                written.append(event)
            await self._inner.emit(event)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    engine.events = _Recorder(engine.events)

    async def _drive_once() -> None:
        async with engine.driving_turn():
            pass

    task = asyncio.ensure_future(_drive_once())
    for _ in range(4):
        await asyncio.sleep(0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert written
