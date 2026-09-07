"""End-to-end scenarios for the public surface a stored run is picked up through.

Every scenario here drives the engine only through entry points a host is
supposed to reach for — ``QueryEngine.run``, ``resume`` and
``resume_approved_tool``, all imported from ``protocore.runtime`` — and observes
only what a host can observe: yielded ``TurnEvent``s, the engine's public state
and history, and the ``state_snapshot`` envelopes that reach the injected event
stream. No private engine symbol is touched, which is the point: the guarantees
asserted below have to hold for a caller that has nothing else to hold on to.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookEvent, HookResult
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime import EventType, LoopState, resume, resume_approved_tool
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


class _CountingTool(Tool):
    """A tool that records every invocation, so "exactly once" is assertable."""

    def __init__(self, name: str = "MyTool") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []
        self.before_return: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="records its calls",
            parameters=ToolParameterSchema(properties={"v": {"type": "string"}}),
        )

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        if self.before_return is not None:
            await self.before_return(dict(arguments))
        return ToolResult(tool_call_id="", content="approved-output", is_error=False)


class _Runtime:
    """The injected adapters, kept together so a scenario can reach them."""

    def __init__(self) -> None:
        self.llm = InMemoryLLMProvider()
        self.tools = InMemoryToolRegistry()
        self.events = InMemoryEventStream()
        self.hooks = InMemoryHookManager()
        self.skills = InMemorySkillStore()
        self.blobs = InMemoryBlobStore()

    def engine(self, *, run_id: str = "run-1", rc: LoopConstants | None = None) -> QueryEngine:
        return QueryEngine(
            config=QueryEngineConfig(
                run_id=run_id,
                tenant_id="tenant-1",
                account_id="tenant-1",
                session_id="sess-1",
                model_name="scripted-model",
                rc=rc or LoopConstants(model_context_window=4_096),
            ),
            llm_provider=self.llm,
            tool_registry=self.tools,
            event_stream=self.events,
            hook_manager=self.hooks,
            skill_store=self.skills,
            blob_store=self.blobs,
        )

    def snapshots_emitted(self, run_id: str = "run-1") -> int:
        return sum(
            1
            for event in self.events.stream_for("tenant-1", run_id)
            if event.name == "state_snapshot"
        )


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _approval_rc() -> LoopConstants:
    return LoopConstants(model_context_window=4_096, approval_gate_web_enabled=True)


def _hold_next_tool_for_approval(runtime: _Runtime) -> None:
    runtime.hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )


async def _run_to_pending_approval(runtime: _Runtime, tool: _CountingTool) -> QueryEngine:
    """Drive a fresh turn that stops holding one tool call for approval."""
    runtime.tools.register(tool)
    _hold_next_tool_for_approval(runtime)
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_q",
        tool_name="MyTool",
        tool_input={"v": "approved"},
    )
    engine = runtime.engine(rc=_approval_rc())
    async for _ in engine.run(_user("use the tool")):
        pass
    assert engine.state is LoopState.AWAITING
    assert tool.calls == []
    return engine


def test_the_low_level_turn_entry_is_not_part_of_the_public_surface() -> None:
    """The weaker entry is gone from the package; the resume entries are there.

    A host that reads the package namespace to find out how to drive a run
    must not be shown an entry that skips the cancellation handle and the
    turn-boundary snapshots.
    """
    import protocore.runtime as runtime_package

    assert "query" not in runtime_package.__all__
    assert "resume" in runtime_package.__all__
    assert "resume_approved_tool" in runtime_package.__all__
    assert all(hasattr(runtime_package, name) for name in runtime_package.__all__)


async def test_resume_re_drives_a_turn_that_died_mid_flight() -> None:
    """A run whose turn never finished is picked back up from its snapshot."""
    runtime = _Runtime()
    engine = runtime.engine()
    engine.history.append(_user("say hello"))
    stored = engine.snapshot()

    runtime.llm.queue_response(text="hello", stop_reason=StopReason.end_turn)
    successor = runtime.engine()
    events = [event async for event in resume(successor, stored)]

    assert [event.type for event in events].count(EventType.MESSAGE_STOP) == 1
    assert successor.state is LoopState.COMPLETED
    assert successor.history_snapshot()[-1].role is MessageRole.assistant


async def test_resume_drives_the_message_that_arrived() -> None:
    """Input a waiting run needs opens a fresh turn on the restored run.

    A run parked in ``AWAITING`` cannot open a turn while it is still marked as
    waiting, so the drive that carries the awaited input is also what ends the
    wait — the caller supplies the answer, not the bookkeeping.
    """
    runtime = _Runtime()
    engine = runtime.engine()
    engine.history.append(_user("ask me something"))
    engine.transition_to(LoopState.RUNNING)
    engine.mark_awaiting_answer("call-question", tool_name="AskUser")
    engine.transition_to(LoopState.AWAITING)
    stored = engine.snapshot()

    runtime.llm.queue_response(text="second answer", stop_reason=StopReason.end_turn)
    successor = runtime.engine()
    events = [event async for event in resume(successor, stored, message=_user("answer"))]

    assert [event.type for event in events].count(EventType.MESSAGE_STOP) == 1
    assert successor.state is LoopState.COMPLETED
    texts = [
        block.text
        for message in successor.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    ]
    assert "answer" in texts
    assert "second answer" in texts


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param({}, id="re-drive"),
        pytest.param({"message": _user("unrelated news")}, id="message"),
    ],
)
async def test_resume_refuses_to_walk_past_a_call_still_awaiting_approval(
    drive: dict[str, Any],
) -> None:
    """Neither a re-drive nor an arriving message decides a parked call.

    Both are news that something happened somewhere else. If either cleared
    the approval, the call would stay in history with no result, and the wire
    would be repaired by inventing a failure for it — telling the model that a
    call nobody approved was tried and failed, which is exactly the repeat the
    durable record exists to prevent. So the resume refuses and names the call.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    stored = engine.snapshot()

    successor = runtime.engine(rc=_approval_rc())
    with pytest.raises(ValueError, match="toolu_q"):
        async for _ in resume(successor, stored, **drive):
            pass

    assert tool.calls == []
    assert successor.pending_approval_tool_call_id() == "toolu_q"


async def test_abandoning_an_approval_says_the_call_never_ran() -> None:
    """A caller that gives up on the decision gets a truthful result, not a failure.

    The call demonstrably did not happen, so the result that closes it says
    so. That is the whole difference from letting the wire repair fill the
    gap: the repair can only offer a synthetic error, and an error invites the
    model to try again a call that was refused a decision, not a run.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    stored = engine.snapshot()

    runtime.llm.queue_response(text="moving on", stop_reason=StopReason.end_turn)
    successor = runtime.engine(rc=_approval_rc())
    async for _ in resume(successor, stored, abandon_approval=True):
        pass

    assert tool.calls == []
    assert successor.pending_approval_tool_call_id() is None
    results = [
        block
        for message in successor.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "toolu_q"
    ]
    assert len(results) == 1
    assert results[0].is_error is False
    assert "never approved" in results[0].content

    # And the request that went out carries that same result — nothing was
    # repaired into an error on the way to the provider.
    sent = [
        block
        for request in runtime.llm.calls
        for message in request.messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "toolu_q"
    ]
    assert sent
    assert not any(block.is_error for block in sent)


async def test_abandoning_an_approval_and_approving_it_are_alternatives() -> None:
    """A caller cannot both run the parked call and give up on it."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    approved = ToolCall(id="toolu_q", name="MyTool", arguments={"v": "approved"})

    successor = runtime.engine(rc=_approval_rc())
    with pytest.raises(ValueError, match="abandon"):
        async for _ in resume(
            successor,
            engine.snapshot(),
            approved_tool_call=approved,
            abandon_approval=True,
        ):
            pass
    assert tool.calls == []


async def test_resume_refuses_a_snapshot_from_another_run_and_drives_nothing() -> None:
    """A foreign snapshot is refused before anything is driven or mutated.

    The refusal is the whole reason the restore and the drive live behind one
    entry: a caller cannot end up half-restored and then driving.
    """
    runtime = _Runtime()
    engine = runtime.engine(run_id="run-1")
    engine.history.append(_user("say hello"))
    foreign = engine.snapshot()

    runtime.llm.queue_response(text="hello", stop_reason=StopReason.end_turn)
    other_run = runtime.engine(run_id="run-2")
    snapshots_before = runtime.snapshots_emitted("run-2")

    with pytest.raises(ValueError):
        async for _ in resume(other_run, foreign):
            pass

    assert runtime.llm.calls == ()
    assert other_run.history_snapshot() == ()
    assert other_run.state is LoopState.PENDING
    assert runtime.snapshots_emitted("run-2") == snapshots_before


async def test_resume_refuses_to_guess_between_two_drives() -> None:
    """Supplying both an approved call and a message is a caller error."""
    runtime = _Runtime()
    engine = runtime.engine()
    engine.history.append(_user("say hello"))
    stored = engine.snapshot()

    with pytest.raises(ValueError, match="not both"):
        async for _ in resume(
            runtime.engine(),
            stored,
            approved_tool_call=ToolCall(id="toolu_q", name="MyTool", arguments={}),
            message=_user("second"),
        ):
            pass

    assert runtime.llm.calls == ()


async def test_resume_runs_the_approved_call_exactly_once() -> None:
    """The approval drive executes the held call and lands its real result."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    stored = engine.snapshot()

    successor = runtime.engine(rc=_approval_rc())
    events = [
        event
        async for event in resume(
            successor,
            stored,
            approved_tool_call=ToolCall(id="toolu_q", name="MyTool", arguments={"v": "approved"}),
        )
    ]

    assert tool.calls == [{"v": "approved"}]
    assert [event.type for event in events].count(EventType.TOOL_RESULT) == 1
    assert EventType.TOOL_CALL_PENDING not in [event.type for event in events]
    results = [
        block
        for message in successor.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [(block.tool_call_id, block.content, block.is_error) for block in results] == [
        ("toolu_q", "approved-output", False)
    ]


async def test_a_redelivered_approval_does_not_run_the_call_twice() -> None:
    """The same approval delivered again is absorbed, not re-executed."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    approved = ToolCall(id="toolu_q", name="MyTool", arguments={"v": "approved"})

    first = runtime.engine(rc=_approval_rc())
    async for _ in resume(first, engine.snapshot(), approved_tool_call=approved):
        pass
    after_first = first.snapshot()

    second = runtime.engine(rc=_approval_rc())
    async for _ in resume(second, after_first, approved_tool_call=approved):
        pass

    assert tool.calls == [{"v": "approved"}]


async def test_the_approval_drive_leaves_a_pickup_point_on_every_exit() -> None:
    """A drive that returns early still persists a snapshot before it does.

    The approval drive used to persist only along the paths that did work, so
    the two that decline it — an approval redelivered after the result already
    landed, and one that names a call the run is not holding — left the pod
    with state nothing else could read back.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_pending_approval(runtime, tool)
    approved = ToolCall(id="toolu_q", name="MyTool", arguments={"v": "approved"})

    async for _ in resume_approved_tool(engine, approved):
        pass
    before_replay = runtime.snapshots_emitted()

    async for _ in resume_approved_tool(engine, approved):
        pass

    assert tool.calls == [{"v": "approved"}]
    assert runtime.snapshots_emitted() > before_replay

    unknown = ToolCall(id="toolu_unknown", name="MyTool", arguments={})
    before_refusal = runtime.snapshots_emitted()
    with pytest.raises(ValueError):
        async for _ in resume_approved_tool(engine, unknown):
            pass
    assert runtime.snapshots_emitted() > before_refusal


async def test_stop_interrupts_an_approved_call_parked_in_an_await() -> None:
    """``stop()`` reaches the approval drive, not only the turn loop.

    The handle it cancels through is bound by the drive itself.  Without it an
    operator's cancel could only set a cooperative flag that a tool parked
    inside an ``await`` never comes back to read.
    """
    runtime = _Runtime()
    started = asyncio.Event()
    tool = _CountingTool()

    async def park(arguments: dict[str, Any]) -> None:
        started.set()
        await asyncio.sleep(10)

    tool.before_return = park
    engine = await _run_to_pending_approval(runtime, tool)

    async def drive() -> None:
        async for _ in resume_approved_tool(
            engine,
            ToolCall(id="toolu_q", name="MyTool", arguments={"v": "approved"}),
        ):
            pass

    driver = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=5)
    engine.stop()

    with pytest.raises(asyncio.CancelledError):
        await driver
