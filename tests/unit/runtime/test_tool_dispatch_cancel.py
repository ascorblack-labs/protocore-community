"""#6 cancel propagation — core tool-dispatch cancel-event race.

The executor places a per-run cancel ``asyncio.Event`` on the helper bag
(``ctx.metadata["protocore.helpers"]["cancel_event"]``). When it fires while a
tool is mid-flight, the dispatcher must cancel the in-flight tool task,
bounded-drain it, and raise :class:`ToolDispatchCancelled` so a leader parked
inside a synchronous long tool (the ``Agent`` tool / whole subagent) unblocks
promptly. With NO cancel event the dispatch path is byte-identical to before.

See ``.tmp/chat-forensics-2026-06-15/CANCEL-CONTRACT.md``.
"""

from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolCall
from protocore.runtime.events import EventType
from protocore.runtime.tool_dispatch import (
    HELPER_RUN_CANCEL_EVENT_KEY,
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatchCancelled,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool


def _ctx_with_cancel(
    cancel_event: asyncio.Event | None,
    *,
    rc: object | None = None,
) -> ToolContext:
    helpers: dict[str, object] = {}
    if cancel_event is not None:
        helpers[HELPER_RUN_CANCEL_EVENT_KEY] = cancel_event
    if rc is not None:
        helpers["rc"] = rc
    return ToolContext(
        run_id="run-1",
        tenant_id="tenant-1",
        session_id="sess-1",
        metadata={"protocore.helpers": helpers} if helpers else {},
    )


def _dispatcher(tool: MockTool) -> ToolDispatcher:
    return ToolDispatcher(
        registry=ToolRegistry([tool]),
        permission_gate=ToolPermissionGate(),
        hook_manager=None,
    )


async def _drain_to_outcome(
    dispatcher: ToolDispatcher,
    *,
    tool_call: ToolCall,
    ctx: ToolContext,
    timeout_seconds: int = 30,
) -> DispatchOutcome:
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        timeout_seconds=timeout_seconds,
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
    assert outcome is not None
    return outcome


@pytest.mark.asyncio
async def test_no_cancel_event_is_byte_identical() -> None:
    """Absent cancel event → the plain await path: tool runs, result returned."""
    tool = MockTool(tool_name="Echo", response_content="done")
    dispatcher = _dispatcher(tool)
    outcome = await _drain_to_outcome(
        dispatcher,
        tool_call=ToolCall(id="t1", name="Echo", arguments={"v": "x"}),
        ctx=_ctx_with_cancel(None),
    )
    assert outcome.success is True
    assert outcome.content == "done"
    assert tool.calls == [{"v": "x"}]


@pytest.mark.asyncio
async def test_unset_cancel_event_lets_tool_finish() -> None:
    """A present-but-UNSET cancel event must not disturb a normal dispatch."""
    cancel_event = asyncio.Event()  # never set
    tool = MockTool(tool_name="Echo", response_content="ok")
    dispatcher = _dispatcher(tool)
    outcome = await _drain_to_outcome(
        dispatcher,
        tool_call=ToolCall(id="t1", name="Echo", arguments={"v": "y"}),
        ctx=_ctx_with_cancel(cancel_event),
    )
    assert outcome.success is True
    assert outcome.content == "ok"
    assert not cancel_event.is_set()


@pytest.mark.asyncio
async def test_already_set_cancel_never_starts_the_tool() -> None:
    """A cancel already set when dispatch reaches the invoke point must not
    run the tool body (a coroutine runs side-effects before its first await), and
    must raise ToolDispatchCancelled with NO tool_result emitted."""
    cancel_event = asyncio.Event()
    cancel_event.set()  # already cancelled before dispatch

    flag = {"invoke_body_ran": False}

    async def _flip(_args: dict[str, object]) -> None:
        # MockTool calls on_invoke at the TOP of invoke(), before any await —
        # exactly the "side-effects before first await" the contract forbids
        # after cancel.
        flag["invoke_body_ran"] = True

    tool = MockTool(tool_name="Slow", on_invoke=_flip, sleep_seconds=30)
    dispatcher = _dispatcher(tool)

    seen_outcome = False
    saw_tool_result = False
    with pytest.raises(ToolDispatchCancelled):
        async for item in dispatcher.dispatch(
            tool_call=ToolCall(id="t1", name="Slow", arguments={}),
            ctx=_ctx_with_cancel(cancel_event),
            visibility_policy=ToolVisibilityPolicy(),
            timeout_seconds=30,
        ):
            if isinstance(item, DispatchOutcome):
                seen_outcome = True
            elif item.type is EventType.TOOL_RESULT:
                saw_tool_result = True

    assert flag["invoke_body_ran"] is False  # tool body never executed
    assert tool.calls == []  # invoke() was never called
    assert saw_tool_result is False
    assert seen_outcome is False  # raised before yielding an outcome


@pytest.mark.asyncio
async def test_cancel_during_flight_raises_and_cancels_tool() -> None:
    """Cancel fires mid-flight → tool task cancelled, ToolDispatchCancelled raised."""
    started = asyncio.Event()
    cancelled_in_tool = asyncio.Event()
    cancel_event = asyncio.Event()

    async def _block(_args: dict[str, object]) -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled_in_tool.set()
            raise

    tool = MockTool(tool_name="Slow", on_invoke=_block)
    dispatcher = _dispatcher(tool)

    async def _run() -> None:
        async for item in dispatcher.dispatch(
            tool_call=ToolCall(id="t1", name="Slow", arguments={}),
            ctx=_ctx_with_cancel(cancel_event),
            visibility_policy=ToolVisibilityPolicy(),
            timeout_seconds=30,
        ):
            del item

    drive = asyncio.ensure_future(_run())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    cancel_event.set()

    with pytest.raises(ToolDispatchCancelled):
        await asyncio.wait_for(drive, timeout=2.0)

    # The in-flight tool task was actually cancelled (bounded-drain reached it).
    assert cancelled_in_tool.is_set()


@pytest.mark.asyncio
async def test_cancel_outcome_is_cancellederror_subclass() -> None:
    """ToolDispatchCancelled must be an asyncio.CancelledError subclass so every
    existing ``except asyncio.CancelledError`` handler treats it identically."""
    assert issubclass(ToolDispatchCancelled, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_tool_timeout_still_classified_under_race() -> None:
    """With a cancel event present (never set), a tool that exceeds its budget
    must still surface DispatchErrorKind.timeout — the race preserves timeout."""
    cancel_event = asyncio.Event()  # never set
    tool = MockTool(tool_name="Slow", sleep_seconds=5.0)
    dispatcher = _dispatcher(tool)
    outcome = await _drain_to_outcome(
        dispatcher,
        tool_call=ToolCall(id="t1", name="Slow", arguments={}),
        ctx=_ctx_with_cancel(cancel_event),
        timeout_seconds=1,
    )
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.timeout


@pytest.mark.asyncio
async def test_cancel_event_emits_no_tool_result() -> None:
    """On cancel the dispatcher raises before yielding a tool_result/outcome —
    a cancelled in-flight tool must not produce a normal result frame."""
    started = asyncio.Event()
    cancel_event = asyncio.Event()
    seen: list[EventType] = []

    async def _block(_args: dict[str, object]) -> None:
        started.set()
        await asyncio.sleep(30)

    tool = MockTool(tool_name="Slow", on_invoke=_block)
    dispatcher = _dispatcher(tool)

    async def _run() -> None:
        async for item in dispatcher.dispatch(
            tool_call=ToolCall(id="t1", name="Slow", arguments={}),
            ctx=_ctx_with_cancel(cancel_event),
            visibility_policy=ToolVisibilityPolicy(),
            timeout_seconds=30,
        ):
            if isinstance(item, DispatchOutcome):
                seen.append(EventType.TOOL_RESULT)  # sentinel: should never happen
            else:
                seen.append(item.type)

    drive = asyncio.ensure_future(_run())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    cancel_event.set()
    with pytest.raises(ToolDispatchCancelled):
        await asyncio.wait_for(drive, timeout=2.0)

    assert EventType.TOOL_RESULT not in seen
