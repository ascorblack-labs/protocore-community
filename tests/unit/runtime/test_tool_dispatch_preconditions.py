"""Dispatcher-level tests for the DAG-precondition mechanism.

Verifies the :class:`~protocore.runtime.tool_dispatch.ToolDispatcher` wiring:

* When the called tool declares ``preconditions`` on its
  :class:`~protocore.contracts.types.ToolDefinition`, an unsatisfied
  precondition returns a ``[PRECONDITION NOT MET: ...]`` envelope without
  invoking the tool.
* When the precondition IS satisfied (a prior tool was recorded on the
  run state), dispatch proceeds normally.
* ``LoopConstants.tool_preconditions_enabled=False`` bypasses the
  check entirely.
* Successful dispatches record satisfaction on the run state so future
  dispatches can see the prior call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.run_state import RunScopedState
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES


@dataclass
class PreconditionedTool(Tool):
    """:class:`Tool` impl that publishes preconditions on its definition."""

    tool_name: str = "FinalizeFile"
    description_text: str = "Finalize the file"
    preconditions: list[str] | None = None
    path_fields: list[str] | None = None
    response_content: str = "ok"
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description_text,
            parameters=ToolParameterSchema(properties={"path": {"type": "string"}}),
            preconditions=self.preconditions,
            path_fields=self.path_fields,
        )

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
        )


def _build_ctx(state: RunScopedState | None = None) -> ToolContext:
    return ToolContext(
        tenant_id="tenant-1",
        run_id="run-1",
        session_id="sess-1",
        run_state=state,
    )


def _build_dispatcher(tools: list[Tool]) -> ToolDispatcher:
    return ToolDispatcher(
        registry=ToolRegistry(tools),
        permission_gate=ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES),
    )


async def _drain(
    dispatcher: ToolDispatcher,
    *,
    tool_call: ToolCall,
    ctx: ToolContext,
) -> tuple[list[TurnEvent], DispatchOutcome]:
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        timeout_seconds=30,
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
        else:
            events.append(item)
    assert outcome is not None
    return events, outcome


# ----------------------------------------------------------------------
# Bare-name preconditions
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsatisfied_bare_precondition_returns_error() -> None:
    """``preconditions=["AppendFile"]`` with empty satisfied set → error."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile"],
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission
    assert "PRECONDITION NOT MET" in outcome.content
    assert "AppendFile" in outcome.content
    # Tool MUST NOT have been invoked.
    assert tool.calls == []


@pytest.mark.asyncio
async def test_satisfied_bare_precondition_dispatches() -> None:
    """Pre-seeded ``AppendFile`` on the run state → FinalizeFile succeeds."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile"],
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True), satisfied_preconditions=set(["AppendFile"]))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True
    assert tool.calls == [{"path": "x.py"}]


# ----------------------------------------------------------------------
# Parameterised preconditions
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsatisfied_parameterised_precondition_returns_error() -> None:
    """``preconditions=["AppendFile:{path}"]`` mismatch on path → error."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile:{path}"],
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True), satisfied_preconditions=set(["AppendFile", "AppendFile:other.py"]))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "wanted.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is False
    assert "PRECONDITION NOT MET" in outcome.content
    assert "AppendFile:wanted.py" in outcome.content
    assert tool.calls == []


@pytest.mark.asyncio
async def test_satisfied_parameterised_precondition_dispatches() -> None:
    """``AppendFile:wanted.py`` already satisfied → FinalizeFile succeeds."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile:{path}"],
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True), satisfied_preconditions=set(["AppendFile", "AppendFile:wanted.py"]))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "wanted.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True
    assert tool.calls == [{"path": "wanted.py"}]


# ----------------------------------------------------------------------
# Success path records satisfaction on the run state
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_dispatch_records_satisfaction() -> None:
    """A successful call must store its (tool, path) on the run state."""
    tool = PreconditionedTool(
        tool_name="AppendFile",
        preconditions=None,  # no preconditions for AppendFile itself
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True))
    ctx = _build_ctx(state)
    call = ToolCall(name="AppendFile", arguments={"path": "doc.md"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success
    satisfied = state.satisfied_preconditions
    assert "AppendFile" in satisfied
    assert "AppendFile:doc.md" in satisfied


@pytest.mark.asyncio
async def test_record_then_check_sequence_works_end_to_end() -> None:
    """Dispatch AppendFile, then FinalizeFile with a path precondition."""
    append = PreconditionedTool(
        tool_name="AppendFile",
        preconditions=None,
    )
    finalize = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile:{path}"],
    )
    dispatcher = _build_dispatcher([append, finalize])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True))
    ctx = _build_ctx(state)

    # First: AppendFile records satisfaction.
    _evts1, outcome1 = await _drain(
        dispatcher,
        tool_call=ToolCall(name="AppendFile", arguments={"path": "big.py"}),
        ctx=ctx,
    )
    assert outcome1.success

    # Second: FinalizeFile sees the satisfaction → succeeds.
    _evts2, outcome2 = await _drain(
        dispatcher,
        tool_call=ToolCall(name="FinalizeFile", arguments={"path": "big.py"}),
        ctx=ctx,
    )
    assert outcome2.success, outcome2.content
    assert finalize.calls == [{"path": "big.py"}]


@pytest.mark.asyncio
async def test_record_then_check_different_path_blocks() -> None:
    """AppendFile for ``a.py`` does NOT satisfy FinalizeFile for ``b.py``."""
    append = PreconditionedTool(tool_name="AppendFile", preconditions=None)
    finalize = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile:{path}"],
    )
    dispatcher = _build_dispatcher([append, finalize])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True))
    ctx = _build_ctx(state)

    await _drain(
        dispatcher,
        tool_call=ToolCall(name="AppendFile", arguments={"path": "a.py"}),
        ctx=ctx,
    )

    _evts, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="FinalizeFile", arguments={"path": "b.py"}),
        ctx=ctx,
    )

    assert outcome.success is False
    assert "PRECONDITION NOT MET" in outcome.content
    assert finalize.calls == []


# ----------------------------------------------------------------------
# Kill-switch
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_via_runtime_constant_bypasses_check() -> None:
    """``tool_preconditions_enabled=False`` → check is skipped."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile"],  # would fail with enforcement on
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=False))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True
    assert tool.calls == [{"path": "x.py"}]


@pytest.mark.asyncio
async def test_disabled_does_not_record_satisfaction() -> None:
    """When disabled, the run's satisfied set is NOT mutated on success."""
    tool = PreconditionedTool(
        tool_name="AppendFile",
        preconditions=None,
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=False))
    ctx = _build_ctx(state)
    call = ToolCall(name="AppendFile", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success
    # With enforcement off, no satisfaction should be recorded.
    assert state.satisfied_preconditions == set()


# ----------------------------------------------------------------------
# No-precondition tools are not affected
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_without_preconditions_dispatches_normally() -> None:
    """Default tools (no preconditions field) must not be gated."""
    tool = PreconditionedTool(tool_name="Write", preconditions=None)
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants())
    ctx = _build_ctx(state)
    call = ToolCall(name="Write", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True


@pytest.mark.asyncio
async def test_a_context_without_run_state_does_not_break_dispatch() -> None:
    """A context carrying no run state still dispatches (no RC = enabled)."""
    tool = PreconditionedTool(tool_name="Write", preconditions=None)
    dispatcher = _build_dispatcher([tool])
    ctx = _build_ctx(state=None)
    call = ToolCall(name="Write", arguments={"path": "x.py"})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True


# ----------------------------------------------------------------------
# Tool-error envelope still uses dispatcher's TOOL_RESULT event shape
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_precondition_failure_emits_tool_result_event() -> None:
    """The pre-empted call still surfaces a TOOL_RESULT envelope for the LLM."""
    tool = PreconditionedTool(
        tool_name="FinalizeFile",
        preconditions=["AppendFile"],
    )
    dispatcher = _build_dispatcher([tool])
    state = RunScopedState(rc=LoopConstants(tool_preconditions_enabled=True))
    ctx = _build_ctx(state)
    call = ToolCall(name="FinalizeFile", arguments={"path": "x.py"})

    events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is False
    tool_results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(tool_results) == 1
    payload = tool_results[0].payload
    assert payload["success"] is False
    assert "PRECONDITION NOT MET" in payload["content_blocks"][0]["text"]
