"""The terminal schema-error cap must ACT on Pydantic ``type='missing'`` (the
``Field required`` error), not only ``string_type``.

A content-less ``Write`` produced
``[{'type': 'missing', 'loc': ('content',), 'msg': 'Field required'}]`` on
every retry, but the terminal cap only matched ``'type': 'string_type'`` — so
the schema-specific TERMINAL guidance never fired and the loop spiralled.
Broadening the terminal cap to also recognise ``type='missing'`` bounds a model
that keeps re-emitting the same required-field-missing mutation call with a
structural intervention instead of an endless vague-error loop.

(The chunk-recovery path is the PRIMARY fix — a truncated call is detected and
never dispatched. This cap is the structural backstop for a missing-field error
that still reaches dispatch, e.g. the model genuinely omits the field rather
than truncating.)
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolCall
from protocore.runtime.events import TurnEvent
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool


def _build_dispatcher(tools: list[MockTool]) -> ToolDispatcher:
    return ToolDispatcher(
        registry=ToolRegistry(tools),
        permission_gate=ToolPermissionGate(),
    )


def _make_helpers_ctx(
    *, helpers: dict[str, Any] | None = None
) -> ToolContext:
    bag: dict[str, Any] = dict(helpers) if helpers else {}
    return ToolContext(
        tenant_id="tenant-miss",
        run_id="run-miss",
        session_id="sess-miss",
        metadata={"protocore.helpers": bag},
    )


async def _drain(
    dispatcher: ToolDispatcher,
    *,
    tool_call: ToolCall,
    ctx: ToolContext,
) -> DispatchOutcome:
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
            assert isinstance(item, TurnEvent)
    assert outcome is not None
    return outcome


# The exact prod 14f0debd Pydantic error: ``content`` missing from a Write.
_MISSING_FIELD_ERROR_MSG = (
    "tool 'Write': invalid arguments: "
    "[{'type': 'missing', 'loc': ('content',), 'msg': 'Field required'}]"
)


@pytest.mark.asyncio
async def test_repeated_missing_field_error_trips_terminal_cap() -> None:
    """C4 — N identical ``type='missing'`` Write errors trip the terminal cap
    (a structural intervention), not an endless vague-error loop."""
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants()  # defaults: generic=4, schema-terminal cap=3
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_MISSING_FIELD_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx = _make_helpers_ctx(helpers={"rc": rc})

    last: DispatchOutcome | None = None
    for _ in range(3):
        last = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={"path": "x.html"}),
            ctx=ctx,
        )
    assert last is not None
    assert last.error_kind is DispatchErrorKind.consecutive_error_cap, (
        "C4: the 3rd identical type='missing' error must trip the terminal cap"
    )
    assert "TERMINAL" in last.content
    # The original Field-required error text must still be present.
    assert "Field required" in last.content


@pytest.mark.asyncio
async def test_first_two_missing_field_errors_stay_original_kind() -> None:
    """Under cap=3 the first two missing-field errors keep their original kind
    (the cap is a backstop, not a first-error reject)."""
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants()
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_MISSING_FIELD_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx = _make_helpers_ctx(helpers={"rc": rc})

    for attempt in range(2):
        outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={"path": "x.html"}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution, (
            f"attempt {attempt + 1}/2 must stay original kind under cap=3"
        )
