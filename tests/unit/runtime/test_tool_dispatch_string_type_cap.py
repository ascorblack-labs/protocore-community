"""Stabilization string_type terminal cap.

Verifies the dispatcher's NEW separate streak counter that fires
when consecutive Pydantic ``string_type`` validation errors on the
SAME tool exceed ``RuntimeConstants.tool_dispatch_string_type_terminal_cap``
(default 3, lowered from 5 so the shape-specific TERMINAL guidance fires
BEFORE the generic ``tool_dispatch_consecutive_error_cap=4`` wraps the error
with vague guidance). The mainline coercion validators on
``WriteInput.content``, ``AppendFileInput.content``, and
``BashInput.command`` handle common malformed shapes silently; this guard
is the safety net for residual cases.

GLM looped 24+ min on ``coding-en-004`` (1421s / 103 tools / 63 string_type
errors) retrying ``Write {content: [array]}`` repeatedly. The mainline
coercion validator prevents this for ``content`` specifically; the terminal
cap below prevents it for any residual or future failure of the same shape.
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
    *,
    run_id: str = "run-st-1",
    helpers: dict[str, Any] | None = None,
) -> tuple[ToolContext, dict[str, Any]]:
    bag: dict[str, Any] = dict(helpers) if helpers else {}
    ctx = ToolContext(
        tenant_id="tenant-st",
        run_id=run_id,
        session_id="sess-st",
        metadata={"protocore.helpers": bag},
    )
    return ctx, bag


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


# Build a ``ToolInvocationError`` message body that contains the literal
# ``'type': 'string_type'`` payload — this is what the host typed-tool
# adapter raises after pydantic ``string_type`` validation failure under
# the output sanitization (``include_input=False``).
_STRING_TYPE_ERROR_MSG = (
    "tool 'Write': invalid arguments: "
    "[{'type': 'string_type', 'loc': ('content',), "
    "'msg': 'Input should be a valid string'}]"
)


@pytest.mark.asyncio
async def test_string_type_below_cap_returns_original_kind() -> None:
    """Under default RC (cap=3, generic cap=4) the first 2 calls keep the
    original ``execution`` kind; the 3rd fires the terminal cap.

    The string_type cap default is 3 so the shape-specific TERMINAL guidance
    fires BEFORE the generic ``tool_dispatch_consecutive_error_cap=4`` wraps
    the error with vague guidance.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants()  # defaults: generic=4, string_type=3
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    for attempt in range(2):
        _events, outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution, (
            f"attempt {attempt + 1}/2 must stay original kind under cap=3"
        )


@pytest.mark.asyncio
async def test_third_string_type_error_trips_terminal_cap() -> None:
    """At string_type cap=3 (default) the 3rd identical call surfaces the
    terminal rewrite with a TERMINAL marker.

    The default of 3 ensures the terminal guidance fires BEFORE the generic
    cap (default 4).
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants()  # defaults: generic=4, string_type=3
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    last_outcome: DispatchOutcome | None = None
    for _ in range(3):
        _events, last_outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
    assert last_outcome is not None
    assert last_outcome.error_kind is DispatchErrorKind.consecutive_error_cap
    assert "TERMINAL" in last_outcome.content
    assert "string_type" in last_outcome.content
    # The original error text must still be present.
    assert "Input should be a valid string" in last_outcome.content


@pytest.mark.asyncio
async def test_string_type_terminal_fires_before_generic_under_defaults() -> None:
    """Under the defaults (generic=4, string_type=3) the string_type terminal
    cap MUST fire on attempt 3 — BEFORE the generic cap could fire on
    attempt 4. The shape-specific TERMINAL guidance must reach the model
    before the vague generic "try a different tool or argument shape" wrapping.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants()  # defaults: generic=4, string_type=3
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    last_outcome: DispatchOutcome | None = None
    for _ in range(3):
        _events, last_outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
    assert last_outcome is not None
    assert last_outcome.error_kind is DispatchErrorKind.consecutive_error_cap
    # TERMINAL is the string_type-specific marker; ensure the string_type
    # path won the short-circuit race AHEAD of the generic cap. The
    # generic cap would have wrapped with "tool+error combination repeated".
    assert "TERMINAL" in last_outcome.content
    assert "tool+error combination repeated" not in last_outcome.content


@pytest.mark.asyncio
async def test_string_type_terminal_wins_over_generic_when_both_fire() -> None:
    """When both caps would fire at the same iteration the string_type
    terminal cap WINS — it short-circuits the generic check so the
    surfaced message carries shape-specific guidance.

    With generic cap=3 + string_type cap=3, the 3rd identical
    string_type error must surface the TERMINAL message, not the generic
    "try a different tool or argument shape" wording. This pins the
    short-circuit ordering even under operator overrides that align
    both caps at the same value.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants(
        tool_dispatch_consecutive_error_cap=3,
        tool_dispatch_string_type_terminal_cap=3,
    )
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    last_outcome: DispatchOutcome | None = None
    for _ in range(3):
        _events, last_outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
    assert last_outcome is not None
    assert last_outcome.error_kind is DispatchErrorKind.consecutive_error_cap
    # TERMINAL is the string_type-specific marker; ensure the string_type
    # path won the short-circuit race.
    assert "TERMINAL" in last_outcome.content


@pytest.mark.asyncio
async def test_non_string_type_error_resets_streak() -> None:
    """A non-``string_type`` error in the middle of the streak resets the
    counter — the failure mode has changed, so the terminal guard should
    not fire on a different shape.

    Uses raised generic cap AND raised string_type cap (20) so the test
    can exercise streak-reset semantics across many iterations without
    either cap firing.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants(
        tool_dispatch_consecutive_error_cap=20,
        tool_dispatch_string_type_terminal_cap=20,
    )
    string_type_tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    other_tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError("unrelated execution failure"),
    )
    ctx, bag = _make_helpers_ctx(helpers={"rc": rc})

    dispatcher_st = _build_dispatcher([string_type_tool])
    for _ in range(3):
        _events, outcome = await _drain(
            dispatcher_st,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution

    # Non-string_type error breaks the streak.
    dispatcher_other = _build_dispatcher([other_tool])
    _events, outcome = await _drain(
        dispatcher_other,
        tool_call=ToolCall(name="Write", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution
    # State must show the streak cell gone (or count reset).
    assert "tool_dispatch.string_type_streak" not in bag

    # Back to string_type errors — fresh count restarts at 1, not 4.
    dispatcher_st_again = _build_dispatcher([string_type_tool])
    for _ in range(4):
        _events, outcome = await _drain(
            dispatcher_st_again,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution


@pytest.mark.asyncio
async def test_string_type_streak_is_per_tool() -> None:
    """Switching tools restarts the streak even when the error type stays.

    Uses raised generic cap AND raised string_type cap so the test
    exercises the string_type counter's per-tool behavior across many
    iterations without either cap firing.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants(
        tool_dispatch_consecutive_error_cap=20,
        tool_dispatch_string_type_terminal_cap=20,
    )
    write_st = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    append_st = MockTool(
        tool_name="AppendFile",
        raise_exception=ToolInvocationError(
            "tool 'AppendFile': invalid arguments: "
            "[{'type': 'string_type', 'loc': ('content',), "
            "'msg': 'Input should be a valid string'}]"
        ),
    )
    dispatcher = _build_dispatcher([write_st, append_st])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    # 4 Write string_type — under cap=20, original kind.
    for _ in range(4):
        _events, outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="Write", arguments={}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution

    # Switching to AppendFile resets the tool name → fresh count=1.
    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="AppendFile", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution


@pytest.mark.asyncio
async def test_successful_call_resets_string_type_streak() -> None:
    """A successful tool call clears the string_type streak — the next
    failure restarts at count=1.

    Uses raised generic AND string_type cap so we can run 4 failures
    without either cap intervening before the success-reset path is
    exercised.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants(
        tool_dispatch_consecutive_error_cap=20,
        tool_dispatch_string_type_terminal_cap=20,
    )
    boom = MockTool(
        tool_name="Mix",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    ok = MockTool(tool_name="Mix", response_content="ok-result")
    ctx, bag = _make_helpers_ctx(helpers={"rc": rc})

    dispatcher_boom = _build_dispatcher([boom])
    for _ in range(4):
        _events, outcome = await _drain(
            dispatcher_boom,
            tool_call=ToolCall(name="Mix", arguments={}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution

    dispatcher_ok = _build_dispatcher([ok])
    _events, outcome = await _drain(
        dispatcher_ok,
        tool_call=ToolCall(name="Mix", arguments={}),
        ctx=ctx,
    )
    assert outcome.success is True
    assert "tool_dispatch.string_type_streak" not in bag


@pytest.mark.asyncio
async def test_rc_override_lowers_string_type_cap() -> None:
    """Operator override via RC sets the cap to 2 — the 2nd identical
    ``string_type`` error trips the terminal guard.

    Generic cap is also raised so it does not pre-empt the string_type
    terminal cap on this short sequence.
    """
    from protocore.contracts.tools import ToolInvocationError

    rc = RuntimeConstants(
        tool_dispatch_consecutive_error_cap=20,
        tool_dispatch_string_type_terminal_cap=2,
    )
    tool = MockTool(
        tool_name="Write",
        raise_exception=ToolInvocationError(_STRING_TYPE_ERROR_MSG),
    )
    dispatcher = _build_dispatcher([tool])
    ctx, _ = _make_helpers_ctx(helpers={"rc": rc})

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Write", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Write", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.consecutive_error_cap
    assert "TERMINAL" in outcome.content


def test_is_string_type_error_recognises_pydantic_payload() -> None:
    """The static detector matches the canonical sanitized error format."""
    assert ToolDispatcher._is_string_type_error(
        DispatchErrorKind.execution,
        _STRING_TYPE_ERROR_MSG,
    )
    # validation kind also matches.
    assert ToolDispatcher._is_string_type_error(
        DispatchErrorKind.validation,
        _STRING_TYPE_ERROR_MSG,
    )


def test_is_string_type_error_rejects_unrelated_messages() -> None:
    assert not ToolDispatcher._is_string_type_error(
        DispatchErrorKind.execution,
        "tool 'Bash' execution failed: command not found",
    )
    assert not ToolDispatcher._is_string_type_error(
        DispatchErrorKind.execution,
        "",
    )
    assert not ToolDispatcher._is_string_type_error(
        DispatchErrorKind.execution,
        "tool 'X' execution failed: some other Pydantic error like missing",
    )
