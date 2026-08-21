"""Soft ``is_error`` results can opt OUT of the per-run tool-error counter
and the generic consecutive-error cap.

Background: ``BashTool`` surfaces a nonzero process exit as
``ToolResult(is_error=True)`` so the model sees ``success=false``. But ordinary
shell predicates use a nonzero exit as DATA (``grep -q`` no-match → 1,
``test``/``[`` false → 1, ``diff`` difference → 1). Counting those as core tool
errors downgraded otherwise-successful runs to ``partial`` (via
``runs.tool_errors_count``) and could trip the consecutive-error cap on a
benign polling loop — the A3b x C4 regression.

The fix: a soft ``is_error`` result may carry
``protocore.count_as_tool_error=False`` and/or
``protocore.consecutive_error_cap_eligible=False`` metadata; the dispatcher
honours both BEFORE ``_record_tool_error`` / ``_apply_consecutive_error_cap``.
Absent flags default to True so every historical soft/hard error path is
unchanged. These tests pin the dispatcher-level contract directly (
the host ``BashTool`` stamps the flags; see its
``tests/integration/tools/test_bash_hardening.py``).
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY,
    ToolCall,
)
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool


class _RecordingCounter:
    """Captures every ``increment_tool_errors_count`` call for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        self.calls.append((run_id, by))


def _build_dispatcher(
    tools: list[MockTool], *, counter: _RecordingCounter | None = None
) -> ToolDispatcher:
    return ToolDispatcher(
        registry=ToolRegistry(tools),
        permission_gate=ToolPermissionGate(),
        tool_error_counter=counter,
    )


def _ctx(
    *, run_id: str = "run-soft", helpers: dict[str, Any] | None = None
) -> tuple[ToolContext, dict[str, Any]]:
    bag: dict[str, Any] = dict(helpers) if helpers else {}
    ctx = ToolContext(
        tenant_id="tenant-soft",
        run_id=run_id,
        session_id="sess-soft",
        metadata={"protocore.helpers": bag},
    )
    return ctx, bag


async def _drain(
    dispatcher: ToolDispatcher, *, tool_call: ToolCall, ctx: ToolContext
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
    assert outcome is not None
    return outcome


_BENIGN_METADATA = {
    TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY: False,
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY: False,
}


# ----------------------------------------------------------------------
# Ordinary nonzero (grep -q / test / diff) — NOT counted, NOT capped
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label",
    ["grep -q missing", "test -f missing", "diff a b"],
)
async def test_benign_nonzero_does_not_increment_tool_errors_count(
    label: str,
) -> None:
    """An ordinary nonzero exit flagged ``count_as_tool_error=False`` must NOT
    bump ``tool_errors_count`` — so the host classifier never downgrades
    the run to ``partial`` for a benign predicate."""
    counter = _RecordingCounter()
    tool = MockTool(
        tool_name="Bash",
        response_content=f"{label}: exit 1",
        response_is_error=True,
        response_metadata=_BENIGN_METADATA,
    )
    dispatcher = _build_dispatcher([tool], counter=counter)
    ctx, _ = _ctx()

    outcome = await _drain(
        dispatcher, tool_call=ToolCall(name="Bash", arguments={}), ctx=ctx
    )

    # The model still sees a non-success result with the exit context.
    assert outcome.success is False
    assert label in outcome.content
    # …but the run's tool-error counter was never touched.
    assert counter.calls == [], (
        "benign nonzero exit wrongly counted as a tool error (A3b x C4 regression)"
    )


@pytest.mark.asyncio
async def test_benign_nonzero_does_not_trip_consecutive_cap() -> None:
    """Repeating a benign nonzero many times must NOT trip the consecutive
    cap — a polling ``grep -q`` loop is not a 'repeat the same failing call'
    signal. cap=2 makes the regression obvious if the gate were ignored."""
    rc = RuntimeConstants(tool_dispatch_consecutive_error_cap=2)
    tool = MockTool(
        tool_name="Bash",
        response_content="grep -q needle: exit 1",
        response_is_error=True,
        response_metadata=_BENIGN_METADATA,
    )
    dispatcher = _build_dispatcher([tool])
    ctx, bag = _ctx(helpers={"rc": rc})

    for _ in range(5):
        outcome = await _drain(
            dispatcher, tool_call=ToolCall(name="Bash", arguments={}), ctx=ctx
        )
        assert outcome.error_kind is DispatchErrorKind.execution, (
            "benign nonzero must keep the plain execution kind, never the cap"
        )
        assert "consecutive" not in outcome.content.lower()
    # The streak cell was never written — the benign path leaves it untouched.
    assert "tool_dispatch.consecutive_error_state" not in bag


# ----------------------------------------------------------------------
# Hard failure (timeout) — STILL counted + cap-eligible
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hard_failure_still_counts_and_caps() -> None:
    """A soft error WITHOUT the opt-out flags (the timeout/legacy case) must
    behave exactly as before: counted toward ``tool_errors_count`` and
    eligible for the consecutive cap."""
    rc = RuntimeConstants(tool_dispatch_consecutive_error_cap=2)
    counter = _RecordingCounter()
    # No response_metadata → flags absent → default (True, True).
    tool = MockTool(
        tool_name="Bash",
        response_content="command timed out after 120s",
        response_is_error=True,
    )
    dispatcher = _build_dispatcher([tool], counter=counter)
    ctx, _ = _ctx(helpers={"rc": rc})

    first = await _drain(
        dispatcher, tool_call=ToolCall(name="Bash", arguments={}), ctx=ctx
    )
    assert first.error_kind is DispatchErrorKind.execution
    second = await _drain(
        dispatcher, tool_call=ToolCall(name="Bash", arguments={}), ctx=ctx
    )
    # cap=2 → the 2nd identical hard error trips the cap.
    assert second.error_kind is DispatchErrorKind.consecutive_error_cap
    assert "timed out" in second.content
    # Both hard failures were counted.
    assert len(counter.calls) == 2


@pytest.mark.asyncio
async def test_explicit_true_flags_behave_like_default() -> None:
    """An explicit ``count_as_tool_error=True`` / cap-eligible=True is the same
    as omitting the flags (defence against a future tool stamping True)."""
    counter = _RecordingCounter()
    tool = MockTool(
        tool_name="Bash",
        response_content="real failure",
        response_is_error=True,
        response_metadata={
            TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY: True,
            TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY: True,
        },
    )
    dispatcher = _build_dispatcher([tool], counter=counter)
    ctx, _ = _ctx()

    outcome = await _drain(
        dispatcher, tool_call=ToolCall(name="Bash", arguments={}), ctx=ctx
    )
    assert outcome.success is False
    assert counter.calls == [(ctx.run_id, 1)]


# ----------------------------------------------------------------------
# Interleaving — a benign nonzero must not RESET a genuine streak
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benign_nonzero_does_not_reset_a_real_streak() -> None:
    """A non-cap-eligible benign nonzero between two genuine failures must
    leave the streak intact (it neither advances nor resets it), so a real
    repeated failure still reaches the cap.

    Sequence at cap=2: real-fail (count=1) → benign (untouched) → real-fail
    (count=2 → cap). If the benign call had reset the streak, the 2nd real
    failure would be a fresh count=1 and never cap.
    """
    rc = RuntimeConstants(tool_dispatch_consecutive_error_cap=2)
    real = MockTool(
        tool_name="Bash",
        response_content="boom failure",
        response_is_error=True,
    )
    benign = MockTool(
        tool_name="Bash",
        response_content="grep -q x: exit 1",
        response_is_error=True,
        response_metadata=_BENIGN_METADATA,
    )
    ctx, _ = _ctx(helpers={"rc": rc})

    out1 = await _drain(
        _build_dispatcher([real]),
        tool_call=ToolCall(name="Bash", arguments={}),
        ctx=ctx,
    )
    assert out1.error_kind is DispatchErrorKind.execution

    out_benign = await _drain(
        _build_dispatcher([benign]),
        tool_call=ToolCall(name="Bash", arguments={}),
        ctx=ctx,
    )
    assert out_benign.error_kind is DispatchErrorKind.execution

    out2 = await _drain(
        _build_dispatcher([real]),
        tool_call=ToolCall(name="Bash", arguments={}),
        ctx=ctx,
    )
    assert out2.error_kind is DispatchErrorKind.consecutive_error_cap, (
        "a benign nonzero must not reset a genuine consecutive-error streak"
    )
