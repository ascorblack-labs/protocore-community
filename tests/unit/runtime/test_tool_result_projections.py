"""One canonical value per tool call, and the projections taken from it.

The value a tool produces and the text the model reads used to be the same
string, which is why shrinking a transcript destroyed evidence: there was
nothing to shrink except the only copy. These tests hold the three channels
apart — the transcript's projection, the watcher's payload, the canonical
value — and check that each reaches exactly where it belongs and nowhere else.
"""

from __future__ import annotations

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolCall, ToolResult, ToolResultBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.tool_dispatch import DispatchOutcome, ToolDispatcher
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool, make_default_ctx


async def _dispatch(
    tool: MockTool, *, arguments: dict[str, str] | None = None
) -> tuple[list[TurnEvent], DispatchOutcome]:
    dispatcher = ToolDispatcher(
        registry=ToolRegistry([tool]),
        permission_gate=ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES),
    )
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    ctx: ToolContext = make_default_ctx()
    async for item in dispatcher.dispatch(
        tool_call=ToolCall(name=tool.name, arguments=arguments or {"v": "x"}),
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


def _result_event(events: list[TurnEvent]) -> TurnEvent:
    results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(results) == 1
    return results[0]


# ----------------------------------------------------------------------
# The canonical value and what is taken from it
# ----------------------------------------------------------------------


def test_a_value_with_no_projection_is_its_own_projection() -> None:
    """The common case costs a tool nothing: it returns a value and stops."""
    result = ToolResult(tool_call_id="c1", content="the whole thing")

    assert result.model_content == "the whole thing"
    assert result.ui_payload is None and result.canonical_ref is None


def test_a_stated_projection_is_what_the_model_reads() -> None:
    result = ToolResult(
        tool_call_id="c1",
        content="a" * 4096,
        model_projection="wrote 4096 bytes",
    )

    assert result.model_content == "wrote 4096 bytes"
    assert result.content == "a" * 4096


def test_an_empty_projection_is_a_projection_and_not_an_absent_one() -> None:
    """``None`` means "no projection"; the empty string means "show nothing".

    A tool whose output is genuinely worth no tokens has to be able to say so,
    and falsy-checking the field would have quietly shown the model the whole
    canonical value instead — the exact opposite of what it asked for.
    """
    result = ToolResult(tool_call_id="c1", content="noise" * 100, model_projection="")

    assert result.model_content == ""


@pytest.mark.asyncio
async def test_the_transcript_carries_the_projection_and_the_event_carries_both() -> None:
    tool = MockTool(
        tool_name="Read",
        response_content="line\n" * 500,
        response_projection="500 lines, first is 'line'",
        response_ui_payload={"lines": 500},
        response_path="/w/report.txt",
    )

    events, outcome = await _dispatch(tool, arguments={"path": "/w/report.txt"})

    # What the model will read.
    assert outcome.content == "500 lines, first is 'line'"
    # The whole value is still reachable from the outcome the host receives.
    assert outcome.canonical_content == "line\n" * 500
    payload = _result_event(events).payload
    assert payload["content_blocks"] == [
        {"type": "text", "text": "500 lines, first is 'line'"}
    ]
    assert payload["ui_payload"] == {"lines": 500}
    assert payload["path"] == "/w/report.txt"


@pytest.mark.asyncio
async def test_the_ui_payload_never_enters_the_transcript() -> None:
    """It rides the event and stops there — no tokens, nothing to compact."""
    tool = MockTool(
        tool_name="Read",
        response_content="body",
        response_ui_payload={"rendered": "<table/>"},
    )

    _, outcome = await _dispatch(tool)

    assert outcome.ui_payload == {"rendered": "<table/>"}
    assert "<table/>" not in outcome.content
    assert (outcome.metadata or {}) == {}


@pytest.mark.asyncio
async def test_a_result_event_states_success_and_mirrors_it_as_is_error() -> None:
    """Two clients, two spellings of the same question, one answer.

    One reads ``success``; the other reads ``is_error`` and treats a missing
    field as "fine". A payload carrying only the first renders a failed call as
    a successful one for the second, which is how a broken tool call reached a
    user looking like it had worked.
    """
    ok_events, _ = await _dispatch(MockTool(tool_name="Read", response_content="ok"))
    failed_events, _ = await _dispatch(
        MockTool(tool_name="Read", response_content="boom", response_is_error=True)
    )

    ok_payload = _result_event(ok_events).payload
    failed_payload = _result_event(failed_events).payload
    assert ok_payload["success"] is True and ok_payload["is_error"] is False
    assert failed_payload["success"] is False and failed_payload["is_error"] is True


@pytest.mark.asyncio
async def test_a_failed_call_keeps_its_message_as_the_only_value() -> None:
    """An error message is written for the model, so it is the value too.

    Nothing about a failure is worth keeping behind a reference: there is no
    larger thing the message is a summary of.
    """
    tool = MockTool(
        tool_name="Read",
        response_content="no such file",
        response_is_error=True,
        response_projection="no such file",
    )

    _, outcome = await _dispatch(tool)

    assert outcome.is_error is True
    assert outcome.canonical_content is None
    assert outcome.content == "no such file"


# ----------------------------------------------------------------------
# The block the transcript keeps
# ----------------------------------------------------------------------


def test_a_block_says_whether_its_text_is_the_whole_value() -> None:
    whole = ToolResultBlock(tool_call_id="c1", content="everything")
    shed = ToolResultBlock(
        tool_call_id="c2", content="[compacted]", canonical_ref="blob-9"
    )

    assert whole.canonical_ref is None
    assert shed.canonical_ref == "blob-9"


def test_a_ui_payload_with_an_unserialisable_number_is_refused() -> None:
    """It goes out as JSON on the event; a NaN there breaks the client, not us."""
    with pytest.raises(ValueError, match="ui_payload"):
        ToolResult(
            tool_call_id="c1", content="ok", ui_payload={"ratio": float("inf")}
        )


@pytest.mark.asyncio
async def test_the_transcript_block_keeps_the_value_the_projection_came_from() -> None:
    """A projected result that nothing has stored is not a lost result."""
    from protocore.runtime.query import _result_block_from_outcome

    tool = MockTool(
        tool_name="Read",
        response_content="line\n" * 500,
        response_projection="500 lines, first is 'line'",
    )
    _, outcome = await _dispatch(tool, arguments={"path": "/w/report.txt"})

    block = _result_block_from_outcome(outcome.tool_call.id, outcome)

    assert block.content == "500 lines, first is 'line'"
    assert block.canonical_content == "line\n" * 500
    assert block.canonical_ref is None
