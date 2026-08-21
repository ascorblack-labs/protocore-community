"""Pathologically nested model data fails by name, not by stack exhaustion.

Every walk in this file runs over a structure the model chose the shape of: a
tool call's arguments, a message's metadata, a partially streamed JSON blob, the
helper bag a tool wrote into. A recursive walk over one of those answers a deep
enough payload with ``RecursionError``, which is not raised by the code that
knows what it was walking and carries nothing about it by the time the run loop
catches it. What these tests pin is that each walk stops at a stated depth and
says which limit it crossed.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.constants import MAX_DATA_NESTING_DEPTH
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from protocore.json_utils import (
    JSONNestingDepthExceeded,
    PartialJSONParser,
    RobustStreamingJSONParser,
    StreamingJSONParser,
    is_strict_json_text,
    parse_complete_json,
    parse_complete_json_any,
)
from protocore.runtime.query import HelperStateTooDeep, _deep_copy_helper_value
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool

# Deep enough to blow CPython's default 1000-frame limit several times over,
# so a guard that merely lowered the recursion budget would not pass.
_PATHOLOGICAL_DEPTH = 5_000


def _deep_json_text(depth: int = _PATHOLOGICAL_DEPTH) -> str:
    return '{"a":' * depth + "1" + "}" * depth


def _deep_dict(depth: int = _PATHOLOGICAL_DEPTH) -> dict[str, Any]:
    value: dict[str, Any] = {"leaf": 1}
    for _ in range(depth):
        value = {"a": value}
    return value


# ----------------------------------------------------------------------
# JSON utilities
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "parse",
    [
        parse_complete_json,
        parse_complete_json_any,
        lambda text: PartialJSONParser().parse(text),
        lambda text: PartialJSONParser().parse_with_flag(text),
        lambda text: StreamingJSONParser().consume(text),
        lambda text: RobustStreamingJSONParser().consume(text),
    ],
)
def test_every_json_entry_point_refuses_a_pathologically_nested_payload(parse) -> None:
    with pytest.raises(JSONNestingDepthExceeded) as excinfo:
        parse(_deep_json_text())
    assert "nesting" in str(excinfo.value)


def test_the_depth_refusal_is_catchable_as_an_ordinary_parse_failure() -> None:
    """A caller that already treats unparseable output as recoverable keeps working.

    ``JSONNestingDepthExceeded`` subclasses ``OutputParserException`` on
    purpose: the callers that route bad model output into a repair prompt
    should treat "too deep" the same way, and only the callers that want to
    distinguish the two need to know the subclass exists.
    """
    from protocore.json_utils import OutputParserException

    with pytest.raises(OutputParserException):
        parse_complete_json_any(_deep_json_text())


def test_repair_is_not_offered_a_payload_that_is_already_too_deep() -> None:
    """Repair closes open brackets, so it can only make an over-deep blob deeper."""
    truncated = '{"a":' * _PATHOLOGICAL_DEPTH  # no closers at all
    with pytest.raises(JSONNestingDepthExceeded):
        PartialJSONParser().parse(truncated)


def test_the_streaming_parser_resets_so_the_next_payload_still_parses() -> None:
    parser = StreamingJSONParser()
    with pytest.raises(JSONNestingDepthExceeded):
        parser.consume(_deep_json_text())
    assert parser.consume('{"ok": 1}') == {"ok": 1}


def test_brackets_inside_strings_do_not_count_toward_the_depth() -> None:
    """A payload whose STRINGS are full of braces is ordinary data, not deep."""
    text = '{"a": "' + "{[" * 1_000 + '"}'
    assert parse_complete_json(text)["a"] == "{[" * 1_000


def test_a_payload_just_under_the_bound_still_parses() -> None:
    depth = MAX_DATA_NESTING_DEPTH - 1
    assert parse_complete_json_any(_deep_json_text(depth)) is not None


def test_the_bound_is_a_parameter_so_a_caller_can_tighten_it() -> None:
    text = _deep_json_text(10)
    assert parse_complete_json_any(text, max_depth=20) is not None
    with pytest.raises(JSONNestingDepthExceeded):
        parse_complete_json_any(text, max_depth=5)


def test_the_strict_json_predicate_answers_false_rather_than_raising() -> None:
    assert is_strict_json_text(_deep_json_text()) is False


# ----------------------------------------------------------------------
# Metadata validation (runs on every streamed event)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda payload: Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="hi")],
            metadata=payload,
        ),
        lambda payload: ToolResultBlock(
            tool_call_id="call-1",
            content="ok",
            metadata=payload,
        ),
    ],
    ids=["message", "tool_result_block"],
)
def test_metadata_nested_past_the_bound_is_rejected_with_the_depth_named(build) -> None:
    with pytest.raises(ValueError) as excinfo:
        build(_deep_dict())
    assert str(MAX_DATA_NESTING_DEPTH) in str(excinfo.value)
    assert "nested deeper" in str(excinfo.value)


def test_ordinary_nested_metadata_is_untouched() -> None:
    message = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="hi")],
        metadata={"a": {"b": [{"c": 1.5}]}},
    )
    assert message.metadata["a"]["b"][0]["c"] == 1.5


def test_the_non_finite_rejection_still_names_the_path_it_found() -> None:
    """The depth bound was added to the same walk; it must not blunt the old check."""
    with pytest.raises(ValueError) as excinfo:
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="hi")],
            metadata={"outer": {"inner": [float("inf")]}},
        )
    assert "outer.inner[0]" in str(excinfo.value)


# ----------------------------------------------------------------------
# Helper-bag snapshot (parallel dispatch)
# ----------------------------------------------------------------------


def test_the_helper_bag_copier_refuses_a_runaway_value() -> None:
    with pytest.raises(HelperStateTooDeep):
        _deep_copy_helper_value(_deep_dict())


def test_the_helper_bag_copier_still_deep_copies_ordinary_state() -> None:
    original = {"streak": {"count": 1}, "seen": {"a"}, "rows": [{"x": 1}]}
    copied = _deep_copy_helper_value(original)
    copied["streak"]["count"] = 99
    copied["rows"][0]["x"] = 99
    assert original["streak"]["count"] == 1
    assert original["rows"][0]["x"] == 1
    assert copied["seen"] == {"a"}


# ----------------------------------------------------------------------
# Tool dispatch
# ----------------------------------------------------------------------


async def _dispatch(tool_call: ToolCall) -> DispatchOutcome:
    dispatcher = ToolDispatcher(
        registry=ToolRegistry([MockTool(tool_name="Echo")]),
        permission_gate=ToolPermissionGate(),
    )
    ctx = ToolContext(
        tenant_id="tenant-depth",
        run_id="run-depth",
        session_id="sess-depth",
        metadata={"protocore.helpers": {}},
    )
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        timeout_seconds=30,
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
            break
    assert outcome is not None
    return outcome


@pytest.mark.asyncio
async def test_a_tool_call_whose_arguments_are_pathologically_nested_fails_as_validation() -> None:
    """The model gets a re-issuable error; the run does not go down with the call.

    ``json.dumps`` is what the dispatcher used to reach first, and it answers a
    deep enough dict with ``RecursionError`` — neither ``TypeError`` nor
    ``ValueError``, so the handler around it never sees it and the exception
    leaves the dispatcher entirely.
    """
    outcome = await _dispatch(ToolCall(name="Echo", arguments=_deep_dict()))
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.validation
    assert str(MAX_DATA_NESTING_DEPTH) in outcome.content


@pytest.mark.asyncio
async def test_ordinary_nested_tool_arguments_dispatch_normally() -> None:
    outcome = await _dispatch(
        ToolCall(name="Echo", arguments={"a": {"b": {"c": [1, 2, 3]}}})
    )
    assert outcome.success is True


def test_the_dashboard_tunable_ceiling_defaults_to_the_structural_floor() -> None:
    """One number, two reachable places — a drift between them is the bug."""
    assert RuntimeConstants().max_data_nesting_depth == MAX_DATA_NESTING_DEPTH
