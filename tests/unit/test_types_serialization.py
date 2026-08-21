"""Pydantic round-trip + multilingual regression."""
from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from protocore.constants import MAX_TOOL_CALL_ARGUMENT_BYTES
from protocore.contracts import PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY
from protocore.contracts.types import (
    AgentEnvelope,
    BlockVisibility,
    ContentBlock,
    ContentBlockKind,
    EnvelopeKind,
    Event,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)


def test_message_round_trip_latin() -> None:
    original = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="Hello world")],
    )
    blob = original.model_dump_json()
    rebuilt = Message.model_validate_json(blob)
    assert rebuilt == original


def test_partial_attempt_marker_is_public_and_round_trips() -> None:
    assert PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY == (
        "protocore.partial_assistant_attempt"
    )
    original = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="incomplete")],
        metadata={PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True},
    )

    restored = Message.model_validate(original.model_dump(mode="json"))
    assert restored.metadata == {PARTIAL_ASSISTANT_ATTEMPT_METADATA_KEY: True}


def test_message_round_trip_cyrillic() -> None:
    """Cyrillic-in-JSON-escape regression."""
    original = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="Привет, мир")],
    )
    blob_ascii = json.dumps(original.model_dump(mode="json"), ensure_ascii=True)
    blob_native = json.dumps(original.model_dump(mode="json"), ensure_ascii=False)
    # Both representations must round-trip cleanly.
    rebuilt_ascii = Message.model_validate_json(blob_ascii)
    rebuilt_native = Message.model_validate_json(blob_native)
    assert rebuilt_ascii.text == "Привет, мир"
    assert rebuilt_native.text == "Привет, мир"
    # JSON-escape format ('\u04xx') used by ensure_ascii=True.
    assert "\\u041f" in blob_ascii  # `П`


def test_tool_use_block_arg_size_cap() -> None:
    """``MAX_TOOL_CALL_ARGUMENT_BYTES`` enforced on construction.

 Stabilization raised the cap from 32 KiB to
 128 KiB. The boundary check is symbolic — exceeding ``cap + 1`` bytes
 must raise regardless of the configured cap value.
 """
    assert MAX_TOOL_CALL_ARGUMENT_BYTES == 128 * 1024
    huge_args = "x" * (MAX_TOOL_CALL_ARGUMENT_BYTES + 1)
    with pytest.raises(ValidationError):
        ToolUseBlock(
            tool_call_id="t1",
            name="Bash",
            arguments_json=huge_args,
        )


def test_tool_use_block_arg_size_at_cap_accepted() -> None:
    """Exactly ``MAX_TOOL_CALL_ARGUMENT_BYTES`` bytes is allowed.

    Validator uses ``len(value.encode("utf-8")) > cap`` — equal is OK.
    Pins the v2 ceiling so a 40-50 KiB Write payload (long-en-004) and the
    full 128 KiB envelope both validate.
    """
    at_cap_args = "x" * MAX_TOOL_CALL_ARGUMENT_BYTES
    block = ToolUseBlock(
        tool_call_id="t1",
        name="Write",
        arguments_json=at_cap_args,
    )
    assert len(block.arguments_json.encode("utf-8")) == MAX_TOOL_CALL_ARGUMENT_BYTES


def test_tool_use_block_arg_pathological_200_kib_rejected() -> None:
    """A pathological 200 KiB Write still fails fast with ValidationError.

 Acceptance criterion: pathological MB-scale outputs
 remain blocked after the cap raise. 200 KiB is still ~5x below the
 256 KiB alternative cap and ~7x below any industry provider ceiling,
 so this assertion is stable against future tuning within the
 documented range.
 """
    pathological = "x" * (200 * 1024)
    assert len(pathological.encode("utf-8")) > MAX_TOOL_CALL_ARGUMENT_BYTES
    with pytest.raises(ValidationError):
        ToolUseBlock(
            tool_call_id="t1",
            name="Write",
            arguments_json=pathological,
        )


def test_message_user_at_most_one_block() -> None:
    """User-role message rejects multiple blocks."""
    with pytest.raises(ValidationError):
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="a"), TextBlock(text="b")],
        )


def test_assistant_can_have_multiple_blocks() -> None:
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ThinkingBlock(text="reasoning"),
            TextBlock(text="answer"),
        ],
    )
    assert len(msg.content_blocks) == 2
    assert msg.text == "answer"  # thinking excluded from .text


def test_tool_call_serialization() -> None:
    call = ToolCall(name="Bash", arguments={"command": "ls"})
    blob = call.model_dump_json()
    rebuilt = ToolCall.model_validate_json(blob)
    assert rebuilt.name == "Bash"
    assert rebuilt.arguments == {"command": "ls"}


def test_tool_result_serialization() -> None:
    result = ToolResult(tool_call_id="t1", content="output", is_error=False)
    blob = result.model_dump_json()
    rebuilt = ToolResult.model_validate_json(blob)
    assert rebuilt == result


def test_event_serialization() -> None:
    event = Event(run_id="r1", name="tool_call_start", payload={"name": "Bash"})
    blob = event.model_dump_json()
    rebuilt = Event.model_validate_json(blob)
    assert rebuilt.name == "tool_call_start"
    assert rebuilt.payload == {"name": "Bash"}


def test_envelope_size_cap_enforced() -> None:
    from protocore.constants import MAX_ENVELOPE_PAYLOAD_CHARS

    with pytest.raises(ValidationError):
        AgentEnvelope(
            kind=EnvelopeKind.task,
            payload="x" * (MAX_ENVELOPE_PAYLOAD_CHARS + 1),
        )


def test_content_block_kind_enum_values() -> None:
    assert ContentBlockKind.text == "text"
    assert ContentBlockKind.tool_use == "tool_use"
    assert ContentBlockKind.tool_result == "tool_result"
    assert ContentBlockKind.thinking == "thinking"
    assert ContentBlockKind.image_ref == "image_ref"


# ---------------------------------------------------------------------------
# H8 — Message.reasoning_content persistence
# ---------------------------------------------------------------------------


def test_message_reasoning_content_roundtrip() -> None:
    """``reasoning_content`` is preserved across serialise → deserialise."""
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="The answer is 42.")],
        reasoning_content="Let me think: 6 * 7 = 42. So yes, 42.",
    )
    blob = msg.model_dump_json()
    rebuilt = Message.model_validate_json(blob)
    assert rebuilt.reasoning_content == "Let me think: 6 * 7 = 42. So yes, 42."
    assert rebuilt.text == "The answer is 42."


def test_message_reasoning_content_defaults_none() -> None:
    """``reasoning_content`` defaults to ``None`` (no thinking provider)."""
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="hi")],
    )
    assert msg.reasoning_content is None


def test_message_reasoning_content_rejected_on_user_role() -> None:
    """``reasoning_content`` is invalid on non-assistant turns."""
    with pytest.raises(ValidationError, match="only valid on assistant"):
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="hi")],
            reasoning_content="should not be here",
        )


def test_message_reasoning_content_rejected_on_system_role() -> None:
    with pytest.raises(ValidationError, match="only valid on assistant"):
        Message(
            role=MessageRole.system,
            content_blocks=[TextBlock(text="sys")],
            reasoning_content="bad",
        )


def test_message_reasoning_content_rejected_on_tool_role() -> None:
    with pytest.raises(ValidationError, match="only valid on assistant"):
        Message(
            role=MessageRole.tool,
            content_blocks=[TextBlock(text="result")],
            reasoning_content="bad",
        )


def test_message_reasoning_content_with_thinking_block_coexist() -> None:
    """A message MAY carry BOTH a ThinkingBlock content block AND
    ``reasoning_content`` — they're separate seams. The block is for
    inline placement in the assistant turn; ``reasoning_content`` is the
    canonical re-injection field for the next provider call.
    """
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[
            ThinkingBlock(text="step-by-step"),
            TextBlock(text="answer"),
        ],
        reasoning_content="step-by-step",
    )
    assert msg.reasoning_content == "step-by-step"
    assert len(msg.content_blocks) == 2
    # `.text` still strips thinking from the visible join.
    assert msg.text == "answer"


def test_message_reasoning_content_cyrillic_preserved() -> None:
    """Multilingual reasoning text round-trips."""
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="готово")],
        reasoning_content="Сначала проверю переменные, затем вычислю результат.",
    )
    blob = msg.model_dump_json()
    rebuilt = Message.model_validate_json(blob)
    assert rebuilt.reasoning_content is not None
    assert "переменные" in rebuilt.reasoning_content


# ---------------------------------------------------------------------------
# Non-finite floats in metadata/payload are rejected, not silently
# nulled (lossless-round-trip invariant). inf/nan serialise to JSON null and
# parse back as None — a SILENT lossy round-trip — so they are fail-closed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_message_metadata_rejects_non_finite_float(bad: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="x")],
            metadata={"k": bad},
        )


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_tool_result_block_metadata_rejects_non_finite_float(bad: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        ToolResultBlock(tool_call_id="t1", content="out", metadata={"k": bad})


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_event_payload_rejects_non_finite_float(bad: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        Event(run_id="r1", name="evt", payload={"k": bad})


def test_message_metadata_rejects_nested_non_finite_float() -> None:
    """The scan recurses through nested dict/list, not just top level."""
    with pytest.raises(ValidationError, match="non-finite"):
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="x")],
            metadata={"outer": {"inner": [1, 2, float("inf")]}},
        )


def test_message_metadata_finite_values_round_trip_lossless() -> None:
    """Finite floats / bool / int / str / None still round-trip byte-exact.

    Pins that the rejection is surgical: only non-finite floats are blocked;
    ordinary control-flag metadata still serialise -> deserialise losslessly.
    """
    meta = {"flag": True, "count": 3, "ratio": 1.5, "label": "ok", "absent": None}
    msg = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="x")],
        metadata=meta,
    )
    rebuilt = Message.model_validate_json(msg.model_dump_json())
    assert rebuilt.metadata == meta


def test_message_metadata_rejects_non_finite_on_json_parse() -> None:
    """A non-standard ``Infinity`` literal in inbound JSON is also rejected.

    Pydantic parses the non-standard JSON ``Infinity`` token to ``float('inf')``;
    the validator must reject it on ``model_validate_json`` too (defence for a
    hand-written / legacy payload), not just at Python construction.
    """
    blob = (
        '{"role":"assistant","content_blocks":[{"kind":"text","text":"x"}],'
        '"metadata":{"x":Infinity}}'
    )
    with pytest.raises(ValidationError, match="non-finite"):
        Message.model_validate_json(blob)


# ---------------------------------------------------------------------------
# A text block's visibility is on the wire only when it has something to say.
# ---------------------------------------------------------------------------


def test_a_public_text_block_serialises_exactly_as_it_always_did() -> None:
    """The durable form of an ordinary block must not change.

    Every user turn and every ordinary answer carries the default, and a reader
    that finds no key reads ``PUBLIC`` — so emitting it would rewrite the stored
    bytes of every block in every durable blob to say nothing.
    """

    assert TextBlock(text="durable hello").model_dump(mode="json") == {
        "kind": "text",
        "text": "durable hello",
    }
    assert TextBlock(text="durable hello").model_dump() == {
        "kind": "text",
        "text": "durable hello",
    }


def test_a_non_public_text_block_carries_its_visibility() -> None:
    block = TextBlock(text="narration", visibility=BlockVisibility.COLLAPSED)
    assert block.model_dump(mode="json") == {
        "kind": "text",
        "text": "narration",
        "visibility": "collapsed",
    }


def test_visibility_survives_a_round_trip_in_both_directions() -> None:
    adapter: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)
    absent = adapter.validate_python({"kind": "text", "text": "x"})
    assert isinstance(absent, TextBlock)
    assert absent.visibility is BlockVisibility.PUBLIC

    marked = TextBlock(text="x", visibility=BlockVisibility.COLLAPSED)
    restored = adapter.validate_python(marked.model_dump(mode="json"))
    assert isinstance(restored, TextBlock)
    assert restored.visibility is BlockVisibility.COLLAPSED


def test_a_split_assistant_turn_serialises_one_marked_block_and_one_plain() -> None:
    message = Message(
        role=MessageRole.assistant,
        content_blocks=[
            TextBlock(text="narration. ", visibility=BlockVisibility.COLLAPSED),
            TextBlock(text="the answer"),
        ],
    )
    assert message.model_dump(mode="json")["content_blocks"] == [
        {"kind": "text", "text": "narration. ", "visibility": "collapsed"},
        {"kind": "text", "text": "the answer"},
    ]
    # And the model's own view of its turn is still one string.
    assert message.text == "narration. the answer"
