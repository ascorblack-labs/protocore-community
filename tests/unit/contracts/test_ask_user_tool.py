"""Unit tests for the core ``AskUser`` tool contract.

RICH MULTI-QUESTION contract (modeled on Anthropic's
``AskUserQuestion`` tool). REPLACES the legacy single-question /
3-render-mode contract (no backward-compat).

Covers:

* :class:`AskUserOption` (label + optional description + optional ASCII
 ``preview``) validation.
* :class:`AskUserQuestion` (question + optional header + options +
 multiSelect + allow_custom) cross-field validation: a question MUST
 carry options OR ``allow_custom=True``; ``multiSelect`` is only
 meaningful with options; option labels unique/non-empty/bounded.
* :class:`AskUserInput` (``questions[]``) — one or more questions, each
 independently single/multi-select with an optional free-text custom
 answer. The three legacy render modes all express in the unified
 schema (plain text = no options + allow_custom; yes/no = two options;
 choices = options + multiSelect as needed).
* :class:`AskUserOutput` (``answers[]`` — per-question ``selected`` +
 optional ``custom``).
* :class:`AskUserTool` ABC compliance (name + definition); the LLM
 JSON-schema regenerates to the rich shape (``questions[]`` with
 inlined ``options``/``multiSelect``/``allow_custom`` — NO dangling
 ``$ref``).
* :meth:`AskUserTool.invoke` raises :class:`AskUserPauseRequested` with
 the validated payload.
* The RC fields ``max_ask_user_calls_per_run`` (default 10) and
 ``ask_user_resume_timeout_seconds`` (default 300) survive.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import ToolContext
from protocore.tools.ask_user import (
    ASK_USER_HEADER_MAX_LENGTH,
    ASK_USER_MAX_OPTIONS,
    ASK_USER_MAX_QUESTIONS,
    ASK_USER_OPTION_LABEL_MAX_LENGTH,
    ASK_USER_PREVIEW_MAX_LENGTH,
    ASK_USER_QUESTION_MAX_LENGTH,
    ASK_USER_TOOL_NAME,
    AskUserAnswer,
    AskUserInput,
    AskUserOption,
    AskUserOutput,
    AskUserPauseRequested,
    AskUserQuestion,
    AskUserTool,
)

# ---------------------------------------------------------------------------
# AskUserOption
# ---------------------------------------------------------------------------


def test_option_label_only() -> None:
    opt = AskUserOption(label="Yes")
    assert opt.label == "Yes"
    assert opt.description is None
    assert opt.preview is None


def test_option_with_description_and_preview() -> None:
    opt = AskUserOption(
        label="Layered",
        description="Adapter pattern, clean boundaries",
        preview="┌──┐\n│ A│\n└──┘",
    )
    assert opt.description == "Adapter pattern, clean boundaries"
    assert "┌──┐" in (opt.preview or "")


def test_option_rejects_empty_label() -> None:
    with pytest.raises(ValidationError):
        AskUserOption(label="")


def test_option_rejects_overlong_label() -> None:
    with pytest.raises(ValidationError):
        AskUserOption(label="x" * (ASK_USER_OPTION_LABEL_MAX_LENGTH + 1))


def test_option_rejects_overlong_preview() -> None:
    with pytest.raises(ValidationError):
        AskUserOption(label="ok", preview="x" * (ASK_USER_PREVIEW_MAX_LENGTH + 1))


def test_option_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AskUserOption.model_validate({"label": "x", "id": "1"})


# ---------------------------------------------------------------------------
# AskUserQuestion
# ---------------------------------------------------------------------------


def test_question_with_options_single_select() -> None:
    q = AskUserQuestion(
        question="Pick an architecture",
        options=[AskUserOption(label="Layered"), AskUserOption(label="Hexagonal")],
    )
    assert len(q.options) == 2
    assert q.multiSelect is False
    assert q.allow_custom is False
    assert q.header is None


def test_question_multi_select_with_options() -> None:
    q = AskUserQuestion(
        question="Pick features",
        options=[AskUserOption(label="A"), AskUserOption(label="B")],
        multiSelect=True,
    )
    assert q.multiSelect is True


def test_question_plain_text_via_allow_custom_no_options() -> None:
    """Plain free-text question = no options + allow_custom=True."""
    q = AskUserQuestion(question="What's the deadline?", allow_custom=True)
    assert q.options == []
    assert q.allow_custom is True


def test_question_yes_no_via_two_options() -> None:
    """yes/no = two literal options, single-select."""
    q = AskUserQuestion(
        question="Proceed?",
        options=[AskUserOption(label="Yes"), AskUserOption(label="No")],
    )
    assert [o.label for o in q.options] == ["Yes", "No"]
    assert q.multiSelect is False


def test_question_header_chip() -> None:
    q = AskUserQuestion(
        question="Pick", header="Arch", options=[AskUserOption(label="A")]
    )
    assert q.header == "Arch"


def test_question_rejects_empty_no_custom() -> None:
    """A question with neither options NOR allow_custom is unanswerable."""
    with pytest.raises(ValidationError) as exc:
        AskUserQuestion(question="dangling?")
    assert "options" in str(exc.value).lower() or "allow_custom" in str(exc.value)


def test_question_rejects_empty_options_list_no_custom() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion(question="dangling?", options=[])


def test_question_multiselect_requires_options() -> None:
    """multiSelect is only meaningful with options."""
    with pytest.raises(ValidationError) as exc:
        AskUserQuestion(question="x", allow_custom=True, multiSelect=True)
    assert "multiselect" in str(exc.value).lower() or "options" in str(exc.value).lower()


def test_question_rejects_duplicate_option_labels() -> None:
    with pytest.raises(ValidationError) as exc:
        AskUserQuestion(
            question="x",
            options=[AskUserOption(label="A"), AskUserOption(label="A")],
        )
    assert "unique" in str(exc.value).lower()


def test_question_rejects_too_many_options() -> None:
    opts = [AskUserOption(label=f"o{i}") for i in range(ASK_USER_MAX_OPTIONS + 1)]
    with pytest.raises(ValidationError):
        AskUserQuestion(question="x", options=opts)


def test_question_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion(question="", allow_custom=True)


def test_question_rejects_overlong_question() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion(
            question="x" * (ASK_USER_QUESTION_MAX_LENGTH + 1), allow_custom=True
        )


def test_question_rejects_overlong_header() -> None:
    with pytest.raises(ValidationError):
        AskUserQuestion(
            question="x",
            header="h" * (ASK_USER_HEADER_MAX_LENGTH + 1),
            allow_custom=True,
        )


def test_question_options_and_custom_together() -> None:
    """options + allow_custom (the 'Other' escape hatch) is valid."""
    q = AskUserQuestion(
        question="Pick or type",
        options=[AskUserOption(label="A")],
        allow_custom=True,
    )
    assert q.allow_custom is True
    assert len(q.options) == 1


# ---------------------------------------------------------------------------
# AskUserInput — questions[]
# ---------------------------------------------------------------------------


def test_input_single_question() -> None:
    payload = AskUserInput(
        questions=[
            AskUserQuestion(question="Proceed?", allow_custom=True),
        ]
    )
    assert len(payload.questions) == 1


def test_input_multi_question() -> None:
    payload = AskUserInput(
        questions=[
            AskUserQuestion(
                question="Architecture?",
                options=[AskUserOption(label="Layered"), AskUserOption(label="Hex")],
            ),
            AskUserQuestion(
                question="Features?",
                options=[AskUserOption(label="A"), AskUserOption(label="B")],
                multiSelect=True,
            ),
            AskUserQuestion(question="Anything else?", allow_custom=True),
        ]
    )
    assert len(payload.questions) == 3
    assert payload.questions[1].multiSelect is True
    assert payload.questions[2].allow_custom is True


def test_input_rejects_empty_questions() -> None:
    with pytest.raises(ValidationError):
        AskUserInput(questions=[])


def test_input_rejects_too_many_questions() -> None:
    qs = [
        AskUserQuestion(question=f"q{i}?", allow_custom=True)
        for i in range(ASK_USER_MAX_QUESTIONS + 1)
    ]
    with pytest.raises(ValidationError):
        AskUserInput(questions=qs)


def test_input_rejects_duplicate_question_text() -> None:
    """duplicate question text is unresumable.

    The output contract pairs answers to questions by ``question`` text;
    two questions with identical text make a submitted answer ambiguous.
    """
    with pytest.raises(ValidationError) as exc:
        AskUserInput(
            questions=[
                AskUserQuestion(question="Pick one", options=[AskUserOption(label="A")]),
                AskUserQuestion(question="Pick one", options=[AskUserOption(label="B")]),
            ]
        )
    assert "unique" in str(exc.value).lower()


def test_input_rejects_duplicate_question_text_via_validate() -> None:
    with pytest.raises(ValidationError):
        AskUserInput.model_validate(
            {
                "questions": [
                    {"question": "Pick one", "options": [{"label": "A"}]},
                    {"question": "Pick one", "options": [{"label": "B"}]},
                ]
            }
        )


def test_input_allows_distinct_question_text() -> None:
    payload = AskUserInput(
        questions=[
            AskUserQuestion(question="Pick one", options=[AskUserOption(label="A")]),
            AskUserQuestion(question="Pick two", options=[AskUserOption(label="B")]),
        ]
    )
    assert len(payload.questions) == 2


def test_input_no_render_mode_field() -> None:
    """The legacy ``render_mode`` field is DELETED — must be rejected as extra."""
    with pytest.raises(ValidationError):
        AskUserInput.model_validate(
            {
                "questions": [{"question": "x", "allow_custom": True}],
                "render_mode": "text",
            }
        )


def test_input_rejects_legacy_single_question_shape() -> None:
    """The old ``{question, render_mode, choices}`` shape no longer validates."""
    with pytest.raises(ValidationError):
        AskUserInput.model_validate({"question": "x", "render_mode": "text"})


def test_input_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        AskUserInput.model_validate(
            {
                "questions": [{"question": "x", "allow_custom": True}],
                "metadata": {},
            }
        )


def test_input_is_frozen() -> None:
    payload = AskUserInput(
        questions=[AskUserQuestion(question="x", allow_custom=True)]
    )
    with pytest.raises(ValidationError):
        payload.questions = []  # type: ignore[misc]


def test_input_option_preview_survives_round_trip() -> None:
    payload = AskUserInput(
        questions=[
            AskUserQuestion(
                question="Pick a layout",
                options=[
                    AskUserOption(label="Grid", preview="┌─┬─┐\n├─┼─┤\n└─┴─┘"),
                    AskUserOption(label="Flow", preview="A → B → C"),
                ],
            )
        ]
    )
    dumped = payload.model_dump()
    assert dumped["questions"][0]["options"][0]["preview"] == "┌─┬─┐\n├─┼─┤\n└─┴─┘"
    reloaded = AskUserInput.model_validate(dumped)
    assert reloaded.questions[0].options[1].preview == "A → B → C"


# ---------------------------------------------------------------------------
# AskUserOutput — answers[]
# ---------------------------------------------------------------------------


def test_output_single_answer_selected() -> None:
    out = AskUserOutput(
        answers=[AskUserAnswer(question="Proceed?", selected=["Yes"])]
    )
    assert out.answers[0].selected == ["Yes"]
    assert out.answers[0].custom is None


def test_output_multi_answer_with_custom() -> None:
    out = AskUserOutput(
        answers=[
            AskUserAnswer(question="Features?", selected=["A", "B"]),
            AskUserAnswer(question="Anything else?", selected=[], custom="ship it"),
        ]
    )
    assert out.answers[0].selected == ["A", "B"]
    assert out.answers[1].custom == "ship it"


def test_output_answer_custom_only() -> None:
    """A free-text answer carries no selected labels, only ``custom``."""
    out = AskUserOutput(
        answers=[AskUserAnswer(question="Deadline?", selected=[], custom="Friday")]
    )
    assert out.answers[0].selected == []
    assert out.answers[0].custom == "Friday"


def test_output_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        AskUserOutput.model_validate(
            {"answers": [{"question": "x", "selected": []}], "source": "user"}
        )


def test_output_requires_answers() -> None:
    with pytest.raises(ValidationError):
        AskUserOutput()  # type: ignore[call-arg]


def test_output_rejects_empty_answers_list() -> None:
    """an empty wrapped ``answers`` list must not validate.

    ``{"answers": []}`` previously slipped through and consumed the
    pending interrupt without answering any posed question.
    """
    with pytest.raises(ValidationError):
        AskUserOutput(answers=[])


def test_answer_rejects_empty_custom_string() -> None:
    """MEDIUM-1 — an empty ``custom`` string is not a valid answer.

    Use ``None`` (omit) when no custom answer is supplied.
    """
    with pytest.raises(ValidationError):
        AskUserAnswer(question="Deadline?", selected=[], custom="")


def test_answer_custom_none_is_valid() -> None:
    answer = AskUserAnswer(question="Pick?", selected=["A"], custom=None)
    assert answer.custom is None


def test_output_json_serialisable() -> None:
    out = AskUserOutput(
        answers=[AskUserAnswer(question="Proceed?", selected=["Yes"], custom=None)]
    )
    encoded = out.model_dump_json()
    decoded = json.loads(encoded)
    assert decoded["answers"][0]["question"] == "Proceed?"
    assert decoded["answers"][0]["selected"] == ["Yes"]


# ---------------------------------------------------------------------------
# Tool ABC compliance + LLM JSON-schema regen
# ---------------------------------------------------------------------------


def test_tool_name_constant() -> None:
    assert ASK_USER_TOOL_NAME == "AskUser"


def test_tool_name_property() -> None:
    tool = AskUserTool()
    assert tool.name == ASK_USER_TOOL_NAME


def test_tool_definition_regenerates_rich_schema() -> None:
    """The model-facing JSON schema MUST carry ``questions[]`` with the
    nested option/multiSelect/allow_custom shape and contain NO dangling
    ``$ref`` (the nested models must be inlined for the provider)."""
    tool = AskUserTool()
    definition = tool.definition
    assert definition.name == ASK_USER_TOOL_NAME
    assert len(definition.description) > 30

    props = definition.parameters.properties
    assert "questions" in props
    assert "questions" in definition.parameters.required

    # No legacy fields leak into the schema.
    assert "render_mode" not in props
    assert "choices" not in props
    assert "question" not in props  # singular legacy field gone

    # The whole schema must be self-contained (no unresolved $ref / $defs)
    # — providers cannot resolve Pydantic's $defs section.
    serialized = json.dumps(definition.model_dump())
    assert "$ref" not in serialized
    assert "$defs" not in serialized

    # The nested rich shape is reachable through the inlined items schema.
    questions_schema = props["questions"]
    assert questions_schema.get("type") == "array"
    item_schema = questions_schema.get("items", {})
    item_props = item_schema.get("properties", {})
    assert "question" in item_props
    assert "options" in item_props
    assert "multiSelect" in item_props
    assert "allow_custom" in item_props
    # The option object carries label/description/preview.
    option_props = item_props["options"].get("items", {}).get("properties", {})
    assert "label" in option_props
    assert "description" in option_props
    assert "preview" in option_props


# ---------------------------------------------------------------------------
# Tool invocation — raises typed pause signal
# ---------------------------------------------------------------------------


def _make_ctx() -> ToolContext:
    return ToolContext(tenant_id="t-1", run_id="r-1", session_id="s-1")


@pytest.mark.asyncio
async def test_invoke_raises_pause_with_validated_payload() -> None:
    tool = AskUserTool()
    ctx = _make_ctx()
    with pytest.raises(AskUserPauseRequested) as exc:
        await tool.invoke(
            ctx,
            {
                "questions": [
                    {
                        "question": "Pick",
                        "options": [{"label": "alpha"}, {"label": "beta"}],
                    }
                ]
            },
        )
    payload = exc.value.payload
    assert isinstance(payload, AskUserInput)
    assert payload.questions[0].question == "Pick"
    assert [o.label for o in payload.questions[0].options] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_invoke_multi_question_round_trip() -> None:
    tool = AskUserTool()
    ctx = _make_ctx()
    with pytest.raises(AskUserPauseRequested) as exc:
        await tool.invoke(
            ctx,
            {
                "questions": [
                    {
                        "question": "Architecture?",
                        "header": "Arch",
                        "options": [
                            {"label": "Layered", "description": "clean"},
                            {"label": "Hex", "preview": "A→B"},
                        ],
                        "multiSelect": False,
                    },
                    {"question": "Notes?", "allow_custom": True},
                ]
            },
        )
    payload = exc.value.payload
    assert len(payload.questions) == 2
    assert payload.questions[0].header == "Arch"
    assert payload.questions[1].allow_custom is True


@pytest.mark.asyncio
async def test_invoke_validation_error_passes_through() -> None:
    """Bad arguments surface as Pydantic ``ValidationError``, NOT a pause."""
    tool = AskUserTool()
    ctx = _make_ctx()
    with pytest.raises(ValidationError):
        # question with neither options nor allow_custom is unanswerable.
        await tool.invoke(ctx, {"questions": [{"question": "dangling?"}]})


# ---------------------------------------------------------------------------
# RC fields (unchanged by the contract upgrade)
# ---------------------------------------------------------------------------


def test_rc_max_ask_user_calls_per_run_default() -> None:
    rc = RuntimeConstants()
    assert rc.max_ask_user_calls_per_run == 10


def test_rc_max_ask_user_calls_per_run_can_be_zero() -> None:
    rc = RuntimeConstants(max_ask_user_calls_per_run=0)
    assert rc.max_ask_user_calls_per_run == 0


def test_rc_max_ask_user_calls_per_run_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        RuntimeConstants(max_ask_user_calls_per_run=-1)


def test_rc_ask_user_resume_timeout_seconds_default() -> None:
    rc = RuntimeConstants()
    assert rc.ask_user_resume_timeout_seconds == 300.0


def test_rc_ask_user_resume_timeout_seconds_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        RuntimeConstants(ask_user_resume_timeout_seconds=0.0)
