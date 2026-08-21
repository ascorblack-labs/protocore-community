"""Unit tests for the universal terminal-answer validation contract.

Pure-core unit tests: model construction, round-trip serialisation,
edge cases of the ``applies_to`` helper, and the
``ALL_RULE_KINDS`` registry shape.

The host-side validator logic + dispatch hook are tested under
``protocore-the host/tests/unit/runtime/``.
"""
from __future__ import annotations

import pytest

from protocore.contracts.terminal_answer_validation import (
    ALL_RULE_KINDS,
    RULE_KIND_NO_DIRECTORY_REFS,
    RULE_KIND_NO_PSEUDO_REFS,
    RULE_KIND_OUTCOME_ALLOWED_OUTCOMES,
    RULE_KIND_OUTCOME_BLOCKED_REFS,
    RULE_KIND_OUTCOME_REQUIRED_REFS,
    RULE_KIND_REQUIRE_OBSERVED_ONLY,
    TerminalAnswerRefRule,
    TerminalAnswerValidationResult,
    TerminalAnswerValidationSpec,
)

# ---------------------------------------------------------------------------
# Rule-kind registry
# ---------------------------------------------------------------------------


def test_all_rule_kinds_registry_contains_six_known_kinds() -> None:
    """The validator vocabulary covers the meta-judge PROCEED 2+3 needs."""
    assert ALL_RULE_KINDS == frozenset(
        {
            RULE_KIND_NO_DIRECTORY_REFS,
            RULE_KIND_NO_PSEUDO_REFS,
            RULE_KIND_REQUIRE_OBSERVED_ONLY,
            RULE_KIND_OUTCOME_BLOCKED_REFS,
            RULE_KIND_OUTCOME_REQUIRED_REFS,
            RULE_KIND_OUTCOME_ALLOWED_OUTCOMES,
        }
    )


def test_rule_kind_constants_are_neutral_strings() -> None:
    """Rule-kind constants contain no benchmark/tenant-specific tokens.

    Universal-core invariant: the contract surface stays
    benchmark-agnostic so future tenants opt in without code change.
    """
    for kind in ALL_RULE_KINDS:
        lower = kind.lower()
        assert "pac" not in lower
        assert "ecom" not in lower
        assert "pcm" not in lower
        assert "bitgn" not in lower


# ---------------------------------------------------------------------------
# Model construction + defaults
# ---------------------------------------------------------------------------


def test_rule_default_fields_are_empty() -> None:
    """A minimal rule omits every optional field."""
    rule = TerminalAnswerRefRule(name="r1", kind=RULE_KIND_NO_DIRECTORY_REFS)
    assert rule.name == "r1"
    assert rule.kind == RULE_KIND_NO_DIRECTORY_REFS
    assert rule.description == ""
    assert rule.applies_to_outcomes == []
    assert rule.blocked_prefixes == []
    assert rule.required_prefixes == []
    assert rule.blocked_substrings == []
    assert rule.allowed_outcomes == []


def test_rule_applies_to_empty_filter_matches_any_outcome() -> None:
    rule = TerminalAnswerRefRule(name="r1", kind=RULE_KIND_NO_DIRECTORY_REFS)
    assert rule.applies_to("OUTCOME_OK") is True
    assert rule.applies_to("OUTCOME_DENIED_SECURITY") is True
    assert rule.applies_to("OUTCOME_UNRELATED") is True


def test_rule_applies_to_non_empty_filter_matches_only_listed_outcomes() -> None:
    rule = TerminalAnswerRefRule(
        name="r1",
        kind=RULE_KIND_OUTCOME_BLOCKED_REFS,
        applies_to_outcomes=["OUTCOME_DENIED_SECURITY"],
        blocked_prefixes=["/checkout/baskets/"],
    )
    assert rule.applies_to("OUTCOME_DENIED_SECURITY") is True
    assert rule.applies_to("OUTCOME_OK") is False


def test_empty_spec_round_trips() -> None:
    """An empty spec serialises and deserialises faithfully."""
    spec = TerminalAnswerValidationSpec()
    assert spec.rules == []
    assert spec.applies_to_outcomes == []
    assert spec.metadata == {}
    blob = spec.model_dump()
    restored = TerminalAnswerValidationSpec.model_validate(blob)
    assert restored.rules == []
    assert restored.metadata == {}


def test_populated_spec_round_trips_through_model_dump() -> None:
    spec = TerminalAnswerValidationSpec(
        rules=[
            TerminalAnswerRefRule(
                name="hygiene-no-dirs",
                kind=RULE_KIND_NO_DIRECTORY_REFS,
            ),
            TerminalAnswerRefRule(
                name="ok-needs-evidence",
                kind=RULE_KIND_OUTCOME_REQUIRED_REFS,
                applies_to_outcomes=["OUTCOME_OK"],
                required_prefixes=["/orders/", "/customers/"],
            ),
        ],
        applies_to_outcomes=["OUTCOME_OK", "OUTCOME_DENIED_SECURITY"],
        metadata={"tenant": "tenant-eval", "spec_version": 1},
    )
    blob = spec.model_dump()
    restored = TerminalAnswerValidationSpec.model_validate(blob)
    assert len(restored.rules) == 2
    assert restored.rules[0].kind == RULE_KIND_NO_DIRECTORY_REFS
    assert restored.rules[1].applies_to_outcomes == ["OUTCOME_OK"]
    assert restored.applies_to_outcomes == [
        "OUTCOME_OK",
        "OUTCOME_DENIED_SECURITY",
    ]
    assert restored.metadata == {"tenant": "tenant-eval", "spec_version": 1}


def test_spec_round_trips_through_json() -> None:
    """JSON serialise/deserialise preserves rule order and payload."""
    spec = TerminalAnswerValidationSpec(
        rules=[
            TerminalAnswerRefRule(
                name="block-dirs",
                kind=RULE_KIND_NO_DIRECTORY_REFS,
            ),
            TerminalAnswerRefRule(
                name="restrict-outcomes",
                kind=RULE_KIND_OUTCOME_ALLOWED_OUTCOMES,
                allowed_outcomes=["OUTCOME_OK", "OUTCOME_DENIED_SECURITY"],
            ),
        ]
    )
    json_blob = spec.model_dump_json()
    restored = TerminalAnswerValidationSpec.model_validate_json(json_blob)
    assert [rule.name for rule in restored.rules] == [
        "block-dirs",
        "restrict-outcomes",
    ]
    assert restored.rules[1].allowed_outcomes == [
        "OUTCOME_OK",
        "OUTCOME_DENIED_SECURITY",
    ]


def test_result_model_default_violations_is_empty_list() -> None:
    result = TerminalAnswerValidationResult(valid=True)
    assert result.valid is True
    assert result.violations == []


def test_result_model_carries_violations_verbatim() -> None:
    result = TerminalAnswerValidationResult(
        valid=False,
        violations=["rule X failed: bad ref /docs/", "rule Y failed: ..."],
    )
    assert result.valid is False
    assert result.violations[0].startswith("rule X failed")


# ---------------------------------------------------------------------------
# Extra-forbid Pydantic guard
# ---------------------------------------------------------------------------


def test_rule_rejects_unknown_field() -> None:
    """``extra='forbid'`` catches schema drift in tenant overrides."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TerminalAnswerRefRule(
            name="x",
            kind=RULE_KIND_NO_DIRECTORY_REFS,
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_spec_rejects_unknown_field() -> None:
    """``extra='forbid'`` catches schema drift in tenant overrides."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TerminalAnswerValidationSpec(unknown="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Pure-core boundary: contract does not import anything upward
# ---------------------------------------------------------------------------


def test_contract_module_does_not_import_upward() -> None:
    """The contract must remain importable from a pure-core process."""
    import importlib

    module = importlib.import_module(
        "protocore.contracts.terminal_answer_validation"
    )
    # All submodule imports must originate inside protocore.contracts.
    for attr in dir(module):
        obj = getattr(module, attr, None)
        mod_name = getattr(obj, "__module__", "") if obj is not None else ""
        if mod_name.startswith("protocore_"):
            raise AssertionError(
                f"contract module references an upward symbol: {attr}"
            )
