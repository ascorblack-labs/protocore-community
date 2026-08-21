"""Unit tests for the universal tool-action precondition contract.

Pure-core unit tests: model construction, round-trip serialisation,
edge cases of optional fields, and the ``ALL_PREDICATE_KINDS`` registry
shape.

The host-side evaluator logic + dispatch hook are tested under
``protocore-the host/tests/unit/runtime/``.
"""
from __future__ import annotations

import pytest

from protocore.contracts.tool_action_preconditions import (
    ALL_PREDICATE_KINDS,
    PREDICATE_KIND_ARGS_MATCH,
    PREDICATE_KIND_DOC_OBSERVED,
    PREDICATE_KIND_REF_OBSERVED,
    ToolActionPreconditionPredicate,
    ToolActionPreconditionResult,
    ToolActionPreconditionRule,
    ToolActionPreconditionSpec,
)

# ---------------------------------------------------------------------------
# Predicate-kind registry
# ---------------------------------------------------------------------------


def test_all_predicate_kinds_registry_contains_three_known_kinds() -> None:
    """The evaluator vocabulary covers the meta-judge PROCEED 4 needs."""
    assert ALL_PREDICATE_KINDS == frozenset(
        {
            PREDICATE_KIND_ARGS_MATCH,
            PREDICATE_KIND_REF_OBSERVED,
            PREDICATE_KIND_DOC_OBSERVED,
        }
    )


def test_predicate_kind_constants_are_neutral_strings() -> None:
    """Predicate-kind constants contain no benchmark/tenant-specific tokens.

    Universal-core invariant: the contract surface stays
    benchmark-agnostic so future tenants opt in without code change.
    """
    for kind in ALL_PREDICATE_KINDS:
        lower = kind.lower()
        assert "pac" not in lower
        assert "ecom" not in lower
        assert "pcm" not in lower
        assert "bitgn" not in lower


# ---------------------------------------------------------------------------
# Predicate construction + defaults
# ---------------------------------------------------------------------------


def test_predicate_defaults_are_empty_lists() -> None:
    """A minimal predicate omits every optional payload."""
    pred = ToolActionPreconditionPredicate(
        name="p1", kind=PREDICATE_KIND_ARGS_MATCH
    )
    assert pred.name == "p1"
    assert pred.kind == PREDICATE_KIND_ARGS_MATCH
    assert pred.description == ""
    assert pred.args_match_patterns == []
    assert pred.required_observed_refs == []


def test_predicate_rejects_unknown_field() -> None:
    """``extra='forbid'`` keeps the contract tight."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolActionPreconditionPredicate(
            name="p1",
            kind=PREDICATE_KIND_ARGS_MATCH,
            unknown_field="x",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Rule construction + defaults
# ---------------------------------------------------------------------------


def test_rule_default_fields_are_empty() -> None:
    """A minimal rule omits every optional field."""
    rule = ToolActionPreconditionRule(
        name="r1", tool_name="remote_exec"
    )
    assert rule.name == "r1"
    assert rule.tool_name == "remote_exec"
    assert rule.description == ""
    assert rule.args_pattern == ""
    assert rule.predicates == []
    assert rule.repair_message == ""
    assert rule.mode_override is None


def test_rule_with_predicates_constructs() -> None:
    """A rule with a mixed predicate list constructs cleanly."""
    rule = ToolActionPreconditionRule(
        name="checkout-evidence-required",
        description="block checkout without security doc",
        tool_name="remote_exec",
        args_pattern=r"^/bin/checkout(\s.*)?$",
        predicates=[
            ToolActionPreconditionPredicate(
                name="security-doc-read",
                kind=PREDICATE_KIND_DOC_OBSERVED,
                required_observed_refs=["/docs/security.md"],
            ),
        ],
        repair_message="Read /docs/security.md before /bin/checkout.",
        mode_override="shadow",
    )
    assert rule.tool_name == "remote_exec"
    assert rule.args_pattern.startswith("^/bin/checkout")
    assert len(rule.predicates) == 1
    assert rule.predicates[0].kind == PREDICATE_KIND_DOC_OBSERVED
    assert rule.mode_override == "shadow"


# ---------------------------------------------------------------------------
# Spec construction + round trip
# ---------------------------------------------------------------------------


def test_empty_spec_round_trips() -> None:
    """An empty spec serialises and deserialises faithfully."""
    spec = ToolActionPreconditionSpec()
    assert spec.rules == []
    assert spec.metadata == {}
    payload = spec.model_dump()
    restored = ToolActionPreconditionSpec.model_validate(payload)
    assert restored == spec


def test_populated_spec_round_trips() -> None:
    """A populated multi-rule spec round-trips by value."""
    spec = ToolActionPreconditionSpec(
        rules=[
            ToolActionPreconditionRule(
                name="checkout-evidence",
                tool_name="remote_exec",
                args_pattern=r"^/bin/checkout(\s.*)?$",
                predicates=[
                    ToolActionPreconditionPredicate(
                        name="docs-read",
                        kind=PREDICATE_KIND_DOC_OBSERVED,
                        required_observed_refs=[
                            "/docs/security.md",
                            "/docs/checkout.md",
                        ],
                    ),
                ],
                repair_message="Read both policy docs first.",
            ),
            ToolActionPreconditionRule(
                name="refund-policy",
                tool_name="remote_exec",
                args_pattern=r"^/bin/payments\s+refund(\s.*)?$",
                predicates=[
                    ToolActionPreconditionPredicate(
                        name="basket-observed",
                        kind=PREDICATE_KIND_REF_OBSERVED,
                        required_observed_refs=["/orders/active.json"],
                    ),
                ],
                mode_override="block",
            ),
        ],
        metadata={"owner": "team-a"},
    )
    payload = spec.model_dump()
    restored = ToolActionPreconditionSpec.model_validate(payload)
    assert restored == spec
    assert restored.metadata["owner"] == "team-a"
    assert len(restored.rules) == 2


def test_spec_rejects_unknown_field() -> None:
    """``extra='forbid'`` prevents drift via typos."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolActionPreconditionSpec.model_validate(
            {"rules": [], "extra_field": "drift"}
        )


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


def test_result_default_fields_make_an_empty_pass() -> None:
    """``allowed=True`` with no violations + ``effective_mode='off'``."""
    result = ToolActionPreconditionResult(allowed=True)
    assert result.allowed is True
    assert result.violations == []
    assert result.rule_name == ""
    assert result.effective_mode == "off"


def test_result_carries_violations_and_rule_name() -> None:
    """A failed result names the rule and lists violations."""
    result = ToolActionPreconditionResult(
        allowed=False,
        violations=["security-doc-read: /docs/security.md not observed"],
        rule_name="checkout-evidence-required",
        effective_mode="block",
    )
    assert result.allowed is False
    assert "security-doc-read" in result.violations[0]
    assert result.rule_name == "checkout-evidence-required"
    assert result.effective_mode == "block"


# ---------------------------------------------------------------------------
# Universal-core: no benchmark-specific symbols
# ---------------------------------------------------------------------------


def test_module_surface_carries_no_benchmark_tokens() -> None:
    """``__all__`` symbols stay benchmark-neutral."""
    from protocore.contracts import tool_action_preconditions as mod

    for sym in mod.__all__:
        lower = sym.lower()
        assert "pac" not in lower
        assert "ecom" not in lower
        assert "pcm" not in lower
        assert "bitgn" not in lower


# ---------------------------------------------------------------------------
# RuntimeConstants RC defaults — meta-judge PROCEED 4 wiring
# ---------------------------------------------------------------------------


def test_runtime_constants_carries_action_precondition_fields_with_safe_defaults() -> None:
    """Default-off RC defaults keep every existing tenant snapshot identical."""
    from protocore.contracts.runtime_constants import RuntimeConstants

    rc = RuntimeConstants()
    # The four PROCEED 4 fields must default to a no-op posture.
    assert rc.tool_action_preconditions_enabled is False
    assert rc.tool_action_preconditions_mode == "off"
    assert rc.tool_action_preconditions_specs == []
    assert rc.tool_action_preconditions_repair_budget == 1


def test_runtime_constants_action_preconditions_specs_round_trip() -> None:
    """The list-of-spec RC round-trips a populated payload via Pydantic."""
    from protocore.contracts.runtime_constants import RuntimeConstants

    spec = ToolActionPreconditionSpec(
        rules=[
            ToolActionPreconditionRule(
                name="r1",
                tool_name="remote_exec",
                args_pattern=r"^/bin/checkout(\s.*)?$",
                predicates=[
                    ToolActionPreconditionPredicate(
                        name="p1",
                        kind=PREDICATE_KIND_DOC_OBSERVED,
                        required_observed_refs=["/docs/security.md"],
                    ),
                ],
            ),
        ],
    )
    rc = RuntimeConstants(
        tool_action_preconditions_enabled=True,
        tool_action_preconditions_mode="shadow",
        tool_action_preconditions_specs=[spec],
    )
    assert rc.tool_action_preconditions_enabled is True
    assert rc.tool_action_preconditions_mode == "shadow"
    assert len(rc.tool_action_preconditions_specs) == 1
    only = rc.tool_action_preconditions_specs[0]
    assert only.rules[0].name == "r1"
    assert only.rules[0].predicates[0].required_observed_refs == [
        "/docs/security.md"
    ]


def test_runtime_constants_action_preconditions_mode_is_literal() -> None:
    """Only off/shadow/block are accepted at snapshot time."""
    from pydantic import ValidationError

    from protocore.contracts.runtime_constants import RuntimeConstants

    for valid in ("off", "shadow", "block"):
        rc = RuntimeConstants(tool_action_preconditions_mode=valid)
        assert rc.tool_action_preconditions_mode == valid
    with pytest.raises(ValidationError):
        RuntimeConstants(tool_action_preconditions_mode="invalid")  # type: ignore[arg-type]


def test_runtime_constants_action_preconditions_repair_budget_is_bounded() -> None:
    """``repair_budget`` is constrained to [0, 3]."""
    from pydantic import ValidationError

    from protocore.contracts.runtime_constants import RuntimeConstants

    for valid in (0, 1, 2, 3):
        rc = RuntimeConstants(tool_action_preconditions_repair_budget=valid)
        assert rc.tool_action_preconditions_repair_budget == valid
    with pytest.raises(ValidationError):
        RuntimeConstants(tool_action_preconditions_repair_budget=-1)
    with pytest.raises(ValidationError):
        RuntimeConstants(tool_action_preconditions_repair_budget=4)
