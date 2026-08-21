"""Universal tool-action precondition contract (pure core).

Defines the data models the host-side evaluator consumes to gate
arbitrary tool invocations on declarative predicates before the tool's
side effect runs. Generalises the existing ``ToolDefinition.preconditions``
DAG mechanism (`runtime.tool_preconditions`) which targets tool-name
ordering only; this contract targets *argument-pattern + observed-state*
preconditions ("``/bin/payments refund`` must not run unless
``/docs/security.md`` was observed in this run").

Universal-core invariants:

* Every rule kind takes string payloads so a tenant expresses its
  contract entirely via per-tenant RC overrides without a code change.
* Predicate ``kind`` is a string-typed enum (NOT a Pydantic ``Literal``)
  to preserve forward-compatibility: an unknown ``kind`` is a runtime
  no-op rather than a snapshot-rejecting Pydantic error so an operator
  can roll forward an evaluator with a new predicate kind without
  invalidating the existing JSON specs.
* Rule ``args_pattern`` is the canonical command/argument matcher; it is
  applied against a canonical form of the tool's args dict via
  the host evaluator (different tools produce different canonical
  forms — different tool types may concatenate path + first args token;
  future tenants register their own canonical args emitter).
* Spec lists are JSON-serialisable via Pydantic so they round-trip
  through the catalog / dashboard path.

Universal tool-action preconditions gate the side effects of destructive
tools whose action policy must be evaluated against observed evidence
(e.g. ``/bin/checkout`` may only run after ``/proc/catalog/...`` was
read; ``/bin/payments refund`` may only run after the policy doc
declared the customer eligible).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Predicate kinds (string-typed, forward-compatible)
# ---------------------------------------------------------------------------

PREDICATE_KIND_ARGS_MATCH = "args_match"
"""Fire only when the canonical args string matches one of
:pyattr:`ToolActionPreconditionPredicate.args_match_patterns` (regex,
fully anchored implicitly via ``re.fullmatch`` semantics on the
canonical form). Useful as a sub-rule narrowing within a broader
``tool_name``-level rule — e.g. the tenant's ``remote_exec`` rule
matches every ``/bin/...`` command, and the
``required_observed_refs`` predicate inside it only fires for the
``/bin/checkout`` subset.

When ``args_match_patterns`` is empty the predicate is a permissive
no-op (passes). When non-empty, the predicate fails if NO pattern
matches the canonical args; ``violations`` reports the rule name +
canonical form for operator debugging."""

PREDICATE_KIND_REF_OBSERVED = "ref_observed"
"""Fire only when every ref in
:pyattr:`ToolActionPreconditionPredicate.required_observed_refs` has
been observed during the current run. The runtime evaluator receives
the observed-state collection from its caller.

When the caller supplies no observed-state collection, the predicate
short-circuits to a pass to avoid noisy false-positives. An empty
collection is distinct: it means observation tracking is active but no
refs have been observed, so the predicate fails closed. The exact-match
semantics protect a tenant from "any ref starting with /docs/" style
overreach; use multiple predicates if a tenant wants prefix matching."""

PREDICATE_KIND_DOC_OBSERVED = "doc_observed"
"""Strict subset of :data:`PREDICATE_KIND_REF_OBSERVED` for refs whose
canonical path starts with ``/docs/``. The semantics are identical to
``ref_observed`` for that path family; the dedicated kind exists so an
operator can declare a doc-only precondition without spelling out the
``/docs/`` prefix in every required ref. Mismatched non-doc refs in
the required list (e.g. ``/orders/0001.json``) are ignored by this
predicate — they never trigger the doc-observed gate.

When the caller supplies no observed-state collection, like
``ref_observed``, the predicate short-circuits to a pass."""

ALL_PREDICATE_KINDS: frozenset[str] = frozenset(
    {
        PREDICATE_KIND_ARGS_MATCH,
        PREDICATE_KIND_REF_OBSERVED,
        PREDICATE_KIND_DOC_OBSERVED,
    }
)
"""Known predicate kinds. Unknown kinds are forward-compatible no-ops:
an evaluator that has not yet been taught about a new predicate kind
reports it as a permissive pass rather than rejecting the snapshot at
Pydantic validation time. Tenants can extend the contract via a future
release without invalidating in-flight spec JSON."""


# ---------------------------------------------------------------------------
# Spec models
# ---------------------------------------------------------------------------


class ToolActionPreconditionPredicate(BaseModel):
    """One predicate evaluated as part of a precondition rule.

    Each predicate carries a ``kind`` discriminator + a small payload
    tuple of optional fields. Evaluators inspect the payload fields they
    care about per ``kind`` and ignore the rest.

    Predicates evaluate to a pass / fail boolean; the parent rule fails
    when ANY predicate fails (AND semantics). When the spec has no
    predicates the rule trivially passes (the rule's purpose then is to
    log the canonical args via shadow mode without enforcing anything).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Human-readable predicate name, surfaced verbatim in "
            "violation messages so the model can repair against a clear "
            "label."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Free-form description; not consumed by the evaluator. "
            "Useful for dashboard/operator audit context."
        ),
    )
    kind: str = Field(
        ...,
        description=(
            "Predicate kind discriminator. Known values are in "
            "``ALL_PREDICATE_KINDS``; unknown kinds are forward-"
            "compatible no-ops so a future tenant-specific predicate "
            "does not break existing snapshots."
        ),
    )
    args_match_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Used by ``args_match`` predicate kind: list of regex "
            "patterns matched against the tool's canonical args string. "
            "An empty list is a permissive no-op (predicate passes). "
            "Patterns are matched with ``re.fullmatch`` against the "
            "canonical form so a tenant must spell out anchors only "
            "when narrowing inside the canonical form (e.g. matching a "
            "substring inside a longer args string)."
        ),
    )
    required_observed_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Used by ``ref_observed`` / ``doc_observed`` predicate "
            "kinds: list of exact ref paths that must appear in the "
            "per-run observed-state collection before the parent rule "
            "passes. Empty list = permissive (predicate passes). The "
            "evaluator does NOT do prefix matching; tenants needing "
            "ANY-of-prefix semantics should list each acceptable ref "
            "explicitly or split into multiple predicates."
        ),
    )


class ToolActionPreconditionRule(BaseModel):
    """One precondition rule: when does it fire, what does it require?

    A rule fires when its ``tool_name`` matches the dispatched tool AND
    its ``args_pattern`` regex matches the canonical args string for
    that call. When the rule fires, every predicate in
    :pyattr:`predicates` is evaluated (AND semantics); any failed
    predicate contributes a violation message to the result.

    Modes:

    * ``mode_override="off"`` — rule never fires regardless of the
      global mode (per-rule kill-switch for operator A/B testing).
    * ``mode_override="shadow"`` — rule logs violations but lets the
      tool dispatch through (overrides the global mode).
    * ``mode_override="block"`` — rule raises
      :class:`ToolInvocationError` carrying the violation messages
      (overrides the global mode).
    * ``mode_override=None`` (default) — rule uses the global
      ``tool_action_preconditions_mode`` RC.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Human-readable rule name, surfaced verbatim in violation "
            "messages so the model can repair against a clear label. "
            "Operators choose names like ``checkout-evidence-required`` "
            "or ``refund-policy-doc-required`` for audit clarity."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Free-form description; not consumed by the evaluator. "
            "Useful for dashboard/operator audit context."
        ),
    )
    tool_name: str = Field(
        ...,
        description=(
            "Tool slug this rule applies to (e.g. ``remote_exec``). "
            "The evaluator matches case-sensitively against the tool's "
            "registered ``tool_name`` ClassVar."
        ),
    )
    args_pattern: str = Field(
        default="",
        description=(
            "Regex matched (``re.fullmatch``) against the tool's "
            "canonical args string. Empty string = rule applies to "
            "every invocation of ``tool_name`` (canonical args matched "
            "regardless of content). The canonical args form is "
            "tool-specific and registered by the host evaluator "
            "(e.g. a remote-exec tool may emit ``path`` + first arg "
            "token); see ``tool_action_preconditions.py`` in the "
            "the host service runtime."
        ),
    )
    predicates: list[ToolActionPreconditionPredicate] = Field(
        default_factory=list,
        description=(
            "Ordered list of predicates AND-combined when the rule "
            "fires. An empty list is allowed — the rule then fires "
            "purely for logging / shadow purposes and contributes no "
            "violations."
        ),
    )
    repair_message: str = Field(
        default="",
        description=(
            "Operator-supplied human-readable repair guidance fed back "
            "to the model when the rule fails in block mode. Prepended "
            "to the individual predicate violation messages so the "
            "model sees both the high-level reason (e.g. ``Checkout "
            "requires reading the security policy first``) and the "
            "exact unmet predicates. Empty string falls back to a "
            "generic templated message keyed on the rule name."
        ),
    )
    mode_override: str | None = Field(
        default=None,
        description=(
            "Per-rule mode override. ``None`` (default) defers to the "
            "global ``tool_action_preconditions_mode`` RC. Allowed "
            "values: ``off`` | ``shadow`` | ``block``. Unknown strings "
            "are forward-compatible no-ops (treated as ``None`` so the "
            "global mode wins). The override is bidirectional: it can "
            "RELAX a rule (e.g. global ``block`` + this rule ``shadow`` "
            "to A/B test a single rule) AND TIGHTEN a rule (e.g. global "
            "``off`` + this rule ``block`` to canary a single rule "
            "before flipping the global). Callers MUST enter the "
            "evaluator whenever ``tool_action_preconditions_enabled`` "
            "is true, regardless of the global mode, so per-rule "
            "overrides can fire."
        ),
    )


class ToolActionPreconditionSpec(BaseModel):
    """Top-level spec for a tenant's tool-action preconditions.

    A list of rules + free-form metadata for tenant-specific extensions.
    The evaluator iterates the rules in order; the FIRST matching rule
    for a (tool_name, args) pair wins and contributes any violations to
    the result. Subsequent rules are skipped so a tenant can express a
    "general-then-specific" cascade by ordering rules carefully (most
    specific args_pattern first).

    A tenant typically stores a list of specs on the
    ``tool_action_preconditions_specs`` RC; the evaluator concatenates
    rules across every spec so an operator can ship per-tool spec packs
    without merging them.
    """

    model_config = ConfigDict(extra="forbid")

    rules: list[ToolActionPreconditionRule] = Field(
        default_factory=list,
        description=(
            "Ordered rule list. The evaluator matches the FIRST rule "
            "whose (tool_name, args_pattern) matches the dispatched "
            "call; subsequent rules are skipped. Order matters: place "
            "the most specific ``args_pattern`` first if rules overlap."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form metadata for tenant-specific extensions. Never "
            "consumed by the universal evaluator; surface only for "
            "dashboard/audit annotations or future tenant-specific "
            "hooks."
        ),
    )


class ToolActionPreconditionResult(BaseModel):
    """Outcome of evaluating a tenant's preconditions against one call.

    ``allowed`` is True when no rule fired OR every fired rule's
    predicates all passed. ``violations`` carries one human-readable
    string per failed predicate; ``rule_name`` identifies the matching
    rule (when one fired) so callers can route the result through
    shadow / block logging consistently. ``effective_mode`` reports
    which mode the evaluator applied (``off`` / ``shadow`` / ``block``)
    after combining the global RC + per-rule override; callers MUST
    respect this when deciding whether to raise.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool = Field(
        ...,
        description=(
            "True when the tool call may proceed (no rule fired, or "
            "every fired predicate passed, or the effective mode is "
            "shadow/off). False only when the effective mode is "
            "``block`` AND at least one predicate failed."
        ),
    )
    violations: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable violation messages, one per failed "
            "predicate. Designed to be model-actionable: the host "
            "evaluator prefixes the rule's ``repair_message`` (when "
            "non-empty) so the model gets both the high-level reason "
            "and the exact predicate failures."
        ),
    )
    rule_name: str = Field(
        default="",
        description=(
            "Name of the matching rule (empty string when no rule "
            "fired). Used by callers for DIAG logging so operators can "
            "trace which rule applied to a given tool call."
        ),
    )
    effective_mode: str = Field(
        default="off",
        description=(
            "The mode actually applied after combining the global RC "
            "with the per-rule ``mode_override``. One of ``off`` | "
            "``shadow`` | ``block``. Callers MUST consult this rather "
            "than the global RC when deciding whether to raise."
        ),
    )


__all__ = [
    "ALL_PREDICATE_KINDS",
    "PREDICATE_KIND_ARGS_MATCH",
    "PREDICATE_KIND_DOC_OBSERVED",
    "PREDICATE_KIND_REF_OBSERVED",
    "ToolActionPreconditionPredicate",
    "ToolActionPreconditionResult",
    "ToolActionPreconditionRule",
    "ToolActionPreconditionSpec",
]
