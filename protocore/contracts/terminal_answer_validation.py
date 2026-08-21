"""Universal terminal-answer validation contract (pure core).

Defines the data models the host-side validator consumes to gate a
tenant's terminal-tool answer (whatever the tenant names it, e.g.
``<tenant>_answer``) on three universal dimensions:

* **Ref hygiene** — reject directory refs, pseudo refs (e.g. a
  query-string masquerading as a path), and refs the model never observed.
* **Outcome consistency** — restrict which outcome strings are valid, block
  certain refs for a given outcome, require certain refs for a given
  outcome, all expressed as data via :class:`TerminalAnswerRefRule`.
* **Observed-source validation** — when the runtime supplies observed refs,
  the validator can enforce ``require_observed_only``.

Pure core: this module owns the **shape** of the validation spec and
result. The validator logic and its observed-state source live in
the host service runtime.

Universal-core invariants:

* No tenant- or benchmark-specific symbols here — every rule kind takes
  string-typed payloads so a tenant can express its contract entirely via
  per-tenant RC overrides without a code change. The concrete outcome
  vocabulary (the set of accepted/blocked outcome strings) is supplied by
  the tenant as data; core owns only the field shape, never a fixed value
  set.
* Rule ``kind`` is a string-typed enum rather than a Python ``Literal``
  to keep forward-compatibility for tenant-specific extensions: an
  unknown ``kind`` is a runtime no-op rather than a snapshot-rejecting
  Pydantic error.
* Spec lists are JSON-serialisable via Pydantic so they round-trip
  through the catalog / dashboard path.

A single :class:`TerminalAnswerValidationSpec` folds outcome-consistency
rules and ref-hygiene rules into one ruleset so a tenant can express both
classes of invariant together.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Rule kinds (string-typed, forward-compatible)
# ---------------------------------------------------------------------------

RULE_KIND_NO_DIRECTORY_REFS = "no_directory_refs"
"""Reject terminal answers whose refs include directory paths (suffix '/').

Operator-specific blocked prefixes can be supplied via
:pyattr:`TerminalAnswerRefRule.blocked_prefixes`."""

RULE_KIND_NO_PSEUDO_REFS = "no_pseudo_refs"
"""Reject pseudo refs that are not real source paths.

Pseudo refs are strings containing characters that disqualify them as
canonical source paths (e.g. a path with an embedded query expression).
The matcher checks for any ref containing the substrings in
:pyattr:`TerminalAnswerRefRule.blocked_substrings`, defaulting to a
small built-in set (``(``, `` ``, ``\\n``) when the list is empty."""

RULE_KIND_REQUIRE_OBSERVED_ONLY = "require_observed_only"
"""Reject refs the model never observed (read/stat/list/tree/find/search).

The validator consumes the observed refs supplied by the runtime; refs
not in that collection are flagged. When no observations are supplied,
this rule is a no-op."""

RULE_KIND_OUTCOME_BLOCKED_REFS = "outcome_blocked_refs"
"""Reject specific ref prefixes when the answer outcome matches.

Applied only when the answer's outcome string is in
:pyattr:`TerminalAnswerRefRule.applies_to_outcomes` (or always when
that list is empty). Useful when a resource may be legitimately read but
must NOT appear as a terminal ref for a particular outcome (e.g. a
sensitive entity that should not be cited on a denial outcome)."""

RULE_KIND_OUTCOME_REQUIRED_REFS = "outcome_required_refs"
"""Require at least one ref starting with any of ``required_prefixes``
when the answer outcome matches the rule. Useful when a given outcome
(e.g. a success outcome) must cite an evidence file."""

RULE_KIND_OUTCOME_ALLOWED_OUTCOMES = "outcome_allowed_outcomes"
"""Restrict the set of permitted outcome strings.

Carries the allowed outcomes via
:pyattr:`TerminalAnswerRefRule.allowed_outcomes`. An empty list is a
no-op (every outcome accepted)."""

ALL_RULE_KINDS: frozenset[str] = frozenset(
    {
        RULE_KIND_NO_DIRECTORY_REFS,
        RULE_KIND_NO_PSEUDO_REFS,
        RULE_KIND_REQUIRE_OBSERVED_ONLY,
        RULE_KIND_OUTCOME_BLOCKED_REFS,
        RULE_KIND_OUTCOME_REQUIRED_REFS,
        RULE_KIND_OUTCOME_ALLOWED_OUTCOMES,
    }
)
"""Known rule kinds. Unknown kinds are no-ops (forward-compat)."""


# ---------------------------------------------------------------------------
# Spec models
# ---------------------------------------------------------------------------


class TerminalAnswerRefRule(BaseModel):
    """One rule in :class:`TerminalAnswerValidationSpec`.

    Each rule carries a ``kind`` discriminator + a small payload tuple of
    optional fields. Validators inspect the payload fields they care
    about per ``kind`` and ignore the rest.

    A rule applies only when its ``applies_to_outcomes`` is empty
    (always-on) OR contains the answer outcome. Outcome-scoped rules
    (e.g. ``outcome_blocked_refs``) typically set this list to the
    outcomes they target; ref-hygiene rules typically leave it empty so
    they fire for every answer.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Human-readable rule name, surfaced verbatim in violation "
            "messages so the model can repair against a clear label."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "Free-form description; not consumed by the validator. "
            "Useful for dashboard/operator audit context."
        ),
    )
    kind: str = Field(
        ...,
        description=(
            "Rule kind discriminator. Known values are in "
            "``ALL_RULE_KINDS``; unknown kinds are forward-compatible "
            "no-ops so a future tenant-specific rule does not break "
            "existing snapshots."
        ),
    )
    applies_to_outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "When non-empty, the rule fires only for answers whose "
            "outcome string is in this list. Empty list = rule fires "
            "for every answer regardless of outcome."
        ),
    )
    # Rule-kind-specific payload — only the fields relevant to a given
    # kind are consumed; the rest are simply ignored.
    blocked_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "Path prefixes that disqualify a ref. Used by "
            "``no_directory_refs`` (auto-suffix '/') and "
            "``outcome_blocked_refs`` (operator-supplied list)."
        ),
    )
    required_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "At least one ref must start with one of these prefixes. "
            "Used by ``outcome_required_refs``. Empty list disables "
            "the rule (no-op)."
        ),
    )
    blocked_substrings: list[str] = Field(
        default_factory=list,
        description=(
            "Substrings that disqualify a ref. Used by "
            "``no_pseudo_refs`` to catch refs that embed a query/"
            "expression rather than addressing a real path (an open "
            "paren is a typical giveaway). Empty list falls back to a "
            "built-in set of pseudo-ref markers documented on "
            "``RULE_KIND_NO_PSEUDO_REFS``."
        ),
    )
    allowed_outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "Permitted outcome strings. Used by "
            "``outcome_allowed_outcomes``. Empty list = every outcome "
            "accepted (rule is a no-op)."
        ),
    )

    def applies_to(self, outcome: str) -> bool:
        """Return True when the rule's outcome filter matches ``outcome``."""
        if not self.applies_to_outcomes:
            return True
        return outcome in self.applies_to_outcomes


class TerminalAnswerValidationSpec(BaseModel):
    """Top-level validation spec for a tenant's terminal-answer contract.

    A list of rules + a free-form metadata dict for tenant-specific
    extensions. The host validator iterates the rules in order;
    every failed rule contributes one entry to the violations list of
    :class:`TerminalAnswerValidationResult`.

    A tenant typically stores a list of specs on the
    ``terminal_answer_validation_specs`` RC; the first spec whose
    ``applies_to_outcomes``-aware match succeeds is used per answer.
    """

    model_config = ConfigDict(extra="forbid")

    rules: list[TerminalAnswerRefRule] = Field(
        default_factory=list,
        description=(
            "Ordered rule list. Each rule contributes at most one "
            "violation message. Order matters only for human-readable "
            "log output; the validator never short-circuits."
        ),
    )
    applies_to_outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "Top-level spec gate: when non-empty the spec is selected "
            "only for answers whose outcome is in this list. The "
            "the host selector picks the FIRST spec whose top-level "
            "gate matches. Empty list = always-eligible. Lets an "
            "operator ship per-outcome spec packs (one for a specific "
            "outcome, one for everything else) without merging them into "
            "a single ruleset."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form metadata for tenant-specific extensions. Never "
            "consumed by the universal validator; surface only for "
            "dashboard/audit annotations or future tenant-specific "
            "hooks."
        ),
    )


class TerminalAnswerValidationResult(BaseModel):
    """Outcome of running a spec against a terminal answer.

    ``valid`` is the AND of every rule. ``violations`` carries one
    human-readable string per failed rule so the host reject path
    can echo them back to the model verbatim for one-turn repair.
    """

    model_config = ConfigDict(extra="forbid")

    valid: bool = Field(
        ...,
        description=(
            "True when every applicable rule passed. False when any "
            "rule produced a violation message."
        ),
    )
    violations: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable violation messages, one per failed rule. "
            "Designed to be model-actionable so the model can repair "
            "without needing to consult external documentation."
        ),
    )


__all__ = [
    "ALL_RULE_KINDS",
    "RULE_KIND_NO_DIRECTORY_REFS",
    "RULE_KIND_NO_PSEUDO_REFS",
    "RULE_KIND_OUTCOME_ALLOWED_OUTCOMES",
    "RULE_KIND_OUTCOME_BLOCKED_REFS",
    "RULE_KIND_OUTCOME_REQUIRED_REFS",
    "RULE_KIND_REQUIRE_OBSERVED_ONLY",
    "TerminalAnswerRefRule",
    "TerminalAnswerValidationResult",
    "TerminalAnswerValidationSpec",
]
