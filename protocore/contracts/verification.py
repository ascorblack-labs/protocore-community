"""Typed, domain-neutral contracts for candidate verification.

The models in this module describe durable runtime facts.  They intentionally
do not infer those facts from generated text: adapters and trusted tools create
evidence records, while higher layers choose profiles and perform checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from protocore.contracts.types import BlockVisibility, ImageRefBlock, TextBlock


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _freeze_json(value: Any) -> Any:
    """Return a recursively immutable representation of JSON-compatible data."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return ordinary JSON-compatible containers from frozen metadata."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json_object(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a JSON *object*, leaving its element types polymorphic.

    The recursive helpers above are shape-polymorphic and can only promise
    ``Any``.  A JSON object, though, has a knowable outer type, so entering the
    recursion through this boundary lets the type checker verify the result
    rather than be told to assume it.
    """
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _thaw_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    """Thaw a JSON *object*, leaving its element types polymorphic."""
    return {key: _thaw_json(item) for key, item in value.items()}


def _canonical_frozen_object(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    """Validate, canonicalise, and deep-freeze a trusted JSON object."""
    try:
        canonical = json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc
    restored = json.loads(canonical)
    if not isinstance(restored, dict):  # Defensive: field annotation requires an object.
        raise ValueError(f"{field_name} must be a JSON object")
    return _freeze_json_object(restored)


def _require_nonempty_identifier(value: str, *, field_name: str) -> str:
    """Reject blank opaque identifiers without assigning them domain meaning."""
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must not be empty or padded")
    return value


def _allow_absent_identifier(value: str, *, field_name: str) -> str:
    """Accept an opaque identifier whose subject may have nothing to identify.

    The empty string is the single spelling of "there is nothing here", so a
    subject that declares nothing says so the same way every time and two
    readings of it fingerprint alike.  Anything else is an ordinary opaque
    identifier and must be well formed: a padded or whitespace-only value is
    malformed rather than a second way of writing absence.
    """
    if not value:
        return value
    return _require_nonempty_identifier(value, field_name=field_name)


def _canonical_digest(value: object) -> str:
    """Hash a JSON-compatible value using one stable cross-process encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DeliveryMode(StrEnum):
    """How a candidate is intended to be delivered to its reader."""

    inline = "inline"
    artifact = "artifact"
    both = "both"


class VerificationMode(StrEnum):
    """Policy posture selected for a verification profile."""

    shadow = "shadow"
    warn = "warn"
    enforce = "enforce"


class VerificationDelivery(StrEnum):
    """Visibility policy for candidate content while checks execute."""

    optimistic = "optimistic"
    gated = "gated"


class VerificationCheckStatus(StrEnum):
    """Outcome of one verification check."""

    passed = "pass"
    failed = "fail"
    error = "error"
    skipped = "skip"


class VerificationSeverity(StrEnum):
    """Ordered impact label supplied by a check implementation."""

    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ReleaseDecision(StrEnum):
    """The policy outcome recorded after verification."""

    release = "release"
    warn = "warn"
    block = "block"


class VerificationState(StrEnum):
    """Durable phase of an optional verification lifecycle."""

    not_requested = "not_requested"
    executing = "executing"
    candidate_ready = "candidate_ready"
    verifying = "verifying"
    repair_requested = "repair_requested"
    pickup = "pickup"
    released = "released"
    warned = "warned"
    blocked = "blocked"
    failed = "failed"
    cancelled = "cancelled"
    budget_exhausted = "budget_exhausted"


TERMINAL_VERIFICATION_STATES = frozenset(
    {
        VerificationState.released,
        VerificationState.warned,
        VerificationState.blocked,
        VerificationState.failed,
        VerificationState.cancelled,
        VerificationState.budget_exhausted,
    }
)


_VERIFICATION_TRANSITIONS: dict[VerificationState, frozenset[VerificationState]] = {
    VerificationState.not_requested: frozenset({VerificationState.executing, VerificationState.cancelled}),
    VerificationState.executing: frozenset(
        {
            VerificationState.candidate_ready,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }
    ),
    VerificationState.candidate_ready: frozenset(
        {
            VerificationState.verifying,
            VerificationState.released,
            VerificationState.warned,
            VerificationState.blocked,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
        }
    ),
    VerificationState.verifying: frozenset(
        {
            VerificationState.repair_requested,
            VerificationState.released,
            VerificationState.warned,
            VerificationState.blocked,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }
    ),
    VerificationState.repair_requested: frozenset(
        {
            VerificationState.executing,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }
    ),
    VerificationState.pickup: frozenset(
        {
            VerificationState.executing,
            VerificationState.candidate_ready,
            VerificationState.verifying,
            VerificationState.repair_requested,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }
    ),
    VerificationState.released: frozenset(),
    VerificationState.warned: frozenset(),
    VerificationState.blocked: frozenset(),
    VerificationState.failed: frozenset(),
    VerificationState.cancelled: frozenset(),
    VerificationState.budget_exhausted: frozenset(),
}


class InvalidVerificationTransitionError(ValueError):
    """Raised when a verification phase change violates the transition table."""

    def __init__(self, from_state: VerificationState, to_state: VerificationState) -> None:
        super().__init__(f"invalid VerificationState transition: {from_state.value} → {to_state.value}")
        self.from_state = from_state
        self.to_state = to_state


def assert_verification_transition(from_state: VerificationState, to_state: VerificationState) -> None:
    """Raise if a durable verification phase change is not legal.

    Reapplying the current phase is explicitly valid so a redelivered lifecycle
    command is idempotent.
    """
    if to_state is from_state:
        return
    if to_state not in _VERIFICATION_TRANSITIONS[from_state]:
        raise InvalidVerificationTransitionError(from_state, to_state)


def is_terminal_verification_state(state: VerificationState) -> bool:
    """Return whether no further lifecycle phase may be entered."""
    return state in TERMINAL_VERIFICATION_STATES


class RequirementsManifest(BaseModel):
    """Frozen, caller-supplied requirements attached to a candidate."""

    model_config = ConfigDict(frozen=True)

    revision: str
    requirements: Mapping[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("requirements", mode="after")
    @classmethod
    def _freeze_requirements(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _canonical_frozen_object(value, field_name="requirements")

    @field_serializer("requirements")
    def _serialize_requirements(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_object(value)


class ArtifactDeclaration(BaseModel):
    """Stable identity and integrity data for one candidate artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    reference: str
    media_type: str
    size_bytes: int = Field(ge=0)
    digest: str


class CitationSpan(BaseModel):
    """Structured relation between a claim and a cited evidence reference."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    evidence_record_id: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_offsets(self) -> CitationSpan:
        if self.end_offset < self.start_offset:
            raise ValueError("citation span end_offset must not precede start_offset")
        return self


class RunTreeOrigin(BaseModel):
    """The durable, typed run-tree location at which evidence was observed.

    ``parent_run_id`` and ``subagent_id`` preserve the immediate producer for
    attribution.  ``root_run_id`` binds that producer to the complete run tree,
    so verification can accept evidence from arbitrarily nested descendants
    without inferring ancestry from text or querying process-local state.

    ``depth`` states how far below the root the producer sat.  The ids cannot
    answer that between them: they name one hop and the root, never the
    distance separating the two.  Within a single run the tree is still
    present to be walked, but evidence outlives the run that observed it, and
    a reader holding a session's worth of it has only these records — so the
    distance is recorded where it is known rather than reconstructed later
    from ids that were never a chain.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    root_run_id: str
    depth: int
    parent_run_id: str | None = None
    subagent_id: str | None = None

    @model_validator(mode="after")
    def _validate_run_tree_binding(self) -> RunTreeOrigin:
        if not self.run_id or self.run_id.strip() != self.run_id:
            raise ValueError("evidence origin run_id must be a non-padded, non-empty identifier")
        if not self.root_run_id or self.root_run_id.strip() != self.root_run_id:
            raise ValueError(
                "evidence origin root_run_id must be a non-padded, non-empty identifier"
            )
        if self.depth < 0:
            raise ValueError("evidence origin depth must not be negative")
        # Depth and the ids describe one position, so they are checked against
        # each other here.  A depth that disagrees with the ids is the signature
        # of a second, independent account of where the run sat, and this field
        # exists precisely to be trusted once the tree is gone.
        if self.run_id == self.root_run_id:
            if self.parent_run_id is not None or self.subagent_id is not None:
                raise ValueError("root evidence origin must not declare an immediate parent")
            if self.depth != 0:
                raise ValueError("root evidence origin must declare depth 0")
        elif (
            not self.parent_run_id
            or self.parent_run_id.strip() != self.parent_run_id
            or not self.subagent_id
            or self.subagent_id.strip() != self.subagent_id
        ):
            raise ValueError(
                "descendant evidence origin must declare non-padded, non-empty immediate parent and subagent_id"
            )
        elif self.parent_run_id == self.run_id:
            raise ValueError("descendant evidence origin must not declare itself as immediate parent")
        elif self.depth == 0:
            raise ValueError("descendant evidence origin must declare a depth below the root")
        return self

    def belongs_to_root(self, run_id: str) -> bool:
        """Return whether this origin belongs to the given durable root run."""
        return self.root_run_id == run_id


class EvidenceRecord(BaseModel):
    """One runtime-authored observation from a trusted producer."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    origin: RunTreeOrigin
    producer_id: str
    producer_revision: str
    subject_id: str
    subject_reference: str
    digest: str
    observed_at: datetime = Field(default_factory=_utcnow)
    metadata: Mapping[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _canonical_frozen_object(value, field_name="evidence metadata")

    @field_serializer("metadata")
    def _serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_json_object(value)


class EvidenceProducerBinding(BaseModel):
    """Trusted producer identity attached to one registered tool revision.

    The runtime, rather than a tool result, owns this binding.  A deployment
    chooses which registered tools receive one; core merely verifies and
    stamps the binding during dispatch.
    """

    model_config = ConfigDict(frozen=True)

    producer_id: str
    producer_revision: str

    @field_validator("producer_id", "producer_revision")
    @classmethod
    def _require_identifier(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)


class InheritedEvidencePrefix(BaseModel):
    """Evidence a ledger opened on top of, vouched for by orchestration.

    Core does not know where a prefix comes from or what groups the runs that
    produced it.  It knows the ids it must accept a citation against, and it
    carries a digest of the content behind them that it cannot check itself,
    because it does not hold that content.  The digest is what makes the
    reference falsifiable by whoever does hold it; for evidence this ledger
    never observed, that is the most core can offer.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    record_ids: tuple[str, ...]
    digest: str

    @field_validator("source_id", "digest")
    @classmethod
    def _require_identifier(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _require_unique_nonempty_record_ids(self) -> InheritedEvidencePrefix:
        if not self.record_ids:
            raise ValueError("inherited evidence prefix must reference at least one record")
        for record_id in self.record_ids:
            _require_nonempty_identifier(record_id, field_name="inherited evidence record_ids")
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("inherited evidence prefix contains duplicate record ids")
        return self


class EvidenceLedger(BaseModel):
    """Append-only trusted evidence retained independently of model history."""

    model_config = ConfigDict(frozen=True)

    ledger_id: str
    # An open ledger belongs to the exact engine attempt that created it.  The
    # binding is independent of its records so a ledger cannot be moved to a
    # sibling before the first observation is appended.
    attempt_owner: RunTreeOrigin
    records: tuple[EvidenceRecord, ...] = ()
    # Evidence this ledger opened on top of rather than observed.  Its records
    # were produced outside this run tree, so they cannot enter ``records``
    # without breaking the owner binding above; carrying them by reference
    # keeps the binding exact and still lets a candidate cite them.
    inherited: InheritedEvidencePrefix | None = None

    @model_validator(mode="after")
    def _require_unique_record_ids(self) -> EvidenceLedger:
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence ledger contains duplicate record ids")
        # Membership in the owner's run tree, not equality with the owner: a
        # ledger holds what its own tree observed, including descendants.
        if any(not record.origin.belongs_to_root(self.attempt_owner.root_run_id) for record in self.records):
            raise ValueError("evidence ledger records are outside the attempt owner's run tree")
        # Tree membership compares the root and nothing else, so on its own it
        # would admit two records naming one run at two different positions.
        # A run occupies one position, and a sealed ledger that answers "who
        # produced this" two ways cannot be reconciled after the fact.
        positions: dict[str, RunTreeOrigin] = {self.attempt_owner.run_id: self.attempt_owner}
        for record in self.records:
            if positions.setdefault(record.origin.run_id, record.origin) != record.origin:
                raise ValueError("evidence ledger states conflicting origins for one run")
        if self.inherited is not None and set(self.inherited.record_ids) & set(record_ids):
            raise ValueError("inherited evidence prefix overlaps this ledger's own records")
        return self

    @property
    def digest(self) -> str:
        """Return a stable digest over the ordered immutable evidence records."""
        content = self.model_dump(mode="json")
        if self.inherited is None:
            # A ledger that opened on nothing digests over its own records
            # alone: an absent component is not content.  Serialising it as a
            # null would tie the digest to the model's field list rather than
            # to what the ledger holds, so introducing an optional facet would
            # stop every reference already issued from matching content that
            # never changed.
            del content["inherited"]
        payload = json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def append(self, record: EvidenceRecord) -> EvidenceLedger:
        """Return a new ledger with one record appended exactly once."""
        if any(existing.record_id == record.record_id for existing in self.records):
            raise ValueError(f"evidence record already exists: {record.record_id}")
        if not record.origin.belongs_to_root(self.attempt_owner.root_run_id):
            raise ValueError("evidence record origin is outside the attempt owner's run tree")
        # ``append`` returns via ``model_copy``, which skips the model
        # validator, so the one-position-per-run rule is checked by hand here
        # exactly as the duplicate-id rule above is.
        if any(
            known.run_id == record.origin.run_id and known != record.origin
            for known in (self.attempt_owner, *(existing.origin for existing in self.records))
        ):
            raise ValueError("evidence record origin conflicts with a recorded origin for the same run")
        if self.inherited is not None and record.record_id in set(self.inherited.record_ids):
            raise ValueError(f"evidence record already exists: {record.record_id}")
        return self.model_copy(update={"records": (*self.records, record)})


class EvidenceLedgerReference(BaseModel):
    """Integrity-pinned reference to the evidence ledger used by a candidate."""

    model_config = ConfigDict(frozen=True)

    ledger_id: str
    digest: str


class CandidateBundle(BaseModel):
    """Sealed candidate output and declarations supplied to verification."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    run_id: str
    generation_attempt: int = Field(ge=0)
    content_reference: str | None = None
    # Candidates contain only reader-facing blocks with scalar immutable fields.
    # Operational tool blocks retain mutable metadata by contract and belong to
    # run history, not to a sealed deliverable.
    content_blocks: tuple[TextBlock | ImageRefBlock, ...] = ()
    artifacts: tuple[ArtifactDeclaration, ...] = ()
    requirements: RequirementsManifest
    evidence_ledger: EvidenceLedgerReference
    citations: tuple[CitationSpan, ...] = ()
    delivery_mode: DeliveryMode
    reader_capabilities: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _require_deliverable_content(self) -> CandidateBundle:
        if not self.content_reference and not self.content_blocks and not self.artifacts:
            raise ValueError("candidate must declare content or an artifact")
        if any(
            isinstance(block, TextBlock) and block.visibility in {BlockVisibility.HIDDEN, BlockVisibility.DEBUG}
            for block in self.content_blocks
        ):
            raise ValueError("candidate must not contain non-reader-visible text blocks")
        return self


class VerificationCheckSelector(BaseModel):
    """An exact check implementation requested by a verification profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    revision: str
    accepted_artifact_types: tuple[str, ...] = ()

    @field_validator("check_id", "revision")
    @classmethod
    def _validate_identifiers(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)

    @field_validator("accepted_artifact_types")
    @classmethod
    def _validate_accepted_artifact_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("accepted artifact types must be unique")
        return tuple(
            _require_nonempty_identifier(artifact_type, field_name="accepted_artifact_types") for artifact_type in value
        )


class VerificationExecutionBindingReference(BaseModel):
    """Opaque identity of one immutable verification execution binding.

    Admission owns materialising the binding.  The core deliberately retains
    only stable identifiers and fingerprints: it must never contain provider
    settings, endpoints, credentials, or provider-specific configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    binding_id: str
    binding_digest: str
    provider_id: str
    provider_contract_revision: str

    @field_validator("binding_id", "binding_digest", "provider_id", "provider_contract_revision")
    @classmethod
    def _validate_identifiers(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)


class ResolvedVerificationCheck(BaseModel):
    """Immutable runtime binding for one check selected at admission.

    The identifiers are deliberately opaque.  The runtime records the exact
    executable surface it resolved, while domain owners remain responsible for
    assigning meaning to the check and its settings.

    ``settings_digest`` is the one identifier here that may be empty.  A check
    needing no configuration to run has no settings to fingerprint, and that is
    the ordinary shape of a check rather than a degenerate one: refusing the
    empty value would leave "declares nothing to configure" as the only
    executable surface that cannot be bound.  Emptiness is therefore a
    statement — this check declares no settings — and not a fact that failed to
    arrive.  A caller that holds settings it could not fingerprint has not
    resolved the check and must decline to build one rather than report the
    absence it does not have; that is why the distinction needs no second
    spelling here.  The field stays required for the same reason: absence is
    stated by whoever resolved the check, never defaulted in on their behalf.

    Every other identifier names something that exists whenever resolution
    succeeded at all, so an empty one there describes a resolution that did not
    happen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    selector_revision: str
    accepted_artifact_types: tuple[str, ...] = ()
    tool_id: str
    tool_revision: str
    tool_schema_digest: str
    settings_digest: str
    capability_contract_revision: str
    execution_binding: VerificationExecutionBindingReference
    deterministic: StrictBool
    idempotent: StrictBool

    @field_validator(
        "check_id",
        "selector_revision",
        "tool_id",
        "tool_revision",
        "tool_schema_digest",
        "capability_contract_revision",
    )
    @classmethod
    def _validate_identifiers(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)

    @field_validator("settings_digest")
    @classmethod
    def _validate_settings_digest(cls, value: str) -> str:
        return _allow_absent_identifier(value, field_name="settings_digest")

    @field_validator("accepted_artifact_types")
    @classmethod
    def _validate_accepted_artifact_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("accepted artifact types must be unique")
        return tuple(
            _require_nonempty_identifier(artifact_type, field_name="accepted_artifact_types") for artifact_type in value
        )


class VerificationExecutionPlan(BaseModel):
    """Frozen ordered executable surface for one resolved verification binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checks: tuple[ResolvedVerificationCheck, ...]

    @model_validator(mode="after")
    def _require_unique_check_ids(self) -> VerificationExecutionPlan:
        check_ids = [check.check_id for check in self.checks]
        if not check_ids:
            raise ValueError("verification execution plan must contain at least one check")
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("verification execution plan contains duplicate check ids")
        return self

    @property
    def digest(self) -> str:
        """Return a deterministic integrity fingerprint for the resolved plan."""
        return _canonical_digest(self.model_dump(mode="json"))

    def snapshot(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible form persisted with a run."""
        return self.model_dump(mode="json")


class ResolvedVerificationBinding(BaseModel):
    """Admission-time immutable binding between a policy revision and a plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    profile_revision: str
    profile_checks: tuple[VerificationCheckSelector, ...]
    execution_plan: VerificationExecutionPlan

    @field_validator("profile_id", "profile_revision")
    @classmethod
    def _validate_profile_identifiers(cls, value: str, info: Any) -> str:
        return _require_nonempty_identifier(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _require_exact_profile_plan_correspondence(self) -> ResolvedVerificationBinding:
        selectors = self.profile_checks
        resolved_checks = self.execution_plan.checks
        if not selectors:
            raise ValueError("verification binding must retain at least one profile check")
        if len(selectors) != len(resolved_checks):
            raise ValueError("verification binding profile checks and execution plan must have equal length")

        selector_ids = [selector.check_id for selector in selectors]
        if len(selector_ids) != len(set(selector_ids)):
            raise ValueError("verification binding contains duplicate profile check ids")

        for selector, resolved in zip(selectors, resolved_checks, strict=True):
            if selector.check_id != resolved.check_id:
                raise ValueError("verification binding selector and execution plan check order differs")
            if selector.revision != resolved.selector_revision:
                raise ValueError("verification binding selector revision does not match resolved check")
            if selector.revision != resolved.tool_revision:
                raise ValueError("verification binding selector revision does not match resolved tool")
            if selector.accepted_artifact_types != resolved.accepted_artifact_types:
                raise ValueError("verification binding accepted artifact types do not match resolved check")
        return self

    @classmethod
    def from_profile(
        cls,
        profile: VerificationProfile,
        execution_plan: VerificationExecutionPlan,
    ) -> ResolvedVerificationBinding:
        """Bind an exact immutable profile selection to its resolved tools."""
        return cls(
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            profile_checks=profile.checks,
            execution_plan=execution_plan,
        )

    @property
    def digest(self) -> str:
        """Return a deterministic integrity fingerprint for the complete binding."""
        return _canonical_digest(self.model_dump(mode="json"))

    def snapshot(self) -> dict[str, Any]:
        """Return a self-authenticating JSON-compatible persisted envelope."""
        return {"binding": self.model_dump(mode="json"), "digest": self.digest}

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> ResolvedVerificationBinding:
        """Restore a persisted binding only if its admission digest still matches."""
        if set(snapshot) != {"binding", "digest"}:
            raise ValueError("verification binding snapshot must contain exactly binding and digest")
        persisted_digest = snapshot["digest"]
        if not isinstance(persisted_digest, str):
            raise ValueError("verification binding snapshot digest must be a string")
        payload = snapshot["binding"]
        if not isinstance(payload, Mapping):
            raise ValueError("verification binding snapshot binding must be an object")
        binding = cls.model_validate(payload)
        if binding.digest != persisted_digest:
            raise ValueError("verification binding snapshot integrity digest does not match")
        return binding


class VerificationBudget(BaseModel):
    """Explicit resource ceilings selected by the caller's profile."""

    model_config = ConfigDict(frozen=True)

    max_repair_cycles: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    max_duration_ms: int = Field(ge=0)
    max_cost_microunits: int = Field(ge=0)
    max_concurrency: int = Field(ge=1)


class VerificationProfile(BaseModel):
    """Run-frozen policy and check selection; domain checks remain external."""

    model_config = ConfigDict(frozen=True)

    profile_id: str
    revision: str
    enabled: StrictBool
    mode: VerificationMode
    delivery: VerificationDelivery
    checks: tuple[VerificationCheckSelector, ...] = ()
    budget: VerificationBudget

    @model_validator(mode="after")
    def _require_unique_check_ids(self) -> VerificationProfile:
        check_ids = [check.check_id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("verification profile contains duplicate check ids")
        return self


class VerificationFinding(BaseModel):
    """Machine-readable observation reported by one check."""

    model_config = ConfigDict(frozen=True)

    code: str
    severity: VerificationSeverity
    evidence_record_ids: tuple[str, ...] = ()
    citation_spans: tuple[CitationSpan, ...] = ()
    repair_hint: str | None = None


class VerificationResourceUse(BaseModel):
    """Measured resource use from a check invocation."""

    model_config = ConfigDict(frozen=True)

    tokens: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    cost_microunits: int = Field(ge=0)


class VerificationCheckResult(BaseModel):
    """Typed outcome of one exact check revision."""

    model_config = ConfigDict(frozen=True)

    check_id: str
    revision: str
    status: VerificationCheckStatus
    severity: VerificationSeverity
    findings: tuple[VerificationFinding, ...] = ()
    deterministic: StrictBool
    idempotent: StrictBool
    resource_use: VerificationResourceUse


class VerificationReport(BaseModel):
    """Aggregate machine decision for a candidate verification attempt."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    candidate_id: str
    profile_id: str
    profile_revision: str
    results: tuple[VerificationCheckResult, ...]
    decision: ReleaseDecision


class CandidateReleasedProjection(BaseModel):
    """One atomic reader-facing projection of an approved sealed candidate.

    This is deliberately not a delayed transcription of LLM stream frames.
    It contains only the immutable reader-facing blocks and declared artifacts
    from the sealed candidate plus the durable decision that authorized them.
    A stream adapter supplies an idempotency key when publishing the associated
    :class:`~protocore.runtime.events.TurnEvent`.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    report_id: str
    decision: ReleaseDecision
    content_reference: str | None = None
    content_blocks: tuple[TextBlock | ImageRefBlock, ...] = ()
    artifacts: tuple[ArtifactDeclaration, ...] = ()
    citations: tuple[CitationSpan, ...] = ()
    evidence_ledger: EvidenceLedgerReference
    delivery_mode: DeliveryMode

    @model_validator(mode="after")
    def _require_reader_delivery_decision(self) -> CandidateReleasedProjection:
        if self.decision not in {ReleaseDecision.release, ReleaseDecision.warn}:
            raise ValueError("candidate release projection requires a release or warn decision")
        if any(
            isinstance(block, TextBlock) and block.visibility in {BlockVisibility.HIDDEN, BlockVisibility.DEBUG}
            for block in self.content_blocks
        ):
            raise ValueError("candidate release projection must not contain non-reader-visible text blocks")
        return self

    @classmethod
    def from_lifecycle(cls, lifecycle: VerificationLifecycle) -> CandidateReleasedProjection:
        """Build a projection only from a terminal release/warn lifecycle."""
        if lifecycle.state not in {VerificationState.released, VerificationState.warned}:
            raise ValueError("candidate release projection requires a released or warned lifecycle")
        candidate = lifecycle.candidate
        report = lifecycle.report
        if candidate is None or report is None:  # Defensive; lifecycle validates this.
            raise ValueError("candidate release projection requires candidate and report")
        return cls(
            candidate_id=candidate.candidate_id,
            report_id=report.report_id,
            decision=report.decision,
            content_reference=candidate.content_reference,
            content_blocks=candidate.content_blocks,
            artifacts=candidate.artifacts,
            citations=candidate.citations,
            evidence_ledger=candidate.evidence_ledger,
            delivery_mode=candidate.delivery_mode,
        )

    def event_payload(self) -> dict[str, Any]:
        """Return the stable JSON payload for a reader-facing projection."""
        return self.model_dump(mode="json", exclude_none=True)


class VerificationLifecycle(BaseModel):
    """Single typed snapshot payload for optional verification runtime state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: VerificationState = VerificationState.not_requested
    ledger: EvidenceLedger | None = None
    candidate: CandidateBundle | None = None
    report: VerificationReport | None = None
    repair_cycles: int = Field(default=0, ge=0)
    restore_error: str | None = None

    @model_validator(mode="after")
    def _validate_semantic_invariants(self) -> VerificationLifecycle:
        if self.candidate is not None and self.ledger is None:
            raise ValueError("lifecycle candidate and ledger must be present together")

        # Evidence is collected before a candidate is sealed.  Pickup is the
        # durable pause/recovery representation for an interrupted execution,
        # so it must retain an open ledger without making it appendable until
        # the lifecycle returns to ``executing``.
        if self.candidate is None and self.ledger is not None and self.state not in {
            VerificationState.executing,
            VerificationState.pickup,
        }:
            raise ValueError("an unsealed evidence ledger is valid only while executing or in pickup")
        if self.candidate is not None and self.state is VerificationState.executing:
            raise ValueError("executing lifecycle must not contain a sealed candidate")

        candidate_required = {
            VerificationState.candidate_ready,
            VerificationState.verifying,
            VerificationState.repair_requested,
            VerificationState.released,
            VerificationState.warned,
            VerificationState.blocked,
        }
        if self.state in candidate_required and self.candidate is None:
            raise ValueError(f"lifecycle state {self.state.value} requires candidate and ledger")

        report_required = {
            VerificationState.repair_requested,
            VerificationState.released,
            VerificationState.warned,
            VerificationState.blocked,
        }
        report_forbidden = {
            VerificationState.not_requested,
            VerificationState.executing,
            VerificationState.candidate_ready,
            VerificationState.verifying,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }
        if self.state in report_required and self.report is None:
            raise ValueError(f"lifecycle state {self.state.value} requires a report")
        if self.state in report_forbidden and self.report is not None:
            raise ValueError(f"lifecycle state {self.state.value} forbids a report")

        if self.candidate is not None and self.ledger is not None:
            reference = self.candidate.evidence_ledger
            if reference.ledger_id != self.ledger.ledger_id or reference.digest != self.ledger.digest:
                raise ValueError("candidate evidence ledger reference does not match lifecycle ledger")
            # A citation is answerable against anything the ledger can produce,
            # which is what it observed plus what it opened on top of.
            record_ids = {record.record_id for record in self.ledger.records}
            if self.ledger.inherited is not None:
                record_ids |= set(self.ledger.inherited.record_ids)
            _require_known_evidence_record_ids(
                (span.evidence_record_id for span in self.candidate.citations),
                record_ids,
                context="candidate citation",
            )
        if self.report is not None:
            if self.candidate is None:
                raise ValueError("verification report requires a candidate")
            if self.report.candidate_id != self.candidate.candidate_id:
                raise ValueError("verification report candidate does not match lifecycle candidate")
            _require_known_evidence_record_ids(
                (
                    evidence_record_id
                    for result in self.report.results
                    for finding in result.findings
                    for evidence_record_id in finding.evidence_record_ids
                ),
                record_ids,
                context="verification finding evidence",
            )
            _require_known_evidence_record_ids(
                (
                    span.evidence_record_id
                    for result in self.report.results
                    for finding in result.findings
                    for span in finding.citation_spans
                ),
                record_ids,
                context="verification finding citation",
            )

        terminal_decisions = {
            VerificationState.released: ReleaseDecision.release,
            VerificationState.warned: ReleaseDecision.warn,
            VerificationState.blocked: ReleaseDecision.block,
        }
        required_decision = terminal_decisions.get(self.state)
        if required_decision is not None and self.report is not None:
            if self.report.decision is not required_decision:
                raise ValueError(f"lifecycle state {self.state.value} requires {required_decision.value} decision")
        if self.restore_error is not None and self.state is not VerificationState.failed:
            raise ValueError("restore_error is valid only for a failed lifecycle")
        return self

    def transition_to(self, state: VerificationState) -> VerificationLifecycle:
        """Return the next legal lifecycle state without mutating prior state."""
        assert_verification_transition(self.state, state)
        payload = {**self.model_dump(), "state": state}
        if state is VerificationState.executing and (
            self.state is VerificationState.repair_requested or self.candidate is not None
        ):
            # A repair, or a pickup made after a candidate was sealed, starts a
            # new attempt. Retaining a sealed candidate while claiming execution
            # would make later evidence append ambiguous. An open-ledger pickup
            # is intentionally not cleared: it resumes the interrupted attempt.
            payload["ledger"] = None
            payload["candidate"] = None
        if state in {
            VerificationState.not_requested,
            VerificationState.executing,
            VerificationState.candidate_ready,
            VerificationState.verifying,
            VerificationState.pickup,
            VerificationState.failed,
            VerificationState.cancelled,
            VerificationState.budget_exhausted,
        }:
            payload["report"] = None
        return self.model_validate(payload)

    def terminalize(self, report: VerificationReport) -> VerificationLifecycle:
        """Atomically attach a report and enter its matching terminal decision state."""
        terminal_state = {
            ReleaseDecision.release: VerificationState.released,
            ReleaseDecision.warn: VerificationState.warned,
            ReleaseDecision.block: VerificationState.blocked,
        }[report.decision]
        assert_verification_transition(self.state, terminal_state)
        return self.model_validate({**self.model_dump(), "state": terminal_state, "report": report})

    def snapshot(self) -> dict[str, Any]:
        """Produce deterministic JSON-compatible state for a run snapshot."""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_snapshot(cls, value: object) -> VerificationLifecycle:
        """Restore typed lifecycle state; malformed durable data fails closed."""
        if not isinstance(value, dict):
            return cls(
                state=VerificationState.failed,
                restore_error="verification snapshot is not an object",
            )
        if "state" not in value:
            return cls(
                state=VerificationState.failed,
                restore_error="verification snapshot is missing state",
            )
        try:
            return cls.model_validate(value)
        except ValidationError:
            return cls(
                state=VerificationState.failed,
                restore_error="verification snapshot failed validation",
            )


def _require_known_evidence_record_ids(
    referenced_ids: Iterable[str],
    record_ids: set[str],
    *,
    context: str,
) -> None:
    """Reject references that do not belong to the attached immutable ledger."""
    unknown_ids = set(referenced_ids) - record_ids
    if unknown_ids:
        raise ValueError(f"{context} references unknown evidence record ids")


__all__ = [
    "TERMINAL_VERIFICATION_STATES",
    "ArtifactDeclaration",
    "CandidateBundle",
    "CandidateReleasedProjection",
    "CitationSpan",
    "DeliveryMode",
    "EvidenceLedger",
    "EvidenceLedgerReference",
    "EvidenceRecord",
    "InheritedEvidencePrefix",
    "InvalidVerificationTransitionError",
    "ReleaseDecision",
    "RequirementsManifest",
    "ResolvedVerificationBinding",
    "ResolvedVerificationCheck",
    "RunTreeOrigin",
    "VerificationBudget",
    "VerificationCheckResult",
    "VerificationCheckSelector",
    "VerificationCheckStatus",
    "VerificationDelivery",
    "VerificationExecutionBindingReference",
    "VerificationExecutionPlan",
    "VerificationFinding",
    "VerificationLifecycle",
    "VerificationMode",
    "VerificationProfile",
    "VerificationReport",
    "VerificationResourceUse",
    "VerificationSeverity",
    "VerificationState",
    "assert_verification_transition",
    "is_terminal_verification_state",
]
