"""Attempt ledger types and runtime model.

The attempt ledger records what artifacts a subagent attempt **declared** it
created or modified, and independently verifies their existence on disk before
the run finalizes. This closes a finalization gap: a
subagent that succeeds at writing the deliverable but exits via
``max_iterations`` without calling ``SubmitAnswer`` would otherwise be
classified as failed even though the artifact is on disk.

The ledger is mutated by the orchestrator during a run, then serialized into
the ExecutionReport for observability. It is per-run state and never
crosses run boundaries.

"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator

DeliverableKind = Literal["file", "directory", "artifact_id"]
"""What kind of thing a declared deliverable is."""


SelfReportedStatus = Literal["success", "partial", "failure", "unknown"]
"""Subagent's own self-classification, recorded but not trusted blindly."""


RuntimeAttemptStatus = Literal[
    "succeeded",
    "failed_max_iter",
    "failed_provider",
    "failed_other",
]
"""Runtime's auto-classification of an attempt (orthogonal to self-report)."""


LedgerOutcome = Literal["completed", "partial", "failed", "unknown"]
"""Final outcome computed from declarations + verifications."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DeliverableDeclaration(BaseModel):
    """A file/directory/artifact a subagent claims to have created or modified.

    Declarations are produced either explicitly by the subagent (via the
    ``declared_deliverables`` field on ``SubagentResult``) or inferred by the
    orchestrator from ``files_changed``. The orchestrator independently
    verifies each declared deliverable via workspace_stat before finalization.
    """

    path: str = Field(min_length=1, description="Workspace-relative path or artifact id.")
    kind: DeliverableKind = "file"
    required: bool = Field(
        default=True,
        description=(
            "If true, the run cannot be 'completed' unless this deliverable "
            "verifies as existing on disk. Set false for optional outputs."
        ),
    )
    min_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Minimum acceptable size, when known in advance. Optional.",
    )
    sha256_expected: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="Expected content hash, when known in advance. Optional.",
    )
    declared_by_agent: str = Field(
        default="",
        description="Agent id that produced the declaration. Filled by runtime.",
    )
    declared_at: datetime = Field(default_factory=_utc_now)
    summary: str | None = Field(
        default=None,
        max_length=400,
        description="Optional short description of what this deliverable contains.",
    )

    model_config = {"extra": "forbid"}


class VerificationRecord(BaseModel):
    """Result of independently verifying a declared deliverable.

    Verification is done by the orchestrator, not the declaring agent — that
    is the whole point. ``exists`` is the load-bearing field: it answers the
    question "is the artifact actually on disk".
    """

    when: datetime = Field(default_factory=_utc_now)
    deliverable_path: str = Field(min_length=1)
    exists: bool
    size_bytes: int | None = Field(default=None, ge=0)
    content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    valid_by_schema: bool | None = Field(
        default=None,
        description=(
            "Optional content validation result (e.g. HTML parses, JSON parses). "
            "None means no schema-level check was attempted."
        ),
    )
    schema_kind: str | None = Field(
        default=None,
        description="Identifier of the schema validator used (e.g. 'html', 'json').",
    )
    verifier_id: str = Field(
        default="workspace_stat",
        description="Identifier of who/what performed the verification.",
    )
    error: str | None = Field(
        default=None,
        max_length=500,
        description="Verification error message when the check itself failed.",
    )

    model_config = {"extra": "forbid"}


class AttemptRecord(BaseModel):
    """One subagent attempt within a run.

    Captures both runtime classification ('did this attempt run to completion')
    and self-reported classification ('did the subagent itself say success').
    These two are orthogonal: a subagent can write the file then run out of
    iterations (self_reported=success, runtime_status=failed_max_iter).
    """

    agent_id: str = Field(min_length=1)
    parent_call_id: str | None = None
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None
    iteration_count: int = Field(default=0, ge=0)
    self_reported_status: SelfReportedStatus = "unknown"
    runtime_status: RuntimeAttemptStatus | None = None
    deliverables_touched: list[str] = Field(default_factory=list)
    failure_class: str | None = None

    model_config = {"extra": "forbid"}


class AttemptLedger(BaseModel):
    """Per-run ledger of declarations, attempts, and verifications.

    Mutated by the orchestrator during a run. Serialized into the
    ExecutionReport at the end. Used by the finalization gate to decide the
    run's outcome based on artifact existence rather than just on whether the
    last subagent called SubmitAnswer.
    """

    run_id: str = Field(min_length=1)
    declared_deliverables: dict[str, DeliverableDeclaration] = Field(default_factory=dict)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    verifications: list[VerificationRecord] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    #: Transient per-path ``(group, batch_index)`` tracking for :meth:`declare`.
    #: Not part of the serialized ledger. The leader's ledger is created once per
    #: run and mutated IN PLACE (it is not snapshot/restore round-tripped), so
    #: this map persists in-memory across leader turns within one hot run — which
    #: is why resolution is keyed by ``group``: each concurrent fan-out group
    #: carries a distinct id, so a LATER group's same-path declaration always
    #: overrides an earlier group's (matching cross-turn last-writer-wins),
    #: while WITHIN a group the highest batch index wins (deterministic batch
    #: order). Only a cold resume (fresh ``AttemptLedger`` via ``model_validate``)
    #: resets it — and ``declared_deliverables`` survives that round-trip, so a
    #: post-resume group correctly overrides the pre-resume declaration.
    _declared_order: dict[str, tuple[str, int]] = PrivateAttr(default_factory=dict)

    @field_validator("declared_deliverables", mode="before")
    @classmethod
    def _coerce_declarations(cls, value: object) -> object:
        # Accept list-of-DeliverableDeclaration input (the natural shape on the
        # wire from a subagent's SubmitAnswer) and convert to path-keyed dict.
        if isinstance(value, list):
            result: dict[str, object] = {}
            for entry in value:
                if isinstance(entry, DeliverableDeclaration):
                    result[entry.path] = entry
                elif isinstance(entry, dict):
                    path = entry.get("path")
                    if isinstance(path, str) and path:
                        result[path] = entry
            return result
        return value

    def declare(
        self,
        declaration: DeliverableDeclaration,
        *,
        order: int | None = None,
        group: str | None = None,
    ) -> None:
        """Record a declared deliverable.

        ``order is None`` (serial / single-child / leader-inferred path):
        last-writer-wins — a later declaration for the same path replaces the
        earlier one, unchanged.

        ``order`` set (a child in a concurrently-dispatched delegation group,
        carrying its 0-based position in the LLM-requested batch, plus the
        group's stable ``group`` id): resolution is scoped PER GROUP —

        * a declaration from a DIFFERENT group than the one currently holding the
          path always wins (groups run sequentially in time, so a different group
          is a later one — this matches the cross-turn last-writer-wins the
          serial path gives); and
        * WITHIN the same group, the HIGHEST batch index wins — i.e. as if the
          group's declarations were applied sequentially in batch order,
          regardless of which child's ``gather`` coroutine finishes (declares)
          first.

        This makes the surviving declaration's provenance deterministic for the
        unusual case of ≥2 parallel siblings declaring the SAME path, without an
        earlier turn's group freezing out a later turn's narrower group.
        """
        path = declaration.path
        if order is None:
            self.declared_deliverables[path] = declaration
            self._declared_order.pop(path, None)
            return
        group_key = group or ""
        prev = self._declared_order.get(path)
        if prev is None or prev[0] != group_key or order >= prev[1]:
            self.declared_deliverables[path] = declaration
            self._declared_order[path] = (group_key, order)

    def record_attempt(self, record: AttemptRecord) -> None:
        self.attempts.append(record)

    def record_verification(self, record: VerificationRecord) -> None:
        self.verifications.append(record)

    def latest_verification_for(self, path: str) -> VerificationRecord | None:
        for record in reversed(self.verifications):
            if record.deliverable_path == path:
                return record
        return None

    def required_declarations(self) -> list[DeliverableDeclaration]:
        return [d for d in self.declared_deliverables.values() if d.required]

    def required_verified_exists(self) -> bool:
        """True iff every required declared deliverable has a verification
        with ``exists=true``. False if any required deliverable is missing or
        unverified.
        """
        for declaration in self.required_declarations():
            latest = self.latest_verification_for(declaration.path)
            if latest is None or not latest.exists:
                return False
        return True

    def required_verified_valid(self) -> bool:
        """True iff every required deliverable that has a schema check also
        passed it (None means no check was attempted; treated as not invalid).
        """
        for declaration in self.required_declarations():
            latest = self.latest_verification_for(declaration.path)
            if latest is None:
                return False
            if latest.valid_by_schema is False:
                return False
        return True

    def unresolved_required_count(self) -> int:
        """Number of required declarations that have not yet been verified as existing.

        A declaration is resolved when its latest verification has ``exists=True``.
        Unverified (no record) or explicitly ``exists=False`` records are unresolved.
        """
        count = 0
        for decl in self.declared_deliverables.values():
            if not decl.required:
                continue
            latest = self.latest_verification_for(decl.path)
            if latest is None or not latest.exists:
                count += 1
        return count

    def declarations_by_agent(self, agent_id: str) -> list[DeliverableDeclaration]:
        """All declarations made by a given agent, in insertion order."""
        return [
            d
            for d in self.declared_deliverables.values()
            if d.declared_by_agent == agent_id
        ]

    def any_self_reported_success(self) -> bool:
        return any(a.self_reported_status == "success" for a in self.attempts)

    def any_partial_deliverable_exists(self) -> bool:
        for declaration in self.declared_deliverables.values():
            latest = self.latest_verification_for(declaration.path)
            if latest is not None and latest.exists:
                return True
        return False

    def compute_outcome(self) -> LedgerOutcome:
        """Decide the run's outcome from declarations and verifications.

        Logic:
          - unknown: no declarations at all (pure-analysis tasks fall back to
            attempt-based classification owned by the orchestrator);
          - completed: at least one required declaration exists AND every
            required declaration verifies as existing and valid;
          - partial: any self-reported success AND at least one declared
            deliverable verifies as existing;
          - failed: anything else.

        When all declarations are optional (required=false),
        ``required_verified_exists()`` and ``required_verified_valid()`` both
        return vacuously True. We explicitly require at least one required
        declaration to be present before classifying ``completed`` so an
        all-optional run with nothing on disk is correctly demoted.
        """
        if not self.declared_deliverables:
            return "unknown"
        has_required = bool(self.required_declarations())
        if (
            has_required
            and self.required_verified_exists()
            and self.required_verified_valid()
        ):
            return "completed"
        # All-optional case: if everything optional was found, treat as completed.
        # Mixed case where the optional half exists but the required half does not
        # falls through to partial/failed below.
        if (
            not has_required
            and self.declared_deliverables
            and all(
                (latest := self.latest_verification_for(d.path)) is not None
                and latest.exists
                and latest.valid_by_schema is not False
                for d in self.declared_deliverables.values()
            )
        ):
            return "completed"
        if self.any_self_reported_success() and self.any_partial_deliverable_exists():
            return "partial"
        return "failed"


__all__ = [
    "AttemptLedger",
    "AttemptRecord",
    "DeliverableDeclaration",
    "DeliverableKind",
    "LedgerOutcome",
    "RuntimeAttemptStatus",
    "SelfReportedStatus",
    "VerificationRecord",
]
