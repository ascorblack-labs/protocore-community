"""Tests for domain-neutral candidate verification contracts."""

from __future__ import annotations

import hashlib
import json

import pytest

from protocore.contracts.types import BlockVisibility, TextBlock, ToolResultBlock
from protocore.contracts.verification import (
    CandidateBundle,
    CitationSpan,
    DeliveryMode,
    EvidenceLedger,
    EvidenceLedgerReference,
    EvidenceRecord,
    InheritedEvidencePrefix,
    InvalidVerificationTransitionError,
    ReleaseDecision,
    RequirementsManifest,
    ResolvedVerificationBinding,
    ResolvedVerificationCheck,
    RunTreeOrigin,
    VerificationBudget,
    VerificationCheckResult,
    VerificationCheckSelector,
    VerificationCheckStatus,
    VerificationDelivery,
    VerificationExecutionBindingReference,
    VerificationExecutionPlan,
    VerificationFinding,
    VerificationLifecycle,
    VerificationMode,
    VerificationProfile,
    VerificationReport,
    VerificationResourceUse,
    VerificationSeverity,
    VerificationState,
    assert_verification_transition,
)


def _owner() -> RunTreeOrigin:
    """Return the root of the run tree the records below belong to."""
    return RunTreeOrigin(run_id="parent-1", root_run_id="parent-1", depth=0)


def _record() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="record-1",
        origin=RunTreeOrigin(
            run_id="run-1",
            root_run_id="parent-1",
            depth=1,
            parent_run_id="parent-1",
            subagent_id="child-1",
        ),
        producer_id="trusted-producer",
        producer_revision="rev-1",
        subject_id="subject-1",
        subject_reference="ref-1",
        digest="content-digest",
    )


def _candidate(ledger: EvidenceLedger) -> CandidateBundle:
    return CandidateBundle(
        candidate_id="candidate-1",
        run_id="run-1",
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )


def _report(candidate: CandidateBundle) -> VerificationReport:
    result = VerificationCheckResult(
        check_id="check-1",
        revision="revision-1",
        status=VerificationCheckStatus.passed,
        severity=VerificationSeverity.info,
        findings=(VerificationFinding(code="verified", severity=VerificationSeverity.info),),
        deterministic=True,
        idempotent=True,
        resource_use=VerificationResourceUse(tokens=0, duration_ms=0, cost_microunits=0),
    )
    return VerificationReport(
        report_id="report-1",
        candidate_id=candidate.candidate_id,
        profile_id="profile-1",
        profile_revision="profile-revision-1",
        results=(result,),
        decision=ReleaseDecision.release,
    )


def test_evidence_ledger_is_append_only_and_digest_is_stable() -> None:
    empty = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner())
    record = _record()
    ledger = empty.append(record)

    assert empty.records == ()
    assert ledger.records == (record,)
    assert ledger.digest == EvidenceLedger.model_validate(ledger.model_dump()).digest
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(record)
    with pytest.raises(Exception):
        ledger.ledger_id = "other"  # type: ignore[misc]


def test_evidence_metadata_is_deeply_immutable_and_canonical() -> None:
    source = {"nested": [{"number": 1}], "other": "value"}
    record = EvidenceRecord(
        record_id="record-1",
        origin=RunTreeOrigin(run_id="run-1", root_run_id="run-1", depth=0),
        producer_id="trusted-producer",
        producer_revision="rev-1",
        subject_id="subject-1",
        subject_reference="ref-1",
        digest="content-digest",
        metadata=source,
    )
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=record.origin, records=(record,))
    digest = ledger.digest

    source["nested"][0]["number"] = 2
    with pytest.raises(TypeError):
        record.metadata["other"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["nested"][0]["number"] = 2  # type: ignore[index]

    assert record.metadata == {"nested": ({"number": 1},), "other": "value"}
    assert ledger.digest == digest


def test_omitted_evidence_metadata_is_immutable() -> None:
    record = _record()
    digest = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner(), records=(record,)).digest

    with pytest.raises(TypeError):
        record.metadata["new"] = "value"  # type: ignore[index]

    assert EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner(), records=(record,)).digest == digest


def test_requirements_are_deeply_immutable_and_canonical() -> None:
    source = {"citation": {"required": True}, "formats": ["html"]}
    manifest = RequirementsManifest(revision="requirements-1", requirements=source)

    source["citation"]["required"] = False
    with pytest.raises(TypeError):
        manifest.requirements["citation"]["required"] = False  # type: ignore[index]
    with pytest.raises(AttributeError):
        manifest.requirements["formats"].append("pdf")  # type: ignore[union-attr]

    assert manifest.requirements == {"citation": {"required": True}, "formats": ("html",)}
    assert manifest.model_dump(mode="json")["requirements"] == {
        "citation": {"required": True},
        "formats": ["html"],
    }


def test_candidate_rejects_operational_content_blocks() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner())
    with pytest.raises(ValueError):
        CandidateBundle(
            candidate_id="candidate-1",
            run_id="run-1",
            generation_attempt=1,
            content_blocks=(ToolResultBlock(tool_call_id="call-1", content="output", metadata={"mutable": True}),),
            requirements=RequirementsManifest(revision="requirements-1"),
            evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
            delivery_mode=DeliveryMode.inline,
        )


@pytest.mark.parametrize("visibility", (BlockVisibility.HIDDEN, BlockVisibility.DEBUG))
def test_candidate_rejects_non_reader_visible_text_blocks(visibility: BlockVisibility) -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner())

    with pytest.raises(ValueError, match="non-reader-visible"):
        CandidateBundle(
            candidate_id="candidate-1",
            run_id="run-1",
            generation_attempt=1,
            content_blocks=(TextBlock(text="internal", visibility=visibility),),
            requirements=RequirementsManifest(revision="requirements-1"),
            evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
            delivery_mode=DeliveryMode.inline,
        )


def test_run_tree_origin_requires_root_binding_and_immediate_descendant_attribution() -> None:
    with pytest.raises(ValueError, match="root_run_id"):
        RunTreeOrigin(run_id="root", root_run_id="", depth=0)

    with pytest.raises(ValueError, match="immediate parent"):
        RunTreeOrigin(run_id="child", root_run_id="root", depth=1, subagent_id="worker-1")

    with pytest.raises(ValueError, match="immediate parent"):
        RunTreeOrigin(
            run_id="child",
            root_run_id="root",
            depth=1,
            parent_run_id="",
            subagent_id="worker-1",
        )


def test_run_tree_origin_binds_depth_to_the_position_the_ids_describe() -> None:
    # Depth and the ids are two statements about one position, so each is
    # checked against the other; a disagreement is the second source this
    # field exists to rule out.
    with pytest.raises(ValueError, match="depth must not be negative"):
        RunTreeOrigin(run_id="root", root_run_id="root", depth=-1)

    with pytest.raises(ValueError, match="root evidence origin must declare depth 0"):
        RunTreeOrigin(run_id="root", root_run_id="root", depth=1)

    with pytest.raises(ValueError, match="depth below the root"):
        RunTreeOrigin(
            run_id="child",
            root_run_id="root",
            depth=0,
            parent_run_id="root",
            subagent_id="worker-1",
        )

    # A descendant may sit arbitrarily deep; the ids name one hop and the root,
    # so nothing but the depth itself says how far down that is.
    grandchild = RunTreeOrigin(
        run_id="grandchild",
        root_run_id="root",
        depth=2,
        parent_run_id="child",
        subagent_id="worker-1",
    )
    assert grandchild.depth == 2
    assert grandchild.belongs_to_root("root")


@pytest.mark.parametrize(
    ("run_id", "root_run_id", "parent_run_id", "subagent_id"),
    (
        (" root", "root", None, None),
        ("root", "root ", None, None),
        ("child", "root", " parent", "worker"),
        ("child", "root", "parent", "worker "),
        ("child", "root", "child", "worker"),
    ),
)
def test_run_tree_origin_rejects_padded_and_self_parent_identifiers(
    run_id: str,
    root_run_id: str,
    parent_run_id: str | None,
    subagent_id: str | None,
) -> None:
    with pytest.raises(ValueError):
        RunTreeOrigin(
            run_id=run_id,
            root_run_id=root_run_id,
            depth=0 if parent_run_id is None else 1,
            parent_run_id=parent_run_id,
            subagent_id=subagent_id,
        )


def test_evidence_ledger_rejects_duplicate_ids_at_construction() -> None:
    record = _record()
    with pytest.raises(ValueError, match="duplicate record ids"):
        EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner(), records=(record, record))


def test_evidence_ledger_requires_an_attempt_owner() -> None:
    with pytest.raises(ValueError, match="attempt_owner"):
        EvidenceLedger(ledger_id="ledger-1")  # type: ignore[call-arg]


def _foreign_record(record_id: str = "foreign-1") -> EvidenceRecord:
    """Return a record observed in a run tree the owner has no part in."""
    return EvidenceRecord(
        record_id=record_id,
        origin=RunTreeOrigin(run_id="other-root", root_run_id="other-root", depth=0),
        producer_id="trusted-producer",
        producer_revision="rev-1",
        subject_id="subject-2",
        subject_reference="ref-2",
        digest="content-digest-2",
    )


def test_evidence_ledger_accepts_a_descendant_of_the_owner_and_refuses_a_foreign_tree() -> None:
    # The owner is the root; the record is two levels below it.  Owner equality
    # would refuse this, and refusing it is what made descendant evidence
    # inexpressible.
    descendant = EvidenceRecord(
        record_id="record-2",
        origin=RunTreeOrigin(
            run_id="grandchild",
            root_run_id="parent-1",
            depth=2,
            parent_run_id="child",
            subagent_id="worker",
        ),
        producer_id="trusted-producer",
        producer_revision="rev-1",
        subject_id="subject-3",
        subject_reference="ref-3",
        digest="content-digest-3",
    )
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(descendant)

    assert ledger.records == (descendant,)

    foreign = _foreign_record()
    with pytest.raises(ValueError, match="outside the attempt owner's run tree"):
        ledger.append(foreign)
    with pytest.raises(ValueError, match="outside the attempt owner's run tree"):
        EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner(), records=(foreign,))


def test_evidence_ledger_refuses_two_positions_for_one_run() -> None:
    # Both origins share the root, so tree membership admits each on its own.
    # They disagree about where the run producing them sits.
    truthful = _record()
    contradicting = truthful.model_copy(
        update={
            "record_id": "record-2",
            "origin": RunTreeOrigin(
                run_id="run-1",
                root_run_id="parent-1",
                depth=1,
                parent_run_id="someone-else",
                subagent_id="child-9",
            ),
        }
    )

    with pytest.raises(ValueError, match="conflicting origins for one run"):
        EvidenceLedger(
            ledger_id="ledger-1",
            attempt_owner=_owner(),
            records=(truthful, contradicting),
        )
    with pytest.raises(ValueError, match="conflicts with a recorded origin"):
        EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(truthful).append(
            contradicting
        )

    # The owner's own position counts as a recorded one, so a record cannot
    # restate the owner's run under a different subagent.
    subagent_owner = RunTreeOrigin(
        run_id="child-1",
        root_run_id="parent-1",
        depth=1,
        parent_run_id="parent-1",
        subagent_id="worker-a",
    )
    with pytest.raises(ValueError, match="conflicting origins for one run"):
        EvidenceLedger(
            ledger_id="ledger-1",
            attempt_owner=subagent_owner,
            records=(
                truthful.model_copy(
                    update={
                        "origin": subagent_owner.model_copy(update={"subagent_id": "worker-b"})
                    }
                ),
            ),
        )

    # Agreeing origins for one run remain ordinary.
    agreeing = truthful.model_copy(update={"record_id": "record-3"})
    ledger = EvidenceLedger(
        ledger_id="ledger-1", attempt_owner=_owner(), records=(truthful, agreeing)
    )
    assert [record.record_id for record in ledger.records] == ["record-1", "record-3"]


def test_inherited_prefix_rejects_empty_padded_and_duplicate_references() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        InheritedEvidencePrefix(source_id="source-1", record_ids=(), digest="prefix-digest")
    with pytest.raises(ValueError, match="duplicate record ids"):
        InheritedEvidencePrefix(
            source_id="source-1", record_ids=("record-9", "record-9"), digest="prefix-digest"
        )
    with pytest.raises(ValueError, match="must not be empty or padded"):
        InheritedEvidencePrefix(source_id=" source-1 ", record_ids=("record-9",), digest="prefix-digest")
    with pytest.raises(ValueError, match="must not be empty or padded"):
        InheritedEvidencePrefix(source_id="source-1", record_ids=("",), digest="prefix-digest")
    with pytest.raises(ValueError, match="must not be empty or padded"):
        InheritedEvidencePrefix(source_id="source-1", record_ids=("record-9",), digest="")


def test_inherited_prefix_must_not_restate_a_record_the_ledger_holds() -> None:
    record = _record()
    prefix = InheritedEvidencePrefix(
        source_id="source-1", record_ids=(record.record_id,), digest="prefix-digest"
    )

    with pytest.raises(ValueError, match="overlaps this ledger's own records"):
        EvidenceLedger(
            ledger_id="ledger-1", attempt_owner=_owner(), records=(record,), inherited=prefix
        )

    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner(), inherited=prefix)
    with pytest.raises(ValueError, match="already exists"):
        ledger.append(record)


def test_evidence_ledger_digest_covers_the_inherited_prefix() -> None:
    def _ledger(record_id: str) -> EvidenceLedger:
        return EvidenceLedger(
            ledger_id="ledger-1",
            attempt_owner=_owner(),
            inherited=InheritedEvidencePrefix(
                source_id="source-1", record_ids=(record_id,), digest="prefix-digest"
            ),
        )

    assert _ledger("earlier-1").digest == _ledger("earlier-1").digest
    assert _ledger("earlier-1").digest != _ledger("earlier-2").digest
    assert (
        _ledger("earlier-1").digest
        != EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).digest
    )


def test_evidence_ledger_digest_ignores_an_absent_prefix() -> None:
    """A ledger that opened on nothing digests over its own records alone.

    The digest is the integrity pin a sealed candidate carries, so it must
    follow the ledger's content and not the model's field list: an optional
    facet that is absent contributes nothing.
    """
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    content = ledger.model_dump(mode="json")

    assert content["inherited"] is None
    del content["inherited"]
    assert ledger.digest == hashlib.sha256(
        json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def test_candidate_may_cite_inherited_evidence_but_not_an_unknown_record() -> None:
    ledger = EvidenceLedger(
        ledger_id="ledger-1",
        attempt_owner=_owner(),
        inherited=InheritedEvidencePrefix(
            source_id="source-1", record_ids=("earlier-1",), digest="prefix-digest"
        ),
    ).append(_record())
    citation = CitationSpan(
        claim_id="claim-1", evidence_record_id="earlier-1", start_offset=0, end_offset=1
    )
    candidate = _candidate(ledger).model_copy(update={"citations": (citation,)})

    lifecycle = VerificationLifecycle(
        state=VerificationState.candidate_ready,
        ledger=ledger,
        candidate=candidate,
    )
    assert lifecycle.candidate is not None

    with pytest.raises(ValueError, match="candidate citation references unknown"):
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=_candidate(ledger).model_copy(
                update={
                    "citations": (
                        citation.model_copy(update={"evidence_record_id": "never-observed"}),
                    )
                }
            ),
        )


def test_candidate_must_reference_the_attached_ledger() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger)
    lifecycle = VerificationLifecycle(
        state=VerificationState.candidate_ready,
        ledger=ledger,
        candidate=candidate,
    )
    assert lifecycle.candidate == candidate

    with pytest.raises(ValueError, match="does not match"):
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate.model_copy(
                update={"evidence_ledger": EvidenceLedgerReference(ledger_id="other", digest="other")}
            ),
        )


def test_lifecycle_rejects_candidate_citation_outside_attached_ledger() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger).model_copy(
        update={
            "citations": (
                CitationSpan(
                    claim_id="claim-1",
                    evidence_record_id="invented-record",
                    start_offset=0,
                    end_offset=1,
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="candidate citation references unknown"):
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )


def test_terminalization_requires_all_report_evidence_to_belong_to_ledger() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    citation = CitationSpan(
        claim_id="claim-1",
        evidence_record_id="record-1",
        start_offset=0,
        end_offset=1,
    )
    candidate = _candidate(ledger).model_copy(update={"citations": (citation,)})
    lifecycle = VerificationLifecycle(
        state=VerificationState.candidate_ready,
        ledger=ledger,
        candidate=candidate,
    )
    valid_finding = VerificationFinding(
        code="verified",
        severity=VerificationSeverity.info,
        evidence_record_ids=("record-1",),
        citation_spans=(citation,),
    )
    valid_report = _report(candidate).model_copy(
        update={
            "results": (
                VerificationCheckResult(
                    check_id="check-1",
                    revision="revision-1",
                    status=VerificationCheckStatus.passed,
                    severity=VerificationSeverity.info,
                    findings=(valid_finding,),
                    deterministic=True,
                    idempotent=True,
                    resource_use=VerificationResourceUse(
                        tokens=0,
                        duration_ms=0,
                        cost_microunits=0,
                    ),
                ),
            )
        }
    )
    assert lifecycle.terminalize(valid_report).state is VerificationState.released

    unknown_evidence_report = valid_report.model_copy(
        update={
            "results": (
                valid_report.results[0].model_copy(
                    update={
                        "findings": (valid_finding.model_copy(update={"evidence_record_ids": ("invented-record",)}),)
                    }
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="verification finding evidence references unknown"):
        lifecycle.terminalize(unknown_evidence_report)

    unknown_span_report = valid_report.model_copy(
        update={
            "results": (
                valid_report.results[0].model_copy(
                    update={
                        "findings": (
                            valid_finding.model_copy(
                                update={
                                    "citation_spans": (
                                        citation.model_copy(update={"evidence_record_id": "invented-record"}),
                                    )
                                }
                            ),
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="verification finding citation references unknown"):
        lifecycle.terminalize(unknown_span_report)


def test_lifecycle_transition_table_supports_repair_pickup_and_idempotency() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger)
    state = VerificationLifecycle(
        state=VerificationState.candidate_ready,
        ledger=ledger,
        candidate=candidate,
    )
    state = state.transition_to(VerificationState.verifying)
    state = VerificationLifecycle(
        state=VerificationState.repair_requested,
        ledger=ledger,
        candidate=candidate,
        report=_report(candidate),
    )
    state = state.transition_to(VerificationState.pickup)
    state = state.transition_to(VerificationState.executing)
    assert state.ledger is None
    assert state.candidate is None
    assert state.transition_to(VerificationState.executing) == state

    with pytest.raises(InvalidVerificationTransitionError):
        assert_verification_transition(VerificationState.released, VerificationState.verifying)


def test_open_ledger_survives_pickup_snapshot_and_execution_resume() -> None:
    record = _record()
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=record.origin).append(record)
    executing = VerificationLifecycle(
        state=VerificationState.executing,
        ledger=ledger,
    )

    pickup = executing.transition_to(VerificationState.pickup)
    restored = VerificationLifecycle.from_snapshot(pickup.snapshot())
    resumed = restored.transition_to(VerificationState.executing)

    assert resumed.ledger == ledger
    assert resumed.candidate is None


def test_terminal_transition_requires_an_atomic_matching_report() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger)
    lifecycle = VerificationLifecycle(
        state=VerificationState.candidate_ready,
        ledger=ledger,
        candidate=candidate,
    )

    with pytest.raises(ValueError, match="requires a report"):
        lifecycle.transition_to(VerificationState.released)

    terminal = lifecycle.terminalize(_report(candidate))

    assert terminal.state is VerificationState.released
    assert terminal.report is not None
    assert terminal.report.decision is ReleaseDecision.release


def test_lifecycle_snapshot_roundtrips_nonempty_typed_state() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger)
    lifecycle = VerificationLifecycle(
        state=VerificationState.repair_requested,
        ledger=ledger,
        candidate=candidate,
        report=_report(candidate),
        repair_cycles=1,
    )

    snapshot = lifecycle.snapshot()
    restored = VerificationLifecycle.from_snapshot(snapshot)

    assert restored == lifecycle
    assert restored.ledger is not None
    assert restored.ledger.digest == ledger.digest


def test_lifecycle_snapshot_malformed_data_fails_closed() -> None:
    malformed = VerificationLifecycle.from_snapshot({"state": "unknown"})

    assert malformed.state is VerificationState.failed
    assert malformed.restore_error == "verification snapshot failed validation"


@pytest.mark.parametrize("snapshot", ({}, {"unknown": "dropped"}))
def test_lifecycle_snapshot_without_discriminator_or_with_unknown_payload_fails_closed(
    snapshot: dict[str, str],
) -> None:
    restored = VerificationLifecycle.from_snapshot(snapshot)

    assert restored.state is VerificationState.failed
    assert restored.restore_error is not None


def test_lifecycle_semantic_invariants_bind_state_candidate_ledger_and_report() -> None:
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()).append(_record())
    candidate = _candidate(ledger)
    report = _report(candidate)

    with pytest.raises(ValueError, match="requires candidate and ledger"):
        VerificationLifecycle(state=VerificationState.candidate_ready)
    assert VerificationLifecycle(
        state=VerificationState.executing,
        ledger=EvidenceLedger(ledger_id="open-ledger", attempt_owner=_record().origin),
    ).candidate is None
    with pytest.raises(ValueError, match="unsealed evidence ledger"):
        VerificationLifecycle(
            state=VerificationState.failed,
            ledger=EvidenceLedger(ledger_id="open-ledger", attempt_owner=_owner()),
        )
    with pytest.raises(ValueError, match="requires a report"):
        VerificationLifecycle(
            state=VerificationState.released,
            ledger=ledger,
            candidate=candidate,
        )
    with pytest.raises(ValueError, match="forbids a report"):
        VerificationLifecycle(
            state=VerificationState.verifying,
            ledger=ledger,
            candidate=candidate,
            report=report,
        )
    with pytest.raises(ValueError, match="candidate does not match"):
        VerificationLifecycle(
            state=VerificationState.repair_requested,
            ledger=ledger,
            candidate=candidate,
            report=report.model_copy(update={"candidate_id": "other"}),
        )
    with pytest.raises(ValueError, match="requires release decision"):
        VerificationLifecycle(
            state=VerificationState.released,
            ledger=ledger,
            candidate=candidate,
            report=report.model_copy(update={"decision": ReleaseDecision.block}),
        )


def _resolved_check(check_id: str = "opaque-check-a") -> ResolvedVerificationCheck:
    return ResolvedVerificationCheck(
        check_id=check_id,
        selector_revision="tool-revision-a",
        accepted_artifact_types=("application/pdf",),
        tool_id="opaque-tool-a",
        tool_revision="tool-revision-a",
        tool_schema_digest="schema-fingerprint-a",
        settings_digest="settings-fingerprint-a",
        capability_contract_revision="capability-revision-a",
        execution_binding=VerificationExecutionBindingReference(
            binding_id="opaque-binding-a",
            binding_digest="binding-fingerprint-a",
            provider_id="opaque-provider-a",
            provider_contract_revision="provider-contract-a",
        ),
        deterministic=True,
        idempotent=True,
    )


def _profile(*, check_id: str = "opaque-check-a") -> VerificationProfile:
    return VerificationProfile(
        profile_id="opaque-profile-a",
        revision="profile-revision-a",
        enabled=True,
        mode=VerificationMode.enforce,
        delivery=VerificationDelivery.gated,
        checks=(
            VerificationCheckSelector(
                check_id=check_id,
                revision="tool-revision-a",
                accepted_artifact_types=("application/pdf",),
            ),
        ),
        budget=VerificationBudget(
            max_repair_cycles=0,
            max_tokens=0,
            max_duration_ms=0,
            max_cost_microunits=0,
            max_concurrency=1,
        ),
    )


def test_resolved_verification_binding_is_frozen_and_has_stable_canonical_digest() -> None:
    plan = VerificationExecutionPlan(checks=(_resolved_check(),))
    binding = ResolvedVerificationBinding.from_profile(_profile(), plan)

    restored = ResolvedVerificationBinding.from_snapshot(binding.snapshot())

    assert restored == binding
    assert restored.execution_plan.digest == plan.digest
    assert restored.digest == binding.digest
    assert binding.snapshot() == {
        "binding": {
            "profile_id": "opaque-profile-a",
            "profile_revision": "profile-revision-a",
            "profile_checks": [
                {
                    "check_id": "opaque-check-a",
                    "revision": "tool-revision-a",
                    "accepted_artifact_types": ["application/pdf"],
                }
            ],
            "execution_plan": {
                "checks": [
                    {
                        "check_id": "opaque-check-a",
                        "selector_revision": "tool-revision-a",
                        "accepted_artifact_types": ["application/pdf"],
                        "tool_id": "opaque-tool-a",
                        "tool_revision": "tool-revision-a",
                        "tool_schema_digest": "schema-fingerprint-a",
                        "settings_digest": "settings-fingerprint-a",
                        "capability_contract_revision": "capability-revision-a",
                        "execution_binding": {
                            "binding_id": "opaque-binding-a",
                            "binding_digest": "binding-fingerprint-a",
                            "provider_id": "opaque-provider-a",
                            "provider_contract_revision": "provider-contract-a",
                        },
                        "deterministic": True,
                        "idempotent": True,
                    }
                ]
            },
        },
        "digest": binding.digest,
    }
    with pytest.raises(Exception):
        binding.profile_id = "changed"  # type: ignore[misc]


def test_resolved_verification_plan_digest_changes_for_any_pinned_surface_change() -> None:
    baseline = VerificationExecutionPlan(checks=(_resolved_check(),))
    changed = VerificationExecutionPlan(
        checks=(_resolved_check().model_copy(update={"tool_revision": "tool-revision-b"}),)
    )

    assert baseline.digest != changed.digest


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("binding_id", ""),
        ("binding_digest", " "),
        ("provider_id", ""),
        ("provider_contract_revision", " "),
    ),
)
def test_verification_execution_binding_reference_requires_complete_opaque_identity(
    field_name: str, value: str
) -> None:
    payload = _resolved_check().execution_binding.model_dump()
    payload[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        VerificationExecutionBindingReference.model_validate(payload)


def test_verification_execution_binding_reference_rejects_provider_configuration() -> None:
    payload = _resolved_check().execution_binding.model_dump()
    payload["endpoint"] = "https://provider.invalid"

    with pytest.raises(ValueError, match="endpoint"):
        VerificationExecutionBindingReference.model_validate(payload)


def test_resolved_verification_plan_digest_changes_for_execution_binding_identity() -> None:
    baseline = VerificationExecutionPlan(checks=(_resolved_check(),))
    changed_binding = _resolved_check().execution_binding.model_copy(
        update={"binding_digest": "binding-fingerprint-b"}
    )
    changed = VerificationExecutionPlan(
        checks=(_resolved_check().model_copy(update={"execution_binding": changed_binding}),)
    )

    assert baseline.digest != changed.digest


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["binding"]["execution_plan"]["checks"][0].update({"tool_revision": "tool-revision-b"}),
        lambda payload: payload["binding"]["execution_plan"]["checks"][0].update({"accepted_artifact_types": []}),
        lambda payload: payload["binding"].update({"profile_revision": "profile-revision-b"}),
    ),
)
def test_resolved_verification_binding_restore_rejects_tampered_snapshot(mutation: object) -> None:
    binding = ResolvedVerificationBinding.from_profile(
        _profile(), VerificationExecutionPlan(checks=(_resolved_check(),))
    )
    snapshot = binding.snapshot()

    mutation(snapshot)  # type: ignore[operator]

    with pytest.raises(ValueError, match=r"integrity digest|selector revision|accepted artifact"):
        ResolvedVerificationBinding.from_snapshot(snapshot)


def test_resolved_verification_binding_requires_exact_ordered_profile_selector_match() -> None:
    first = _resolved_check("opaque-check-a")
    second = _resolved_check("opaque-check-b")
    profile = _profile().model_copy(
        update={
            "checks": (
                VerificationCheckSelector(
                    check_id="opaque-check-a",
                    revision="tool-revision-a",
                    accepted_artifact_types=("application/pdf",),
                ),
                VerificationCheckSelector(
                    check_id="opaque-check-b",
                    revision="tool-revision-a",
                    accepted_artifact_types=("application/pdf",),
                ),
            )
        }
    )
    plan = VerificationExecutionPlan(checks=(second, first))

    with pytest.raises(ValueError, match="check order differs"):
        ResolvedVerificationBinding.from_profile(profile, plan)


def test_resolved_verification_binding_rejects_missing_or_extra_resolved_checks() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ResolvedVerificationBinding.from_profile(
            _profile(), VerificationExecutionPlan(checks=(_resolved_check(), _resolved_check("opaque-check-b")))
        )
    profile_with_two = _profile().model_copy(
        update={
            "checks": (
                _profile().checks[0],
                VerificationCheckSelector(
                    check_id="opaque-check-b",
                    revision="tool-revision-a",
                    accepted_artifact_types=("application/pdf",),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="equal length"):
        ResolvedVerificationBinding.from_profile(
            profile_with_two, VerificationExecutionPlan(checks=(_resolved_check(),))
        )


def test_resolved_verification_binding_restore_rejects_coordinated_reordering() -> None:
    first = _resolved_check("opaque-check-a")
    second = _resolved_check("opaque-check-b")
    profile = _profile().model_copy(
        update={
            "checks": (
                VerificationCheckSelector(
                    check_id="opaque-check-a",
                    revision="tool-revision-a",
                    accepted_artifact_types=("application/pdf",),
                ),
                VerificationCheckSelector(
                    check_id="opaque-check-b",
                    revision="tool-revision-a",
                    accepted_artifact_types=("application/pdf",),
                ),
            )
        }
    )
    binding = ResolvedVerificationBinding.from_profile(profile, VerificationExecutionPlan(checks=(first, second)))
    snapshot = binding.snapshot()
    payload = snapshot["binding"]
    payload["profile_checks"].reverse()
    payload["execution_plan"]["checks"].reverse()

    with pytest.raises(ValueError, match="integrity digest"):
        ResolvedVerificationBinding.from_snapshot(snapshot)


@pytest.mark.parametrize(
    "change",
    (
        {"selector_revision": "tool-revision-b", "tool_revision": "tool-revision-b"},
        {"accepted_artifact_types": ()},
        {"tool_revision": "tool-revision-b"},
    ),
)
def test_resolved_verification_binding_rejects_incompatible_selector_surface(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"selector revision|accepted artifact"):
        ResolvedVerificationBinding.from_profile(
            _profile(),
            VerificationExecutionPlan(checks=(_resolved_check().model_copy(update=change),)),
        )


@pytest.mark.parametrize("value", ("true", "false", "yes", 0, 1))
def test_resolved_verification_check_rejects_coercible_capability_flags(value: object) -> None:
    payload = _resolved_check().model_dump()
    payload["deterministic"] = value
    payload["idempotent"] = value

    with pytest.raises(ValueError):
        ResolvedVerificationCheck.model_validate(payload)


@pytest.mark.parametrize("value", ("true", "false", "yes", 0, 1))
def test_verification_profile_rejects_coercible_enabled_flag(value: object) -> None:
    payload = _profile().model_dump()
    payload["enabled"] = value

    with pytest.raises(ValueError):
        VerificationProfile.model_validate(payload)


@pytest.mark.parametrize("field_name", ("deterministic", "idempotent"))
@pytest.mark.parametrize("value", ("true", "false", "yes", 0, 1))
def test_verification_check_result_rejects_coercible_capability_flags(field_name: str, value: object) -> None:
    candidate = _candidate(EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()))
    payload = _report(candidate).results[0].model_dump()
    payload[field_name] = value

    with pytest.raises(ValueError):
        VerificationCheckResult.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        ({"checks": []}, "at least one check"),
        ({"checks": [_resolved_check().model_dump(), _resolved_check().model_dump()]}, "duplicate check ids"),
        ({"checks": [{**_resolved_check().model_dump(), "check_id": ""}]}, "must not be empty"),
        ({"checks": [{**_resolved_check().model_dump(), "tool_id": " padded "}]}, "must not be empty or padded"),
        ({"checks": [{**_resolved_check().model_dump(), "unexpected": "value"}]}, "Extra inputs are not permitted"),
    ),
)
def test_resolved_verification_plan_rejects_empty_duplicate_or_malformed_records(
    payload: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        VerificationExecutionPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        (
            {
                "profile_id": "",
                "profile_revision": "revision-a",
                "execution_plan": {"checks": [_resolved_check().model_dump()]},
            },
            "must not be empty",
        ),
        (
            {
                "profile_id": "profile-a",
                "profile_revision": " revision-a",
                "execution_plan": {"checks": [_resolved_check().model_dump()]},
            },
            "must not be empty or padded",
        ),
        (
            {
                "profile_id": "profile-a",
                "profile_revision": "revision-a",
                "execution_plan": {"checks": [_resolved_check().model_dump()]},
                "unexpected": "value",
            },
            "Extra inputs are not permitted",
        ),
    ),
)
def test_resolved_verification_binding_rejects_malformed_admission_data(payload: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ResolvedVerificationBinding.model_validate(payload)


# Whether an opaque identifier may be empty is a decision about what it names,
# so both answers are written down here rather than left implicit in a
# validator's field list.  Splitting them this way is what lets the tests below
# be derived from the contract instead of from the values a particular resolver
# happens to produce.
_CHECK_IDENTIFIERS_THAT_MUST_BE_STATED = frozenset(
    {
        "check_id",
        "selector_revision",
        "tool_id",
        "tool_revision",
        "tool_schema_digest",
        "capability_contract_revision",
    }
)
_CHECK_IDENTIFIERS_THAT_MAY_BE_ABSENT = frozenset({"settings_digest"})


def _declared_check_identifiers() -> frozenset[str]:
    """Return every opaque string identifier the resolved-check contract declares."""
    return frozenset(
        name for name, field in ResolvedVerificationCheck.model_fields.items() if field.annotation is str
    )


def _check_declaring_no_settings(*, settings_digest: str = "") -> ResolvedVerificationCheck:
    """Build a check that states every field itself, declaring no settings.

    Deliberately not derived from the fully populated fixture above, and it
    carries no accepted artifact types.  A check assembled by copying a richer
    one inherits values nobody chose for it, which is exactly how an identifier
    only ever exercised while populated keeps a policy no realistic minimal
    input ever tested.
    """
    return ResolvedVerificationCheck(
        check_id="opaque-check-a",
        selector_revision="tool-revision-a",
        accepted_artifact_types=(),
        tool_id="opaque-tool-a",
        tool_revision="tool-revision-a",
        tool_schema_digest="schema-fingerprint-a",
        settings_digest=settings_digest,
        capability_contract_revision="capability-revision-a",
        execution_binding=VerificationExecutionBindingReference(
            binding_id="opaque-binding-a",
            binding_digest="binding-fingerprint-a",
            provider_id="opaque-provider-a",
            provider_contract_revision="provider-contract-a",
        ),
        deterministic=True,
        idempotent=True,
    )


def test_every_declared_check_identifier_has_a_stated_emptiness_policy() -> None:
    """An identifier added later must not inherit a policy nobody chose for it.

    The two policies are not interchangeable: one says the fact this identifier
    names is present whenever the check resolved, the other says its absence is
    itself a fact worth recording.  Reading the field list off the model rather
    than repeating it means a newly declared identifier fails here until someone
    decides which it is, instead of quietly taking the strictness of whichever
    fields it was declared beside.
    """

    assert _CHECK_IDENTIFIERS_THAT_MUST_BE_STATED.isdisjoint(_CHECK_IDENTIFIERS_THAT_MAY_BE_ABSENT)
    assert (
        _CHECK_IDENTIFIERS_THAT_MUST_BE_STATED | _CHECK_IDENTIFIERS_THAT_MAY_BE_ABSENT
    ) == _declared_check_identifiers()


@pytest.mark.parametrize("field_name", sorted(_CHECK_IDENTIFIERS_THAT_MUST_BE_STATED))
def test_a_check_identifier_that_must_be_stated_rejects_an_empty_value(field_name: str) -> None:
    """Each of these names something that exists whenever a check resolved at all.

    A resolution that could not name the check, pin the revision it was selected
    at, say which tool it bound, or state the contract that tool answers under is
    a resolution that did not happen.  Recording it as an empty identifier would
    put a check into a frozen plan while claiming nothing about what will run.
    """

    payload = _resolved_check().model_dump()
    payload[field_name] = ""

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        ResolvedVerificationCheck.model_validate(payload)


@pytest.mark.parametrize("field_name", sorted(_CHECK_IDENTIFIERS_THAT_MAY_BE_ABSENT))
def test_a_check_identifier_that_may_be_absent_keeps_an_empty_value_verbatim(field_name: str) -> None:
    """Absence is carried, not translated.

    Readers downstream compare this identifier for equality against the very
    same empty value its subject published, so the persisted form has to remain
    an empty string.  Rewriting it into a placeholder, a null, or an omitted key
    would state a settings identity the check never claimed and break the
    comparison it exists to serve.
    """

    payload = _resolved_check().model_dump()
    payload[field_name] = ""

    check = ResolvedVerificationCheck.model_validate(payload)

    assert getattr(check, field_name) == ""
    assert check.model_dump(mode="json")[field_name] == ""


@pytest.mark.parametrize("value", (" ", "\t", "\n", " padded ", "padded ", " padded"))
@pytest.mark.parametrize("field_name", sorted(_declared_check_identifiers()))
def test_a_check_identifier_never_accepts_a_blank_or_padded_value(field_name: str, value: str) -> None:
    """Absence must have exactly one spelling, including where absence is allowed.

    Two resolutions of an unchanged check are compared by whole-object equality
    and hashed into the plan digest, so a second way of writing "nothing here"
    would make an unchanged binding look changed.  Permitting an empty value
    therefore permits precisely the empty value: whitespace is a malformed
    identifier, not a quieter way of declaring nothing.
    """

    payload = _resolved_check().model_dump()
    payload[field_name] = value

    with pytest.raises(ValueError, match=f"{field_name} must not be empty or padded"):
        ResolvedVerificationCheck.model_validate(payload)


def test_a_check_that_declares_no_settings_binds_and_survives_persistence() -> None:
    """The simplest check there is must be bindable, not merely constructible.

    A check holding no endpoint, key or account of its own reads what it is
    given and answers.  That is the archetypal check rather than an edge case,
    and its settings digest is empty because there is nothing to fingerprint.
    The value has to travel the whole admission path unchanged — into the plan,
    into the integrity digest, through the persisted envelope and back out of
    it — since refusing it anywhere along that path would leave "declares
    nothing to configure" as the only executable surface that cannot be bound.
    """

    check = _check_declaring_no_settings()
    plan = VerificationExecutionPlan(checks=(check,))
    binding = ResolvedVerificationBinding.from_profile(
        _profile().model_copy(
            update={
                "checks": (
                    VerificationCheckSelector(check_id="opaque-check-a", revision="tool-revision-a"),
                )
            }
        ),
        plan,
    )

    snapshot = binding.snapshot()
    assert snapshot["binding"]["execution_plan"]["checks"][0]["settings_digest"] == ""

    restored = ResolvedVerificationBinding.from_snapshot(snapshot)

    assert restored == binding
    assert restored.execution_plan.checks[0].settings_digest == ""
    assert restored.execution_plan.digest == plan.digest


def test_declaring_no_settings_is_a_distinct_surface_from_declaring_some() -> None:
    """An absent digest states "no settings", never "whichever settings".

    The digest exists so that a configuration change makes the bound executable
    a different one.  If having no settings collapsed onto the same identity as
    having some, a tool that gained a settings spec could keep answering under a
    binding that was frozen before the spec existed.
    """

    without_settings = _check_declaring_no_settings()
    with_settings = _check_declaring_no_settings(settings_digest="settings-fingerprint-a")

    assert without_settings != with_settings
    assert (
        VerificationExecutionPlan(checks=(without_settings,)).digest
        != VerificationExecutionPlan(checks=(with_settings,)).digest
    )
