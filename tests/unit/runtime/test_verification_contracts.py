"""Tests for domain-neutral candidate verification contracts."""

from __future__ import annotations

import hashlib
import json

import pytest

from protocore.contracts.evidence import (
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
    RunTreeOrigin,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationFinding,
    VerificationLifecycle,
    VerificationReport,
    VerificationResourceUse,
    VerificationSeverity,
    VerificationState,
    assert_verification_transition,
)
from protocore.contracts.types import BlockVisibility, TextBlock, ToolResultBlock


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


@pytest.mark.parametrize("field_name", ("deterministic", "idempotent"))
@pytest.mark.parametrize("value", ("true", "false", "yes", 0, 1))
def test_verification_check_result_rejects_coercible_capability_flags(field_name: str, value: object) -> None:
    candidate = _candidate(EvidenceLedger(ledger_id="ledger-1", attempt_owner=_owner()))
    payload = _report(candidate).results[0].model_dump()
    payload[field_name] = value

    with pytest.raises(ValueError):
        VerificationCheckResult.model_validate(payload)
