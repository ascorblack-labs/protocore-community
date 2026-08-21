"""Controlled collection and sealing of typed runtime evidence."""

from __future__ import annotations

import pytest

from protocore.contracts.types import TextBlock
from protocore.contracts.verification import (
    CandidateBundle,
    DeliveryMode,
    EvidenceLedger,
    EvidenceLedgerReference,
    EvidenceRecord,
    InheritedEvidencePrefix,
    ReleaseDecision,
    RequirementsManifest,
    RunTreeOrigin,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationLifecycle,
    VerificationReport,
    VerificationResourceUse,
    VerificationSeverity,
    VerificationState,
)


def _record(engine, record_id: str, **origin_overrides: str | int | None) -> EvidenceRecord:  # type: ignore[no-untyped-def]
    origin: dict[str, str | int | None] = {
        "run_id": engine.config.run_id,
        "root_run_id": engine.config.root_run_id,
        "depth": engine.config.run_depth,
        "parent_run_id": engine.config.parent_run_id,
        "subagent_id": engine.config.subagent_id,
    }
    origin.update(origin_overrides)
    return EvidenceRecord(
        record_id=record_id,
        origin=RunTreeOrigin(**origin),
        producer_id="trusted-producer",
        producer_revision="revision-1",
        subject_id=f"subject-{record_id}",
        subject_reference=f"reference-{record_id}",
        digest=f"digest-{record_id}",
    )


def _candidate(engine, *, ledger_id: str, digest: str) -> CandidateBundle:  # type: ignore[no-untyped-def]
    return CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=0,
        content_blocks=(TextBlock(text="reader-facing candidate"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger_id, digest=digest),
        delivery_mode=DeliveryMode.inline,
    )


def _report(candidate: CandidateBundle) -> VerificationReport:
    return VerificationReport(
        report_id="report-1",
        candidate_id=candidate.candidate_id,
        profile_id="profile-1",
        profile_revision="revision-1",
        results=(
            VerificationCheckResult(
                check_id="check-1",
                revision="revision-1",
                status=VerificationCheckStatus.failed,
                severity=VerificationSeverity.medium,
                deterministic=True,
                idempotent=True,
                resource_use=VerificationResourceUse(tokens=0, duration_ms=0, cost_microunits=0),
            ),
        ),
        decision=ReleaseDecision.block,
    )


def test_root_defaults_to_run_and_collection_accepts_only_exact_engine_origin(engine_factory) -> None:
    engine = engine_factory(run_id="root-run")

    assert engine.config.root_run_id == "root-run"
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_record(engine, "record-1"),))

    assert engine.verification_lifecycle.state is VerificationState.executing
    assert engine.verification_lifecycle.ledger is not None
    assert [record.record_id for record in engine.verification_lifecycle.ledger.records] == ["record-1"]


def test_child_config_requires_explicit_root_binding(engine_factory) -> None:
    with pytest.raises(ValueError, match="requires root_run_id"):
        engine_factory(run_id="child-run", parent_run_id="root-run", subagent_id="worker")


def test_append_rejects_bad_origin_and_duplicates_without_partial_mutation(engine_factory) -> None:
    engine = engine_factory(run_id="child", root_run_id="root", parent_run_id="parent", subagent_id="worker")
    engine.begin_evidence_collection(ledger_id="ledger-1")
    first = _record(engine, "record-1")
    engine.append_tool_evidence((first,))

    with pytest.raises(ValueError, match="root_run_id"):
        engine.append_tool_evidence((_record(engine, "record-2"), _record(engine, "record-3", root_run_id="other")))
    with pytest.raises(ValueError, match="already exists"):
        engine.append_tool_evidence((_record(engine, "record-4"), _record(engine, "record-4")))
    with pytest.raises(ValueError, match="already exists"):
        engine.append_tool_evidence((_record(engine, "record-5"), first))

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["record-1"]


def test_seal_requires_exact_open_ledger_and_prevents_later_append(engine_factory) -> None:
    engine = engine_factory()
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_record(engine, "record-1"),))
    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None

    with pytest.raises(ValueError, match="does not match current ledger"):
        engine.seal_candidate(_candidate(engine, ledger_id="other", digest=ledger.digest))
    assert engine.verification_lifecycle.state is VerificationState.executing

    engine.seal_candidate(_candidate(engine, ledger_id=ledger.ledger_id, digest=ledger.digest))
    assert engine.verification_lifecycle.state is VerificationState.candidate_ready
    with pytest.raises(ValueError, match="open executing ledger"):
        engine.append_tool_evidence((_record(engine, "record-2"),))


def test_repair_starts_a_fresh_ledger_without_mutating_prior_candidate(engine_factory) -> None:
    engine = engine_factory()
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_record(engine, "record-1"),))
    old_ledger = engine.verification_lifecycle.ledger
    assert old_ledger is not None
    candidate = _candidate(engine, ledger_id=old_ledger.ledger_id, digest=old_ledger.digest)
    engine.seal_candidate(candidate)
    engine.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.repair_requested,
            ledger=old_ledger,
            candidate=candidate,
            report=_report(candidate),
            repair_cycles=1,
        )
    )

    engine.begin_evidence_collection(ledger_id="ledger-2")

    lifecycle = engine.verification_lifecycle
    assert lifecycle.state is VerificationState.executing
    assert lifecycle.candidate is None
    assert lifecycle.ledger is not None
    assert lifecycle.ledger.ledger_id == "ledger-2"
    assert lifecycle.ledger.records == ()
    assert old_ledger.digest == candidate.evidence_ledger.digest


@pytest.mark.asyncio
async def test_root_and_open_ledger_survive_snapshot_restore_and_history_changes(engine_factory) -> None:
    source = engine_factory(run_id="child", root_run_id="root", parent_run_id="parent", subagent_id="worker")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_record(source, "record-1"),))
    snapshot = source.snapshot()

    assert snapshot["root_run_id"] == "root"
    restored = engine_factory(run_id="child", root_run_id="root", parent_run_id="parent", subagent_id="worker")
    await restored.resume_from_snapshot(snapshot)
    ledger = restored.verification_lifecycle.ledger
    assert ledger is not None
    digest = ledger.digest

    # History may be compacted independently; the evidence ledger is outside
    # it and its candidate reference therefore remains unchanged.
    restored.history.clear()
    restored.seal_candidate(_candidate(restored, ledger_id=ledger.ledger_id, digest=digest))
    assert restored.verification_lifecycle.candidate is not None
    assert restored.verification_lifecycle.candidate.evidence_ledger.digest == digest


@pytest.mark.asyncio
async def test_restore_fails_closed_when_snapshot_root_differs_from_engine(engine_factory) -> None:
    source = engine_factory(run_id="child", root_run_id="root-a", parent_run_id="parent", subagent_id="worker")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_record(source, "record-1"),))

    restored = engine_factory(run_id="child", root_run_id="root-b", parent_run_id="parent", subagent_id="worker")
    await restored.resume_from_snapshot(source.snapshot())

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.restore_error == "verification snapshot run binding failed"


def test_replace_rejects_open_ledger_from_a_sibling_without_mutation(engine_factory) -> None:
    source = engine_factory(run_id="child-a", root_run_id="root", parent_run_id="parent", subagent_id="worker-a")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_record(source, "record-a"),))
    source_lifecycle = source.verification_lifecycle

    destination = engine_factory(
        run_id="child-b", root_run_id="root", parent_run_id="parent", subagent_id="worker-b"
    )
    with pytest.raises(ValueError, match="attempt owner does not match engine identity"):
        destination.replace_verification_lifecycle(source_lifecycle)

    assert destination.verification_lifecycle == VerificationLifecycle()
    assert source.verification_lifecycle == source_lifecycle


@pytest.mark.asyncio
async def test_restore_rejects_open_ledger_from_a_sibling_without_mutation(engine_factory) -> None:
    source = engine_factory(run_id="child-a", root_run_id="root", parent_run_id="parent", subagent_id="worker-a")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_record(source, "record-a"),))
    source_lifecycle = source.verification_lifecycle

    destination = engine_factory(
        run_id="child-b", root_run_id="root", parent_run_id="parent", subagent_id="worker-b"
    )
    await destination.resume_from_snapshot(source.snapshot())

    assert destination.verification_lifecycle.state is VerificationState.failed
    assert destination.verification_lifecycle.restore_error == "verification snapshot run binding failed"
    assert destination.verification_lifecycle.ledger is None
    assert source.verification_lifecycle == source_lifecycle


def test_replace_rejects_empty_open_ledger_from_a_sibling(engine_factory) -> None:
    source = engine_factory(
        run_id="same-run", root_run_id="root", parent_run_id="parent-a", subagent_id="worker-a"
    )
    source.begin_evidence_collection(ledger_id="ledger-1")
    destination = engine_factory(
        run_id="same-run", root_run_id="root", parent_run_id="parent-b", subagent_id="worker-b"
    )

    with pytest.raises(ValueError, match="attempt owner does not match engine identity"):
        destination.replace_verification_lifecycle(source.verification_lifecycle)

    assert destination.verification_lifecycle == VerificationLifecycle()


@pytest.mark.asyncio
async def test_empty_open_ledger_restores_only_to_its_exact_attempt_owner(engine_factory) -> None:
    source = engine_factory(
        run_id="same-run", root_run_id="root", parent_run_id="parent-a", subagent_id="worker-a"
    )
    source.begin_evidence_collection(ledger_id="ledger-1")
    snapshot = source.snapshot()

    accepted = engine_factory(
        run_id="same-run", root_run_id="root", parent_run_id="parent-a", subagent_id="worker-a"
    )
    await accepted.resume_from_snapshot(snapshot)
    assert accepted.verification_lifecycle == source.verification_lifecycle

    rejected = engine_factory(
        run_id="same-run", root_run_id="root", parent_run_id="parent-b", subagent_id="worker-b"
    )
    await rejected.resume_from_snapshot(snapshot)
    assert rejected.verification_lifecycle.state is VerificationState.failed
    assert rejected.verification_lifecycle.restore_error == "verification snapshot run binding failed"


@pytest.mark.asyncio
async def test_restore_fails_closed_for_legacy_open_ledger_without_attempt_owner(engine_factory) -> None:
    source = engine_factory()
    source.begin_evidence_collection(ledger_id="ledger-1")
    snapshot = source.snapshot()
    del snapshot["verification"]["ledger"]["attempt_owner"]

    restored = engine_factory()
    await restored.resume_from_snapshot(snapshot)

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.restore_error == "verification snapshot failed validation"


@pytest.mark.asyncio
async def test_open_ledger_with_descendant_and_inherited_evidence_survives_restore(
    engine_factory,
) -> None:
    source = engine_factory(run_id="root-run")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_record(source, "record-1"),))
    open_ledger = source.verification_lifecycle.ledger
    assert open_ledger is not None
    # A subagent's observation and a prefix from outside this tree: neither can
    # be re-observed by the run that resumes, so refusing them on restore would
    # cost the run evidence it already holds.
    descendant = _record(
        source,
        "record-2",
        run_id="sub-run",
        depth=1,
        parent_run_id="root-run",
        subagent_id="researcher",
    )
    spliced = EvidenceLedger(
        ledger_id=open_ledger.ledger_id,
        attempt_owner=open_ledger.attempt_owner,
        records=(*open_ledger.records, descendant),
        inherited=InheritedEvidencePrefix(
            source_id="prefix-source-1",
            record_ids=("earlier-record-1",),
            digest="prefix-digest-1",
        ),
    )
    source.replace_verification_lifecycle(
        VerificationLifecycle(state=VerificationState.executing, ledger=spliced)
    )

    restored = engine_factory(run_id="root-run")
    await restored.resume_from_snapshot(source.snapshot())

    assert restored.verification_lifecycle.state is VerificationState.executing
    assert restored.verification_lifecycle == source.verification_lifecycle
    ledger = restored.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["record-1", "record-2"]
    assert ledger.inherited is not None
    assert ledger.inherited.record_ids == ("earlier-record-1",)
