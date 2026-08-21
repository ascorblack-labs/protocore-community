"""Full-history snapshot serialization is offloaded off-loop.

``QueryEngine.snapshot()` builds ``history: [m.model_dump(mode="json") ...]``
synchronously, and ``_persist_snapshot`` is awaited at MANY turn/tool
boundaries. Under large histories this is a pure-CPU section on the single
executor event loop that can starve a neighbour run's pending provider
socket read, producing a FALSE stream stall.

The fix moves the heavy ``snapshot()`` construction to a worker thread via
``asyncio.to_thread(...)`` at the ``_persist_snapshot`` call site. The
serialization is pure-CPU and only reads engine state (no event-loop-only
objects), and ``await asyncio.to_thread`` blocks this run's own coroutine
until the thread finishes, so no concurrent history mutation can race.

Contract: the emitted ``state_snapshot`` payload MUST be byte-identical to
the synchronous ``snapshot()`` output.
"""

from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.contracts.verification import (
    CandidateBundle,
    DeliveryMode,
    EvidenceLedger,
    EvidenceLedgerReference,
    EvidenceRecord,
    RequirementsManifest,
    RunTreeOrigin,
    VerificationLifecycle,
    VerificationState,
)


@pytest.mark.asyncio
async def test_persist_snapshot_offloads_serialization_to_thread(engine_factory, monkeypatch) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    for i in range(20):
        engine.history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=f"message-{i}" * 100)],
            )
        )

    calls: list[object] = []
    real_to_thread = asyncio.to_thread

    async def _counting_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("protocore.runtime.query_engine.asyncio.to_thread", _counting_to_thread)

    expected = engine.snapshot()
    await engine._persist_snapshot()

    # The serialization ran in a worker thread.
    assert len(calls) == 1

    # The emitted snapshot payload equals the synchronous snapshot().
    stream = engine.events.stream_for(  # type: ignore[attr-defined]
        engine.config.tenant_id, engine.config.run_id
    )
    emitted = [e for e in stream if e.name == "state_snapshot"]
    assert emitted, "no state_snapshot event emitted"
    assert emitted[-1].payload["snapshot"] == expected


@pytest.mark.asyncio
async def test_persist_snapshot_payload_roundtrips_history(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="hello 世界")],
        )
    )

    await engine._persist_snapshot()

    stream = engine.events.stream_for(  # type: ignore[attr-defined]
        engine.config.tenant_id, engine.config.run_id
    )
    emitted = [e for e in stream if e.name == "state_snapshot"]
    snapshot = emitted[-1].payload["snapshot"]
    assert snapshot["run_id"] == engine.config.run_id
    assert len(snapshot["history"]) == 1
    assert snapshot["history"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_snapshot_roundtrips_nonempty_verification_lifecycle(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    origin = RunTreeOrigin(run_id=engine.config.run_id, root_run_id=engine.config.run_id, depth=0)
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=origin).append(
        EvidenceRecord(
            record_id="record-1",
            origin=origin,
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )
    engine.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )
    )

    snapshot = engine.snapshot()
    restored = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    await restored.resume_from_snapshot(snapshot)

    assert "verification" in snapshot
    assert restored.verification_lifecycle == engine.verification_lifecycle
    assert restored.verification_lifecycle.ledger is not None
    assert restored.verification_lifecycle.ledger.digest == ledger.digest


@pytest.mark.asyncio
async def test_resume_treats_absent_verification_as_legacy_but_malformed_as_failed(
    engine_factory,
) -> None:
    source = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    legacy_snapshot = source.snapshot()
    restored = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    await restored.resume_from_snapshot(legacy_snapshot)

    assert restored.verification_lifecycle == VerificationLifecycle()

    malformed_snapshot = {**legacy_snapshot, "verification": {"state": "unknown"}}
    await restored.resume_from_snapshot(malformed_snapshot)

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.restore_error is not None


@pytest.mark.parametrize("verification", ({}, {"unknown": "dropped"}))
async def test_resume_fails_closed_for_present_truncated_or_unknown_verification_snapshot(
    engine_factory,
    verification: dict[str, str],
) -> None:
    source = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    snapshot = {**source.snapshot(), "verification": verification}
    restored = engine_factory(rc=RuntimeConstants(model_context_window=4_096))

    await restored.resume_from_snapshot(snapshot)

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.state is not VerificationState.released
    assert restored.verification_lifecycle.restore_error is not None


def test_engine_rejects_candidate_for_another_run(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    origin = RunTreeOrigin(run_id="another-run", root_run_id="another-run", depth=0)
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=origin).append(
        EvidenceRecord(
            record_id="record-1",
            origin=origin,
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id="another-run",
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )

    with pytest.raises(ValueError, match="does not match engine run_id"):
        engine.replace_verification_lifecycle(
            VerificationLifecycle(
                state=VerificationState.candidate_ready,
                ledger=ledger,
                candidate=candidate,
            )
        )


def test_engine_rejects_evidence_from_outside_candidate_run_tree(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    unrelated = RunTreeOrigin(
        run_id="unrelated-run",
        root_run_id="unrelated-parent",
        depth=1,
        parent_run_id="unrelated-parent",
        subagent_id="child",
    )
    unrelated_record = EvidenceRecord(
        record_id="record-1",
        origin=unrelated,
        producer_id="trusted-producer",
        producer_revision="revision-1",
        subject_id="subject-1",
        subject_reference="reference-1",
        digest="digest-1",
    )
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=unrelated).append(unrelated_record)
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )

    with pytest.raises(ValueError, match="outside the candidate run tree"):
        engine.replace_verification_lifecycle(
            VerificationLifecycle(
                state=VerificationState.candidate_ready,
                ledger=ledger,
                candidate=candidate,
            )
        )

    # The other half of the same rule: an unrelated observation cannot be
    # smuggled in under an owner the engine does accept.
    with pytest.raises(ValueError, match="outside the attempt owner's run tree"):
        EvidenceLedger(
            ledger_id="ledger-1",
            attempt_owner=RunTreeOrigin(
                run_id=engine.config.run_id, root_run_id=engine.config.run_id, depth=0
            ),
            records=(unrelated_record,),
        )


def test_engine_accepts_evidence_from_a_direct_subagent(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    ledger = EvidenceLedger(
        ledger_id="ledger-1",
        attempt_owner=RunTreeOrigin(
            run_id=engine.config.run_id, root_run_id=engine.config.run_id, depth=0
        ),
    ).append(
        EvidenceRecord(
            record_id="record-1",
            origin=RunTreeOrigin(
                run_id="subagent-run",
                root_run_id=engine.config.run_id,
                depth=1,
                parent_run_id=engine.config.run_id,
                subagent_id="researcher",
            ),
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )

    engine.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )
    )


def test_engine_accepts_evidence_from_a_nested_subagent(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    ledger = EvidenceLedger(
        ledger_id="ledger-1",
        attempt_owner=RunTreeOrigin(
            run_id=engine.config.run_id, root_run_id=engine.config.run_id, depth=0
        ),
    ).append(
        EvidenceRecord(
            record_id="record-1",
            origin=RunTreeOrigin(
                run_id="grandchild-run",
                root_run_id=engine.config.run_id,
                depth=2,
                parent_run_id="child-run",
                subagent_id="researcher-2",
            ),
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )

    engine.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )
    )


@pytest.mark.asyncio
async def test_resume_fails_closed_for_a_legacy_evidence_origin_without_root_binding(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    origin = RunTreeOrigin(run_id=engine.config.run_id, root_run_id=engine.config.run_id, depth=0)
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=origin).append(
        EvidenceRecord(
            record_id="record-1",
            origin=origin,
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )
    engine.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )
    )
    snapshot = engine.snapshot()
    del snapshot["verification"]["ledger"]["records"][0]["origin"]["root_run_id"]

    restored = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    await restored.resume_from_snapshot(snapshot)

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.restore_error == "verification snapshot failed validation"


@pytest.mark.asyncio
async def test_resume_fails_closed_for_a_foreign_run_verification_snapshot(engine_factory) -> None:
    source = engine_factory(run_id="source-run", rc=RuntimeConstants(model_context_window=4_096))
    origin = RunTreeOrigin(run_id=source.config.run_id, root_run_id=source.config.run_id, depth=0)
    ledger = EvidenceLedger(ledger_id="ledger-1", attempt_owner=origin).append(
        EvidenceRecord(
            record_id="record-1",
            origin=origin,
            producer_id="trusted-producer",
            producer_revision="revision-1",
            subject_id="subject-1",
            subject_reference="reference-1",
            digest="digest-1",
        )
    )
    candidate = CandidateBundle(
        candidate_id="candidate-1",
        run_id=source.config.run_id,
        generation_attempt=1,
        content_blocks=(TextBlock(text="candidate output"),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )
    source.replace_verification_lifecycle(
        VerificationLifecycle(
            state=VerificationState.candidate_ready,
            ledger=ledger,
            candidate=candidate,
        )
    )

    restored = engine_factory(run_id="destination-run", rc=RuntimeConstants(model_context_window=4_096))
    await restored.resume_from_snapshot(source.snapshot())

    assert restored.verification_lifecycle.state is VerificationState.failed
    assert restored.verification_lifecycle.restore_error == "verification snapshot run binding failed"
