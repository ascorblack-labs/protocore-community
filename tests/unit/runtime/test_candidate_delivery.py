"""Typed, no-replay candidate delivery primitives."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.types import BlockVisibility, Message, MessageRole, StopReason, TextBlock
from protocore.contracts.verification import (
    CandidateBundle,
    CandidateReleasedProjection,
    CitationSpan,
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
    VerificationDelivery,
    VerificationLifecycle,
    VerificationReport,
    VerificationResourceUse,
    VerificationSeverity,
    VerificationState,
)
from protocore.runtime.candidate_delivery import CandidateDeliveryGate
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query import query


def _interleaved_tool_then_text_stream(request: object) -> AsyncIterator[ProviderDelta]:
    """Yield the provider shape where a text block follows tool start."""
    del request

    async def _stream() -> AsyncIterator[ProviderDelta]:
        yield ProviderDelta(
            kind=ProviderDeltaKind.tool_use_start,
            tool_call_id="call-1",
            tool_name="Read",
        )
        yield ProviderDelta(kind=ProviderDeltaKind.text, content="unverified generated answer")
        yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")

    return _stream()


def _assert_interleaved_reader_envelope_is_held(events: list[TurnEvent]) -> None:
    """An interleaved tool/text reader turn cannot leak partial grouping."""
    assert not any(
        event.type
        in {
            EventType.MESSAGE_START,
            EventType.CONTENT_BLOCK_START,
            EventType.CONTENT_BLOCK_DELTA,
            EventType.CONTENT_BLOCK_STOP,
            EventType.TOOL_SURFACE_ADVERTISED,
            EventType.TOOL_USE_START,
            EventType.TOOL_USE_INPUT_DELTA,
            EventType.TOOL_USE_STOP,
            EventType.TOOL_RESULT,
            EventType.TOOL_CALL_PENDING,
            EventType.MESSAGE_STOP,
        }
        for event in events
    )
    assert all("unverified generated answer" not in str(event.payload) for event in events)


def _candidate(engine, *, text: str = "sealed answer") -> CandidateBundle:  # type: ignore[no-untyped-def]
    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    return CandidateBundle(
        candidate_id="candidate-1",
        run_id=engine.config.run_id,
        generation_attempt=0,
        content_blocks=(TextBlock(text=text),),
        requirements=RequirementsManifest(revision="requirements-1"),
        evidence_ledger=EvidenceLedgerReference(ledger_id=ledger.ledger_id, digest=ledger.digest),
        delivery_mode=DeliveryMode.inline,
    )


def _release_report(candidate: CandidateBundle, *, decision: ReleaseDecision = ReleaseDecision.release) -> VerificationReport:
    return VerificationReport(
        report_id="report-1",
        candidate_id=candidate.candidate_id,
        profile_id="profile-1",
        profile_revision="revision-1",
        results=(
            VerificationCheckResult(
                check_id="check-1",
                revision="revision-1",
                status=VerificationCheckStatus.passed,
                severity=VerificationSeverity.info,
                deterministic=True,
                idempotent=True,
                resource_use=VerificationResourceUse(tokens=0, duration_ms=0, cost_microunits=0),
            ),
        ),
        decision=decision,
    )


def _event(event_type: EventType) -> TurnEvent:
    return TurnEvent(type=event_type, run_id="run-1", payload={"opaque": "payload"})


def _evidence_record(engine) -> EvidenceRecord:  # type: ignore[no-untyped-def]
    return EvidenceRecord(
        record_id="record-1",
        origin=RunTreeOrigin(
            run_id=engine.config.run_id,
            root_run_id=engine.config.root_run_id,
            depth=engine.config.run_depth,
            parent_run_id=engine.config.parent_run_id,
            subagent_id=engine.config.subagent_id,
        ),
        producer_id="trusted-producer",
        producer_revision="revision-1",
        subject_id="subject-1",
        subject_reference="reference-1",
        digest="evidence-digest-1",
    )


def test_default_and_optimistic_gates_preserve_every_existing_event_object() -> None:
    events = tuple(_event(event_type) for event_type in EventType)

    for gate in (CandidateDeliveryGate(), CandidateDeliveryGate(VerificationDelivery.optimistic)):
        assert all(gate.permits(event) for event in events)


def test_gated_delivery_holds_only_content_block_frames_without_payload_inspection() -> None:
    gate = CandidateDeliveryGate(VerificationDelivery.gated, expected_run_id="run-1", expected_root_run_id="run-1")
    held = {
        EventType.CONTENT_BLOCK_START,
        EventType.CONTENT_BLOCK_DELTA,
        EventType.CONTENT_BLOCK_STOP,
    }

    for event_type in EventType:
        assert gate.permits(_event(event_type)) is (
            event_type not in held and event_type is not EventType.CANDIDATE_RELEASED
        )


def test_gated_delivery_rejects_a_forged_candidate_release_event() -> None:
    gate = CandidateDeliveryGate(VerificationDelivery.gated, expected_run_id="run-1", expected_root_run_id="run-1")
    forged = TurnEvent(
        id="forged-claim",
        type=EventType.CANDIDATE_RELEASED,
        run_id="run-1",
        payload={"content_blocks": [{"kind": "text", "text": "unverified answer"}]},
    )

    assert not gate.permits(forged)
    assert not hasattr(gate, "public_release_event")
    assert not hasattr(gate, "public_release_events")


@pytest.mark.asyncio
async def test_seal_and_persist_is_idempotent_only_for_the_exact_candidate(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(run_id="run-1")
    engine.begin_evidence_collection(ledger_id="ledger-1")
    candidate = _candidate(engine)

    await engine.seal_candidate_and_persist(candidate)
    await engine.seal_candidate_and_persist(candidate)

    events = in_memory_runtime["events"].stream_for("tenant-test", "run-1")
    assert len(events) == 2
    assert all(event.name == "state_snapshot" for event in events)
    assert all(event.payload["snapshot"]["verification"]["candidate"]["candidate_id"] == "candidate-1" for event in events)
    with pytest.raises(ValueError, match="different verification candidate"):
        await engine.seal_candidate_and_persist(_candidate(engine, text="replacement"))


@pytest.mark.asyncio
async def test_public_verification_lifecycle_checkpoint_never_requires_history_access(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(run_id="run-1")
    engine.begin_evidence_collection(ledger_id="ledger-1")

    await engine.persist_verification_lifecycle()

    events = in_memory_runtime["events"].stream_for("tenant-test", "run-1")
    assert len(events) == 1
    assert events[0].name == "state_snapshot"
    assert events[0].payload["snapshot"]["verification"]["ledger"]["ledger_id"] == "ledger-1"


@pytest.mark.asyncio
async def test_gated_engine_run_never_yields_reader_candidate_frames_before_release(
    engine_factory, in_memory_runtime
) -> None:
    """The production ``QueryEngine.run`` boundary, not a helper, owns gating."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    in_memory_runtime["llm"].queue_response(text="unverified generated answer", stop_reason=StopReason.end_turn)

    public_events = [
        event
        async for event in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    ]

    assert not any(
        event.type
        in {
            EventType.CONTENT_BLOCK_START,
            EventType.CONTENT_BLOCK_DELTA,
            EventType.CONTENT_BLOCK_STOP,
            EventType.MESSAGE_STOP,
        }
        for event in public_events
    )
    assert all("unverified generated answer" not in str(event.payload) for event in public_events)

    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_evidence_record(engine),))
    candidate = _candidate(engine, text="unverified generated answer")
    engine.seal_candidate(candidate)
    engine.replace_verification_lifecycle(
        engine.verification_lifecycle.terminalize(_release_report(candidate))
    )

    release = engine.candidate_released_projection()
    assert release.event_payload()["content_blocks"] == [
        {"kind": "text", "text": "unverified generated answer"}
    ]
    # Reader event emission is deliberately unavailable until a durable
    # adapter coordinates claim, emit-once, and checkpointing.
    assert not hasattr(engine, "verified_candidate_release_events")


@pytest.mark.asyncio
async def test_public_query_iterator_uses_the_same_gated_delivery_boundary(
    engine_factory, in_memory_runtime
) -> None:
    """Prepared-engine callers cannot bypass ``QueryEngine.run`` gating."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    engine.turn_count = 1
    in_memory_runtime["llm"].queue_response(text="unverified generated answer", stop_reason=StopReason.end_turn)

    public_events = [event async for event in query(engine)]

    assert not any(
        event.type
        in {
            EventType.CONTENT_BLOCK_START,
            EventType.CONTENT_BLOCK_DELTA,
            EventType.CONTENT_BLOCK_STOP,
            EventType.MESSAGE_STOP,
        }
        for event in public_events
    )
    assert all("unverified generated answer" not in str(event.payload) for event in public_events)


@pytest.mark.asyncio
async def test_gated_engine_run_holds_reader_envelope_when_tool_start_precedes_text(
    engine_factory,
) -> None:
    """A provider may emit text while its tool call remains open."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.llm.stream_with_tools = _interleaved_tool_then_text_stream  # type: ignore[method-assign]

    events = [
        event
        async for event in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    ]

    _assert_interleaved_reader_envelope_is_held(events)


@pytest.mark.asyncio
async def test_gated_public_query_holds_reader_envelope_when_tool_start_precedes_text(
    engine_factory,
) -> None:
    """The prepared-engine iterator cannot flush the deferred envelope either."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    engine.turn_count = 1
    engine.llm.stream_with_tools = _interleaved_tool_then_text_stream  # type: ignore[method-assign]

    events = [event async for event in query(engine)]

    _assert_interleaved_reader_envelope_is_held(events)


def _assert_tool_only_round_is_well_grouped(events: list[TurnEvent]) -> None:
    """A retained tool-only turn must begin before any child tool envelope."""
    reader_turn_types = {
        EventType.MESSAGE_START,
        EventType.MESSAGE_STOP,
        EventType.TOOL_SURFACE_ADVERTISED,
        EventType.TOOL_USE_START,
        EventType.TOOL_USE_INPUT_DELTA,
        EventType.TOOL_USE_STOP,
        EventType.TOOL_RESULT,
        EventType.TOOL_CALL_PENDING,
    }
    first_tool_frame = next(
        index
        for index, event in enumerate(events)
        if event.type in reader_turn_types - {EventType.MESSAGE_START, EventType.MESSAGE_STOP}
    )
    first_start = next(
        index for index, event in enumerate(events) if event.type is EventType.MESSAGE_START
    )
    first_stop = next(
        index
        for index, event in enumerate(events[first_start:], start=first_start)
        if event.type is EventType.MESSAGE_STOP
    )

    assert first_start < first_tool_frame < first_stop
    assert events[first_start].payload["turn_id"] == events[first_stop].payload["turn_id"]
    assert any(event.type is EventType.TOOL_USE_START for event in events[first_start:first_stop])
    assert not any(event.type in reader_turn_types for event in events[:first_start])
    assert all("unverified generated answer" not in str(event.payload) for event in events)


@pytest.mark.asyncio
async def test_gated_engine_run_releases_a_tool_only_round_in_valid_order(
    engine_factory, in_memory_runtime
) -> None:
    """A tool-only round is public only as one complete reader envelope."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-1", tool_name="unregistered-tool", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="unverified generated answer", stop_reason=StopReason.end_turn)

    events = [
        event
        async for event in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    ]

    _assert_tool_only_round_is_well_grouped(events)


@pytest.mark.asyncio
async def test_gated_public_query_releases_a_tool_only_round_in_valid_order(
    engine_factory, in_memory_runtime
) -> None:
    """Prepared-engine iteration follows the same turn-envelope projection."""
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="question")]))
    engine.turn_count = 1
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="call-1", tool_name="unregistered-tool", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="unverified generated answer", stop_reason=StopReason.end_turn)

    events = [event async for event in query(engine)]

    _assert_tool_only_round_is_well_grouped(events)


@pytest.mark.asyncio
async def test_resume_rejects_gated_delivery_downgrade_before_public_iteration(engine_factory) -> None:
    source = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    source_snapshot = source.snapshot()
    recovered = engine_factory(run_id="run-1")

    with pytest.raises(ValueError, match="verification delivery snapshot binding"):
        await recovered.resume_from_snapshot(source_snapshot)


@pytest.mark.asyncio
async def test_gated_release_is_stable_across_crash_reconnect_and_reconstructed_gate(
    engine_factory, in_memory_runtime
) -> None:
    """A retry cannot create a second answer by choosing a new caller key."""
    source = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_evidence_record(source),))
    candidate = _candidate(source)
    await source.seal_candidate_and_persist(candidate)
    source.replace_verification_lifecycle(
        source.verification_lifecycle.terminalize(_release_report(candidate))
    )

    projection = source.candidate_released_projection()
    recovered = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    await recovered.resume_from_snapshot(source.snapshot())
    assert recovered.candidate_released_projection() == projection
    assert not hasattr(recovered, "verified_candidate_release_events")


def test_gated_blocked_terminal_is_explicit_and_contains_no_candidate_content(engine_factory) -> None:
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_evidence_record(engine),))
    candidate = _candidate(engine, text="must not be delivered")
    engine.seal_candidate(candidate)
    engine.replace_verification_lifecycle(
        engine.verification_lifecycle.terminalize(
            _release_report(candidate, decision=ReleaseDecision.block)
        )
    )

    assert engine.verification_lifecycle.state.value == "blocked"
    assert not hasattr(engine, "verified_candidate_terminal_events")


def test_gated_release_rejects_cross_run_and_cross_root_candidates(engine_factory) -> None:
    root = engine_factory(run_id="root-a", verification_delivery=VerificationDelivery.gated)
    root.begin_evidence_collection(ledger_id="ledger-a")
    root.append_tool_evidence((_evidence_record(root),))
    candidate = _candidate(root)
    root.seal_candidate(candidate)
    root.replace_verification_lifecycle(root.verification_lifecycle.terminalize(_release_report(candidate)))

    other_run_gate = CandidateDeliveryGate(
        VerificationDelivery.gated,
        expected_run_id="root-b",
        expected_root_run_id="root-b",
    )
    with pytest.raises(ValueError, match="run_id"):
        other_run_gate.release(root.verification_lifecycle)

    child = engine_factory(
        run_id="child-a",
        root_run_id="root-a",
        parent_run_id="root-a",
        subagent_id="sub-a",
        verification_delivery=VerificationDelivery.gated,
    )
    child.begin_evidence_collection(ledger_id="ledger-child")
    child.append_tool_evidence((_evidence_record(child),))
    child_candidate = _candidate(child)
    child.seal_candidate(child_candidate)
    child.replace_verification_lifecycle(child.verification_lifecycle.terminalize(_release_report(child_candidate)))
    wrong_root_gate = CandidateDeliveryGate(
        VerificationDelivery.gated,
        expected_run_id="child-a",
        expected_root_run_id="root-b",
    )
    with pytest.raises(ValueError, match="root_run_id"):
        wrong_root_gate.release(child.verification_lifecycle)


@pytest.mark.asyncio
async def test_released_projection_is_typed_stable_and_survives_snapshot_reconnect(engine_factory) -> None:
    source = engine_factory(run_id="run-1")
    source.begin_evidence_collection(ledger_id="ledger-1")
    source.append_tool_evidence((_evidence_record(source),))
    citation = CitationSpan(
        claim_id="claim-1",
        evidence_record_id="record-1",
        start_offset=2,
        end_offset=8,
    )
    candidate = _candidate(source).model_copy(update={"citations": (citation,)})
    await source.seal_candidate_and_persist(candidate)
    source.replace_verification_lifecycle(
        source.verification_lifecycle.terminalize(_release_report(candidate))
    )

    gate = CandidateDeliveryGate(
        VerificationDelivery.gated,
        expected_run_id=source.config.run_id,
        expected_root_run_id=source.config.root_run_id,
    )
    emitted = gate.release(source.verification_lifecycle)
    assert emitted.event_payload() == {
        "candidate_id": "candidate-1",
        "report_id": "report-1",
        "decision": "release",
        "content_blocks": [{"kind": "text", "text": "sealed answer"}],
        "artifacts": [],
        "citations": [
            {
                "claim_id": "claim-1",
                "evidence_record_id": "record-1",
                "start_offset": 2,
                "end_offset": 8,
            }
        ],
        "evidence_ledger": {"ledger_id": "ledger-1", "digest": candidate.evidence_ledger.digest},
        "delivery_mode": "inline",
    }

    restored = engine_factory(run_id="run-1")
    await restored.resume_from_snapshot(source.snapshot())
    replay = gate.release(restored.verification_lifecycle)
    # The typed projection survives a reconnect.  Only the durable adapter can
    # assign a projection claim and publish it through its emit-once primitive.
    assert replay == emitted
    assert restored.candidate_released_projection().citations == (citation,)

    assert not hasattr(gate, "public_release_event")


def test_referenced_candidate_projection_preserves_ordered_citations(engine_factory) -> None:
    engine = engine_factory(run_id="run-1")
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_evidence_record(engine),))
    citation = CitationSpan(
        claim_id="claim-in-blob",
        evidence_record_id="record-1",
        start_offset=10,
        end_offset=22,
    )
    candidate = _candidate(engine).model_copy(
        update={
            "content_reference": "blob://sealed-candidate-1",
            "content_blocks": (),
            "citations": (citation,),
        }
    )
    engine.seal_candidate(candidate)
    engine.replace_verification_lifecycle(
        engine.verification_lifecycle.terminalize(_release_report(candidate))
    )

    gate = CandidateDeliveryGate(
        VerificationDelivery.gated,
        expected_run_id=engine.config.run_id,
        expected_root_run_id=engine.config.root_run_id,
    )
    projection = gate.release(engine.verification_lifecycle)
    assert projection.event_payload()["content_reference"] == "blob://sealed-candidate-1"
    assert projection.event_payload()["content_blocks"] == []
    assert projection.event_payload()["citations"] == [citation.model_dump(mode="json")]


@pytest.mark.asyncio
async def test_projection_never_exists_before_release_or_after_block(engine_factory) -> None:
    engine = engine_factory()
    engine.begin_evidence_collection(ledger_id="ledger-1")
    candidate = _candidate(engine)
    await engine.seal_candidate_and_persist(candidate)

    with pytest.raises(ValueError, match="released or warned lifecycle"):
        engine.candidate_released_projection()

    engine.replace_verification_lifecycle(
        engine.verification_lifecycle.terminalize(_release_report(candidate, decision=ReleaseDecision.block))
    )
    with pytest.raises(ValueError, match="released or warned lifecycle"):
        engine.candidate_released_projection()


def test_projection_rejects_hidden_candidate_block_even_if_constructed_outside_candidate_contract() -> None:
    with pytest.raises(ValueError, match="non-reader-visible"):
        CandidateReleasedProjection(
            candidate_id="candidate-1",
            report_id="report-1",
            decision=ReleaseDecision.release,
            content_blocks=(TextBlock(text="secret", visibility=BlockVisibility.HIDDEN),),
            evidence_ledger=EvidenceLedgerReference(ledger_id="ledger-1", digest="digest-1"),
            delivery_mode=DeliveryMode.inline,
        )


def test_gated_release_accepts_a_ledger_that_opened_on_an_earlier_prefix(engine_factory) -> None:
    engine = engine_factory(run_id="run-1", verification_delivery=VerificationDelivery.gated)
    engine.begin_evidence_collection(ledger_id="ledger-1")
    engine.append_tool_evidence((_evidence_record(engine),))
    open_ledger = engine.verification_lifecycle.ledger
    assert open_ledger is not None

    # Orchestration vouches for evidence this run never observed and could not
    # have: it belongs to a different run tree entirely.
    spliced = EvidenceLedger(
        ledger_id=open_ledger.ledger_id,
        attempt_owner=open_ledger.attempt_owner,
        records=open_ledger.records,
        inherited=InheritedEvidencePrefix(
            source_id="prefix-source-1",
            record_ids=("earlier-record-1",),
            digest="prefix-digest-1",
        ),
    )
    engine.replace_verification_lifecycle(
        VerificationLifecycle(state=VerificationState.executing, ledger=spliced)
    )

    citation = CitationSpan(
        claim_id="claim-1",
        evidence_record_id="earlier-record-1",
        start_offset=0,
        end_offset=6,
    )
    candidate = _candidate(engine).model_copy(update={"citations": (citation,)})
    engine.seal_candidate(candidate)
    engine.replace_verification_lifecycle(
        engine.verification_lifecycle.terminalize(_release_report(candidate))
    )

    gate = CandidateDeliveryGate(
        VerificationDelivery.gated,
        expected_run_id=engine.config.run_id,
        expected_root_run_id=engine.config.root_run_id,
    )
    projection = gate.release(engine.verification_lifecycle)

    assert projection.citations == (citation,)
