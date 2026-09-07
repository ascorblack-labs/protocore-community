"""What a paused run is waiting for, as a value with an identity.

The single latch these tests replace could say one thing: *a* call is parked.
It could not say which kind of wait it was, could not carry what the person was
being shown, and could not hold a second call at all. Everything below is a
statement about one of those three gaps, checked at the contract level; the
scenarios in ``tests/scenarios/test_interrupt_*.py`` check the same guarantees
end to end through the public resume surface.
"""
from __future__ import annotations

import pytest

from protocore.contracts.interrupt import (
    InterruptDecision,
    InterruptKind,
    InterruptResolution,
    InterruptResolutionError,
    PendingInterrupt,
    deserialise_interrupts,
    find_interrupt,
    find_interrupt_for_call,
    interrupts_of_kind,
    park_interrupt,
    plan_resolution,
    release_interrupt,
    serialise_interrupts,
)
from protocore.contracts.snapshot import (
    PENDING_INTERRUPTS_SNAPSHOT_KEY,
    SNAPSHOT_SCHEMA_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotSchemaError,
    migrate_snapshot,
)
from protocore.runtime.loop_state import (
    LoopState,
    UnwitnessedAwaitError,
    assert_awaiting_is_witnessed,
)


def _approval(interrupt_id: str, call: str = "call-1") -> PendingInterrupt:
    return PendingInterrupt(
        interrupt_id=interrupt_id,
        kind=InterruptKind.approval,
        tool_call_id=call,
        tool_name="WriteFile",
        payload={"reason": "writes to disk"},
        created_at_ms=1_000,
    )


def _question(interrupt_id: str, call: str = "call-q") -> PendingInterrupt:
    return PendingInterrupt(
        interrupt_id=interrupt_id,
        kind=InterruptKind.question,
        tool_call_id=call,
        tool_name="AskUser",
        payload={"questions": [{"question": "which file?"}]},
        created_at_ms=1_000,
    )


# ---------------------------------------------------------------------------
# The value itself
# ---------------------------------------------------------------------------


def test_an_interrupt_survives_the_round_trip_it_exists_for() -> None:
    """A wait is written by one process and read by another; nothing may be lost.

    The payload is the part that has no other home: it is what the person is
    shown, and a round trip that dropped it would leave the next process able
    to say a call is parked and unable to say what it is parked for.
    """
    original = _question("int-1")
    assert PendingInterrupt.from_dict(original.to_dict()) == original


def test_a_kind_this_build_cannot_place_is_refused_rather_than_defaulted() -> None:
    """Defaulting an unknown wait would default it towards running something.

    ``approval`` is the kind whose resolution executes a tool. A reader that
    quietly read an unrecognised kind as an approval would let a wait it does
    not understand be answered with "yes, run it".
    """
    with pytest.raises(InterruptResolutionError, match="unknown kind"):
        PendingInterrupt.from_dict({"interrupt_id": "int-1", "kind": "webhook", "tool_call_id": "c"})


def test_an_interrupt_with_no_identity_is_refused() -> None:
    """Without an id there is nothing for a resolution map to be keyed on."""
    with pytest.raises(InterruptResolutionError, match="no interrupt_id"):
        PendingInterrupt.from_dict({"kind": "approval", "tool_call_id": "c"})
    with pytest.raises(InterruptResolutionError, match="no tool_call_id"):
        PendingInterrupt.from_dict({"interrupt_id": "int-1", "kind": "approval"})


def test_a_durable_set_that_is_not_a_list_is_refused_not_read_as_empty() -> None:
    """Reading a corrupt record as "nothing was parked" is the silent failure.

    A run that WAS waiting and comes back believing it was not will finalise
    over a call nobody answered.
    """
    with pytest.raises(InterruptResolutionError, match="must be a list"):
        deserialise_interrupts({"interrupt_id": "int-1"})
    assert deserialise_interrupts(None) == ()


def test_the_open_set_round_trips_in_the_order_it_was_parked() -> None:
    """Order is the order the calls appear in history, and is load-bearing."""
    parked = (_approval("int-1", "c1"), _question("int-2", "c2"))
    assert deserialise_interrupts(serialise_interrupts(parked)) == parked


def test_expiry_is_a_deadline_the_host_sets_and_absence_means_none() -> None:
    """An operator decision may take a week; a wait that quietly stopped being
    answerable overnight is worse than one still waiting."""
    assert _approval("int-1").is_expired(10**12) is False
    with_deadline = PendingInterrupt(
        interrupt_id="int-1",
        kind=InterruptKind.approval,
        tool_call_id="c1",
        created_at_ms=1_000,
        expires_at_ms=2_000,
    )
    assert with_deadline.is_expired(1_999) is False
    assert with_deadline.is_expired(2_000) is True


# ---------------------------------------------------------------------------
# The open set
# ---------------------------------------------------------------------------


def test_parking_the_same_call_twice_keeps_the_identity_already_handed_out() -> None:
    """A re-drive that re-parks a call must not rename the wait.

    The host has the first id on a card in front of a person. Minting a second
    would leave that person's answer addressed to an interrupt that no longer
    exists, and the position must hold too, so the batch renders in the order
    the model asked for it.
    """
    parked = park_interrupt((), _approval("int-1", "c1"))
    parked = park_interrupt(parked, _question("int-2", "c2"))
    reparked = park_interrupt(parked, _approval("int-9", "c1"))

    assert [item.interrupt_id for item in reparked] == ["int-1", "int-2"]
    assert reparked[0].tool_call_id == "c1"
    assert reparked[0].created_at_ms == 1_000


def test_a_wait_is_released_by_either_of_its_two_names() -> None:
    parked = (_approval("int-1", "c1"), _question("int-2", "c2"))
    assert release_interrupt(parked, interrupt_id="int-1") == (parked[1],)
    assert release_interrupt(parked, tool_call_id="c2") == (parked[0],)


def test_lookups_answer_by_identity_and_by_the_call_they_park() -> None:
    parked = (_approval("int-1", "c1"), _question("int-2", "c2"))
    assert find_interrupt(parked, "int-2") is parked[1]
    assert find_interrupt(parked, "int-9") is None
    assert find_interrupt_for_call(parked, "c1") is parked[0]
    assert interrupts_of_kind(parked, InterruptKind.question) == (parked[1],)


# ---------------------------------------------------------------------------
# The resolution map
# ---------------------------------------------------------------------------


def test_a_batch_is_resolved_in_one_map_in_the_order_it_was_parked() -> None:
    """The whole reason the wait is a list: three parked calls, one decision set.

    The plan comes back in park order rather than map order, so the results
    land in the order the model asked for them whatever order a client's map
    happens to iterate in.
    """
    parked = (_approval("int-1", "c1"), _approval("int-2", "c2"), _question("int-3", "c3"))
    resolutions = {
        "int-3": InterruptResolution("int-3", InterruptDecision.answer, answer="the second"),
        "int-2": InterruptResolution("int-2", InterruptDecision.deny, reason="too broad"),
        "int-1": InterruptResolution(
            "int-1", InterruptDecision.approve, updated_input={"path": "fixed.md"}
        ),
    }
    plan = plan_resolution(parked, resolutions)
    assert [interrupt.interrupt_id for interrupt, _ in plan] == ["int-1", "int-2", "int-3"]
    assert plan[0][1].updated_input == {"path": "fixed.md"}


def test_a_resolution_for_an_interrupt_that_is_not_open_is_refused() -> None:
    """A stale id answers nothing, and acting on it would answer nothing."""
    with pytest.raises(InterruptResolutionError, match="no interrupt 'int-9' is open"):
        plan_resolution(
            (_approval("int-1"),),
            {
                "int-1": InterruptResolution("int-1", InterruptDecision.approve),
                "int-9": InterruptResolution("int-9", InterruptDecision.approve),
            },
        )


def test_a_resolution_filed_under_the_wrong_key_is_refused() -> None:
    """The key and the body disagree about which wait is being answered."""
    with pytest.raises(InterruptResolutionError, match="filed under 'int-1' names"):
        plan_resolution(
            (_approval("int-1"),),
            {"int-1": InterruptResolution("int-2", InterruptDecision.approve)},
        )


@pytest.mark.parametrize(
    ("parked", "decision"),
    [
        (_approval("int-1"), InterruptDecision.answer),
        (_question("int-1"), InterruptDecision.approve),
        (_question("int-1"), InterruptDecision.deny),
    ],
)
def test_a_decision_that_does_not_answer_this_kind_of_wait_is_refused(
    parked: PendingInterrupt, decision: InterruptDecision
) -> None:
    """The distinction the single latch could not draw, enforced.

    Approving a question writes "approved" where the transcript wants a reply;
    answering an approval hands a decision nobody made to the tool that was
    never allowed to run.
    """
    body = InterruptResolution("int-1", decision, answer="x" if decision is InterruptDecision.answer else None)
    with pytest.raises(InterruptResolutionError, match="cannot be resolved by"):
        plan_resolution((parked,), {"int-1": body})


def test_an_empty_answer_is_refused() -> None:
    """An empty reply would be written into the transcript as the answer to a
    question the model really asked."""
    with pytest.raises(InterruptResolutionError, match="answered with nothing"):
        plan_resolution(
            (_question("int-1"),),
            {"int-1": InterruptResolution("int-1", InterruptDecision.answer, answer="   ")},
        )


def test_corrected_arguments_on_a_decision_that_never_runs_are_refused() -> None:
    """Nothing but an approval runs the call, so the correction would vanish."""
    with pytest.raises(InterruptResolutionError, match="carries corrected arguments"):
        plan_resolution(
            (_approval("int-1"),),
            {
                "int-1": InterruptResolution(
                    "int-1", InterruptDecision.deny, updated_input={"path": "x"}
                )
            },
        )


def test_a_map_that_leaves_an_interrupt_undecided_is_refused_by_default() -> None:
    """Silence about a wait is far more often a dropped one than a deliberate one."""
    parked = (_approval("int-1", "c1"), _approval("int-2", "c2"))
    with pytest.raises(InterruptResolutionError, match=r"\['int-2'\] would be"):
        plan_resolution(parked, {"int-1": InterruptResolution("int-1", InterruptDecision.deny)})


def test_a_partial_map_is_accepted_when_the_caller_says_so_outright() -> None:
    """An operator who decided two of three and will come back to the last."""
    parked = (_approval("int-1", "c1"), _approval("int-2", "c2"))
    plan = plan_resolution(
        parked,
        {"int-1": InterruptResolution("int-1", InterruptDecision.deny)},
        allow_partial=True,
    )
    assert [interrupt.interrupt_id for interrupt, _ in plan] == ["int-1"]


def test_a_resolution_round_trips_through_its_durable_form() -> None:
    body = InterruptResolution(
        "int-1", InterruptDecision.approve, updated_input={"path": "a"}, reason="ok"
    )
    assert InterruptResolution.from_dict(body.to_dict()) == body
    with pytest.raises(InterruptResolutionError, match="unknown decision"):
        InterruptResolution.from_dict({"interrupt_id": "int-1", "decision": "maybe"})


# ---------------------------------------------------------------------------
# The state that carries the wait
# ---------------------------------------------------------------------------


def test_awaiting_with_nothing_parked_is_refused() -> None:
    """A run that stops with nothing recorded that could resume it.

    Nothing names the answer that would move it, and it will not finish
    either — so the state is refused at the transition rather than discovered
    later as a run that simply never came back.
    """
    with pytest.raises(UnwitnessedAwaitError, match="no pending interrupt"):
        assert_awaiting_is_witnessed(LoopState.AWAITING, 0)
    assert_awaiting_is_witnessed(LoopState.AWAITING, 1)
    assert_awaiting_is_witnessed(LoopState.AWAITING, 3)
    # Every other state says what the run is doing on its own.
    assert_awaiting_is_witnessed(LoopState.RUNNING, 0)


# ---------------------------------------------------------------------------
# The lift from the shape that held one id
# ---------------------------------------------------------------------------


def test_a_payload_holding_one_parked_id_is_lifted_to_one_typed_approval() -> None:
    """The id was, in practice, always a call parked at a gate — and nothing more.

    The lift says exactly that and invents nothing around it: no question, no
    deadline, no tool name that was never recorded.
    """
    lifted = migrate_snapshot(
        {SNAPSHOT_SCHEMA_KEY: 5, "pending_approval_tool_call_id": "toolu_7"}
    )
    assert lifted[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION
    assert "pending_approval_tool_call_id" not in lifted
    parked = deserialise_interrupts(lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY])
    assert len(parked) == 1
    assert parked[0].kind is InterruptKind.approval
    assert parked[0].tool_call_id == "toolu_7"
    assert parked[0].expires_at_ms is None


def test_the_lifted_identity_is_the_same_on_every_read_of_the_same_payload() -> None:
    """Two readers of one stored run must name the same wait.

    A fresh id per read would hand two processes two names for one thing, and
    an answer addressed to one of them would miss.
    """
    payload = {SNAPSHOT_SCHEMA_KEY: 5, "pending_approval_tool_call_id": "toolu_7"}
    first = migrate_snapshot(dict(payload))[PENDING_INTERRUPTS_SNAPSHOT_KEY]
    second = migrate_snapshot(dict(payload))[PENDING_INTERRUPTS_SNAPSHOT_KEY]
    assert first == second


def test_a_payload_with_nothing_parked_lifts_to_an_empty_set() -> None:
    lifted = migrate_snapshot({SNAPSHOT_SCHEMA_KEY: 5, "pending_approval_tool_call_id": None})
    assert lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY] == []


def test_a_parked_call_whose_intent_says_it_ran_lifts_to_a_question() -> None:
    """One field held both parks, and the intent record is what tells them apart.

    Lifted as an approval, a stopped ask-user run comes back answerable by
    nothing: an answer is refused because the wait says approval, and approving
    would re-run the tool that already asked.
    """
    lifted = migrate_snapshot(
        {
            SNAPSHOT_SCHEMA_KEY: 5,
            "pending_approval_tool_call_id": "toolu_q",
            "open_intents": [
                {
                    "tool_call_id": "toolu_q",
                    "tool_name": "AskUser",
                    "state": "PAUSED_ASK_USER",
                }
            ],
        }
    )
    parked = deserialise_interrupts(lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY])
    assert [item.kind for item in parked] == [InterruptKind.question]
    assert parked[0].tool_name == "AskUser"


def test_a_parked_call_still_held_at_the_gate_stays_an_approval() -> None:
    lifted = migrate_snapshot(
        {
            SNAPSHOT_SCHEMA_KEY: 5,
            "pending_approval_tool_call_id": "toolu_a",
            "open_intents": [
                {
                    "tool_call_id": "toolu_a",
                    "tool_name": "Bash",
                    "state": "PENDING_APPROVAL",
                }
            ],
        }
    )
    parked = deserialise_interrupts(lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY])
    assert [item.kind for item in parked] == [InterruptKind.approval]


def test_a_payload_that_waits_and_names_nothing_is_refused_at_the_lift() -> None:
    """Nothing recorded in it could ever resume the run, so it is not lifted."""
    with pytest.raises(SnapshotSchemaError, match="AWAITING with no parked call"):
        migrate_snapshot(
            {
                SNAPSHOT_SCHEMA_KEY: 5,
                "state": "awaiting",
                "pending_approval_tool_call_id": None,
            }
        )


def test_a_wait_past_its_deadline_can_no_longer_be_approved() -> None:
    """A deadline nobody enforced was a knob that promised what it did not do.

    An expired approval is not a standing one: the deadline is the statement
    that nobody may act on this call any more, and running it late is exactly
    what it forbids.
    """
    expired = PendingInterrupt(
        interrupt_id="int_1",
        kind=InterruptKind.approval,
        tool_call_id="call-1",
        expires_at_ms=1_000,
    )

    with pytest.raises(InterruptResolutionError, match="expired at 1000"):
        plan_resolution(
            (expired,),
            {"int_1": InterruptResolution("int_1", InterruptDecision.approve)},
            now_ms=1_001,
        )


def test_an_expired_wait_can_still_be_abandoned() -> None:
    """Otherwise the run is stuck on it forever, which is worse than late."""
    expired = PendingInterrupt(
        interrupt_id="int_1",
        kind=InterruptKind.approval,
        tool_call_id="call-1",
        expires_at_ms=1_000,
    )

    plan = plan_resolution(
        (expired,),
        {"int_1": InterruptResolution("int_1", InterruptDecision.abandon)},
        now_ms=1_001,
    )

    assert [item.interrupt_id for item, _ in plan] == ["int_1"]


def test_a_wait_inside_its_deadline_resolves_normally() -> None:
    live = PendingInterrupt(
        interrupt_id="int_1",
        kind=InterruptKind.approval,
        tool_call_id="call-1",
        expires_at_ms=1_000,
    )

    plan = plan_resolution(
        (live,),
        {"int_1": InterruptResolution("int_1", InterruptDecision.approve)},
        now_ms=999,
    )

    assert len(plan) == 1
