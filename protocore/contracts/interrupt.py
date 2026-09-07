"""What a run is waiting for, stated as a value rather than as a latch.

A run stops for a person more than one way. A gate parks a call and asks
whether it may run. A tool runs far enough to ask a question and waits for the
answer. A call is handed to something outside the run entirely and its result
arrives later, by a route the run does not drive. Those are three different
waits with three different answers, and until they are three different values
they are one boolean somewhere and a payload nobody can type.

That is what the single latch cost. One field holding one call id said *a*
call is parked and said nothing about which kind of wait it was, so an answered
question and an approved call arrived at the same door and the loop had to
guess which one it had been given. Guessing wrong in one direction runs a tool
nobody approved; guessing wrong in the other tells the model a question it
asked came back as a failure. And a batch of three parked calls could not be
said at all: the latch held one id, so three calls parked together became three
rounds of stop, ask, resume, each with its own snapshot and its own redraw.

So the wait is a value with an identity. :class:`PendingInterrupt` says which
call is parked, which kind of wait it is, what the person is being shown, when
it started and when it stops being answerable. The run holds however many of
them are open at once. Resuming supplies a **map** — one
:class:`InterruptResolution` per open interrupt, keyed by interrupt id — so
three parked calls are answered in one act, and answering two of three is a
refusal rather than a half-resume that leaves the third parked behind a run
that has moved on.

The refusals are the substance of this module. A resolution naming an
interrupt that is not open, a decision that does not belong to the kind it
answers (approving a question, answering an approval), an answer with nothing
in it, a map that leaves an interrupt unanswered — each of those is a caller
that has lost track of what the run is waiting for, and each is refused by
name. A partial map is accepted only when the caller says outright that it
means to leave the rest parked, because the alternative reading — a client
that dropped an interrupt on the floor — is the more likely one.

``updated_input`` is the one thing an approval may change. A person looking at
a parked call often does not want to answer yes or no to it as written; they
want to run it with the path corrected. Carrying the corrected arguments on
the approval makes that a decision rather than a denial followed by a retry
the model has to be talked into.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final
from uuid import uuid4


class InterruptKind(StrEnum):
    """What kind of wait an interrupt stands for.

    The distinction is not cosmetic: it decides which resolutions are legal,
    and therefore what a resumed run is allowed to do with the answer.
    """

    #: A gate parked the call before the tool was reached. Nothing ran.
    approval = "approval"
    #: The tool ran far enough to ask a person something and is holding.
    question = "question"
    #: The call was handed outside the run; its result arrives by another route.
    external_call = "external_call"


class InterruptDecision(StrEnum):
    """What was decided about one open interrupt."""

    #: Run the parked call, optionally with corrected arguments.
    approve = "approve"
    #: Do not run it. The refusal is recorded as the call's result.
    deny = "deny"
    #: The answer to what was asked, or the result that arrived from outside.
    answer = "answer"
    #: No decision will ever come. The call is closed as never having run.
    abandon = "abandon"


#: Which decisions answer which kind of wait. Approving a question or
#: answering an approval is a caller that has confused two different waits,
#: and running either one would act on a decision nobody made.
LEGAL_DECISIONS: Final[Mapping[InterruptKind, frozenset[InterruptDecision]]] = {
    InterruptKind.approval: frozenset(
        {InterruptDecision.approve, InterruptDecision.deny, InterruptDecision.abandon}
    ),
    InterruptKind.question: frozenset(
        {InterruptDecision.answer, InterruptDecision.abandon}
    ),
    InterruptKind.external_call: frozenset(
        {InterruptDecision.answer, InterruptDecision.abandon}
    ),
}


class InterruptResolutionError(ValueError):
    """A resolution map does not answer the interrupts that are actually open.

    A :class:`ValueError`, like every other refusal the resume entry raises, so
    a caller that already treats a refused resume as "leave the run alone"
    keeps doing so unchanged.
    """


def new_interrupt_id() -> str:
    """Mint an identity for one wait."""
    return f"int_{uuid4().hex[:16]}"


@dataclass(frozen=True, slots=True)
class PendingInterrupt:
    """One thing this run is waiting for, with an identity a host can quote.

    ``payload`` is what the person is shown — the approval reason and the
    arguments for a gate, the question and its choices for an ask. It travels
    verbatim; nothing in the core reads inside it, because what belongs in the
    card is the host's business and typing it here would make every new field
    a core release.

    ``expires_at_ms`` is when the wait stops being answerable. ``None`` means
    it does not expire, which is the right reading for an operator decision
    that may take a week.
    """

    interrupt_id: str
    kind: InterruptKind
    tool_call_id: str
    tool_name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0
    expires_at_ms: int | None = None

    def is_expired(self, now_ms: int) -> bool:
        """Whether this wait can no longer be answered at ``now_ms``."""
        return self.expires_at_ms is not None and now_ms >= self.expires_at_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "interrupt_id": self.interrupt_id,
            "kind": self.kind.value,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "payload": dict(self.payload),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingInterrupt:
        """Read one back, refusing a record whose kind this build cannot place.

        A kind that is not recognised is not defaulted to ``approval``: the
        default would be a decision about a wait this build does not
        understand, and approving is the one direction that runs something.
        """
        raw_kind = data.get("kind")
        try:
            kind = InterruptKind(str(raw_kind))
        except ValueError as exc:
            raise InterruptResolutionError(
                f"pending interrupt names an unknown kind {raw_kind!r}"
            ) from exc
        interrupt_id = data.get("interrupt_id")
        tool_call_id = data.get("tool_call_id")
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise InterruptResolutionError("pending interrupt has no interrupt_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise InterruptResolutionError(
                f"pending interrupt {interrupt_id!r} has no tool_call_id"
            )
        payload = data.get("payload")
        expires = data.get("expires_at_ms")
        created = data.get("created_at_ms")
        tool_name = data.get("tool_name")
        return cls(
            interrupt_id=interrupt_id,
            kind=kind,
            tool_call_id=tool_call_id,
            tool_name=tool_name if isinstance(tool_name, str) else "",
            payload=dict(payload) if isinstance(payload, Mapping) else {},
            created_at_ms=created if isinstance(created, int) and not isinstance(created, bool) else 0,
            expires_at_ms=(
                expires if isinstance(expires, int) and not isinstance(expires, bool) else None
            ),
        )


@dataclass(frozen=True, slots=True)
class InterruptResolution:
    """The decision on one open interrupt.

    ``updated_input`` belongs to :attr:`InterruptDecision.approve` alone: it is
    the corrected arguments the call is to run with, and the call is then
    recorded as having run with those, not with the ones the model proposed.
    ``answer`` carries the text that settles a question or an external call.
    ``reason`` is free text kept beside a denial or an abandonment so the
    record says why, and is never shown to the model in place of the result.
    """

    interrupt_id: str
    decision: InterruptDecision
    updated_input: dict[str, Any] | None = None
    answer: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "interrupt_id": self.interrupt_id,
            "decision": self.decision.value,
            "updated_input": dict(self.updated_input) if self.updated_input is not None else None,
            "answer": self.answer,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InterruptResolution:
        raw_decision = data.get("decision")
        try:
            decision = InterruptDecision(str(raw_decision))
        except ValueError as exc:
            raise InterruptResolutionError(
                f"resolution names an unknown decision {raw_decision!r}"
            ) from exc
        interrupt_id = data.get("interrupt_id")
        if not isinstance(interrupt_id, str) or not interrupt_id:
            raise InterruptResolutionError("resolution has no interrupt_id")
        updated = data.get("updated_input")
        answer = data.get("answer")
        reason = data.get("reason")
        return cls(
            interrupt_id=interrupt_id,
            decision=decision,
            updated_input=dict(updated) if isinstance(updated, Mapping) else None,
            answer=answer if isinstance(answer, str) else None,
            reason=reason if isinstance(reason, str) else None,
        )


def serialise_interrupts(
    interrupts: Iterable[PendingInterrupt],
) -> list[dict[str, Any]]:
    """The durable form of everything a run is waiting for."""
    return [item.to_dict() for item in interrupts]


def deserialise_interrupts(value: Any) -> tuple[PendingInterrupt, ...]:
    """Read the durable form back, or refuse it.

    A value that is not a list of mappings is refused rather than read as
    "nothing was parked": a run that WAS waiting and comes back believing it
    was not is a run that will finalise over a call it never answered.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise InterruptResolutionError(
            f"pending interrupts must be a list, got {type(value).__name__}"
        )
    parsed: list[PendingInterrupt] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise InterruptResolutionError(
                f"pending interrupt must be a mapping, got {type(item).__name__}"
            )
        parsed.append(PendingInterrupt.from_dict(item))
    return tuple(parsed)


def park_interrupt(
    interrupts: tuple[PendingInterrupt, ...],
    interrupt: PendingInterrupt,
) -> tuple[PendingInterrupt, ...]:
    """Add ``interrupt`` to the open set, keyed on the call it parks.

    A call can be waiting for one thing at a time, so a second park of the same
    call replaces the first IN PLACE — keeping its position, so the order the
    run parked them in is the order a host renders them, and keeping the
    identity the host was already given, so a re-drive that re-parks a call
    does not hand out a second id for the same wait.
    """
    for index, existing in enumerate(interrupts):
        if existing.tool_call_id == interrupt.tool_call_id:
            replaced = PendingInterrupt(
                interrupt_id=existing.interrupt_id,
                kind=interrupt.kind,
                tool_call_id=interrupt.tool_call_id,
                tool_name=interrupt.tool_name or existing.tool_name,
                payload=interrupt.payload,
                created_at_ms=existing.created_at_ms,
                expires_at_ms=interrupt.expires_at_ms,
            )
            return (*interrupts[:index], replaced, *interrupts[index + 1 :])
    return (*interrupts, interrupt)


def release_interrupt(
    interrupts: tuple[PendingInterrupt, ...],
    *,
    interrupt_id: str | None = None,
    tool_call_id: str | None = None,
) -> tuple[PendingInterrupt, ...]:
    """Drop the wait that has been answered, by either of its two names."""
    return tuple(
        item
        for item in interrupts
        if not (
            (interrupt_id is not None and item.interrupt_id == interrupt_id)
            or (tool_call_id is not None and item.tool_call_id == tool_call_id)
        )
    )


def find_interrupt(
    interrupts: Iterable[PendingInterrupt], interrupt_id: str
) -> PendingInterrupt | None:
    for item in interrupts:
        if item.interrupt_id == interrupt_id:
            return item
    return None


def find_interrupt_for_call(
    interrupts: Iterable[PendingInterrupt], tool_call_id: str
) -> PendingInterrupt | None:
    for item in interrupts:
        if item.tool_call_id == tool_call_id:
            return item
    return None


def interrupts_of_kind(
    interrupts: Iterable[PendingInterrupt], kind: InterruptKind
) -> tuple[PendingInterrupt, ...]:
    return tuple(item for item in interrupts if item.kind is kind)


def plan_resolution(
    interrupts: tuple[PendingInterrupt, ...],
    resolutions: Mapping[str, InterruptResolution],
    *,
    allow_partial: bool = False,
    now_ms: int | None = None,
) -> tuple[tuple[PendingInterrupt, InterruptResolution], ...]:
    """Pair every open interrupt with its decision, or refuse the map.

    Returned in the order the run parked them, which is the order the calls
    appear in history — so a batch resolved together lands its results in the
    order the model asked for them, and not in whatever order a client's map
    happened to iterate.

    Every refusal here is a caller that has lost track of what the run is
    waiting for:

    * a map naming an interrupt that is not open — the id is stale, or it
      belongs to a different run, and acting on it would answer nothing;
    * a decision that does not answer this kind of wait — approving a question
      or answering an approval, either of which acts on a decision the person
      did not make;
    * an ``approve`` carrying no call to run, or an ``answer`` carrying nothing
      to answer with, which would write an empty result over a real question;
    * ``updated_input`` on anything but an approval, since nothing else runs
      the call and the corrected arguments would be silently dropped;
    * a decision other than ``abandon`` on a wait whose deadline has passed,
      when the caller states the time. An expired approval is not a standing
      one: the deadline is the host's statement that nobody may act on this
      call any more, and running it late is precisely what it forbids.
      ``abandon`` stays legal, because an expired wait still has to be
      clearable or the run is stuck on it forever. A caller that passes no
      ``now_ms`` is not asking the question and no deadline is checked;
    * a map that leaves an open interrupt undecided, unless the caller says
      outright it means to. Silence about an interrupt is far more often a
      client that dropped one than a client that meant to leave it parked, and
      resuming on it moves the run past a call still waiting for a person.
    """
    keyed = {item.interrupt_id: item for item in interrupts}
    for interrupt_id, resolution in resolutions.items():
        if resolution.interrupt_id != interrupt_id:
            raise InterruptResolutionError(
                f"resolution filed under {interrupt_id!r} names interrupt "
                f"{resolution.interrupt_id!r}"
            )
        pending = keyed.get(interrupt_id)
        if pending is None:
            raise InterruptResolutionError(
                f"no interrupt {interrupt_id!r} is open on this run; open: "
                f"{sorted(keyed)}"
            )
        legal = LEGAL_DECISIONS[pending.kind]
        if resolution.decision not in legal:
            raise InterruptResolutionError(
                f"interrupt {interrupt_id!r} is a {pending.kind.value} and cannot "
                f"be resolved by {resolution.decision.value}; it takes one of "
                f"{sorted(item.value for item in legal)}"
            )
        if (
            resolution.decision is InterruptDecision.answer
            and not (resolution.answer or "").strip()
        ):
            raise InterruptResolutionError(
                f"interrupt {interrupt_id!r} is answered with nothing; an empty "
                "answer would be written to the transcript as the reply to a "
                "question that was really asked"
            )
        if (
            now_ms is not None
            and pending.is_expired(now_ms)
            and resolution.decision is not InterruptDecision.abandon
        ):
            raise InterruptResolutionError(
                f"interrupt {interrupt_id!r} expired at {pending.expires_at_ms}; "
                f"it cannot be resolved by {resolution.decision.value} any more. "
                "Abandon it to clear the wait."
            )
        if (
            resolution.updated_input is not None
            and resolution.decision is not InterruptDecision.approve
        ):
            raise InterruptResolutionError(
                f"interrupt {interrupt_id!r} carries corrected arguments on a "
                f"{resolution.decision.value}, which never runs the call"
            )

    if not allow_partial:
        undecided = [item.interrupt_id for item in interrupts if item.interrupt_id not in resolutions]
        if undecided:
            raise InterruptResolutionError(
                f"this run is waiting on {len(interrupts)} interrupt(s) and the "
                f"resolution map answers {len(resolutions)}; {undecided} would be "
                "left parked. Answer them all, or pass allow_partial to say that "
                "is what you mean."
            )

    return tuple(
        (item, resolutions[item.interrupt_id])
        for item in interrupts
        if item.interrupt_id in resolutions
    )


__all__ = [
    "LEGAL_DECISIONS",
    "InterruptDecision",
    "InterruptKind",
    "InterruptResolution",
    "InterruptResolutionError",
    "PendingInterrupt",
    "deserialise_interrupts",
    "find_interrupt",
    "find_interrupt_for_call",
    "interrupts_of_kind",
    "new_interrupt_id",
    "park_interrupt",
    "plan_resolution",
    "release_interrupt",
    "serialise_interrupts",
]
