"""Generic projection rules for verification-gated candidate delivery.

This module deliberately operates on typed turn events and sealed candidate
contracts.  It never inspects generated prose, URLs, citations, or tool names.
The service that owns a public event stream decides when to use the gate and
durably coordinates any terminal projection.
"""

from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.verification import (
    CandidateReleasedProjection,
    VerificationDelivery,
    VerificationLifecycle,
)
from protocore.runtime.events import EventType, TurnEvent

_HELD_CANDIDATE_EVENT_TYPES = frozenset(
    {
        EventType.CONTENT_BLOCK_START,
        EventType.CONTENT_BLOCK_DELTA,
        EventType.CONTENT_BLOCK_STOP,
    }
)


@dataclass(frozen=True)
class CandidateDeliveryGate:
    """Decide whether an internal turn event may reach a reader-facing stream.

    ``None`` and ``optimistic`` retain the existing stream byte-for-byte: the
    caller receives the original event object unchanged.  ``gated`` holds every
    content-block frame, rather than attempting to infer whether a delta is an
    answer.  Tool, state, heartbeat, and subagent progress events remain
    available to the caller.  Held frames are intentionally never buffered or
    replayed here; a successful decision is represented by one separate typed
    candidate projection.
    """

    delivery: VerificationDelivery | None = None
    expected_run_id: str | None = None
    expected_root_run_id: str | None = None
    @property
    def is_gated(self) -> bool:
        """Return whether reader-facing content must be held."""
        return self.delivery is VerificationDelivery.gated

    def __post_init__(self) -> None:
        """Require immutable routing identity whenever delivery is gated."""
        if not self.is_gated:
            return
        if not self.expected_run_id or self.expected_run_id.strip() != self.expected_run_id:
            raise ValueError("gated delivery requires a non-padded expected run_id")
        if (
            not self.expected_root_run_id
            or self.expected_root_run_id.strip() != self.expected_root_run_id
        ):
            raise ValueError("gated delivery requires a non-padded expected root_run_id")

    def permits(self, event: TurnEvent) -> bool:
        """Return whether an ordinary event may reach the public stream.

        In gated mode a release-shaped ``TurnEvent`` is not sufficient proof
        of authorization.  It is rejected even when its payload happens to
        resemble a projection; a durable delivery coordinator must validate
        the typed lifecycle through :meth:`release` before it publishes.
        """
        return not self.is_gated or (
            event.type not in _HELD_CANDIDATE_EVENT_TYPES
            and event.type is not EventType.CANDIDATE_RELEASED
        )

    def release(self, lifecycle: VerificationLifecycle) -> CandidateReleasedProjection:
        """Validate a terminal lifecycle and return its typed projection.

        The lifecycle contract verifies the sealed candidate, report decision,
        evidence ledger, and citation bindings.  This boundary deliberately
        does not manufacture reader events: a durable delivery adapter must
        claim the projection, atomically emit it once, checkpoint completion,
        and recover either terminal outcome.  Core has no persistence contract
        capable of making that sequence exactly-once.
        """
        if not self.is_gated:
            raise ValueError("candidate release capability requires gated delivery")
        projection = CandidateReleasedProjection.from_lifecycle(lifecycle)
        candidate = lifecycle.candidate
        if candidate is None:  # Defensive; from_lifecycle already validates this.
            raise ValueError("candidate release projection requires candidate")
        ledger = lifecycle.ledger
        # Defensive; the lifecycle requires candidate and ledger together, and
        # the ledger's owner is a required field.
        if ledger is None:
            raise ValueError("candidate release projection requires a bound evidence ledger")
        if candidate.run_id != self.expected_run_id:
            raise ValueError("candidate release run_id does not match delivery boundary")
        if ledger.attempt_owner.root_run_id != self.expected_root_run_id:
            raise ValueError("candidate release root_run_id does not match delivery boundary")
        return projection


__all__ = ["CandidateDeliveryGate"]
