"""Fork / clone a session path without mutating the source."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from protocore.contracts.runtime_constants import RuntimeConstants


@dataclass(slots=True)
class SessionBranch:
    session_id: str
    parent_session_id: str | None
    history: list[object]
    audit: dict[str, str]


def refuse_tree_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("session_tree_disabled")


def fork_session(
    history: list[object],
    *,
    upto_index: int,
    parent_session_id: str,
    rc: RuntimeConstants,
    actor: str = "user",
) -> SessionBranch:
    refuse_tree_when_disabled(rc.session_tree_enabled)
    cap = rc.session_tree_max_copy_messages
    copied = list(history[: max(0, upto_index) + 1][:cap])
    return SessionBranch(
        session_id=str(uuid4()),
        parent_session_id=parent_session_id,
        history=copied,
        audit={"parent_session_id": parent_session_id, "actor": actor, "kind": "fork"},
    )


def clone_session(
    history: list[object],
    *,
    settled: bool,
    parent_session_id: str,
    rc: RuntimeConstants,
    actor: str = "user",
) -> SessionBranch:
    refuse_tree_when_disabled(rc.session_tree_enabled)
    if not settled:
        raise ValueError("clone_requires_settled")
    branch = fork_session(
        history,
        upto_index=len(history) - 1,
        parent_session_id=parent_session_id,
        rc=rc,
        actor=actor,
    )
    branch.audit["kind"] = "clone"
    return branch


__all__ = ["SessionBranch", "clone_session", "fork_session", "refuse_tree_when_disabled"]
