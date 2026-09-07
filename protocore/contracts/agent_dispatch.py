"""The subagent registry, and the dispatch that gives a child run an address.

Implemented by the host's subagent dispatcher.

Dispatch used to be a function call that blocked until the child was finished.
That shape decides more than it looks like it does: a caller with no handle
cannot ask how far along the child is, cannot stop it, and cannot start it and
carry on — so the parent held its turn and its slot in the tree budget for the
whole descendant run, and a child that outlived its parent's process was simply
lost. Dispatch therefore answers with a :data:`SubagentHandle`, which is the
pool's :class:`~protocore.contracts.background.WorkHandle` over a record of kind
``agent``: the same address, wait and stop a background command has, because a
child run is the same kind of thing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from protocore.contracts.background import WorkHandle
from protocore.contracts.types import SubagentDef, SubagentResult, SubagentTask

#: A launched child run: :meth:`WorkHandle.identity` names it, :meth:`wait`
#: produces its :class:`SubagentResult`, :meth:`stop` cancels it. Deliberately
#: an alias rather than a type of its own — there is one kind of launched work
#: in this system and one handle over it.
type SubagentHandle = WorkHandle[SubagentResult]


@runtime_checkable
class IDelegationTool(Protocol):
    """A registered tool whose calls start child runs.

    How the loop RECOGNISES delegation. It used to read a boolean class flag off
    whatever object the registry held, which meant any object that happened to
    carry an attribute of that name was treated as spawning child runs, and the
    real contract — that such a call costs the tree N runs and blocks on
    descendants — was written down nowhere.

    So the contract is the two questions the loop actually has to ask, and a
    tool answers them by implementing them. A flag cannot satisfy this and is
    not meant to: an object that declares no way to be asked how many runs a
    call starts has not made the promise the loop needs.
    """

    def child_run_count(self, arguments: Mapping[str, Any]) -> int:
        """How many child runs ONE call with these arguments starts.

        One call is not one run: a call carrying a batch of tasks starts one
        full child run per element, and a tree charged once per CALL advertises
        a cap it is off by the batch width from.
        """

    def is_background_call(self, arguments: Mapping[str, Any]) -> bool:
        """Whether this call returns as soon as its children are launched.

        A background delegation does not block the parent's turn and must not
        hold the parent's tree-budget slot for the children's lifetime.
        """


class AgentDispatchError(Exception):
    """Base for agent-dispatch domain errors."""


class SubagentNotFoundError(AgentDispatchError):
    """Requested subagent_id is not registered for the tenant.

    Carries structured ``requested``, ``available``, and ``suggestion``
    attributes so callers can render a self-healing error message
    ("Subagent 'docgen' is not registered. Available: coder, researcher,
    reviewer. Did you mean 'reviewer'?") without parsing the str() body.
    All fields are optional so existing call sites that only pass a message
    stay backward-compatible.
    """

    def __init__(
        self,
        message: str,
        *,
        requested: str | None = None,
        available: Sequence[str] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.requested = requested
        self.available: tuple[str, ...] = (
            tuple(available) if available is not None else ()
        )
        self.suggestion = suggestion


@runtime_checkable
class IAgentDispatch(Protocol):
    """Adapter Protocol over the per-tenant subagent registry."""

    async def list_subagents(self, tenant_id: str) -> Sequence[SubagentDef]:
        """Return all subagents available to a tenant (dashboard-managed)."""
        ...

    async def get(self, tenant_id: str, subagent_id: str) -> SubagentDef:
        """Fetch one subagent definition. Raise :class:`SubagentNotFoundError`."""
        ...

    async def dispatch(self, task: SubagentTask) -> SubagentHandle:
        """Launch the child run and answer with its handle, without waiting.

        The child has an id before it has produced anything, so a caller may
        keep the handle and collect later (``await handle.wait()``), abandon the
        wait and let the pool deliver the outcome as a wake, or stop the child.
        A caller that wants the old blocking behaviour writes the wait it always
        implicitly did.
        """
        ...


__all__ = [
    "AgentDispatchError",
    "IAgentDispatch",
    "IDelegationTool",
    "SubagentHandle",
    "SubagentNotFoundError",
]
