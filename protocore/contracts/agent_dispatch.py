"""IAgentDispatch Protocol — subagent registry + sync dispatcher.

Implemented by the host's subagent dispatcher.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from protocore.contracts.types import SubagentDef, SubagentResult, SubagentTask


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

    async def dispatch(self, task: SubagentTask) -> SubagentResult:
        """Synchronous subagent dispatch — blocks until subagent returns."""
        ...


__all__ = ["AgentDispatchError", "IAgentDispatch", "SubagentNotFoundError"]
