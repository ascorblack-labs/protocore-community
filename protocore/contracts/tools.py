"""Tool ABC + ToolError hierarchy.

Core defines the contract only. The baseline tools a host is expected to
provide (Read, Write, Edit, Grep, Glob, Bash, WebFetch, Skill, Agent,
ToolSearch, TodoWrite) are all host-side adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import ToolDefinition, ToolResult
from protocore.contracts.verification import EvidenceProducerBinding, RunTreeOrigin

#: ``ToolContext.metadata`` key carrying a delegation child's position in its
#: concurrently-dispatched batch (0-based, LLM-requested order). Set only on the
#: concurrent delegation fan-out; absent on serial/single dispatch. The host
#: forwards it onto the child so same-path parent-ledger declarations resolve in
#: batch order (deterministic) rather than gather-completion order.
SUBAGENT_DISPATCH_ORDER_METADATA_KEY: Final[str] = "protocore.subagent_dispatch_order"

#: ``ToolContext.metadata`` key carrying a stable identity for the whole
#: concurrent fan-out GROUP (shared by every child in one ``asyncio.gather``;
#: distinct across groups and across turns). Paired with
#: :data:`SUBAGENT_DISPATCH_ORDER_METADATA_KEY` so the parent ledger can scope
#: batch-order resolution PER GROUP — a later group's same-path declaration wins
#: over an earlier group's regardless of batch width, matching cross-turn
#: last-writer-wins.
SUBAGENT_DISPATCH_GROUP_METADATA_KEY: Final[str] = "protocore.subagent_dispatch_group"

#: ``ToolContext.metadata`` key carrying the executing child's tree-budget permit
#: handle (a :class:`~protocore.runtime.subagent_budget.SubagentTreePermit`, an
#: in-memory object — not serialized, like the cancel ``asyncio.Event`` on the
#: helper bag). The parent acquires a tree slot at the dispatch site and stamps
#: the handle here so the host runner can lodge it in the CHILD's helper bag;
#: the child engine then release-while-awaits around its OWN nested delegation
#: gather. Absent on serial/single dispatch and on the root leader (which owns no
#: permit) ⇒ the child simply never releases/reacquires a tree slot.
SUBAGENT_TREE_PERMIT_METADATA_KEY: Final[str] = "protocore.subagent_tree_permit"


class ToolError(Exception):
    """Base class for tool-domain errors."""


class ToolInvocationError(ToolError):
    """Tool raised during invocation; caught by dispatcher."""


class ToolPolicyDenied(ToolError):
    """Tool invocation blocked by safety policy."""


class ToolContext(BaseModel):
    """Per-invocation context — injected by the loop.

    Tools must NOT mutate the context; treat as read-only.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tenant_id: str
    """Scope id (``tenants.id``). Keys hooks, secrets, RC, workspace, sessions."""
    account_id: str = ""
    """Owning account id (``tenants.account_id``).

    The skill bank is ACCOUNT-WIDE and flat (``skills.account_id``), so the
    ``Skill`` tool and any skill-store read MUST key on this — NOT ``tenant_id``,
    which is the (possibly different) scope id. Empty only when no account could
    be resolved for the scope; a skill lookup then resolves nothing rather than
    silently keying on the wrong id.
    """
    run_id: str
    session_id: str
    workspace_id: str | None = None
    evidence_origin: RunTreeOrigin | None = None
    """Trusted immutable run-tree identity for evidence produced by this tool.

    Root callers retain the historical minimal context shape.  QueryEngine
    always supplies the complete origin, including parent/subagent identifiers
    for descendants, so tools never infer provenance from text or metadata.
    """
    evidence_producer_binding: EvidenceProducerBinding | None = None
    """Immutable registered producer binding for this execution.

    The dispatcher installs it after resolving the actual registered tool.
    Tools may read this identity to construct an observation, but the
    dispatcher stamps it again before ledger admission.
    """
    evidence_admission_deferred: bool = False
    """Whether a parallel replay owns ordered admission for this invocation."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Adapter-bound tool. All 11 default tools live in the host.

    Concrete subclass implements:
        - :attr:`name`: stable identifier surfaced to LLM
        - :attr:`definition`: full :class:`ToolDefinition` for surface
        - :meth:`invoke`: actual side-effect
    """

    @property
    def evidence_producer(self) -> EvidenceProducerBinding | None:
        """Trusted binding configured for evidence this registered tool may emit.

        ``None`` is the safe default: returning evidence without a registry
        binding fails closed.  Tool implementations cannot select the binding
        for an individual invocation; the dispatcher stamps it onto records.
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name. Matches ``definition.name``."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Full tool definition (name + description + JSON Schema)."""

    @abstractmethod
    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke the tool with validated arguments.

        Contract:
            - Caller has already JSON-Schema-validated ``arguments``.
            - Caller has already passed safety policy gate.
            - On error: raise :class:`ToolInvocationError`; never swallow.
            - On policy denial: raise :class:`ToolPolicyDenied`.
        """


__all__ = [
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInvocationError",
    "ToolPolicyDenied",
]
