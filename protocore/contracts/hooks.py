"""IHookManager Protocol + HookResult + per-event HookSpec definitions.

Spec lives in core; executor (HTTP POST URL / LLM-as-hook prompt) is the host.
8 hook events — see :class:`~protocore.contracts.types.HookEvent`.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import HookEvent

HookActionLiteral = Literal["allow", "deny", "modify"]
HookExecutorKind = Literal["http", "prompt", "judge"]
HookFailureMode = Literal["continue", "fail_run"]


class HookActionKind:
    """Hook action constants — return shape from a hook execution."""

    ALLOW: Final[HookActionLiteral] = "allow"
    DENY: Final[HookActionLiteral] = "deny"
    MODIFY: Final[HookActionLiteral] = "modify"


class HookResult(BaseModel):
    """Result of invoking hooks for one event."""

    model_config = ConfigDict(frozen=True)

    action: HookActionLiteral = HookActionKind.ALLOW
    reason: str = ""
    """Human-readable explanation (logged + surfaced in audit)."""

    modifications: dict[str, Any] = Field(default_factory=dict)
    """Optional payload patches (only honored for ``action='modify'``)."""

    raw_results: list[dict[str, Any]] = Field(default_factory=list)
    """Per-hook raw results (for telemetry; not interpreted by loop)."""


class HookSpec(BaseModel):
    """Registered hook descriptor (HTTP endpoint, LLM prompt, or LLM judge).

    ``config`` shape depends on ``executor``:

    * ``http``: ``{target_url: str, headers: dict[str, str], timeout_ms: int,
      max_retries: int, allowed_internal_targets: list[str]}``
    * ``prompt``: ``{prompt_md: str, provider: str, model: str,
      response_schema: dict, timeout_ms: int, max_tokens: int}``
    * ``judge``: ``{policy_md: str, provider_id: str | None,
      model: str | None, timeout_ms: int, fail_mode: "allow" | "deny"}``
      — LLM-as-judge inspector (adversary review).
      ``policy_md`` is the user-authored ``adversary.md`` body or skill
      reference; the executor renders ``payload`` into the prompt + asks
      a dedicated LLM for an allow/deny/require_approval decision.

    The persistent row also carries a ``name`` (human-readable label, unique per
    tenant + event surface in the dashboard), a ``failure_mode``, and a
    ``bundled`` flag (seeded reference hooks cannot be DELETEd). All three
    round-trip through the PG layer per migration 026.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: str
    event: HookEvent
    executor: HookExecutorKind
    config: dict[str, Any] = Field(default_factory=dict)
    matchers: dict[str, Any] = Field(default_factory=dict)
    """JSONB filter expression — ``{tool_name: {$in: [...]}, ...}``."""

    priority: int = 100
    failure_mode: HookFailureMode = "continue"
    enabled: bool = True
    bundled: bool = False
    name: str | None = None
    """Human-readable label — surfaces in dashboard + audit."""


class HookInvocation(BaseModel):
    """Audit record emitted on every hook fire."""

    model_config = ConfigDict(frozen=True)

    hook_id: str
    hook_event: HookEvent
    outcome: Literal["success", "blocked", "mutated", "error", "timeout", "skipped"]
    duration_ms: int
    error: str | None = None


@runtime_checkable
class IHookManager(Protocol):
    """Adapter Protocol over the hook registry + executor."""

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        """Run all registered hooks for ``event`` in declared order.

        Contract:
            - Hooks are invoked in priority order (lower = first).
            - Sync (Pre*) hooks block; first DENY short-circuits.
            - Async (Post*) hooks fire concurrently.
            - Matchers filter on payload fields before dispatching.
            - Error from a hook is logged + isolated (per failure_mode).
            - Returns the AGGREGATE result.
        """
        ...

    async def register(self, spec: HookSpec) -> None:
        """Register a new hook for a tenant."""
        ...

    async def unregister(self, hook_id: str, tenant_id: str) -> None:
        """Remove a registered hook."""
        ...

    async def list(self, tenant_id: str, *, event: HookEvent | None = None) -> Sequence[HookSpec]:
        """List registered hooks (optionally filtered by event)."""
        ...


__all__ = [
    "HookActionKind",
    "HookExecutorKind",
    "HookFailureMode",
    "HookInvocation",
    "HookResult",
    "HookSpec",
    "IHookManager",
]
