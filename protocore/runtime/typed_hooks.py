"""Host-registered typed hooks. No tenant JS, no exec in the loop pod."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from protocore.contracts.runtime_constants import RuntimeConstants

HookName = str
HookDecision = Literal["allow", "deny", "require_approval", "rewrite"]

PUBLISHED_HOOKS: tuple[str, ...] = (
    "before_run",
    "before_tool",
    "after_tool",
    "transform_context",
    "before_compact",
    "after_compact",
)


@dataclass(slots=True)
class HookOutcome:
    decision: HookDecision
    reason: str = ""
    rewrite: dict[str, Any] | None = None
    approval_token: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "rewrite": self.rewrite,
            "approval_token": self.approval_token,
        }


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {name: [] for name in PUBLISHED_HOOKS}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        if name not in PUBLISHED_HOOKS:
            raise ValueError("unknown_hook")
        self._handlers[name].append(handler)

    def names(self) -> list[str]:
        return [name for name, items in self._handlers.items() if items]


def refuse_hooks_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("typed_hooks_disabled")


def dispatch_hook(
    registry: HookRegistry,
    name: str,
    payload: dict[str, Any],
    rc: RuntimeConstants,
) -> HookOutcome:
    if not rc.typed_hooks_enabled:
        return HookOutcome(decision="allow")
    if name not in PUBLISHED_HOOKS:
        raise ValueError("unknown_hook")
    handlers = registry._handlers.get(name) or []
    outcome = HookOutcome(decision="allow")
    for handler in handlers:
        result = handler(payload)
        if not isinstance(result, HookOutcome):
            continue
        if result.decision == "deny":
            return result
        if result.decision == "require_approval":
            return result
        if result.decision == "rewrite":
            outcome = result
    return outcome


__all__ = [
    "PUBLISHED_HOOKS",
    "HookDecision",
    "HookName",
    "HookOutcome",
    "HookRegistry",
    "dispatch_hook",
    "refuse_hooks_when_disabled",
]
