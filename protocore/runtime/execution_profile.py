"""Published execution profiles as a tool-visibility mask, not a second loop."""
from __future__ import annotations

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy

WRITE_CLASS: frozenset[str] = frozenset(
    {"Write", "Edit", "Bash", "AppendFile", "TodoWrite", "FinalizeFile"}
)


def parse_csv_names(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def apply_execution_profile(
    policy: ToolVisibilityPolicy,
    *,
    profile: str,
    rc: RuntimeConstants,
) -> ToolVisibilityPolicy:
    """Intersect an existing policy with the published plan allowlist.

    ``deep|direct`` stay orthogonal: this function never touches run_mode.
    When the flag is off, or the profile is not plan, the policy is unchanged.
    """
    if not rc.execution_profile_plan_enabled or profile != "plan":
        return policy
    allowed = parse_csv_names(rc.execution_profile_plan_tools)
    visible = set(policy.visible)
    if visible:
        visible &= set(allowed)
    else:
        visible = set(allowed)
    blocked = set(policy.blocked) | (WRITE_CLASS - allowed)
    return ToolVisibilityPolicy(
        visible=visible,
        blocked=blocked,
        pinned=set(policy.pinned) & (visible | set(policy.pinned) - WRITE_CLASS),
        forced_pinned=frozenset(
            name for name in policy.forced_pinned if name in allowed or name not in WRITE_CLASS
        ),
    )


def plan_forbids(name: str, *, profile: str, rc: RuntimeConstants) -> bool:
    if not rc.execution_profile_plan_enabled or profile != "plan":
        return False
    allowed = parse_csv_names(rc.execution_profile_plan_tools)
    return name not in allowed


__all__ = ["WRITE_CLASS", "apply_execution_profile", "parse_csv_names", "plan_forbids"]
