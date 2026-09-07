"""Published execution profiles as a tool-visibility mask, not a second loop."""
from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    STATE_CHANGING_ROLES,
    ToolRoleMap,
)
from protocore.logging_utils import get_logger

_logger = get_logger(__name__)


def parse_csv_names(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def write_class(roles: ToolRoleMap) -> frozenset[str]:
    """The host's tools that change state — the ones a plan-only run loses.

    Derived from the role map, because which name is a writing tool is the
    host's fact, not the core's. An empty answer means the host declared no
    state-changing tool: the profile then masks nothing, which is reported by
    :func:`apply_execution_profile` rather than passed off as a plan.
    """
    return roles.names_with(*STATE_CHANGING_ROLES)


def apply_execution_profile(
    policy: ToolVisibilityPolicy,
    *,
    profile: str,
    rc: LoopConstants,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> ToolVisibilityPolicy:
    """Intersect an existing policy with the published plan allowlist.

    ``deep|direct`` stay orthogonal: this function never touches run_mode.
    When the flag is off, or the profile is not plan, the policy is unchanged.
    """
    if not rc.execution_profile_plan_enabled or profile != "plan":
        return policy
    allowed = parse_csv_names(rc.execution_profile_plan_tools)
    writing = write_class(roles)
    if not writing:
        # The mask still intersects the allowlist, but nothing is being held
        # back BY ROLE — the run is called a plan while its host has named no
        # tool that could break one. Say so; a silent plan profile that admits
        # every write is the failure this mask exists to prevent.
        _logger.warning(
            "execution profile %r is masking a surface whose role map declares "
            "no state-changing tool; nothing is withheld by role",
            profile,
        )
    visible = set(policy.visible)
    if visible:
        visible &= set(allowed)
    else:
        visible = set(allowed)
    blocked = set(policy.blocked) | (writing - allowed)
    return ToolVisibilityPolicy(
        visible=visible,
        blocked=blocked,
        pinned=set(policy.pinned) & (visible | set(policy.pinned) - writing),
        forced_pinned=frozenset(
            name for name in policy.forced_pinned if name in allowed or name not in writing
        ),
    )


def plan_forbids(name: str, *, profile: str, rc: LoopConstants) -> bool:
    if not rc.execution_profile_plan_enabled or profile != "plan":
        return False
    allowed = parse_csv_names(rc.execution_profile_plan_tools)
    return name not in allowed


__all__ = ["apply_execution_profile", "parse_csv_names", "plan_forbids", "write_class"]
