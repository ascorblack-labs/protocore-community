"""What a child run is allowed to do, computed from its parent and nothing else.

Narrowing only. A delegated run may be given less than the run that started it
and can never be given more — not a tool, not a permission, not another hop of
depth — because a run that could widen on the way down would make every bound
above it advisory. Stated as a pure function so it can be applied wherever the
question is asked, and it is asked twice: once when the catalogue for the child
is resolved, and again on each of the child's tool calls.

Twice, because the two used to be able to disagree. The catalogue narrowing and
the four-stage permission gate were separate pieces of arithmetic over separate
inputs, and nothing anywhere required them to reach the same answer — so a tool
withheld from a child's advertised surface could still be dispatched by a child
that named it, and a tool the gate refused could still be advertised. Neither
half is redundant: the catalogue keeps a child from being shown what it may not
have, and the gate keeps it from having what it was not shown.

No input, no output, no clock. Given the same parent and the same definition it
answers the same thing, which is what makes it safe to run on every call.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    ToolRole,
    ToolRoleMap,
    narrow_tool_capabilities,
)
from protocore.contracts.types import SubagentDef

#: A depth ceiling of ``0`` means unbounded, matching the sentinel every other
#: tree bound uses.
_UNBOUNDED_DEPTH = 0


@dataclass(frozen=True, slots=True)
class ParentCapabilities:
    """What the run doing the delegating has, as the child's ceiling."""

    tools: frozenset[str] = frozenset()
    permission_mode: str = ""
    depth: int = 0


@dataclass(frozen=True, slots=True)
class ChildCapabilities:
    """What the delegated run gets. A subset of its parent's, always."""

    tools: frozenset[str] = frozenset()
    permission_mode: str = ""
    depth: int = 0
    may_spawn: bool = True


def stricter_permission_mode(
    parent: str, child: str, *, order: Sequence[str] = ()
) -> str:
    """The stricter of two permission modes, or the parent's when unrankable.

    ``order`` is the host's own ladder of modes, loosest first — core ships no
    mode names, because a mode is the host's word for how much a person is
    asked before a call runs.

    An empty child mode inherits, which is the common case: a definition that
    says nothing about permissions is not a statement that it wants the
    default. A mode neither side can rank leaves the parent's in place: a mode
    that cannot be shown to be stricter has not been shown to be safe to adopt,
    and adopting it is the one direction this function is not allowed to go.
    """
    if not child:
        return parent
    if not parent:
        return child
    if child == parent:
        return parent
    try:
        return max(parent, child, key=list(order).index)
    except ValueError:
        return parent


def narrow_child_capabilities(
    parent: ParentCapabilities,
    definition: SubagentDef,
    *,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
    max_depth: int = _UNBOUNDED_DEPTH,
    permission_mode_order: Sequence[str] = (),
) -> ChildCapabilities:
    """The capabilities a child run gets under ``definition``.

    The tool set is ``parent ∩ declared, minus blocked, minus never_delegated``. The
    intersection is what makes this narrowing: a definition naming a tool its
    parent does not have gets nothing, rather than the tool. A definition that
    declares no tools inherits the parent's set — it has said nothing about
    tools, which is different from having said "none".

    The names that are never delegated come from the host's role declarations,
    so no tool name is spelled here. A tool in that role is withheld from every
    child at every depth, whatever else the definition says.

    Depth goes up by one and never down. A child at the ceiling loses the right
    to spawn, and loses it in the only way that can be enforced: the delegating
    tools come out of its set, so a run that cannot delegate is not carrying a
    tool that would let it.
    """
    declared = frozenset(definition.tool_whitelist)
    tools = parent.tools & declared if declared else frozenset(parent.tools)
    tools -= frozenset(definition.blocked_tools)
    tools = narrow_tool_capabilities(tools, roles=roles)

    depth = parent.depth + 1
    may_spawn = max_depth <= _UNBOUNDED_DEPTH or depth < max_depth
    if not may_spawn:
        tools -= roles.names_with(ToolRole.delegates_work)

    return ChildCapabilities(
        tools=tools,
        permission_mode=stricter_permission_mode(
            parent.permission_mode,
            definition.permission_mode,
            order=permission_mode_order,
        ),
        depth=depth,
        may_spawn=may_spawn,
    )


__all__ = [
    "ChildCapabilities",
    "ParentCapabilities",
    "narrow_child_capabilities",
    "stricter_permission_mode",
]
