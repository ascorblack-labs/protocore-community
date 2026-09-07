"""What a tool DOES, stated by the host that registered it.

The runtime has to make decisions that depend on the KIND of a tool — is this
call the one that produced bytes on disk, is this the read that discharges a
read-back obligation, may a plan-only run reach this tool, does the permission
gate owe this call a shell-safety check. Every one of those questions used to
be answered by comparing the tool's name against a string spelled inside the
core, which quietly assumed that every installation names its tools the way the
first one did. A host that calls its shell tool something else lost the shell
deny-patterns, its large-file writes stopped converging, and nothing anywhere
said so: the comparison simply never matched.

So the core stops knowing names and starts knowing ROLES. The host declares,
once, at tool registration, which of its names carry which roles; the runtime
asks the map. A role the map does not mention is a capability this installation
does not have, and the features that need it say so out loud rather than going
inert (see :meth:`ToolRoleMap.names_with` and the callers' warnings).

The map also carries the ARGUMENT spellings that go with those roles — which
key holds the shell command, which holds the body of a write, which holds a
terminal tool's answer. Those are the host's field names too, and the runtime
reads raw arguments before any input model has resolved an alias, so it has to
be told them rather than guess.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class ToolRole(StrEnum):
    """One capability a registered tool has.

    A tool carries as many roles as it has capabilities: a tool that both
    overwrites a file and reports its size carries ``writes_path`` alone, while
    a tool that can create a file and append to it carries both
    ``writes_path`` and ``appends_path``. Roles are about what the call DOES,
    never about who is allowed to make it — permission is a separate decision
    that happens to consult these.
    """

    reads_path = "reads_path"
    """Returns the content of one file named by its path argument."""

    writes_path = "writes_path"
    """Creates or overwrites a whole file with the body it is given."""

    appends_path = "appends_path"
    """Adds to the end of an existing file without rewriting it."""

    edits_path = "edits_path"
    """Replaces part of a file in place; carries no whole-file body."""

    finalizes_path = "finalizes_path"
    """Seals an artifact that was written in pieces."""

    searches_workspace = "searches_workspace"
    """Answers a question about the workspace without returning a whole file."""

    runs_shell = "runs_shell"
    """Executes a command line."""

    fetches_url = "fetches_url"
    """Reaches a network host named in its arguments."""

    delegates_work = "delegates_work"
    """Starts a child run."""

    records_plan = "records_plan"
    """Writes the run's own plan or task list, not the workspace."""

    discovers_tools = "discovers_tools"
    """Widens the run's own tool surface."""

    asks_user = "asks_user"
    """Suspends the run until a person answers."""

    never_delegated = "never_delegated"
    """Never offered to a child run, whatever else the child is allowed."""


class ToolArgumentSlot(StrEnum):
    """A value the runtime reads out of raw tool arguments.

    One logical slot, many spellings: the runtime sees the arguments as the
    model emitted them, before the tool's own input model has resolved its
    validation aliases, so it must try every name the host accepts.
    """

    path = "path"
    content = "content"
    shell_command = "shell_command"
    url = "url"
    answer = "answer"


#: Roles whose calls change the workspace. A run held to a read-only profile
#: must not reach a tool carrying any of them.
WORKSPACE_MUTATION_ROLES: frozenset[ToolRole] = frozenset(
    {
        ToolRole.writes_path,
        ToolRole.appends_path,
        ToolRole.edits_path,
        ToolRole.finalizes_path,
    }
)

#: Roles whose calls produce bytes at a path — the byte production the
#: large-file convergence driver measures.
BYTE_PRODUCING_ROLES: frozenset[ToolRole] = frozenset(
    {ToolRole.writes_path, ToolRole.appends_path}
)

#: Roles whose calls look at the workspace without changing it.
WORKSPACE_INSPECTION_ROLES: frozenset[ToolRole] = frozenset(
    {ToolRole.reads_path, ToolRole.searches_workspace}
)

#: Roles whose calls only look at state. Repeating one of them costs a second
#: look and nothing else, which is what makes an interrupted call of such a
#: tool safe to simply re-issue.
REPEAT_SAFE_ROLES: frozenset[ToolRole] = frozenset(
    {ToolRole.reads_path, ToolRole.searches_workspace, ToolRole.discovers_tools}
)

#: Roles that change something outside the model's own message stream. The
#: published execution profiles mask exactly these away.
STATE_CHANGING_ROLES: frozenset[ToolRole] = (
    WORKSPACE_MUTATION_ROLES | {ToolRole.runs_shell, ToolRole.records_plan}
)


def _freeze_roles(
    declarations: Mapping[str, Iterable[ToolRole]],
) -> Mapping[str, frozenset[ToolRole]]:
    return MappingProxyType(
        {str(name): frozenset(roles) for name, roles in declarations.items()}
    )


def _freeze_aliases(
    declarations: Mapping[ToolArgumentSlot, Iterable[str]],
) -> Mapping[ToolArgumentSlot, tuple[str, ...]]:
    return MappingProxyType(
        {slot: tuple(dict.fromkeys(names)) for slot, names in declarations.items()}
    )


@dataclass(frozen=True, slots=True)
class ToolRoleMap:
    """The host's statement of what its tools do, as the runtime sees it.

    Empty by construction: the core ships no names, because it has none to
    ship. Build one with :meth:`declare` at registration time and hand it to
    the engine; a run whose map is empty simply has no tool in any role, and
    every feature that needs one reports that rather than pretending.
    """

    roles: Mapping[str, frozenset[ToolRole]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    argument_aliases: Mapping[ToolArgumentSlot, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @classmethod
    def declare(
        cls,
        roles: Mapping[str, Iterable[ToolRole]] | None = None,
        *,
        argument_aliases: Mapping[ToolArgumentSlot, Iterable[str]] | None = None,
    ) -> ToolRoleMap:
        """Build a map from a host's declarations, normalising the containers."""
        return cls(
            roles=_freeze_roles(roles or {}),
            argument_aliases=_freeze_aliases(argument_aliases or {}),
        )

    def with_tool(self, name: str, *roles: ToolRole) -> ToolRoleMap:
        """The same map plus one more tool. Registration builds a map this way."""
        merged = dict(self.roles)
        merged[name] = frozenset(merged.get(name, frozenset())) | frozenset(roles)
        return ToolRoleMap(
            roles=_freeze_roles(merged), argument_aliases=self.argument_aliases
        )

    # -- asking about a name ------------------------------------------------

    def roles_of(self, name: str | None) -> frozenset[ToolRole]:
        if not name:
            return frozenset()
        return self.roles.get(name, frozenset())

    def has_role(self, name: str | None, role: ToolRole) -> bool:
        return role in self.roles_of(name)

    def has_any_role(self, name: str | None, roles: Iterable[ToolRole]) -> bool:
        return bool(self.roles_of(name) & frozenset(roles))

    # -- asking about a role ------------------------------------------------

    def names_with(self, *roles: ToolRole) -> frozenset[str]:
        """Every declared name carrying at least one of ``roles``."""
        wanted = frozenset(roles)
        return frozenset(
            name for name, held in self.roles.items() if held & wanted
        )

    def sole_name(self, role: ToolRole) -> str | None:
        """The one name in ``role``, or ``None`` when there is not exactly one.

        Some decisions name a single tool — a forced ``tool_choice`` carries
        exactly one name, and so does the seal of an unfinished artifact.
        Ambiguity there is not resolvable by the runtime: two tools in the
        role means the host has to say which, and until it does the feature
        that needs one name has nothing to force.
        """
        candidates = sorted(self.names_with(role))
        return candidates[0] if len(candidates) == 1 else None

    def aliases(self, slot: ToolArgumentSlot) -> tuple[str, ...]:
        """Every spelling the host accepts for ``slot``, in priority order."""
        return self.argument_aliases.get(slot, ())

    def __bool__(self) -> bool:
        return bool(self.roles)


#: A map that says nothing. The default for a run whose host declared no roles.
EMPTY_TOOL_ROLE_MAP: ToolRoleMap = ToolRoleMap()


def narrow_tool_capabilities(
    offered: Iterable[str],
    *,
    roles: ToolRoleMap,
    withhold: Iterable[ToolRole] = (ToolRole.never_delegated,),
) -> frozenset[str]:
    """The subset of ``offered`` a narrower context may keep.

    Narrowing only, and that is the whole point: the result is always a subset
    of what was already offered, so this function can take capability away and
    can never hand any back. A caller that wants to widen a surface has to say
    so somewhere this cannot reach — which is what makes it safe to apply on
    every hop down a delegation chain, including one whose map came from a
    child's own registration.
    """
    kept = frozenset(offered)
    for role in withhold:
        kept -= roles.names_with(role)
    return kept


__all__ = [
    "BYTE_PRODUCING_ROLES",
    "EMPTY_TOOL_ROLE_MAP",
    "REPEAT_SAFE_ROLES",
    "STATE_CHANGING_ROLES",
    "WORKSPACE_INSPECTION_ROLES",
    "WORKSPACE_MUTATION_ROLES",
    "ToolArgumentSlot",
    "ToolRole",
    "ToolRoleMap",
    "narrow_tool_capabilities",
]
