"""Reading one named value out of raw tool arguments.

The runtime inspects arguments as the model emitted them — before the tool's
own input model has resolved a validation alias — so a single logical value
arrives under whichever spelling the model reached for. Every place that needed
such a value used to carry its own list of spellings, and the lists had drifted
apart: seven of them, disagreeing about the same two questions ("which key
holds the path", "which key holds the body"), so a call could be tracked by one
subsystem and invisible to the next.

There is one list now, and the host owns most of it: the spellings live on
:class:`~protocore.contracts.tool_roles.ToolRoleMap`, declared once where the
tools are registered.

The single exception is the path slot, whose canonical pair is stated below.
The runtime resolves a path for its OWN state — which file a run is converging
on, which read discharges a read-back obligation — on every installation and
for tools it knows only by role, so it cannot be left with nothing when a host
declares no spellings. Every other slot is read against one particular host
tool, and its spellings are that host's to name.
"""
from __future__ import annotations

from typing import Any, Final

from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRoleMap

#: The two spellings a file target reaches the runtime under. ``path`` is the
#: canonical field; ``file_path`` is the alias models reach for just as often.
_PATH_ARG_KEYS: Final[tuple[str, ...]] = ("path", "file_path")


def argument_names(slot: ToolArgumentSlot, *, roles: ToolRoleMap) -> tuple[str, ...]:
    """Every spelling to try for ``slot``, in priority order.

    A declared list REPLACES the canonical pair rather than extending it, so a
    host that declares only ``file_path`` for the path slot loses ``path``. That
    is deliberate — a host that names its spellings names all of them — but it
    means a partial declaration is narrower than no declaration at all.

    An empty result means the host declared nothing and the slot has no floor.
    A caller that must read the value to do its job — a safety policy inspecting
    a command or an address — treats that as a failure to check, not a pass.
    """
    declared = roles.aliases(slot)
    if declared:
        return declared
    return _PATH_ARG_KEYS if slot is ToolArgumentSlot.path else ()


def string_argument(
    arguments: Any, slot: ToolArgumentSlot, *, roles: ToolRoleMap
) -> str | None:
    """The first non-empty string ``arguments`` carries for ``slot``, or None."""
    if not isinstance(arguments, dict):
        return None
    for name in argument_names(slot, roles=roles):
        value = arguments.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def present_string_argument(
    arguments: Any, slot: ToolArgumentSlot, *, roles: ToolRoleMap
) -> str | None:
    """As :func:`string_argument`, but an empty string counts as present.

    A permission gate has to evaluate ``{"command": ""}`` as a command it saw
    and found harmless, not as a call with no command in it at all.
    """
    if not isinstance(arguments, dict):
        return None
    for name in argument_names(slot, roles=roles):
        value = arguments.get(name)
        if isinstance(value, str):
            return value
    return None


__all__ = ["argument_names", "present_string_argument", "string_argument"]
