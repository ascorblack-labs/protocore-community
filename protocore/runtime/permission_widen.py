"""Session-scoped approval widening: exact, program, or multiplexer verb."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from protocore.contracts.runtime_constants import LoopConstants

GrantKind = Literal["exact", "program", "multiplexer_verb"]
_METACHAR = re.compile(r"[|&;<>`$(){}]|&&|\|\|")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*= ")


@dataclass(frozen=True, slots=True)
class CommandGrant:
    kind: GrantKind
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CommandGrant:
        """Read back a grant this class wrote.

        A grant crosses a process boundary as the two strings ``to_dict``
        names, and whoever picks it up needs the object again: the match is
        made by :func:`grant_covers`, which asks the grant for its kind. A row
        naming a kind this build does not have is read as ``exact`` — the
        narrowest of the three, so an unreadable grant widens nothing.
        """
        kind = str(raw.get("kind", "exact"))
        narrowed: GrantKind = (
            "program"
            if kind == "program"
            else "multiplexer_verb"
            if kind == "multiplexer_verb"
            else "exact"
        )
        return cls(narrowed, str(raw.get("value", "")))


def has_metachar_or_env(command: str) -> bool:
    stripped = command.strip()
    if _METACHAR.search(stripped):
        return True
    first = stripped.split(None, 1)[0] if stripped else ""
    return bool(_ENV_ASSIGN.match(first) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first))


def preview_widen(command: str, rc: LoopConstants) -> CommandGrant:
    """What a widen button would store. Metachar / VAR= stay exact."""
    parts = command.strip().split()
    if not parts or has_metachar_or_env(command):
        return CommandGrant("exact", command.strip())
    program = parts[0]
    multiplexers = {
        item.strip()
        for item in rc.permission_widening_multiplexer_verbs.split(",")
        if item.strip()
    }
    if program in multiplexers and len(parts) >= 2:
        return CommandGrant("multiplexer_verb", f"{program} {parts[1]}")
    return CommandGrant("program", program)


def grant_covers(grant: CommandGrant, command: str) -> bool:
    """Session grant match. Operator allowlist prefix semantics are NOT reused."""
    text = command.strip()
    if grant.kind == "exact":
        return text == grant.value
    if has_metachar_or_env(text):
        return False
    parts = text.split()
    if not parts:
        return False
    if grant.kind == "program":
        return parts[0] == grant.value
    if grant.kind == "multiplexer_verb":
        want = grant.value.split()
        return len(parts) >= 2 and parts[0] == want[0] and parts[1] == want[1]
    return False


def apply_widen(
    grants: list[CommandGrant],
    command: str,
    *,
    kind: GrantKind,
    rc: LoopConstants,
) -> list[CommandGrant]:
    if not rc.permission_widening_enabled:
        raise ValueError("permission_widening_disabled")
    preview = preview_widen(command, rc)
    if kind == "exact":
        stored = CommandGrant("exact", command.strip())
    elif preview.kind == "exact":
        stored = preview
    else:
        stored = preview if kind == preview.kind else CommandGrant("exact", command.strip())
    return [*grants, stored]


__all__ = [
    "CommandGrant",
    "GrantKind",
    "apply_widen",
    "grant_covers",
    "has_metachar_or_env",
    "preview_widen",
]
