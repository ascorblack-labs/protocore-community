"""Session-scoped approval widening: exact, program, or multiplexer verb."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from protocore.contracts.runtime_constants import RuntimeConstants

GrantKind = Literal["exact", "program", "multiplexer_verb"]
_METACHAR = re.compile(r"[|&;<>`$(){}]|&&|\|\|")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*= ")


@dataclass(frozen=True, slots=True)
class CommandGrant:
    kind: GrantKind
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


def has_metachar_or_env(command: str) -> bool:
    stripped = command.strip()
    if _METACHAR.search(stripped):
        return True
    first = stripped.split(None, 1)[0] if stripped else ""
    return bool(_ENV_ASSIGN.match(first) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first))


def preview_widen(command: str, rc: RuntimeConstants) -> CommandGrant:
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
    rc: RuntimeConstants,
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
