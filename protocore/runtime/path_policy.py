"""Published path protection for Write/Edit/Bash. Deny is a tool error, not a crash."""
from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

from protocore.contracts.runtime_constants import RuntimeConstants

_COMMAND_SPLIT = re.compile(r"[\s;|&<>()`]+")


def _globs(rc: RuntimeConstants) -> list[str]:
    return [item.strip() for item in rc.path_protection_deny_globs.split(",") if item.strip()]


def deny_reason(
    path: str,
    *,
    workspace_root: str,
    user_id: str,
    rc: RuntimeConstants,
) -> str | None:
    """Return a deny reason or None. Flags off → always None."""
    if not rc.path_protection_enabled:
        return None
    raw = path.strip()
    if not raw:
        return "empty_path"
    posix = PurePosixPath(raw)
    if rc.path_protection_workspace_only:
        if posix.is_absolute() or raw.startswith("/"):
            return "outside_workspace"
        if ".." in posix.parts:
            return "outside_workspace"
        root = workspace_root.strip("/")
        if user_id and user_id != root:
            # A path that names a different workspace owner.
            if user_id not in posix.parts and any(
                part != root and ("-" in part or part.startswith("usr"))
                for part in posix.parts
            ):
                return "foreign_workspace"
        if user_id and any(part != user_id and part != root for part in posix.parts[:1]) and "-" in (posix.parts[0] if posix.parts else ""):
            if posix.parts[0] != user_id:
                return "foreign_workspace"
    for pattern in _globs(rc):
        if fnmatch.fnmatch(raw, pattern) or fnmatch.fnmatch(str(posix), pattern):
            return "protected_path"
        if pattern.startswith("/") and (raw.startswith(pattern.rstrip("*")) or str(posix).startswith(pattern.rstrip("*"))):
            return "protected_path"
        if pattern == "/etc/**" and (raw.startswith("/etc/") or raw == "/etc" or raw.startswith("etc/")):
            return "protected_path"
    return None


def paths_in_command(command: str) -> list[str]:
    """Filesystem-looking tokens in a Bash command. URLs are skipped."""
    out: list[str] = []
    for raw in _COMMAND_SPLIT.split(command or ""):
        tok = raw.strip().strip("'\"")
        if not tok or tok in {".", ".."}:
            continue
        if "://" in tok:
            continue
        if tok.startswith("/") or tok.startswith("./") or tok.startswith("../"):
            out.append(tok)
        elif "/" in tok and not tok.startswith("-"):
            out.append(tok)
    return out


def foreign_user_path(path: str, user_id: str) -> bool:
    posix = PurePosixPath(path)
    for part in posix.parts:
        if part == user_id:
            return False
    for part in posix.parts:
        if len(part) >= 8 and part != user_id and ("-" in part or part.startswith("usr")):
            return True
    return False


__all__ = ["deny_reason", "foreign_user_path", "paths_in_command"]
