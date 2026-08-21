"""Directory-scoped AGENTS.md discovery and on-touch activation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from protocore.contracts.runtime_constants import RuntimeConstants

HIDDEN_PREFIX = "."


@dataclass(frozen=True, slots=True)
class RuleFile:
    path: str
    body: str
    origin: str  # project_mount | trusted_store | workspace


def skip_dir(name: str, rc: RuntimeConstants) -> bool:
    if name.startswith(HIDDEN_PREFIX):
        return True
    skipped = {item.strip() for item in rc.rules_skip_dir_names.split(",") if item.strip()}
    return name in skipped


def classify_rule_origin(path: str, *, project_roots: tuple[str, ...] = ()) -> str:
    """Session-writable by default. ``project_mount`` only under an explicit root."""
    posix = PurePosixPath(path)
    for raw in project_roots:
        root = PurePosixPath(str(raw).strip("/") or ".")
        if str(root) == ".":
            return "project_mount"
        try:
            posix.relative_to(root)
        except ValueError:
            continue
        return "project_mount"
    return "workspace"


def _as_path_body(item: tuple[str, ...]) -> tuple[str, str]:
    if len(item) < 2:
        raise ValueError("rule file tuple must be (path, body)")
    return str(item[0]), str(item[1])


def discover_agents_md(
    files: list[tuple[str, str]] | list[tuple[str, ...]],
    rc: RuntimeConstants,
    *,
    project_roots: tuple[str, ...] = (),
) -> list[RuleFile]:
    """``files`` is (posix_path, body). Origin is classified here, not by the caller."""
    if not rc.rules_discovery_enabled:
        return []
    found: list[RuleFile] = []
    for item in files:
        path, body = _as_path_body(tuple(item))
        posix = PurePosixPath(path)
        if posix.name not in {"AGENTS.md", "AGENTS.MD"}:
            continue
        if any(skip_dir(part, rc) for part in posix.parts[:-1]):
            continue
        origin = classify_rule_origin(path, project_roots=project_roots)
        found.append(RuleFile(path=path, body=body[: rc.rules_max_body_bytes], origin=origin))
    return found


def ancestor_rule_paths(file_path: str) -> list[str]:
    posix = PurePosixPath(file_path)
    parents = list(posix.parents)
    out: list[str] = []
    for parent in reversed(parents):
        if str(parent) in {".", ""}:
            out.append("AGENTS.md")
        else:
            out.append(f"{parent}/AGENTS.md")
    return out


def is_trusted(rule: RuleFile, rc: RuntimeConstants) -> bool:
    if rule.origin in {"project_mount", "trusted_store"}:
        return True
    if rc.rules_workspace_trust == "always":
        return True
    if rc.rules_workspace_trust == "never":
        return False
    return False


def activate_on_filesystem_touch(
    *,
    touched_path: str,
    tool_name: str,
    discovered: list[RuleFile],
    already_active: list[str],
    rc: RuntimeConstants,
) -> list[str]:
    """Bash / run_command does not activate. Workspace-written files need trust."""
    if not rc.rules_discovery_enabled:
        return list(already_active)
    if tool_name in {"Bash", "run_command"}:
        return list(already_active)
    wanted = set(ancestor_rule_paths(touched_path))
    active = list(already_active)
    for rule in discovered:
        if rule.path not in wanted:
            continue
        if rule.path in active:
            continue
        if not is_trusted(rule, rc):
            continue
        if len(active) >= rc.rules_max_active:
            break
        active.append(rule.path)
    return active


def bodies_for_prompt(
    discovered: list[RuleFile],
    active_paths: list[str],
    rc: RuntimeConstants,
) -> list[str]:
    by_path = {item.path: item for item in discovered}
    bodies: list[str] = []
    for path in active_paths:
        rule = by_path.get(path)
        if rule is None:
            continue
        bodies.append(rule.body[: rc.rules_max_body_bytes])
    return bodies


__all__ = [
    "RuleFile",
    "activate_on_filesystem_touch",
    "ancestor_rule_paths",
    "bodies_for_prompt",
    "classify_rule_origin",
    "discover_agents_md",
    "is_trusted",
    "skip_dir",
]
