"""Tool dependency graph with preconditions.

Optional ``preconditions`` on :class:`~protocore.contracts.types.ToolDefinition`
a list of prerequisite patterns that must appear in the execution trace
before the tool can run.

Format: ``"tool_name"``, ``"tool_name:{param}"``, or
``"tool_name:path_prefix*"`` where ``{param}`` is substituted from the
current call's arguments and ``*`` means prefix-match against a prior
recorded satisfaction entry.

No LLM calls. All checks are O(n) per precondition where *n* is
the trace length.

Ported verbatim from the v1 ``protocore.runtime.tool_preconditions`` module
shipped at commit ``7dfa1ff``. The v2 adaptation:

* uses the helper bag
 (``ToolContext.metadata["protocore.helpers"]``) for cross-call state
 instead of v1's ``AgentContext.metadata``. The :class:`ToolDispatcher`
 reads/writes the satisfied set under the
 :data:`SATISFIED_PRECONDITIONS_KEY` helper key.
* The check is wired into
 :meth:`~protocore.runtime.tool_dispatch.ToolDispatcher.dispatch` BEFORE
 the permission gate fires so a precondition violation short-circuits
 with a ``[PRECONDITION NOT MET: ...]`` tool error envelope —
 matching the v1 ``dispatch.py:1179-1216`` flow.

The mechanism is the foundation for ``AppendFile`` /
``FinalizeFile`` workflow (CORE-1 / CORE-2 from the long-output-generation
synthesis): ``FinalizeFile`` declares
``preconditions=["AppendFile:{path}"]`` so the model must actually append
content before declaring the file ready.
"""

from __future__ import annotations

import logging
import posixpath
import re
from typing import Any

logger = logging.getLogger(__name__)


# Helper-bag key for the satisfied-precondition set (per-run state). The
# dispatcher reads/writes this under
# ``ctx.metadata["protocore.helpers"][SATISFIED_PRECONDITIONS_KEY]``.
# Stored as a sorted list (JSON-friendly) and rehydrated to a set on read.
SATISFIED_PRECONDITIONS_KEY: str = "tool_preconditions.satisfied"


# Matches ``{param_name}`` placeholders in precondition patterns.
_PARAM_RE = re.compile(r"\{(\w+)\}")

# Fields that carry filesystem paths and should be normalized.
_DEFAULT_PATH_FIELDS: list[str] = [
    "path",
    "file_path",
    "source_path",
    "destination_path",
    "target_path",
]

_PRECONDITION_PATH_FIELDS_BY_TOOL_SUFFIX: dict[str, tuple[str, ...]] = {
    "copy_path": ("destination_path",),
    "move_path": ("destination_path",),
}


def _normalize_path(value: str) -> str:
    """Normalize a workspace-relative path for consistent comparison.

    Strips leading ``./``, collapses ``//``, removes trailing ``/``, and
    resolves ``..`` segments via :func:`posixpath.normpath`. Empty / ``.``
    preserves the caller's literal intent so root-level checks behave
    consistently.
    """
    normed = posixpath.normpath(value)
    # ``posixpath.normpath`` converts empty / '.' to '.'; preserve intent.
    if normed == ".":
        return value.rstrip("/") or "."
    return normed


def resolve_precondition(
    pattern: str,
    arguments: dict[str, Any],
) -> str:
    """Substitute ``{param}`` placeholders in *pattern* with *arguments* values.

    Path-like values are normalized so that recording and checking use the
    same canonical form (v1 V006). Unresolved placeholders are left as-is
    (they simply won't match).
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = arguments.get(key)
        if value is None:
            return match.group(0)
        raw = str(value)
        # Normalize path-like values in path-qualified preconditions.
        if key in _DEFAULT_PATH_FIELDS:
            return _normalize_path(raw)
        return raw

    return _PARAM_RE.sub(_replace, pattern)


def check_preconditions(
    *,
    preconditions: list[str],
    arguments: dict[str, Any],
    satisfied: set[str],
) -> str | None:
    """Check if all *preconditions* are satisfied.

    Returns ``None`` if all preconditions are met, or a human-readable
    denial reason for the first unmet precondition.
    """
    for pattern in preconditions:
        resolved = resolve_precondition(pattern, arguments)
        if not _is_precondition_satisfied(resolved=resolved, satisfied=satisfied):
            return (
                f"Precondition not met: '{resolved}' "
                f"(from pattern '{pattern}'). "
                f"Required tool must be called first."
            )
    return None


def _is_precondition_satisfied(*, resolved: str, satisfied: set[str]) -> bool:
    """Return whether a resolved precondition matches prior tool satisfactions."""
    if resolved in satisfied:
        return True
    if not resolved.endswith("*"):
        return False
    prefix = resolved[:-1]
    return any(entry.startswith(prefix) for entry in satisfied)


def _satisfaction_path_fields(
    *,
    tool_name: str,
    path_fields: list[str] | None,
) -> list[str]:
    """Return path fields that should satisfy future preconditions.

    Copy/move operations may read from any source path, but only the
    destination path proves that an artifact was persisted in a required
    location.
    """
    effective = path_fields or _DEFAULT_PATH_FIELDS
    for suffix, overridden_fields in _PRECONDITION_PATH_FIELDS_BY_TOOL_SUFFIX.items():
        if tool_name.endswith(suffix):
            return [field for field in overridden_fields if field in effective] or list(
                overridden_fields
            )
    return effective


def record_satisfaction(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    satisfied: set[str],
    path_fields: list[str] | None = None,
) -> None:
    """Record that *tool_name* was successfully called, satisfying future preconditions.

    Records both the bare tool name and ``tool_name:path_value`` for each
    path-like argument found. Paths are normalized (v1 V006) so that
    :func:`check_preconditions` comparisons are consistent.
    """
    satisfied.add(tool_name)
    effective_path_fields = _satisfaction_path_fields(
        tool_name=tool_name,
        path_fields=path_fields,
    )
    for field in effective_path_fields:
        value = arguments.get(field)
        if isinstance(value, str) and value:
            satisfied.add(f"{tool_name}:{_normalize_path(value)}")


def compute_masked_tools(
    *,
    tool_definitions: list[Any],
    satisfied: set[str],
) -> dict[str, str]:
    """Return ``{tool_name: denial_reason}`` for tools with unmet bare-name preconditions.

    Only **bare-name** preconditions (those without ``{param}`` placeholders)
    are evaluated. Parameterized preconditions like ``read_file:{path}``
    cannot be checked without concrete arguments and remain enforced post-hoc
    in the dispatcher.
    """
    masked: dict[str, str] = {}
    for tool in tool_definitions:
        preconditions: list[str] = getattr(tool, "preconditions", None) or []
        if not preconditions:
            continue
        bare = [p for p in preconditions if "{" not in p]
        if not bare:
            continue
        denial = check_preconditions(
            preconditions=bare,
            arguments={},
            satisfied=satisfied,
        )
        if denial is not None:
            masked[getattr(tool, "name", "")] = denial
    return masked


def load_satisfied_set(
    helpers: dict[str, Any] | None,
    *,
    history: list[tuple[str, dict[str, Any]]] | None = None,
) -> set[str]:
    """Load the per-run satisfied-precondition set from the helper bag.

 adaptation of v1's ``_resolved_satisfied_preconditions``.
 The v1 version rebuilt the set every dispatch by replaying
 ``ExecutionReport.tool_call_details``. v2 keeps the set durable on
 the helper bag (``ctx.metadata["protocore.helpers"]``) so each
 dispatch reads the cumulative state in O(1).

 The set is persisted as a sorted list (JSON-serialisable) and
 hydrated to a Python ``set`` on read.

 when the helper bag is missing the satisfied key
 (``SATISFIED_PRECONDITIONS_KEY`` absent or empty) AND a *history*
 fallback is provided, rebuild the set from the transcript-derived
 ``(tool_name, arguments)`` pairs. This handles the cross-pod
 re-drive case where the helper bag is built fresh per pod: a run
 that called ``AppendFile(foo)``, then the pod restarted, then a new
 pod resumed with an empty helper bag, must NOT re-block a follow-up
 ``FinalizeFile(foo)`` with ``[PRECONDITION NOT MET: AppendFile:foo]``
 just because the in-bag satisfied set is empty. The transcript
 (in the engine snapshot) is the durable source of truth, and the
 helper bag is just a per-run cache.

 ``helpers=None`` is treated as "no state yet" — falls through to
 the history fallback when provided. The history list is a
 chronological list of ``(tool_name, arguments)`` tuples; each entry
 is replayed through :func:`record_satisfaction` so path
 normalisation + the bare-name entry match the live recording
 semantics.
 """
    stored: set[str] = set()
    if helpers is not None:
        raw = helpers.get(SATISFIED_PRECONDITIONS_KEY)
        if isinstance(raw, set):
            stored = {entry for entry in raw if isinstance(entry, str)}
        elif isinstance(raw, (list, tuple)):
            stored = {entry for entry in raw if isinstance(entry, str)}
    if stored:
        return stored
    if history is None:
        return set()
    rebuilt: set[str] = set()
    for tool_name, arguments in history:
        record_satisfaction(
            tool_name=tool_name,
            arguments=arguments or {},
            satisfied=rebuilt,
        )
    return rebuilt


def store_satisfied_set(
    helpers: dict[str, Any] | None,
    satisfied: set[str],
) -> None:
    """Persist the satisfied set on the helper bag.

    Stored as a sorted list so the snapshot is JSON-friendly (matches
    the v1 ``context.metadata`` storage shape). The mutation is in-place
    on the helper-bag dict — callers must hold a reference to the bag
    (``ctx.metadata["protocore.helpers"]``) for the change to be visible
    to subsequent dispatches.

    ``helpers=None`` is a no-op (legacy test wiring without a bag).
    """
    if helpers is None:
        return
    helpers[SATISFIED_PRECONDITIONS_KEY] = sorted(satisfied)


def derive_satisfied_from_messages(
    messages: Any,
) -> set[str]:
    """Rebuild the satisfied set by replaying every tool_use in *messages*.

 the cross-pod re-drive case. The helper bag is rebuilt
 fresh on every new pod (``service_runtime.build_helper_bag``), and
 its ``SATISFIED_PRECONDITIONS_KEY`` starts empty even when the
 engine snapshot carries a long transcript of prior ``tool_use``
 blocks. Replay each ``ToolUseBlock`` in the transcript through
 :func:`record_satisfaction` so the same bare-name + ``tool:path``
 entries are produced that live recording would have produced.

 Only ``assistant``-role messages contribute (the model is the only
 role that emits ``tool_use``). ``tool``-role messages carry the
 result, not the call site, so they add nothing to the satisfied set.
 Arguments are JSON-decoded from the ``arguments_json`` blob; a
 decode failure yields ``{}`` (defensive — the live call already
 succeeded so an undecodable argument shape is "no path to record").

 The replay is order-preserving (the loop walks ``messages`` in
 sequence) so any preconditions that depend on call order see the
 same final set live recording would have.
 """
    import json

    from protocore.contracts.types import ContentBlockKind, MessageRole, ToolUseBlock

    rebuilt: set[str] = set()
    for message in messages:
        if getattr(message, "role", None) is not MessageRole.assistant:
            continue
        for block in getattr(message, "content_blocks", []) or []:
            if not isinstance(block, ToolUseBlock):
                continue
            if getattr(block, "kind", None) is not ContentBlockKind.tool_use:
                continue
            arguments: dict[str, Any] = {}
            raw = getattr(block, "arguments_json", None)
            if isinstance(raw, str) and raw:
                try:
                    decoded = json.loads(raw)
                except (TypeError, ValueError):
                    decoded = None
                if isinstance(decoded, dict):
                    arguments = decoded
            record_satisfaction(
                tool_name=block.name,
                arguments=arguments,
                satisfied=rebuilt,
            )
    return rebuilt


__all__ = [
    "SATISFIED_PRECONDITIONS_KEY",
    "check_preconditions",
    "compute_masked_tools",
    "derive_satisfied_from_messages",
    "load_satisfied_set",
    "record_satisfaction",
    "resolve_precondition",
    "store_satisfied_set",
]
