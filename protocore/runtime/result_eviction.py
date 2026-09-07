"""Evict unmarked results of read-shaped tools from the next LLM request.

Which tools count as read-shaped is a tenant policy
(:attr:`LoopConstants.result_eviction_tool_names`), not a core invariant.

Persist (engine.history) is never mutated. Only the context-build view
is rewritten. Compacted-tool-result placeholders are left untouched.
"""
from __future__ import annotations

import json
import posixpath
from collections.abc import Iterable, Sequence

from protocore.constants import PROTOCOL_COMPACTED_TOOL_RESULT_V1
from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    WORKSPACE_INSPECTION_ROLES,
    WORKSPACE_MUTATION_ROLES,
    ToolArgumentSlot,
    ToolRoleMap,
)
from protocore.contracts.types import ContentBlock, Message, ToolResultBlock, ToolUseBlock
from protocore.runtime.tool_arguments import argument_names, string_argument

#: The roles whose call makes an earlier view of the same path untrue. Every
#: role that changes a file is here, not only the one that rewrites it whole:
#: an appended line and a replaced hunk falsify a pinned read exactly as much
#: as an overwrite does, and a rule that only noticed overwrites would keep
#: feeding the model the version before the edit.
PATH_INVALIDATING_ROLES = WORKSPACE_MUTATION_ROLES


def evictable_tool_names(
    rc: LoopConstants, roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP
) -> frozenset[str]:
    """Which tools' unmarked results may be dropped from the next request.

    A tenant may name them outright (``result_eviction_tool_names``); when it
    does not, they are the host's workspace-inspection tools, which are
    read-shaped by definition — their result is a view of something still on
    disk, so dropping it loses nothing that cannot be looked at again.

    An empty list is the second case, not a third one: it says nothing about
    which tools, so the roles answer, and the answer is usually WIDER than the
    default names. A tenant that wants nothing evicted turns eviction off.
    """
    configured = frozenset(rc.result_eviction_tool_names)
    if configured:
        return configured
    return roles.names_with(*WORKSPACE_INSPECTION_ROLES)


def is_compacted_placeholder(content: str) -> bool:
    return PROTOCOL_COMPACTED_TOOL_RESULT_V1 in content


def tool_names_by_call_id(history: Sequence[Message]) -> dict[str, str]:
    """Every call id in ``history`` mapped to the name of the tool it called.

    One forward pass. A call id names exactly one call, so traversal order
    cannot change the answer and a later block cannot legitimately shadow an
    earlier one for the same id.

    Callers that need a name for more than one call id build this map ONCE and
    index it, rather than calling :func:`tool_name_for_result` per id: the
    per-id lookup walks the whole transcript, so a loop over it costs the
    transcript squared exactly where transcripts are largest.
    """
    names: dict[str, str] = {}
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock) and block.tool_call_id not in names:
                names[block.tool_call_id] = block.name
    return names


def tool_name_for_result(history: Sequence[Message], tool_call_id: str) -> str | None:
    """The name of the tool whose ``ToolUseBlock`` carries ``tool_call_id``.

    A call id names exactly one call, so traversal order cannot change the
    answer. This is the one lookup every history reader that needs a name for
    a call id goes through; None means the originating call is no longer in
    the history handed in.

    This is the single-id form. Resolving many ids over the same history goes
    through :func:`tool_names_by_call_id` instead.
    """
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock) and block.tool_call_id == tool_call_id:
                return block.name
    return None


def _normalised_path(path: str | None) -> str | None:
    """One spelling for a path, so two names for one file compare equal.

    Pure text, deliberately: the runtime is asked whether a call rewrote the
    file another call read, and it has to answer from the transcript, on a
    machine that may not be the one either call ran on. ``./a/b`` and ``a/b/``
    are the same file to everyone who reads them, and nothing here touches a
    filesystem to find that out.
    """
    if not path:
        return None
    return posixpath.normpath(path) or None


def _call_path(
    block: ToolUseBlock, *, roles: ToolRoleMap, spellings: tuple[str, ...]
) -> str | None:
    """The path argument of a call, read the way the runtime reads arguments.

    ``spellings`` is the same list the read below would try, passed in so the
    common case costs a substring test rather than a JSON parse: this runs over
    the whole transcript on every request build, and most calls in a long
    transcript carry no path at all.
    """
    if not any(name in block.arguments_json for name in spellings):
        return None
    try:
        arguments = json.loads(block.arguments_json)
    except (TypeError, ValueError):
        return None
    return _normalised_path(
        string_argument(arguments, ToolArgumentSlot.path, roles=roles)
    )


def pins_invalidated_by_writes(
    history: Sequence[Message], *, roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP
) -> frozenset[str]:
    """Call ids whose pinned result the run has since made untrue.

    A pin says "keep this in front of the model". It was never meant to say
    "keep it forever": pin the result of reading a file, have the agent rewrite
    that file two turns later, and the pin goes on holding the version before
    the rewrite in the context — the one place the model is guaranteed to look,
    now guaranteed to be wrong. Nothing lifted the pin, because nothing was
    watching for the write.

    This watches. A result carrying a path is invalidated by any LATER write
    whose tool holds a path-changing role and whose path argument names the
    same file. The write counts from where its OWN result lands, not from where
    the model asked for it: one assistant message may carry several tool calls,
    and a read requested beside a write in that same message still answers about
    the file as it was before the write. Anchoring on the result also means a
    write that FAILED lifts nothing — it changed no file, so no pin it might
    have falsified is falsified.

    Order is otherwise the transcript's: only writes after the result count, so
    a read taken after the write is untouched, which is what makes re-reading
    the file the way to get a durable pin back.

    Roles, never names (:mod:`protocore.contracts.tool_roles`): the host says
    which of its tools change files, and an installation that names its writer
    something else keeps this behaviour instead of silently losing it.

    Computed from the transcript rather than remembered, so a resumed run
    reaches the same answer as the run that was paused, and a pin no snapshot
    recorded as lifted is still not honoured over a file it no longer describes.
    """
    if not roles:
        return frozenset()
    writing_names = roles.names_with(*PATH_INVALIDATING_ROLES)
    if not writing_names:
        return frozenset()
    spellings = argument_names(ToolArgumentSlot.path, roles=roles)
    #: Whether each answered call failed. A call absent here is still in
    #: flight: the transcript ends before its answer, so its own use block is
    #: the only place it can be counted from.
    answered: dict[str, bool] = {
        block.tool_call_id: block.is_error
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }
    #: Every result seen so far that speaks about a given path.
    results_by_path: dict[str, list[str]] = {}
    call_paths: dict[str, str] = {}
    writing_calls: set[str] = set()
    invalidated: set[str] = set()
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                path = _call_path(block, roles=roles, spellings=spellings)
                if path is None:
                    continue
                call_paths[block.tool_call_id] = path
                if block.name not in writing_names:
                    continue
                if block.tool_call_id in answered:
                    writing_calls.add(block.tool_call_id)
                else:
                    # Nothing will come back to say whether it succeeded, and a
                    # write the run has already asked for is not a file the
                    # pinned read still describes.
                    invalidated.update(results_by_path.get(path, ()))
                continue
            if not isinstance(block, ToolResultBlock):
                continue
            # The result states its own path when the tool knew one; when it
            # does not, the call that produced it did, and both spellings are
            # kept because a tool may report an absolute path for an argument
            # the model wrote relative.
            paths = {
                _normalised_path(block.path),
                call_paths.get(block.tool_call_id),
            }
            known = [path for path in paths if path is not None]
            if block.tool_call_id in writing_calls and not block.is_error:
                for path in known:
                    invalidated.update(results_by_path.get(path, ()))
            for path in known:
                results_by_path.setdefault(path, []).append(block.tool_call_id)
    return frozenset(invalidated)


def is_pinned_result(
    block: ToolResultBlock,
    pinned_ids: Iterable[str],
    *,
    keep_marked: bool,
    invalidated_ids: Iterable[str] = (),
) -> bool:
    """Whether this result must survive the next request build.

    ``invalidated_ids`` are the pins the run has since falsified
    (:func:`pins_invalidated_by_writes`). They lose to nothing else: a result
    that no longer describes the file it names is not worth keeping under any
    of the three ways of asking for it, and keeping it is worse than dropping
    it, because the model reads a stale file as the current one.
    """
    if not keep_marked:
        return False
    if block.tool_call_id in frozenset(invalidated_ids):
        return False
    if block.tool_call_id in pinned_ids:
        return True
    retention = block.metadata.get("retention")
    if retention == "pinned":
        return True
    keep = block.metadata.get("keep")
    return keep is True or keep == "true"


def evict_history_for_llm(
    history: Sequence[Message],
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
    pinned_ids: Iterable[str] = (),
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> tuple[list[Message], list[str]]:
    """Return a shallow-copied history with unmarked read-shaped results replaced.

    The tool names that qualify come from
    :attr:`LoopConstants.result_eviction_tool_names`; the text that replaces an
    evicted result comes from the ``result_eviction`` template, rendered once
    per evicted block with that block's call id.

    Compacted placeholders are never rewritten. Persist is not touched.
    """
    if not rc.result_eviction_enabled:
        return list(history), []
    evictable = evictable_tool_names(rc, roles)
    if not evictable:
        return list(history), []
    pinned = set(pinned_ids)
    # Hoisted: one pass over the transcript, not one per result block.
    tool_names = tool_names_by_call_id(history)
    invalidated = pins_invalidated_by_writes(history, roles=roles)
    evicted_ids: list[str] = []
    rewritten: list[Message] = []
    for message in history:
        new_blocks: list[ContentBlock] = []
        changed = False
        for block in message.content_blocks:
            if not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue
            if is_compacted_placeholder(block.content):
                new_blocks.append(block)
                continue
            name = tool_names.get(block.tool_call_id)
            if name not in evictable:
                new_blocks.append(block)
                continue
            if is_pinned_result(
                block,
                pinned,
                keep_marked=rc.result_eviction_keep_marked,
                invalidated_ids=invalidated,
            ):
                new_blocks.append(block)
                continue
            placeholder = prompts.render(
                "result_eviction", {"tool_call_id": block.tool_call_id}
            )
            new_blocks.append(
                block.model_copy(
                    update={
                        "content": placeholder,
                        "metadata": {**block.metadata, "evicted": True},
                    }
                )
            )
            evicted_ids.append(block.tool_call_id)
            changed = True
        if changed:
            rewritten.append(message.model_copy(update={"content_blocks": new_blocks}))
        else:
            rewritten.append(message)
    return rewritten, evicted_ids


def apply_line_cap(content: str, max_lines: int) -> str:
    """Cap tool output lines. ``max_lines <= 0`` is a no-op."""
    if max_lines <= 0:
        return content
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    kept = lines[:max_lines]
    kept.append(f"...[{len(lines) - max_lines} lines truncated]")
    return "\n".join(kept)


__all__ = [
    "PATH_INVALIDATING_ROLES",
    "apply_line_cap",
    "evict_history_for_llm",
    "evictable_tool_names",
    "is_compacted_placeholder",
    "is_pinned_result",
    "pins_invalidated_by_writes",
    "tool_name_for_result",
    "tool_names_by_call_id",
]
