"""Evict unmarked results of read-shaped tools from the next LLM request.

Which tools count as read-shaped is a tenant policy
(:attr:`RuntimeConstants.result_eviction_tool_names`), not a core invariant.

Persist (engine.history) is never mutated. Only the context-build view
is rewritten. Compacted-tool-result placeholders are left untouched.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from protocore.constants import PROTOCOL_COMPACTED_TOOL_RESULT_V1
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import ContentBlock, Message, ToolResultBlock, ToolUseBlock

#: Fallback for callers that predate ``result_eviction_tool_names``. The live
#: set is read from RuntimeConstants; this only names the historical default.
EVICTABLE_TOOLS: frozenset[str] = frozenset({"Read", "Grep", "read", "grep"})


def is_compacted_placeholder(content: str) -> bool:
    return PROTOCOL_COMPACTED_TOOL_RESULT_V1 in content


def tool_name_for_result(history: Sequence[Message], tool_call_id: str) -> str | None:
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock) and block.tool_call_id == tool_call_id:
                return block.name
    return None


def is_pinned_result(
    block: ToolResultBlock,
    pinned_ids: Iterable[str],
    *,
    keep_marked: bool,
) -> bool:
    if not keep_marked:
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
    rc: RuntimeConstants,
    pinned_ids: Iterable[str] = (),
) -> tuple[list[Message], list[str]]:
    """Return a shallow-copied history with unmarked read-shaped results replaced.

    The tool names that qualify come from
    :attr:`RuntimeConstants.result_eviction_tool_names`.

    Compacted placeholders are never rewritten. Persist is not touched.
    """
    if not rc.result_eviction_enabled:
        return list(history), []
    evictable = frozenset(rc.result_eviction_tool_names)
    if not evictable:
        return list(history), []
    pinned = set(pinned_ids)
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
            name = tool_name_for_result(history, block.tool_call_id)
            if name not in evictable:
                new_blocks.append(block)
                continue
            if is_pinned_result(block, pinned, keep_marked=rc.result_eviction_keep_marked):
                new_blocks.append(block)
                continue
            placeholder = rc.result_eviction_placeholder.format(
                tool_call_id=block.tool_call_id
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
    "EVICTABLE_TOOLS",
    "apply_line_cap",
    "evict_history_for_llm",
    "is_compacted_placeholder",
    "is_pinned_result",
    "tool_name_for_result",
]
