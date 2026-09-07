"""Compaction as a retained-tail checkpoint the next LLM request cannot read through."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    WORKSPACE_INSPECTION_ROLES,
    WORKSPACE_MUTATION_ROLES,
    ToolRoleMap,
)
from protocore.contracts.types import Message, MessageRole


def tracked_tool_names(
    rc: LoopConstants, roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP
) -> tuple[str, ...]:
    """The tool calls whose bare fact must outlive compaction.

    A tenant may name them outright (``compaction_tracked_tool_names``). When
    it does not, they are every tool that touched the workspace — read or
    written — as the host declared its own tools, so an installation that
    renamed its file tools keeps its file history across a compaction instead
    of losing it to a name the core guessed.
    """
    configured = tuple(rc.compaction_tracked_tool_names)
    if configured:
        return configured
    return tuple(
        sorted(roles.names_with(*WORKSPACE_MUTATION_ROLES, *WORKSPACE_INSPECTION_ROLES))
    )


@dataclass(slots=True)
class CompactCheckpoint:
    entry_id: str
    summary: str
    retained_from_index: int
    file_op_facts: list[str] = field(default_factory=list)
    instructions: str = ""
    reason: str = "manual"

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "summary": self.summary,
            "retained_from_index": self.retained_from_index,
            "file_op_facts": list(self.file_op_facts),
            "instructions": self.instructions,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CompactCheckpoint:
        """Read back a checkpoint this class wrote.

        The folded-away turns survive nowhere else, so a checkpoint that
        crosses a process boundary has to come back as the object the loop
        reads rather than as the row it travelled as. A row missing a field is
        read at the value a checkpoint that never had one would carry, so a
        partial row is a poorer checkpoint rather than a refusal.
        """
        raw_index = raw.get("retained_from_index")
        index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else 0
        facts = raw.get("file_op_facts")
        return cls(
            entry_id=str(raw.get("entry_id", "")),
            summary=str(raw.get("summary", "")),
            retained_from_index=index,
            file_op_facts=[str(fact) for fact in facts] if isinstance(facts, list) else [],
            instructions=str(raw.get("instructions", "")),
            reason=str(raw.get("reason", "manual")),
        )


def collect_file_op_facts(
    history: list[Message],
    tracked_tool_names: Sequence[str] = (),
) -> list[str]:
    """Keep one line per tracked tool call so the bare fact outlives compaction.

    ``tracked_tool_names`` is a tenant policy
    (:attr:`LoopConstants.compaction_tracked_tool_names`), not a core
    invariant: a non-coding backend names its own domain verbs here.
    """
    tracked = frozenset(tracked_tool_names)
    facts: list[str] = []
    for message in history:
        if message.role is not MessageRole.assistant:
            continue
        for block in message.content_blocks:
            name = getattr(block, "name", "")
            if name in tracked:
                args = getattr(block, "arguments_json", "") or ""
                facts.append(f"{name}:{args[:200]}")
    return facts


def build_checkpoint(
    history: list[Message],
    *,
    keep_recent_turns: int,
    instructions: str,
    reason: str,
    enabled: bool,
    tracked_tool_names: Sequence[str] = (),
) -> CompactCheckpoint | None:
    if not enabled:
        return None
    user_indexes = [
        idx for idx, message in enumerate(history) if message.role is MessageRole.user
    ]
    if len(user_indexes) <= keep_recent_turns:
        retained = 0
    else:
        retained = user_indexes[-keep_recent_turns]
    dropped = history[:retained]
    summary_bits = [f"compacted {len(dropped)} messages"]
    if instructions:
        summary_bits.append(f"focus: {instructions}")
    return CompactCheckpoint(
        entry_id=f"ckpt_{len(history)}_{retained}",
        summary="; ".join(summary_bits),
        retained_from_index=retained,
        file_op_facts=collect_file_op_facts(dropped, tracked_tool_names),
        instructions=instructions,
        reason=reason,
    )


def apply_checkpoint(
    history: list[Message],
    checkpoint: CompactCheckpoint | None,
) -> list[Message]:
    """Next LLM request: summary + file-op facts + retained tail. Persist is untouched."""
    if checkpoint is None:
        return list(history)
    tail = history[checkpoint.retained_from_index :]
    prefix = [
        Message(
            role=MessageRole.user,
            content_blocks=[],
        )
    ]
    # Build via text helper if available; keep a single user summary message.
    from protocore.contracts.types import TextBlock

    body = checkpoint.summary
    if checkpoint.file_op_facts:
        body += "\nfile-ops:\n" + "\n".join(checkpoint.file_op_facts)
    prefix = [
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text=body)],
        )
    ]
    return prefix + tail


def overflow_should_compact(*, used_tokens: int, window: int, rc: LoopConstants) -> bool:
    return used_tokens > max(0, window - rc.compaction_reserve_tokens)


__all__ = [
    "CompactCheckpoint",
    "apply_checkpoint",
    "build_checkpoint",
    "collect_file_op_facts",
    "overflow_should_compact",
    "tracked_tool_names",
]
