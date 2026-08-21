"""Steer / follow-up queues, live model/thinking, and settled helper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import Message, MessageRole, TextBlock

QueueKind = Literal["steer", "follow_up"]
QueueMode = Literal["one-at-a-time", "all"]


@dataclass(slots=True)
class QueuedPrompt:
    """One pending steer or follow-up item."""

    id: str
    kind: QueueKind
    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "attachments": list(self.attachments),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QueuedPrompt:
        return cls(
            id=str(raw["id"]),
            kind=raw["kind"],
            text=str(raw.get("text") or ""),
            attachments=list(raw.get("attachments") or []),
        )


def new_queued_prompt(
    kind: QueueKind,
    text: str,
    *,
    attachments: list[dict[str, Any]] | None = None,
    item_id: str | None = None,
) -> QueuedPrompt:
    return QueuedPrompt(
        id=item_id or f"q_{uuid4().hex[:12]}",
        kind=kind,
        text=text,
        attachments=list(attachments or []),
    )


def enqueue(
    queue: list[QueuedPrompt],
    item: QueuedPrompt,
    rc: RuntimeConstants,
) -> list[QueuedPrompt]:
    """Append if under caps. Raises ValueError when the item or queue is over cap."""
    if not rc.steer_follow_up_enabled:
        raise ValueError("steer_follow_up_disabled")
    if len(item.text) > rc.max_queued_chars:
        raise ValueError("queued_item_too_long")
    if len(queue) >= rc.max_queued_items:
        raise ValueError("queue_full")
    return [*queue, item]


def place_items(
    queue: list[QueuedPrompt],
    mode: QueueMode,
) -> tuple[list[QueuedPrompt], list[QueuedPrompt]]:
    """Split ``queue`` into (placed, remaining) according to ``mode``."""
    if not queue:
        return [], []
    if mode == "all":
        return list(queue), []
    return [queue[0]], list(queue[1:])


def placed_as_user_message(items: SequenceLike) -> Message | None:
    """Join placed items into one user message."""
    texts = [item.text.strip() for item in items if item.text.strip()]
    if not texts:
        return None
    body = "\n\n".join(texts)
    return Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=body)],
    )


def validate_thinking_for_mode(run_mode: str, thinking_enabled: bool) -> None:
    """Raise ValueError when deep would turn thinking off."""
    if run_mode not in ("direct", "deep"):
        raise ValueError("invalid_run_mode")
    if run_mode == "deep" and not thinking_enabled:
        raise ValueError("deep_requires_thinking")


def lease_key(scope_id: str, session_id: str, lane: str) -> str:
    return f"pc:lease:{scope_id}:{session_id}:{lane}"


def restore_queued_prompts(engine: Any) -> list[str]:
    """Drain steer + follow-up queues and return their texts.

    This is the shipped cancel helper: HTTP cancel, ``query()`` stop, and
    the composer restore path all call this rather than copying lists.
    """
    texts: list[str] = []
    for raw in list(getattr(engine, "_steer_queue", []) or []) + list(
        getattr(engine, "_follow_up_queue", []) or []
    ):
        if isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
        else:
            text = str(getattr(raw, "text", "") or "").strip()
        if text:
            texts.append(text)
    engine._steer_queue = []
    engine._follow_up_queue = []
    return texts


# SequenceLike avoids importing Sequence in the public helper signature noise.
SequenceLike = list[QueuedPrompt]


__all__ = [
    "QueueKind",
    "QueueMode",
    "QueuedPrompt",
    "enqueue",
    "lease_key",
    "new_queued_prompt",
    "place_items",
    "placed_as_user_message",
    "restore_queued_prompts",
    "validate_thinking_for_mode",
]
