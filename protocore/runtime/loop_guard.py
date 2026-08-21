"""Repeating-stream and identical-tool loop guard.

Pure functions over stream text and tool fingerprints. The query loop
calls these; when the RC flag is off the helpers are no-ops.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from protocore.contracts.runtime_constants import RuntimeConstants

GuardKind = Literal["stream_repeat", "identical_tool"]


@dataclass(frozen=True, slots=True)
class LoopGuardHit:
    """One firing of the loop guard."""

    kind: GuardKind
    nudge_index: int
    stripped_chars: int
    fingerprint: str


def _normalize_passage(text: str) -> str:
    return " ".join(text.split())


def repeating_tail_cut(
    buffer: str,
    *,
    min_chars: int,
    window_tokens: int,
) -> tuple[str, int]:
    """Return ``(kept, stripped_chars)`` if ``buffer`` ends in a repeated passage.

    The detector looks at the last ``window_tokens`` whitespace-separated
    tokens as a candidate passage. If that passage already appears earlier
    in the buffer (at least once, consecutively at the end), the repeated
    tail is stripped. Short buffers below ``min_chars`` are left intact.
    """
    if len(buffer) < min_chars or window_tokens <= 0:
        return buffer, 0
    tokens = buffer.split()
    if len(tokens) < 2:
        return buffer, 0
    max_window = min(window_tokens, len(tokens) // 2)
    best_kept: list[str] | None = None
    for width in range(max_window, 0, -1):
        window = tokens[-width:]
        passage = " ".join(window)
        if len(passage) < min_chars:
            continue
        copies = 1
        end = len(tokens) - width
        while end >= width and tokens[end - width : end] == window:
            copies += 1
            end -= width
        if copies < 2:
            continue
        best_kept = tokens[: end + width]
        break
    if best_kept is None:
        return buffer, 0
    kept = " ".join(best_kept)
    stripped = len(buffer) - len(kept)
    if stripped <= 0:
        return buffer, 0
    return kept, stripped


def inspect_stream_repeat(
    text_buffer: str,
    reasoning_buffer: str,
    rc: RuntimeConstants,
) -> tuple[str, str, LoopGuardHit | None]:
    """Cut a repeating tail from text and/or thinking.

    Returns the (possibly shortened) buffers and a hit if either channel
    was cut. When the flag is off the buffers are returned unchanged.
    """
    if not rc.loop_guard_enabled:
        return text_buffer, reasoning_buffer, None
    new_text, text_stripped = repeating_tail_cut(
        text_buffer,
        min_chars=rc.loop_guard_repeat_min_chars,
        window_tokens=rc.loop_guard_repeat_window_tokens,
    )
    new_reason, reason_stripped = repeating_tail_cut(
        reasoning_buffer,
        min_chars=rc.loop_guard_repeat_min_chars,
        window_tokens=rc.loop_guard_repeat_window_tokens,
    )
    stripped = text_stripped + reason_stripped
    if stripped <= 0:
        return text_buffer, reasoning_buffer, None
    return (
        new_text,
        new_reason,
        LoopGuardHit(
            kind="stream_repeat",
            nudge_index=0,
            stripped_chars=stripped,
            fingerprint=_normalize_passage(new_text or new_reason)[:80],
        ),
    )


def canonical_tool_fingerprint(name: str, arguments: Any) -> str:
    """Stable fingerprint for identical-tool detection."""
    if isinstance(arguments, str):
        raw = arguments
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = arguments
    else:
        parsed = arguments
        raw = ""
    if isinstance(parsed, dict):
        payload = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    elif isinstance(parsed, str):
        payload = parsed
    else:
        payload = raw or json.dumps(parsed, sort_keys=True, default=str)
    return f"{name}:{payload}"


def identical_tool_should_block(
    fingerprint: str,
    prior_counts: dict[str, int],
    rc: RuntimeConstants,
) -> bool:
    """Return True when this fingerprint has already hit the execute limit."""
    if not rc.loop_guard_enabled:
        return False
    seen = prior_counts.get(fingerprint, 0)
    return seen >= rc.loop_guard_identical_tool_limit


__all__ = [
    "GuardKind",
    "LoopGuardHit",
    "canonical_tool_fingerprint",
    "identical_tool_should_block",
    "inspect_stream_repeat",
    "repeating_tail_cut",
]
