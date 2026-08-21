"""Split a tool result into model content and UI details."""
from __future__ import annotations

from typing import Any

from protocore.contracts.runtime_constants import RuntimeConstants


def split_result(
    content: str,
    *,
    rc: RuntimeConstants,
    details: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return (model_content, ui_details). Flags off → content unchanged, details None."""
    if not rc.tool_result_split_enabled:
        return content, None
    limit = rc.tool_result_content_max_chars
    if len(content) <= limit:
        payload = dict(details or {})
        payload.setdefault("full_content", content)
        return content, payload
    pointer = f"[truncated {len(content) - limit} chars; full result in details]"
    model = content[:limit] + "\n" + pointer
    payload = dict(details or {})
    payload["full_content"] = content
    payload["truncated_chars"] = len(content) - limit
    return model, payload


def llm_content_only(content: str, details: dict[str, Any] | None) -> str:
    """Next provider body never includes details."""
    return content


__all__ = ["llm_content_only", "split_result"]
