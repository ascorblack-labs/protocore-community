"""Projecting a tool result down to what the next request can afford.

The canonical value of a call lives on
:class:`~protocore.contracts.types.ToolResult`, and the transcript carries a
projection of it (:class:`~protocore.contracts.types.ToolResultBlock`). A tool
usually takes that projection itself, because only the tool knows which part of
its output is the part worth reading. This module handles the case where it
did not: a result already in the transcript is longer than the run can afford
to send again, and the runtime has to shorten it without being told how.

What it does is deliberately dull — keep the head, say how much was cut, point
at where the whole value still is. The interesting property is not the summary
but where it is applied: the request VIEW, never the transcript. Persist keeps
the canonical value; only the copy handed to the provider is shortened, so the
same run can be resumed, replayed or compacted afterwards against a history
that never lost anything.
"""
from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.runtime_constants import LoopConstants


@dataclass(frozen=True, slots=True)
class ResultProjection:
    """One result's text as the next request should carry it."""

    content: str
    """The projected text. Identical to the canonical value when nothing was cut."""

    dropped_chars: int = 0
    """How much of the canonical value this projection leaves out."""

    @property
    def is_shortened(self) -> bool:
        return self.dropped_chars > 0


def project_result_content(
    content: str, *, rc: LoopConstants, canonical_ref: str | None = None
) -> ResultProjection:
    """Shorten ``content`` for the next request, or hand it back untouched.

    ``canonical_ref`` is named in the pointer when the caller has one, so the
    model is told the value still exists rather than being left to assume the
    tool returned this much and no more. A caller with no reference passes
    none and the pointer says only how much is missing — still true, and still
    better than a silent cut.
    """
    if not rc.tool_result_split_enabled:
        return ResultProjection(content=content)
    limit = rc.tool_result_content_max_chars
    if len(content) <= limit:
        return ResultProjection(content=content)
    dropped = len(content) - limit
    where = f"; full result at {canonical_ref}" if canonical_ref else ""
    pointer = f"[truncated {dropped} chars{where}]"
    return ResultProjection(content=content[:limit] + "\n" + pointer, dropped_chars=dropped)


__all__ = ["ResultProjection", "project_result_content"]
