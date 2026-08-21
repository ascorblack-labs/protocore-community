"""Shared classification of CHUNKABLE content-mutation tools.

The runtime's truncation chunk-recovery (Write->AppendFile->FinalizeFile) can
only repair a tool whose large ``content`` body was cut at the output cap. The
ONE predicate here is the single source of truth used by BOTH:

* the host's OpenAI-compatible provider adapter — to decide whether a streamed
  ``tool_use_stop`` missing a required ``content`` field should be flagged
  ``truncated_by_output_cap`` (chunk-recovery) vs left for the normal
  dispatch -> missing-field cap; and
* core ``protocore.runtime.query`` — to decide whether a truncated tool call
  gets the structured chunk-recovery message vs the generic resume prompt.

A tool qualifies iff its required parameter set includes ``content`` AND it is
EITHER explicitly flagged (``ToolParameterSchema.chunkable_content_mutation`` /
the ``x-protocore-chunkable-content-mutation`` wire extension) OR on the narrow
built-in allowlist. A dynamic/tenant tool that merely declares a ``content``
field — without the flag and not on the allowlist — does NOT qualify, so it is
never misrouted into the file-chunk protocol.
"""
from __future__ import annotations

from collections.abc import Iterable

# The field whose absence (when required) signals a chunkable output-cap
# truncation, as opposed to a small structural field (Bash ``command`` / Read
# ``path``).
CHUNKABLE_CONTENT_FIELD: str = "content"

# JSON-schema wire extension key for the explicit opt-in. Mirrors the typed
# ``ToolParameterSchema.chunkable_content_mutation`` field for adapters that
# carry the marker in a raw schema dict rather than the typed model.
CHUNKABLE_CONTENT_MUTATION_SCHEMA_KEY: str = "x-protocore-chunkable-content-mutation"

# Narrow built-in allowlist of the runtime's own chunkable content-mutation
# tools. These take a full ``content`` body and have a working append path
# (Write creates/overwrites; AppendFile appends), so the
# Write(header)->AppendFile(chunks)->FinalizeFile recovery is valid for them.
# ``Edit`` is intentionally absent — it has no full-``content`` parameter, so it
# can never match the ``content``-in-required test anyway.
CHUNKABLE_CONTENT_MUTATION_ALLOWLIST: frozenset[str] = frozenset({"Write", "AppendFile"})


def is_chunkable_content_mutation(
    *,
    tool_name: str | None,
    required: Iterable[str] | None,
    chunkable_flag: bool | None,
) -> bool:
    """Return True iff a tool is a chunkable content-mutation tool.

    Parameters
    ----------
    tool_name:
        The tool's registered name (checked against the built-in allowlist).
    required:
        The tool's required parameter names. ``content`` MUST be among them —
        an optional ``content`` field is not the cut-body shape.
    chunkable_flag:
        The explicit opt-in (``ToolParameterSchema.chunkable_content_mutation``
        or the ``x-protocore-chunkable-content-mutation`` wire extension).

    The two acceptance routes (flag OR allowlist) are deliberately OR-ed so a
    per-tenant content-mutation tool can opt in via the flag without a code
    change, while the runtime's own Write/AppendFile work out of the box.
    """
    required_set = set(required or ())
    if CHUNKABLE_CONTENT_FIELD not in required_set:
        return False
    if chunkable_flag is True:
        return True
    return bool(tool_name) and tool_name in CHUNKABLE_CONTENT_MUTATION_ALLOWLIST


__all__ = [
    "CHUNKABLE_CONTENT_FIELD",
    "CHUNKABLE_CONTENT_MUTATION_ALLOWLIST",
    "CHUNKABLE_CONTENT_MUTATION_SCHEMA_KEY",
    "is_chunkable_content_mutation",
]
