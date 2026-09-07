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
the ``x-protocore-chunkable-content-mutation`` wire extension) OR declared by
the host as a tool that produces bytes at a path. A dynamic/tenant tool that
merely declares a ``content`` field — without the flag and without such a role
— does NOT qualify, so it is never misrouted into the file-chunk protocol.
"""
from __future__ import annotations

from collections.abc import Iterable

from protocore.contracts.tool_roles import (
    BYTE_PRODUCING_ROLES,
    EMPTY_TOOL_ROLE_MAP,
    ToolRoleMap,
)

# The field whose absence (when required) signals a chunkable output-cap
# truncation, as opposed to a small structural field (Bash ``command`` / Read
# ``path``).
CHUNKABLE_CONTENT_FIELD: str = "content"

# JSON-schema wire extension key for the explicit opt-in. Mirrors the typed
# ``ToolParameterSchema.chunkable_content_mutation`` field for adapters that
# carry the marker in a raw schema dict rather than the typed model.
CHUNKABLE_CONTENT_MUTATION_SCHEMA_KEY: str = "x-protocore-chunkable-content-mutation"

def chunkable_content_mutation_names(roles: ToolRoleMap) -> frozenset[str]:
    """The host's tools whose cut ``content`` body the chunk protocol can repair.

    A tool qualifies by ROLE: it takes a whole body and there is an append path
    to continue it (create-or-overwrite plus append), which is exactly what the
    header-then-chunks-then-seal recovery needs. An in-place edit tool never
    qualifies — it carries no whole-body parameter, so it cannot match the
    ``content``-in-required test either way.
    """
    return roles.names_with(*BYTE_PRODUCING_ROLES)


def is_chunkable_content_mutation(
    *,
    tool_name: str | None,
    required: Iterable[str] | None,
    chunkable_flag: bool | None,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> bool:
    """Return True iff a tool is a chunkable content-mutation tool.

    Parameters
    ----------
    tool_name:
        The tool's registered name (checked against ``roles``).
    required:
        The tool's required parameter names. ``content`` MUST be among them —
        an optional ``content`` field is not the cut-body shape.
    chunkable_flag:
        The explicit opt-in (``ToolParameterSchema.chunkable_content_mutation``
        or the ``x-protocore-chunkable-content-mutation`` wire extension).

    roles:
        What the host said its tools do. The two acceptance routes (flag OR
        role) are deliberately OR-ed so a per-tenant content-mutation tool can
        opt in via the flag without a code change, while the host's own
        byte-producing tools qualify from their registration alone.
    """
    required_set = set(required or ())
    if CHUNKABLE_CONTENT_FIELD not in required_set:
        return False
    if chunkable_flag is True:
        return True
    return bool(tool_name) and tool_name in chunkable_content_mutation_names(roles)


__all__ = [
    "CHUNKABLE_CONTENT_FIELD",
    "CHUNKABLE_CONTENT_MUTATION_SCHEMA_KEY",
    "chunkable_content_mutation_names",
    "is_chunkable_content_mutation",
]
