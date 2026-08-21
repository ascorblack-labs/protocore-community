"""Universal lean tool-surface contract (pure core).

Defines the canonical specs for a small, universal agent-facing tool pool
that any backend can implement. A capable model hand-composes commands against
the runtime instead of filling a dozen benchmark-shaped typed schemas, and
grounding/citation discipline is tracked by a structured ``read`` vs
``read_silent`` split.

Universal-core invariants:

* **Domain-neutral.** No benchmark, tenant-id, scorer-key or rubric-shaped
  logic. Every name + description here is domain-neutral. The host side
  binds these canonical specs to a concrete backend — core only owns the *shape*.
* **Lean by construction.** Exactly seven canonical verbs — ``exec``,
  ``read``, ``read_silent``, ``write``, ``find``, ``search``, ``answer``.
* **Grounding-tracked reads.** ``read`` records the path it returns so a
  downstream grounding/citation layer can treat it as observed evidence;
  ``read_silent`` returns identical content but is explicitly NOT recorded,
  letting the model browse siblings without polluting its citation set.
* **Backend-bound, profile-selected.** This module owns the *contracts*;
  whether a given tenant is shown the lean pool vs a legacy typed surface
  is governed by ``RuntimeConstants.tool_surface_profile`` + the per-tool
  ``tool_surface_*_enabled`` flags. Core never registers a concrete
  ``Tool`` here (default implementations live in the host).

The canonical tool *names* are stable identifiers a backend SHOULD adopt
for its lean agent-facing surface so the surface is uniform across
backends. A backend MAY prefix/namespace them internally, but the
agent-visible name should match these constants for a consistent product.
"""
from __future__ import annotations

from collections.abc import Sequence

from protocore.contracts.types import ToolDefinition, ToolParameterSchema

# ---------------------------------------------------------------------------
# Profile identifiers (string-typed, forward-compatible)
# ---------------------------------------------------------------------------

TOOL_SURFACE_PROFILE_LEGACY = "legacy"
"""Legacy profile — the backend keeps whatever bespoke/typed tool surface it
already registered. Default for existing tenants so behaviour is unchanged
unless a tenant opts in."""

TOOL_SURFACE_PROFILE_LEAN = "lean"
"""Lean profile — the backend exposes only the small universal pool defined
here (model composes shell via ``exec``; grounding-tracked ``read`` /
``read_silent``; ``write`` / ``find`` / ``search``; a generic ``answer``)."""

TOOL_SURFACE_PROFILES: tuple[str, str] = (
    TOOL_SURFACE_PROFILE_LEGACY,
    TOOL_SURFACE_PROFILE_LEAN,
)
"""All recognised tool-surface profiles. ``RuntimeConstants`` validates the
``tool_surface_profile`` field against this set (as a ``Literal``)."""

# ---------------------------------------------------------------------------
# Canonical lean tool names (stable agent-facing identifiers)
# ---------------------------------------------------------------------------

LEAN_TOOL_EXEC = "exec"
"""Run a runtime binary by path with an argv list and optional stdin. The
runtime exposes its executables through a tool registry (e.g. a query /
compute binary the environment provides); the model picks the binary path
and composes its arguments / stdin. This is NOT a shell — there is no
``/bin/sh`` and no shell metacharacter handling. To read a file use
:data:`LEAN_TOOL_READ`; to list or locate files use :data:`LEAN_TOOL_FIND`
/ :data:`LEAN_TOOL_SEARCH`. The single power tool that lets one capable
model invoke any runtime binary without a bespoke typed verb per
operation."""

LEAN_TOOL_READ = "read"
"""Read a file and RECORD its path as observed evidence (grounding-tracked).
Use for any source you may cite — the recorded path feeds the
grounding/citation layer so the model's own minimal reads become the
citation set."""

LEAN_TOOL_READ_SILENT = "read_silent"
"""Read a file WITHOUT recording it as observed evidence. Use to browse /
compare sibling or candidate files you do NOT intend to cite, so the
citation set stays minimal. Identical content to ``read``; only the
grounding side effect differs."""

LEAN_TOOL_WRITE = "write"
"""Create or modify a file / record (the mutation verb). Optimistic-write
backends may accept a content hash for compare-and-swap; that is a backend
concern, not part of the universal name."""

LEAN_TOOL_FIND = "find"
"""Locate files/records by name or path pattern (filename-level discovery)."""

LEAN_TOOL_SEARCH = "search"
"""Search file/record *contents* by query (content-level discovery)."""

LEAN_TOOL_ANSWER = "answer"
"""Produce the final, terminal answer for the task with its supporting
citations. A generic terminal verb — every backend's lean surface MUST
guarantee a terminal answer is always producible. Besides the answer
``message`` and the grounding ``refs``, the answer carries an ``outcome``
disposition code whose accepted values are defined by the backend's own
answer contract (core only owns the field's *shape*, never a fixed value
set)."""

LEAN_TOOL_NAMES: tuple[str, ...] = (
    LEAN_TOOL_EXEC,
    LEAN_TOOL_READ,
    LEAN_TOOL_READ_SILENT,
    LEAN_TOOL_WRITE,
    LEAN_TOOL_FIND,
    LEAN_TOOL_SEARCH,
    LEAN_TOOL_ANSWER,
)
"""The canonical lean pool, in a stable order. Backends SHOULD expose
exactly this set (modulo per-tool RC enables) under the lean profile."""

# ---------------------------------------------------------------------------
# Canonical parameter schemas (domain-neutral JSON Schema fragments)
# ---------------------------------------------------------------------------
#
# These describe the MINIMAL universal arg shape for each verb. A backend may
# widen a schema (extra optional fields are allowed at its adapter), but it
# SHOULD accept at least these fields so the agent-facing contract is uniform.

_EXEC_SCHEMA = ToolParameterSchema(
    properties={
        "path": {
            "type": "string",
            "description": (
                "Path of the runtime binary to invoke (a binary the "
                "environment exposes via its runtime tool registry, e.g. a "
                "query or compute binary). Not a shell command line."
            ),
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Argument vector (argv) passed to the binary after its path. "
                "One element per argument; no shell quoting or word-splitting."
            ),
        },
        "stdin": {
            "type": "string",
            "description": (
                "Optional standard-input body fed to the binary (e.g. a query "
                "string for a query binary). Empty string for binaries that "
                "take no stdin."
            ),
        },
    },
    required=["path"],
)

_READ_SCHEMA = ToolParameterSchema(
    properties={
        "path": {
            "type": "string",
            "description": "Path of the file/record to read.",
        },
    },
    required=["path"],
)

# read_silent shares the read arg shape — the only difference is the
# (non-)recording side effect, documented in the description.
_READ_SILENT_SCHEMA = _READ_SCHEMA

_WRITE_SCHEMA = ToolParameterSchema(
    properties={
        "path": {
            "type": "string",
            "description": "Path of the file/record to create or modify.",
        },
        "content": {
            "type": "string",
            "description": "New content to write.",
        },
    },
    required=["path", "content"],
)

_FIND_SCHEMA = ToolParameterSchema(
    properties={
        "pattern": {
            "type": "string",
            "description": "Name/path glob or pattern to locate.",
        },
    },
    required=["pattern"],
)

_SEARCH_SCHEMA = ToolParameterSchema(
    properties={
        "query": {
            "type": "string",
            "description": "Content query to search for across files/records.",
        },
    },
    required=["query"],
)

_ANSWER_SCHEMA = ToolParameterSchema(
    properties={
        "message": {
            "type": "string",
            "description": "The final answer text for the task.",
        },
        "outcome": {
            "type": "string",
            "description": (
                "Outcome / disposition code for the answer, drawn from the "
                "set the runtime's answer contract defines (the backend "
                "publishes its accepted values + their meaning). Set it to the "
                "value that matches how the task resolved."
            ),
        },
        "refs": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The minimal set of source paths whose content supports the "
                "answer."
            ),
        },
    },
    required=["message"],
)

# ---------------------------------------------------------------------------
# Canonical descriptions (generic — NO baked-in step hints / paths / verbs)
# ---------------------------------------------------------------------------
#
# Descriptions stay capability-only: what the verb does, never a mandated
# bootstrap sequence, exact paths, or "call X before answering" coaching.
# The model decides how to use the pool.

_DESCRIPTIONS: dict[str, str] = {
    LEAN_TOOL_EXEC: (
        "Run a runtime binary by path with argv + optional stdin (e.g. a "
        "query/compute binary the environment provides via its runtime tool "
        "registry). This is NOT a shell — to read a file use the read tool; "
        "to list or locate files use find/search."
    ),
    LEAN_TOOL_READ: (
        "Read a file/record and record its path as observed evidence. Use this "
        "for any source whose content may support your answer — recorded reads "
        "form your citation set."
    ),
    LEAN_TOOL_READ_SILENT: (
        "Read a file/record WITHOUT recording it as evidence. Use this to "
        "browse or compare candidates you do not intend to cite, keeping your "
        "citation set minimal."
    ),
    LEAN_TOOL_WRITE: (
        "Create or modify a file/record. Use only after reading the governing "
        "record/policy; confirm the result before reporting success."
    ),
    LEAN_TOOL_FIND: (
        "Locate files/records by name or path pattern."
    ),
    LEAN_TOOL_SEARCH: (
        "Search file/record contents by query."
    ),
    LEAN_TOOL_ANSWER: (
        "Produce your final answer for the task. Set outcome to the "
        "disposition code your runtime's answer contract defines, and include "
        "in refs the minimal set of sources whose content supports your "
        "conclusion."
    ),
}

_SCHEMAS: dict[str, ToolParameterSchema] = {
    LEAN_TOOL_EXEC: _EXEC_SCHEMA,
    LEAN_TOOL_READ: _READ_SCHEMA,
    LEAN_TOOL_READ_SILENT: _READ_SILENT_SCHEMA,
    LEAN_TOOL_WRITE: _WRITE_SCHEMA,
    LEAN_TOOL_FIND: _FIND_SCHEMA,
    LEAN_TOOL_SEARCH: _SEARCH_SCHEMA,
    LEAN_TOOL_ANSWER: _ANSWER_SCHEMA,
}

# Which canonical verbs RECORD grounding evidence on read. ``read`` does;
# ``read_silent`` deliberately does not. Exposed so a host adapter can
# assert its (non-)recording behaviour against the contract.
GROUNDING_TRACKED_TOOLS: frozenset[str] = frozenset({LEAN_TOOL_READ})
"""Canonical verbs whose successful invocation records observed evidence."""


def lean_tool_definition(name: str) -> ToolDefinition:
    """Return the canonical :class:`ToolDefinition` for a lean verb.

    Args:
        name: one of :data:`LEAN_TOOL_NAMES`.

    Raises:
        KeyError: if ``name`` is not a recognised lean verb.
    """
    return ToolDefinition(
        name=name,
        description=_DESCRIPTIONS[name],
        parameters=_SCHEMAS[name],
    )


def lean_tool_surface() -> tuple[ToolDefinition, ...]:
    """Return the full lean pool as canonical :class:`ToolDefinition` specs.

    Order matches :data:`LEAN_TOOL_NAMES` (stable for KV-prefix cache
    stability + deterministic test assertions).
    """
    return tuple(lean_tool_definition(n) for n in LEAN_TOOL_NAMES)


def is_lean_tool(name: str) -> bool:
    """Whether ``name`` is a canonical lean-pool verb."""
    return name in _SCHEMAS


def select_tool_surface_profile(
    profile: str,
    *,
    valid: Sequence[str] = TOOL_SURFACE_PROFILES,
) -> str:
    """Normalise/validate a tool-surface profile string.

    Returns ``profile`` if it is a recognised profile, else falls back to
    :data:`TOOL_SURFACE_PROFILE_LEGACY` (the safe, behaviour-preserving
    default). Pure helper so the host wiring has one canonical place
    to resolve the RC value without duplicating the legacy fallback.
    """
    return profile if profile in valid else TOOL_SURFACE_PROFILE_LEGACY


__all__ = [
    "GROUNDING_TRACKED_TOOLS",
    "LEAN_TOOL_ANSWER",
    "LEAN_TOOL_EXEC",
    "LEAN_TOOL_FIND",
    "LEAN_TOOL_NAMES",
    "LEAN_TOOL_READ",
    "LEAN_TOOL_READ_SILENT",
    "LEAN_TOOL_SEARCH",
    "LEAN_TOOL_WRITE",
    "TOOL_SURFACE_PROFILES",
    "TOOL_SURFACE_PROFILE_LEAN",
    "TOOL_SURFACE_PROFILE_LEGACY",
    "is_lean_tool",
    "lean_tool_definition",
    "lean_tool_surface",
    "select_tool_surface_profile",
]
