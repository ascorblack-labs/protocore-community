"""``IWorkspace`` Protocol — universal, scoped, searchable agent workspace.

A *session/task-scoped, searchable, atomic, lifecycle-bound local workspace*: a
place the agent can DUMP intermediate data (a SQL result set, a discovered
schema, ``jq`` output, working notes) once and re-read / search it MANY times
WITHOUT re-querying a flaky remote (the dump-once / re-read-many pattern — a
stability lever, not just ergonomics).

Why a NEW contract (and how it differs from what already exists)
===============================================================
A host typically already has a per-session **byte store** backing
Read/Write/Edit/Grep over ``{tenant}/sessions/{sid}/workspace/{path}``. Such a
store is durable (an ``IBlobStore`` over object storage) and atomic, but it is
NOT searchable, NOT scoped beyond ``session``, has no typed manifest, no
lifecycle, and no agent-facing "dump scratch + re-read" surface. This module
adds the contract for those; an adapter is expected to implement it by
EXTENDING the byte store it already has and adding a full-text/BM25 index over
the *manifest + searchable text* — not by duplicating the byte plane.

A session-scoped, searchable, auto-GC'd scratch workspace is a deliberate
addition over a bare per-session byte store: it gives the agent a typed,
lexically-searchable scratch area distinct from curated long-term memory and
from the task filesystem.

Design north-star (all behaviours below are the *contract*, not hints)
=====================================================================
* **Scope grammar** — every unit carries a ``(scope, scope_key)`` address:
  ``session | task | project | knowledge_base``. The default is ``session``
  (isolated, no cross-task/cross-session leak — the safest, most-isolated
  scope). ``task`` is for a single bounded sub-task inside a session;
  ``project`` is durable across sessions for a repo/workspace;
  ``knowledge_base`` is a durable per-knowledge-base tree whose planes carry
  different write permissions (see :class:`WorkspaceScope`). Controlled
  per-tenant via RuntimeConstants, never hard-coded. This deliberately mirrors
  a SUBSET of
  :class:`~protocore.contracts.memory.MemoryScope` so a tenant gets ONE
  consistent scoping mental model across memory + workspace.
* **Lifecycle-bound** — each unit is either :attr:`WorkspaceLifecycle.scratch`
  (ephemeral; eligible for per-scope soft-cap / TTL garbage collection — the
  "auto-GC'd" property) or :attr:`WorkspaceLifecycle.durable` (kept until
  explicitly deleted). Scratch is the default for dump-once data.
* **Atomic writes** — :meth:`IWorkspace.write` persists a unit atomically: the
  manifest row AND its bounded body (stored in the same ``workspace_units``
  ``BYTEA content`` column) are upserted in a SINGLE transaction. There is no
  separate byte plane / temp-then-rename and therefore no orphan-blob window. A
  crash mid-write never leaves a half-written unit visible. Idempotent per
  ``path`` (a re-write of the same path REPLACEs in place — see "idempotent
  file-per-unit resume" below).
* **Files-over-inline manifest** — the unit's body is stored as bytes
  (files-over-inline: large dumps live as files, not pasted into the prompt);
  the agent re-reads or SEARCHES them on demand. :meth:`IWorkspace.read` returns
  the body; :meth:`IWorkspace.list` returns the manifest (paths + sizes +
  lifecycle, NOT bodies); :meth:`IWorkspace.search` returns ranked hits over the
  searchable text. The model never has to hold a dump in context to re-use it.
* **Idempotent, file-per-unit resume** — writing the same ``(scope, scope_key,
  path)`` twice yields exactly one unit (REPLACE, version bumped). This makes a
  retried/resumed step safe: re-dumping ``orders.json`` after a reconnect does
  not create ``orders (1).json`` — it overwrites, and a subsequent read sees the
  latest. The returned :class:`WorkspaceWriteOutcome` says whether the write
  CREATED or REPLACED so the runtime/model can observe the idempotency.
* **SEARCHABLE (lexical/FTS, mandatory v1)** — :meth:`IWorkspace.search` ranks
  units by lexical relevance over their searchable text, using the SAME search
  shape as :class:`~protocore.contracts.memory.IMemory` (Postgres
  ``to_tsvector('simple', …)`` + ``ts_rank`` with a ``pg_trgm`` substring
  fallback) so a tenant has ONE consistent recall surface across memory and
  workspace. Lexical/BM25 is non-negotiable: the whole point is finding the
  exact ``SKU-12345`` / ``order_id`` / ``/proc/catalog/x.json`` token a dump
  contains. The contract is backend-agnostic so a hybrid-vector v2 is a
  non-breaking drop-in (:attr:`WorkspaceUnit.embedding` reserved).
* **Non-fatal, off the hot path, bounded** — a workspace op must never take down
  the agent. A slow/failed op surfaces as
  :class:`WorkspaceStoreUnavailableError`, which the runtime treats as "no
  workspace this turn" — never an error that aborts the run. Per-scope soft caps
  (bytes + unit count) bound growth.
* **Horizontal-scale-safe** — durable state lives in Postgres + the durable byte
  store (object store), NOT pod-local-only filesystem. No module-level / per-pod
  authority. Every method is tenant-scoped.

Reference shape: a durable byte store plus a full-text/BM25 manifest index,
atomic write, per-scope GC. Core defines the contract only and never imports an
implementation (boundary guard: ``tests/test_core_import_boundary.py``).
"""
from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    field_serializer,
    field_validator,
)

#: Sentinel marking a base64-encoded :attr:`WorkspaceUnit.content` body in JSON
#: mode. Pydantic v2 decodes a ``bytes`` field as UTF-8 when serialising to JSON,
#: which RAISES on non-UTF8 binary bodies (the unit explicitly supports binary
#: bodies — see the class docstring). We therefore base64-encode the body on
#: JSON output and decode it back on input, so the unit's
#: "serialize -> deserialize round-trips losslessly" invariant holds for ANY
#: bytes (incl. ``\xff\xfe\x00\x80``). The prefix makes the encoding
#: self-describing so the validator never mis-reads a literal text body as
#: base64.
_WORKSPACE_CONTENT_B64_PREFIX = "base64:"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspaceError(Exception):
    """Base class for workspace-subsystem domain errors.

Named distinctly from whatever error types an adapter's own byte plane
    raises; this is the *core contract* error hierarchy the ``IWorkspace``
    Protocol raises.
    """


class WorkspaceNotFoundError(WorkspaceError):
    """Requested workspace unit (path) is unknown for the given tenant+scope."""


class WorkspacePathError(WorkspaceError):
    """The supplied path is unsafe / malformed.

    Workspace-relative paths only — no leading ``/``, no ``..`` traversal, no
    backslashes, no control characters, no empty segments. The contract is
    fail-closed: an unsafe path is rejected, never silently rewritten to escape
    the scope root.
    """


class WorkspaceQuotaExceededError(WorkspaceError):
    """A write would exceed a hard per-tenant/per-scope limit.

    Distinct from the *soft* cap (which triggers best-effort GC of scratch units
    on write, see :meth:`IWorkspace.write`): this is the HARD ceiling
    (``workspace_max_bytes`` per unit / the absolute scope byte budget) above
    which the write is refused with a corrective error the model can read and
    route around (e.g. "dump a smaller slice"). Refusing is deliberate — silently
    truncating a dump would corrupt the dump-once/re-read-many guarantee.
    """


class WorkspaceStoreUnavailableError(WorkspaceError):
    """Backing store is temporarily unreachable.

    Callers (the runtime) MUST treat this as **non-fatal**: a read/search that
    raises this degrades to "no workspace this turn"; a write that raises it is
    surfaced to the model as an error tool result (it can retry or route around),
    never an error that aborts the run. Mirrors
    :class:`~protocore.contracts.memory.MemoryStoreUnavailableError`.
    """


# ---------------------------------------------------------------------------
# Scope + lifecycle grammar
# ---------------------------------------------------------------------------


class WorkspaceScope(StrEnum):
    """The scope a workspace unit is filed under.

    A unit's *address* is the tuple ``(tenant_id, scope, scope_key, path)``.
    This is a deliberate SUBSET of
    :class:`~protocore.contracts.memory.MemoryScope` (session/project shared;
    plus ``task`` and ``knowledge_base``, which are workspace-specific) so a
    tenant has one consistent scoping model across the two subsystems without
    the noise of memory's ``global``/``user``/``agent``/``custom`` tiers (a
    scratch workspace is not a tenant-wide or per-user durable store — that is
    what memory is for).

    * :attr:`session` — per conversation/session. ``scope_key`` = the session
      id. **The default** (isolated, no cross-session leak).
    * :attr:`task` — per bounded sub-task within a session (e.g. a single
      delegated goal / a single bounded step). ``scope_key`` = the task
      id. Lets a sub-task keep its own scratch that is GC'd when the task ends,
      without polluting the session scratch.
    * :attr:`project` — per project / repository / workspace, durable across
      sessions. ``scope_key`` = the project id. For product tenants that want a
      persistent scratch/notes area for a repo.
    * :attr:`knowledge_base` — per knowledge base, durable across sessions.
      ``scope_key`` = the knowledge-base id. Unlike the other scopes its tree is
      NOT uniformly writable: it has three top-level planes with different write
      permissions — the ingested SOURCES the agent may only read (never mutate),
      an agent-owned WIKI the agent writes and rewrites freely, and a single
      schema/instruction file describing the tree's own conventions, which BOTH
      the user and the agent may edit. Its units are ``durable``, so the
      scratch soft-cap GC never LRU-evicts them — the same guarantee project
      knowledge relies on.

    The enum *value* is the lowercase string stored in Postgres and surfaced on
    the wire.
    """

    session = "session"
    task = "task"
    project = "project"
    knowledge_base = "knowledge_base"


class WorkspaceLifecycle(StrEnum):
    """How long a workspace unit lives.

    * :attr:`scratch` — ephemeral. Eligible for the per-scope soft-cap / TTL
      garbage collector (the "auto-GC'd" property): when a scope exceeds its byte
      / unit-count cap, the least-recently-accessed *scratch* units are evicted
      first. This is the default for dump-once intermediate data.
    * :attr:`durable` — kept until explicitly deleted (or the scope itself is
      cleared). GC never evicts a durable unit on cap pressure (it counts toward
      the cap but is not auto-deleted). Use for a result the agent wants to
      guarantee survives the rest of the run.

    Naming mirrors the "scratch | dir" lifecycle; ``durable`` is the
    canonical name for the kept lane (clearer than "dir").
    """

    scratch = "scratch"
    durable = "durable"


#: Default scratch over durable: dump-once intermediate data is ephemeral.
DEFAULT_WORKSPACE_LIFECYCLE: WorkspaceLifecycle = WorkspaceLifecycle.scratch


# ---------------------------------------------------------------------------
# Address + unit types
# ---------------------------------------------------------------------------


class WorkspaceScopeRef(BaseModel):
    """A resolved ``(scope, scope_key)`` address. Frozen.

    ``scope_key`` is required for every scope (unlike memory's ``global`` which
    has a fixed empty key) — there is no tenant-wide workspace bucket.
    """

    model_config = ConfigDict(frozen=True)

    scope: WorkspaceScope
    scope_key: str = Field(min_length=1)

    def as_tuple(self) -> tuple[str, str]:
        """``(scope_value, scope_key)`` — the canonical storage key fragment."""
        return (self.scope.value, self.scope_key)


class WorkspaceUnit(BaseModel):
    """A single workspace unit (one file-per-unit dump).

    A unit is a *typed manifest row + a bounded byte body* stored together: the
    body bytes live in the manifest row's ``BYTEA content`` column (the bodies
    are capped by ``workspace_max_bytes``), and the rest of this model (minus
    ``content``) is the manifest, carrying the FTS-indexed ``searchable_text``.
    The model never has to hold the body in context to re-use it:
    :meth:`IWorkspace.list` returns manifests, :meth:`IWorkspace.read` fetches one
    body, :meth:`IWorkspace.search` ranks over ``searchable_text``.

    Idempotency identity
    --------------------
    A unit's logical identity is ``(tenant_id, scope, scope_key, path)``. Writing
    the same address twice REPLACEs in place (``version`` bumped) — never a
    duplicate / never a ``foo (1).json`` rename. This is the "idempotent
    file-per-unit resume" guarantee: a retried dump after a reconnect overwrites.

    Versioning (drift-guard)
    ------------------------
    ``version`` is bumped by the store on every REPLACE. A caller that read a
    unit at ``version=N`` may pass ``expected_version=N`` to
    :meth:`IWorkspace.write` / :meth:`IWorkspace.delete`; the store raises
    :class:`WorkspaceConflictError` if the live row advanced (concurrent writer).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Store-assigned surrogate id (UUID).")
    tenant_id: str
    scope: WorkspaceScope
    scope_key: str = Field(min_length=1)
    path: str = Field(
        description=(
            "Workspace-relative path (forward slashes), unique within "
            "``(tenant, scope, scope_key)``. The agent's chosen name for the "
            "dump, e.g. ``orders_2026.json`` or ``notes/schema.md``. Validated: "
            "no leading ``/``, no ``..``, no backslashes, no control chars."
        ),
    )
    lifecycle: WorkspaceLifecycle = Field(
        default=DEFAULT_WORKSPACE_LIFECYCLE,
        description=(
            "scratch (GC-eligible) or durable (kept). Default scratch for "
            "dump-once data."
        ),
    )
    size_bytes: int = Field(
        ge=0,
        description="Byte length of the stored body (uncompressed).",
    )
    sha256: str = Field(
        description=(
            "Hex sha256 of the body bytes. Lets a caller detect an unchanged "
            "re-write (idempotent no-op) and verify integrity on read."
        ),
    )
    content_type: str = Field(
        default="text/plain; charset=utf-8",
        description=(
            "Best-effort MIME type of the body. Text bodies are FTS-indexed; "
            "binary bodies are stored but not text-searchable (their PATH still "
            "is)."
        ),
    )
    searchable_text: str = Field(
        default="",
        description=(
            "The text that :meth:`IWorkspace.search` indexes (FTS/BM25). For a "
            "text body this is (a bounded prefix of) the body; for a binary body "
            "it is empty (only the path/metadata are searchable). The store sets "
            "this from the body on write; callers never invent it."
        ),
    )
    summary: str | None = Field(
        default=None,
        description=(
            "Optional one-line human/agent summary of what the unit contains "
            "(e.g. 'Top-50 orders by revenue, 2026 Q1'). Indexed alongside "
            "``searchable_text`` so a search can match the gist of a dump even "
            "when the exact body tokens differ. Caller-supplied or None."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary structured side-channel (e.g. the SQL that produced the "
            "dump, row counts, source table). NOT a fallback body store; search "
            "ranks ``searchable_text``/``summary``, not ``metadata``."
        ),
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Provenance: canonical source references the dump was derived from "
            "(file paths, run ids, query ids). Dovetails with the grounding / "
            "citation discipline. Never invented by the store."
        ),
    )
    embedding: list[float] | None = Field(
        default=None,
        description=(
            "RESERVED for the v2 hybrid-vector upgrade. v1 leaves it ``None`` "
            "and ranks lexically. Present so dense retrieval is non-breaking."
        ),
    )
    content: bytes | None = Field(
        default=None,
        description=(
            "The body bytes. ONLY populated on :meth:`IWorkspace.read` (and on "
            "the :class:`WorkspaceWriteOutcome` echo when the caller asked for "
            "it). :meth:`IWorkspace.list` / :meth:`IWorkspace.search` return "
            "units with ``content=None`` (manifest-only) — the whole point is "
            "files-over-inline: you fetch the body only when you actually need "
            "it."
        ),
    )
    version: int = Field(
        default=1,
        ge=1,
        description=(
            "Optimistic-concurrency version. Bumped by the store on every "
            "REPLACE; used by the drift-guard (see class docstring)."
        ),
    )
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = Field(
        default=None,
        description=(
            "Set when the unit is returned by :meth:`IWorkspace.read` / "
            ":meth:`IWorkspace.search` (recency signal for the scratch GC's "
            "least-recently-accessed eviction). ``None`` = never read."
        ),
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times the unit has been read/surfaced.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def _decode_content(cls, value: Any) -> Any:
        """Accept a base64-encoded body string from JSON and decode to bytes.

        The complement of :meth:`_serialize_content`. On Python-mode input the
        body is already ``bytes`` (or ``None``) and passes through untouched. On
        JSON-mode input it arrives as a ``str``: a value carrying the
        :data:`_WORKSPACE_CONTENT_B64_PREFIX` is base64-decoded back to the exact
        original bytes (lossless round-trip for non-UTF8 binary bodies); a bare
        ``str`` (legacy / hand-written UTF-8 text body) is encoded as UTF-8 so a
        plain JSON string still loads. ``None`` stays ``None`` (the manifest-only
        case).
        """
        if isinstance(value, str):
            if value.startswith(_WORKSPACE_CONTENT_B64_PREFIX):
                encoded = value[len(_WORKSPACE_CONTENT_B64_PREFIX):]
                try:
                    return base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(
                        "WorkspaceUnit.content base64 body is malformed"
                    ) from exc
            return value.encode("utf-8")
        return value

    @field_serializer("content", when_used="json")
    def _serialize_content(
        self, value: bytes | None, _info: FieldSerializationInfo
    ) -> str | None:
        """Emit the body as a base64 string in JSON mode (lossless for binary).

        Pydantic v2 serialises a ``bytes`` field to JSON by UTF-8-decoding it,
        which RAISES :class:`pydantic_core.PydanticSerializationError` on a
        non-UTF8 binary body. The unit documents binary bodies as supported, so
        we instead base64-encode the bytes (prefixed with
        :data:`_WORKSPACE_CONTENT_B64_PREFIX` so the decode side is
        unambiguous). ``None`` (the manifest-only case) stays ``None``. Python
        mode (``model_dump()``) keeps the raw ``bytes`` — only JSON output is
        re-encoded.
        """
        if value is None:
            return None
        return _WORKSPACE_CONTENT_B64_PREFIX + base64.b64encode(value).decode("ascii")


class WorkspaceWriteDecision(StrEnum):
    """Which branch an idempotent :meth:`IWorkspace.write` took."""

    created = "created"
    """No prior unit at this ``(scope, scope_key, path)`` — a new unit was
    stored."""

    replaced = "replaced"
    """A prior unit existed at this path and was REPLACED in place (body +
    manifest overwritten, ``version`` bumped). The returned unit is the new
    state."""

    unchanged = "unchanged"
    """A prior unit existed and the new body is byte-identical (same sha256) AND
    no metadata/lifecycle change — the write was a no-op. The returned unit is
    the pre-existing row, unchanged (idempotent re-dump of the same bytes)."""


class WorkspaceWriteOutcome(BaseModel):
    """Outcome of :meth:`IWorkspace.write`.

    ``decision`` makes the idempotency observable: the runtime can tell the model
    "stored (created)" / "overwrote your earlier dump (replaced)" / "already had
    exactly this (unchanged)" rather than silently double-writing or
    duplicate-renaming. ``unit`` is the surviving unit in all branches
    (manifest-only — ``content=None`` — unless the caller requested an echo).
    """

    model_config = ConfigDict(frozen=True)

    unit: WorkspaceUnit
    decision: WorkspaceWriteDecision
    evicted_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Paths of scratch units the soft-cap GC evicted as a side effect of "
            "this write (least-recently-accessed first). Empty when the scope "
            "was within its cap. Lets the caller/model know a stale dump was "
            "reclaimed."
        ),
    )


class WorkspaceConflictError(WorkspaceError):
    """Optimistic-concurrency (drift-guard) failure on a write/delete.

    Raised by :meth:`IWorkspace.write` / :meth:`IWorkspace.delete` when an
    ``expected_version`` was supplied and the live unit's version no longer
    matches — a concurrent writer mutated the same unit. The store NEVER
    blind-overwrites a drifted unit. Mirrors
    :class:`~protocore.contracts.memory.MemoryConflictError`.
    """


class WorkspaceHit(BaseModel):
    """A ranked search result from :meth:`IWorkspace.search`.

    Mirrors :class:`~protocore.contracts.memory.MemoryHit` exactly (a
    manifest-only :class:`WorkspaceUnit` + an opaque ``score``) so a tenant's
    memory and workspace recall surfaces are shaped identically.
    """

    model_config = ConfigDict(frozen=True)

    unit: WorkspaceUnit
    score: float = Field(
        description=(
            "Opaque relevance score, higher = better. v1 = lexical rank "
            "(``ts_rank`` + trgm similarity). NOT stable across backends — use "
            "only for ORDER BY / thresholding within one result set."
        ),
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IWorkspace(Protocol):
    """Universal, per-tenant, scoped, searchable agent workspace subsystem.

    Every method is **tenant-scoped** (first positional ``tenant_id``) for
    multi-tenant isolation, and **non-fatal by contract**: a backing-store
    outage surfaces as :class:`WorkspaceStoreUnavailableError`, which the runtime
    treats as "no workspace this turn" rather than aborting the run.

    The reference adapter is durable-byte-store + Postgres-FTS backed (N-pod
    safe). The Protocol is the seam tests fake against (the in-memory fake in
    ``protocore/tests_support/adapters.py``) and the seam a future vector/hybrid
    backend substitutes behind.
    """

    async def write(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
        content: bytes,
        *,
        lifecycle: WorkspaceLifecycle = DEFAULT_WORKSPACE_LIFECYCLE,
        content_type: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_refs: Sequence[str] | None = None,
        max_bytes: int | None = None,
        max_units_per_scope: int | None = None,
        max_scope_bytes: int | None = None,
        searchable_text_max_bytes: int | None = None,
        expected_version: int | None = None,
    ) -> WorkspaceWriteOutcome:
        """Atomically + idempotently persist one workspace unit (dump).

        The canonical *dump* entrypoint. Semantics:

        1. **Path safety.** ``path`` is validated (workspace-relative; no
           leading ``/``, no ``..``, no backslashes, no control chars, no empty
           segments). An unsafe path raises :class:`WorkspacePathError` — never a
           silent rewrite.
        2. **Hard quota.** If ``content`` exceeds the resolved per-unit
           ``max_bytes`` (``workspace_max_bytes``) OR the write would push the
           scope past the resolved absolute ``max_scope_bytes``, raise
           :class:`WorkspaceQuotaExceededError` (refuse, do not truncate).
        3. **Idempotent REPLACE.** Look up the unit at ``(tenant, scope,
           scope_key, path)``:
           * none → **CREATE** a new unit (``decision=created``).
           * exists + byte-identical body (same sha256) + no metadata/lifecycle
             change → **no-op** (``decision=unchanged``, the existing unit
             returned).
           * exists + different body/metadata/lifecycle → **REPLACE** in place,
             bump ``version`` (``decision=replaced``).
           Writing the same path twice therefore yields exactly ONE unit — the
           "idempotent file-per-unit resume" guarantee (a re-dump after a
           reconnect overwrites, never ``foo (1).json``).
        4. **Atomicity.** The manifest row and its bounded body (a ``BYTEA``
           column on the same ``workspace_units`` row) are upserted in ONE
           transaction — there is no separate byte plane / temp-then-rename and
           thus no orphan-blob or dangling-row window. A crash mid-write never
           leaves a half-written unit visible to a subsequent read/list/search.
        5. **Searchable text.** The store derives ``searchable_text`` from the
           body (a bounded prefix for a text body; empty for a binary body) and
           indexes it (+ ``summary``) for :meth:`search`. Callers never supply
           ``searchable_text``.
        6. **Soft-cap GC.** After the write, if the scope exceeds the resolved
           soft caps (``max_units_per_scope`` count, ``max_scope_bytes`` bytes —
           the same byte budget acts as both a hard pre-check and a post-write
           trim target), the store evicts least-recently-accessed **scratch**
           units (never ``durable`` ones) until back within the cap, and reports
           the evicted paths on the outcome. Bounded growth without a background
           job (the "auto-GC'd" property).

        **Drift-guard.** When ``expected_version`` is supplied AND the path
        already exists, raise :class:`WorkspaceConflictError` if the live unit's
        ``version`` no longer equals ``expected_version`` (concurrent writer).
        Ignored on the CREATE branch (nothing to conflict with).

        The ``max_*`` knobs and ``searchable_text_max_bytes`` default (when
        ``None``) to the values the host adapter resolves from the
        per-tenant RuntimeConstants (``workspace_max_bytes`` /
        ``workspace_max_units_per_scope`` / ``workspace_max_scope_bytes`` /
        ``workspace_searchable_text_max_bytes``) — core never hard-codes the
        numbers; the per-tenant runtime passes the RESOLVED values in per call so
        a pod-wide store honours per-tenant config (the adapter's own field is
        the static fallback). Same rationale as
        :meth:`~protocore.contracts.memory.IMemory.write`.

        Args:
            tenant_id: Tenant isolation key (required).
            scope: The :class:`WorkspaceScope` to file under.
            scope_key: The scope's key (required for every scope — no tenant-wide
                bucket).
            path: Workspace-relative unit path (the dump's name).
            content: The body bytes (a SQL result set, schema, notes, …).
            lifecycle: scratch (GC-eligible, default) or durable (kept).
            content_type: Optional MIME type; the store guesses from the path
                extension when ``None``.
            summary: Optional one-line gist, indexed for search.
            metadata: Optional structured side-channel (e.g. the source query).
            source_refs: Optional provenance references.
            max_bytes: Per-tenant resolved per-unit hard cap (``None`` = adapter
                fallback). 0 = unbounded.
            max_units_per_scope: Per-tenant resolved soft cap on unit count
                (``None`` = adapter fallback, 0 = unbounded).
            max_scope_bytes: Per-tenant resolved absolute byte budget for the
                scope (``None`` = adapter fallback, 0 = unbounded).
            searchable_text_max_bytes: Per-tenant resolved cap (bytes) on how
                much of a text body is extracted into the FTS-indexed
                ``searchable_text`` (``None`` = adapter fallback, 0 = index the
                whole body). The body is always stored + fully readable
                regardless; only the indexed prefix is bounded.
            expected_version: Optimistic-concurrency guard (see above).

        Returns:
            :class:`WorkspaceWriteOutcome` — the surviving unit (manifest-only,
            ``content=None``) + which branch fired + any GC-evicted paths.

        Raises:
            WorkspacePathError: unsafe/malformed path, or empty ``scope_key``.
            WorkspaceQuotaExceededError: body over the hard per-unit/scope cap.
            WorkspaceConflictError: drift-guard tripped on the REPLACE branch.
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal
                to the run).
        """
        ...

    async def read(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
    ) -> WorkspaceUnit:
        """Read one unit's full body by path. The *re-read* entrypoint.

        Returns the :class:`WorkspaceUnit` with ``content`` POPULATED (the body
        bytes) — this is the dump-once / re-read-many path: the agent fetches the
        body it dumped earlier without re-querying the remote. Best-effort
        reinforcement: the store SHOULD bump ``last_accessed_at`` /
        ``access_count`` on the read row (never fatally) so the scratch GC's LRU
        eviction favours genuinely-stale dumps.

        Raises:
            WorkspaceNotFoundError: no unit at ``(tenant, scope, scope_key,
                path)``.
            WorkspacePathError: unsafe/malformed path.
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        scopes: Sequence[WorkspaceScope] | None = None,
        scope_keys: dict[WorkspaceScope, str] | None = None,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
        limit: int = 20,
    ) -> Sequence[WorkspaceHit]:
        """Lexical/BM25 search over the tenant's workspace units.

        The *find-what-I-dumped* entrypoint, shaped IDENTICALLY to
        :meth:`~protocore.contracts.memory.IMemory.search` (same ranked-hit
        return, same scope fan-out, same non-fatal semantics) so memory +
        workspace are one consistent recall surface for a tenant.

        Ranks units by lexical relevance over ``searchable_text`` + ``summary``
        + ``path``. **Lexical/BM25 is mandatory** — this is what finds the exact
        ``SKU-12345`` / ``order_id`` / ``/proc/catalog/x.json`` token a dump
        contains (a vector-only search would miss it). v1 = ``to_tsvector`` +
        ``plainto_tsquery`` ranked by ``ts_rank`` with a ``pg_trgm`` substring
        fallback for partial tokens; ``simple`` config = RU + EN safe.

        Hits are **manifest-only** (``unit.content is None``): search tells you
        WHICH dumps match; you then :meth:`read` the one you want. (Returning
        every matching body would defeat files-over-inline.)

        Scope fan-out:
            * ``scopes=None`` → the store's default set (the host resolver
              narrows it per-tenant; the most-isolated default = ``(session,)``).
            * ``scope_keys`` maps each requested scope to its concrete key. A
              scope present in ``scopes`` but absent from ``scope_keys`` is
              skipped (every workspace scope needs a key — there is no
              fixed-key scope), so no other key's units can leak.
            * ``lifecycles`` optionally filters to scratch-only / durable-only.

        An empty ``query`` returns the most-recently-updated units in the
        requested scopes (recency order) so "what's in my scratch?" works without
        a search term. Best-effort reinforcement on the returned rows, as in
        :meth:`read`.

        Returns:
            Up to ``limit`` :class:`WorkspaceHit` ordered by descending score
            (empty sequence when nothing matches, never ``None``).

        Raises:
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

    async def list(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        *,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[WorkspaceUnit]:
        """List the manifest of units in one scope (NOT ranked retrieval).

        Returns units (manifest-only, ``content=None``) in deterministic
        ``updated_at DESC, id`` order, optionally filtered by ``lifecycles``.
        This is the agent-facing "ls my scratch" surface AND the admin "inspect
        a scope" surface — paths + sizes + lifecycle + summaries, no bodies.
        Pagination via ``limit`` / ``offset``.

        Raises:
            WorkspacePathError: empty ``scope_key``.
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

    async def delete(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        """Delete one unit by path (tenant-scoped). The *discard* entrypoint.

        ``expected_version`` activates the drift-guard: if supplied and the live
        unit's ``version`` differs, raise :class:`WorkspaceConflictError` instead
        of deleting.

        Returns:
            ``True`` if a unit was removed, ``False`` if the path was already
            absent (idempotent delete — absence is not an error). Removes BOTH
            the manifest row AND the durable body bytes.

        Raises:
            WorkspaceConflictError: drift-guard tripped.
            WorkspacePathError: unsafe/malformed path.
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

    async def clear_scope(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        *,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
    ) -> int:
        """Bulk-delete every unit in a scope (the lifecycle teardown hook).

        This is the *lifecycle-bound* cleanup the runtime calls when a scope
        ends — e.g. a session is finalized or a task completes — so scratch data
        does not accumulate forever (the auto-GC'd property, applied at scope
        end rather than on cap pressure). When ``lifecycles`` is supplied, only
        units in those lanes are removed (e.g. clear only ``scratch`` on session
        end while keeping ``durable`` results); when ``None``, every unit in the
        scope is removed. Removes both manifest rows and durable bodies.

        Returns:
            The number of units removed (``0`` when the scope was already empty —
            idempotent).

        Raises:
            WorkspacePathError: empty ``scope_key``.
            WorkspaceStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...


__all__ = [
    "DEFAULT_WORKSPACE_LIFECYCLE",
    "IWorkspace",
    "WorkspaceConflictError",
    "WorkspaceError",
    "WorkspaceHit",
    "WorkspaceLifecycle",
    "WorkspaceNotFoundError",
    "WorkspacePathError",
    "WorkspaceQuotaExceededError",
    "WorkspaceScope",
    "WorkspaceScopeRef",
    "WorkspaceStoreUnavailableError",
    "WorkspaceUnit",
    "WorkspaceWriteDecision",
    "WorkspaceWriteOutcome",
]
