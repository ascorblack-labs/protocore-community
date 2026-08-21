"""Agent-facing memory tools — ``Remember`` / ``Recall`` / ``Forget``.

Three CRUD verbs: a tiny tool surface with the *complexity inside the
subsystem*, never tool sprawl. ``Recall`` doubles as search (it is a ranked
lexical/BM25 query), so a separate ``search`` verb would be redundant surface.

Why these tools live in CORE
============================
:class:`~protocore.contracts.memory.IMemory` is itself a *core* contract, so —
unlike ``AskUser`` (which needs Redis + the event bus, both host-only) —
the memory tools can hold their store dependency directly and keep all of their
logic in the universal core. The host builds the concrete
``PgMemoryStore``, injects it into these tool instances, and decides whether to
register them at all (RC-gated by
:attr:`~protocore.contracts.runtime_constants.RuntimeConstants.memory_enabled`).
This keeps the agent-facing behaviour universal + unit-testable against the
in-memory fake, with zero per-tenant targeting.

Scope resolution
================
The tools are scope-aware but **never hard-code a scope policy**. They read the
ambient ``(scope, scope_key)`` from :attr:`ToolContext.metadata` under
:data:`MEMORY_SCOPE_CONTEXT_KEY` / :data:`MEMORY_SCOPE_KEY_CONTEXT_KEY`, which
the host dispatcher populates from the resolved
:attr:`RuntimeConstants.memory_default_scope` for the tenant (the most-isolated
configuration = ``session``; broader tenants may use
``user``/``project``/``global``). When the metadata is absent the tools fall
back to ``session`` scope keyed by :attr:`ToolContext.session_id` — the safest,
most-isolated default. The model MAY override the scope per call via the optional
``scope`` argument when the tenant policy permits broader scopes (the host
layer can clip this).

Untrusted content
=================
A memory's ``text`` is untrusted (model-authored, sister-session-authored, or
written by a compromised tool) and re-enters context on recall, so it SHOULD be
scanned for prompt-injection / exfiltration content. ``remember`` and ``recall``
accept an optional :class:`~protocore.contracts.memory.IMemoryContentScanner`
(populated by the host): a flagged write is refused, and a flagged recalled
row is replaced by a ``[BLOCKED]`` placeholder rather than rendered verbatim.

Non-fatal contract
==================
A memory op must never abort the run. These tools translate a
:class:`~protocore.contracts.memory.MemoryStoreUnavailableError` into an *error
tool result* (``is_error=True``) the model can read and route around, rather
than letting it propagate as a dispatch failure. (Validation errors still raise
so the dispatcher surfaces a corrective message.)
"""
from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from protocore.contracts.memory import (
    DEFAULT_RECALL_SCOPES,
    IMemory,
    IMemoryContentScanner,
    MemoryScope,
    MemoryStoreUnavailableError,
    blocked_memory_placeholder,
)
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Canonical tool names (stable, surfaced to the LLM)
# ---------------------------------------------------------------------------

REMEMBER_TOOL_NAME: Final[str] = "Remember"
RECALL_TOOL_NAME: Final[str] = "Recall"
FORGET_TOOL_NAME: Final[str] = "Forget"

MEMORY_TOOL_NAMES: Final[tuple[str, ...]] = (
    REMEMBER_TOOL_NAME,
    RECALL_TOOL_NAME,
    FORGET_TOOL_NAME,
)

# ---------------------------------------------------------------------------
# ToolContext.metadata keys the host dispatcher populates per call.
# Core reads them; core never decides their values (that is the RC resolver's
# job — keeps the scope policy configurable + universal).
# ---------------------------------------------------------------------------

MEMORY_SCOPE_CONTEXT_KEY: Final[str] = "memory_default_scope"
MEMORY_SCOPE_KEY_CONTEXT_KEY: Final[str] = "memory_default_scope_key"
MEMORY_ALLOWED_SCOPES_CONTEXT_KEY: Final[str] = "memory_allowed_scopes"
# Per-call resolved scope-key map the host dispatcher injects for the
# non-session scopes (user/project/agent/custom) so recall/remember can address
# the tenant's own bucket without the model supplying the key. Value shape:
# ``{scope_value: scope_key}`` (a plain dict; ``MemoryScope`` keys are
# serialised to their string value by the resolver).
MEMORY_SCOPE_KEYS_CONTEXT_KEY: Final[str] = "memory_scope_keys"
# Defense-in-depth gate: the host dispatcher injects the resolved
# ``memory_enabled`` bool. When present AND explicitly ``False`` the memory
# tools refuse with a corrective tool error even if a stale visibility policy
# let the tool through. Absence is treated as "no opinion" (the tools run) so
# pure-core unit tests and any non-RC caller are unaffected.
MEMORY_ENABLED_CONTEXT_KEY: Final[str] = "memory_enabled"
# Per-call resolved store-tuning RCs: the dispatcher injects the
# per-tenant ``memory_write_similarity_threshold`` / ``memory_max_records_per_scope``
# so a pod-wide store honours per-tenant config (the store's own dataclass
# default is only the static fallback). Absent → the tool passes ``None`` and
# the store falls back to its field.
MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY: Final[str] = (
    "memory_write_similarity_threshold"
)
MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY: Final[str] = "memory_max_records_per_scope"

# Defensive caps so a single tool call cannot wedge the store / blow a row.
REMEMBER_TEXT_MAX_LENGTH: Final[int] = 8000
RECALL_QUERY_MAX_LENGTH: Final[int] = 2000
RECALL_DEFAULT_LIMIT: Final[int] = 10
RECALL_MAX_LIMIT: Final[int] = 50

# Default ``kind`` an empty/whitespace value coalesces to . The
# idempotency bucket is keyed on ``kind``, so a blank kind MUST normalise to the
# same bucket the default produces — otherwise the same fact saved with kind=""
# and kind="fact" lands in two buckets, defeating the idempotency guarantee, and
# an empty-kind recall could never target it.
DEFAULT_MEMORY_KIND: Final[str] = "fact"


def _normalise_kind(value: str | None) -> str | None:
    """Strip ``kind`` and coalesce empty/whitespace to :data:`DEFAULT_MEMORY_KIND`.

 Shared by :class:`RememberInput` and :class:`RecallInput` so write and recall
 agree on the empty-kind meaning and the idempotency bucket stays stable
 . ``None`` (the recall "no filter" sentinel) is preserved; a present
 but blank string becomes the default kind, never ``""``.
 """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or DEFAULT_MEMORY_KIND


# ---------------------------------------------------------------------------
# Dispatch-level gate (defense-in-depth) + per-call store tuning
# ---------------------------------------------------------------------------


def _memory_disabled(context: ToolContext) -> bool:
    """Return ``True`` only when the dispatcher injected an EXPLICIT False.

    The advertisement gate (``_apply_memory_visibility_policy`` in the host)
    is the primary control — a tenant with ``memory_enabled=False`` never sees
    these tools. This is the defense-in-depth layer for a direct call that
    slips past advertisement (e.g. a stale whitelist, a replayed transcript):
    the host dispatcher injects the resolved ``memory_enabled`` into
    metadata, and we refuse when it is explicitly ``False``. Absence is "no
    opinion" (return ``False``) so pure-core tests and non-RC callers are
    unaffected — same convention as the other per-tool dispatch guards.
    """
    md = context.metadata or {}
    if MEMORY_ENABLED_CONTEXT_KEY not in md:
        return False
    return md.get(MEMORY_ENABLED_CONTEXT_KEY) is False


def _resolved_similarity_threshold(context: ToolContext) -> float | None:
    """Per-tenant ``memory_write_similarity_threshold`` from metadata (or None)."""
    raw = (context.metadata or {}).get(MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolved_max_records_per_scope(context: ToolContext) -> int | None:
    """Per-tenant ``memory_max_records_per_scope`` from metadata (or None)."""
    raw = (context.metadata or {}).get(MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Scope resolution helper (shared by all three tools)
# ---------------------------------------------------------------------------


def _resolve_scope(
    context: ToolContext,
    requested_scope: str | None,
    requested_scope_key: str | None,
) -> tuple[MemoryScope, str]:
    """Resolve the effective ``(scope, scope_key)`` for a memory op.

 Precedence:
 1. explicit ``requested_scope`` argument (model override) — validated
 against the per-tenant allow-list in ``metadata`` if present.
 2. ``metadata[MEMORY_SCOPE_CONTEXT_KEY]`` — the RC-resolved tenant default.
 3. fall back to ``session`` scope keyed by ``context.session_id`` (the
 most-isolated default).

 ``scope_key`` resolution (precedence):
 1. explicit ``requested_scope_key`` (model-supplied).
 2. the dispatcher-injected per-scope key for THIS scope
 (:func:`_injected_scope_key` — the addressed-bucket path).
 3. the metadata default-scope key — but ONLY when the requested scope
 equals the metadata default scope (: never borrow the default
 scope's key, e.g. the user id, for a different requested scope).
 4. for ``session`` — the context session id; for ``global`` — the empty
 string.
 5. otherwise (a non-session, non-global scope with no resolvable key):
 raise a corrective ``ValueError`` (: never silently rebind to
 ``session_id`` under the requested scope — that mis-files the record;
 this mirrors the recall path and the :meth:`IMemory.write` contract).
 """
    md = context.metadata or {}

    # 1/2 — pick the scope.
    scope_str = requested_scope or md.get(MEMORY_SCOPE_CONTEXT_KEY) or MemoryScope.session.value
    try:
        scope = MemoryScope(scope_str)
    except ValueError as exc:
        raise ValueError(
            f"unknown memory scope {scope_str!r}; valid: "
            f"{', '.join(s.value for s in MemoryScope)}"
        ) from exc

    # enforce per-tenant allow-list when the dispatcher supplied one. This
    # applies to BOTH an explicit model override AND a resolved default —
    # otherwise a tenant whose ``memory_default_scope`` was (mis)configured
    # outside its own allow-list could still write there. ``allowed`` is
    # non-None for a present allow-list (empty → safe session-only set), so
    # this also denies non-session scopes under a blanked allow-list.
    allowed = _allowed_scope_values(context)
    if allowed is not None and scope.value not in allowed:
        raise ValueError(
            f"memory scope {scope.value!r} not permitted for this tenant; "
            f"allowed: {', '.join(sorted(allowed))}"
        )

    # 3 — pick the scope key.
    if scope is MemoryScope.global_:
        return scope, ""

    # : only borrow the metadata default-scope key when the requested
    # scope IS the metadata default scope. The default key is resolved for the
    # tenant's default scope (today: ``user`` → the user id); applying it to a
    # different requested scope (project/agent/custom) files the record under
    # the wrong id (a silent cross-scope key bleed).
    default_scope = md.get(MEMORY_SCOPE_CONTEXT_KEY)
    default_scope_key = (
        md.get(MEMORY_SCOPE_KEY_CONTEXT_KEY)
        if default_scope and str(default_scope) == scope.value
        else None
    )

    key = (
        requested_scope_key
        or _injected_scope_key(md, scope)
        or default_scope_key
        or (context.session_id if scope is MemoryScope.session else "")
    )
    if not key:
        # : a non-session, non-global scope with no resolvable key is a
        # corrective error — NOT a silent rebind to the session id (which would
        # file an agent/project/custom record under an unrelated session key).
        raise ValueError(
            f"scope_key required for memory scope {scope.value!r}: no key was "
            f"supplied and none is resolvable for this scope. Pass an explicit "
            f"'scope_key', or use the 'session' / 'global' scope."
        )
    return scope, key


#: Safe fallback allow-list when the per-tenant ``memory_allowed_scopes`` is
#: PRESENT but parses to empty . A present-but-empty allow-list means
#: "lock memory down", NOT "no restriction" — so it collapses to the single
#: most-isolated scope (``session``) rather than being read as allow-all.
_EMPTY_ALLOWLIST_FALLBACK: Final[frozenset[str]] = frozenset({MemoryScope.session.value})


def _allowed_scope_values(context: ToolContext) -> set[str] | None:
    """Parse the per-tenant allow-list from metadata into a set of scope values.

 Accepts the list/tuple/set form (what the dispatcher injects) AND a
 comma-separated string (the raw RC shape) so either is enforceable.

 Three outcomes (key-ABSENT must not be conflated with
 present-but-EMPTY):

 * key absent (``None``) → return ``None`` = "no allow-list, no restriction"
 (preserves the legacy default-tenant behaviour).
 * key present but parses to an EMPTY set (``""`` / ``" "`` / ``","`` /
 ``[]``) → return the safe session-only fallback
 :data:`_EMPTY_ALLOWLIST_FALLBACK`. An operator who blanks the allow-list
 gets memory locked to the most-isolated scope, NOT every scope opened.
 * a non-empty set → return it verbatim.

 The non-``None`` return is always an explicit restriction the callers
 (:func:`_resolve_scope` / :func:`_allowed_recall_scopes`) enforce.
 """
    raw = (context.metadata or {}).get(MEMORY_ALLOWED_SCOPES_CONTEXT_KEY)
    if raw is None:
        return None
    if isinstance(raw, str):
        values = {part.strip() for part in raw.split(",") if part.strip()}
    elif isinstance(raw, (list, tuple, set)):
        values = {str(a).strip() for a in raw if str(a).strip()}
    else:
        # An unexpected type is a present-but-uninterpretable allow-list; fail
        # closed to the safe session-only set rather than silently allow-all.
        return set(_EMPTY_ALLOWLIST_FALLBACK)
    # PRESENT but empty → DENY everything but the safe session-only fallback.
    return values or set(_EMPTY_ALLOWLIST_FALLBACK)


def _injected_scope_key(md: dict[str, Any], scope: MemoryScope) -> str | None:
    """Look up the dispatcher-injected resolved key for ``scope`` (or None).

    The dispatcher injects ``{scope_value: scope_key}`` under
    :data:`MEMORY_SCOPE_KEYS_CONTEXT_KEY` for the non-session scopes it can
    resolve (user/project/agent). Session is keyed by the context session id,
    so it is intentionally not required here.
    """
    keys = md.get(MEMORY_SCOPE_KEYS_CONTEXT_KEY)
    if not isinstance(keys, dict):
        return None
    val = keys.get(scope.value)
    return str(val) if val else None


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class RememberInput(BaseModel):
    """``remember`` LLM-facing input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=REMEMBER_TEXT_MAX_LENGTH,
        description=(
            "The fact / note / decision to remember, as a single self-contained "
            "sentence or short paragraph. Write durable, reusable knowledge "
            "(e.g. a discovered schema, an ID mapping, a convention, a lesson) — "
            "NOT raw transcript or a large data dump."
        ),
    )
    kind: str = Field(
        default=DEFAULT_MEMORY_KIND,
        max_length=64,
        description=(
            "Optional typed category: one of profile, preference, fact, "
            "decision, entity, reflection, skill, observation, other. "
            "Defaults to 'fact'."
        ),
    )
    scope: str | None = Field(
        default=None,
        description=(
            "Optional scope override: global, user, project, session, agent, "
            "or custom. Omit to use the tenant default (usually session). May "
            "be restricted by tenant policy."
        ),
    )
    scope_key: str | None = Field(
        default=None,
        description=(
            "Optional key for the chosen scope (e.g. a project id). Omit to use "
            "the ambient key. Ignored for the 'global' scope."
        ),
    )

    @field_validator("kind")
    @classmethod
    def _coalesce_kind(cls, value: str) -> str:
        # : strip + coalesce empty/whitespace → the default kind so the
        # idempotency bucket is stable (kind="" must not be a distinct bucket).
        normalised = _normalise_kind(value)
        return normalised if normalised is not None else DEFAULT_MEMORY_KIND


class RememberTool(Tool):
    """``remember`` — idempotently persist a memory.

    Backed by :meth:`IMemory.write` (two-stage idempotent: CREATE/MERGE/SKIP).
    The result tells the model which branch fired so a re-stated fact is
    visibly de-duplicated rather than silently double-written.
    """

    name_: ClassVar[str] = REMEMBER_TOOL_NAME
    description_: ClassVar[str] = (
        "Save a durable fact, decision, or lesson to long-term memory so you "
        "can recall it later (this turn or a future one). Idempotent: "
        "re-saving the same fact merges or skips, never duplicates. Use for "
        "reusable knowledge (schemas, ID mappings, conventions, things that "
        "failed), not raw data dumps."
    )
    #: Multilingual retrieval hint (EN+RU). Joined into the ToolSearch BM25
    #: corpus only — NEVER advertised to the model (the wire description stays
    #: ``description_``). Keeps RU discovery queries (e.g. "память") matching
    #: per the multilingual mandate without bloating the LLM-visible schema.
    search_hint: ClassVar[str] = (
        "memory remember save store fact note persist long-term "
        "память запомнить запомни сохранить заметка факт"
    )

    def __init__(self, store: IMemory, *, scanner: IMemoryContentScanner | None = None) -> None:
        self._store = store
        # : optional untrusted-content scanner. The host injects the
        # concrete threat-pattern scanner; pure-core callers pass None (no scan).
        self._scanner = scanner

    @property
    def name(self) -> str:
        return self.name_

    @property
    def definition(self) -> ToolDefinition:
        return _definition_from_model(self.name_, self.description_, RememberInput)

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        payload = RememberInput.model_validate(arguments)
        call_id = _call_id(context)
        if _memory_disabled(context):
            return _memory_disabled_error(call_id)
        # Scan the untrusted text BEFORE it is persisted; a flagged write is
        # refused with a corrective error so poison never reaches the store
        # (and therefore can never be re-injected on a later recall).
        # ``kind`` is ALSO untrusted (the model picks it) and is rendered
        # verbatim into recall output + the auto-recall <memory-context>
        # snapshot, so a benign text + malicious kind would otherwise bypass
        # both the write rejection and the recall [BLOCKED] placeholder.
        # Scan both; reject if EITHER trips.
        if self._scanner is not None:
            threat = self._scanner.scan(payload.text) or self._scanner.scan(
                payload.kind
            )
            if threat:
                return _blocked_write_error(call_id, threat)
        scope, scope_key = _resolve_scope(context, payload.scope, payload.scope_key)
        try:
            result = await self._store.write(
                context.tenant_id,
                scope,
                scope_key,
                payload.text,
                kind=payload.kind,
                similarity_threshold=_resolved_similarity_threshold(context),
                max_records_per_scope=_resolved_max_records_per_scope(context),
            )
        except MemoryStoreUnavailableError as exc:
            return _non_fatal_error(call_id, f"memory unavailable: {exc}")
        return ToolResult(
            tool_call_id=call_id,
            content=(
                f"{result.decision.value} memory {result.record.id} "
                f"(scope={scope.value}; kind={result.record.kind})"
            ),
            metadata={
                "memory_id": result.record.id,
                "decision": result.decision.value,
                "scope": scope.value,
                "scope_key": scope_key,
                "version": result.record.version,
            },
        )


# ---------------------------------------------------------------------------
# recall (doubles as search)
# ---------------------------------------------------------------------------


class RecallInput(BaseModel):
    """``recall`` LLM-facing input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        default="",
        max_length=RECALL_QUERY_MAX_LENGTH,
        description=(
            "Search text. Lexical/keyword ranked — use exact tokens (SKUs, IDs, "
            "table names, paths) when you have them. Empty query returns your "
            "most recent memories in scope."
        ),
    )
    scope: str | None = Field(
        default=None,
        description=(
            "Optional single-scope filter (global, user, project, session, "
            "agent, custom). Omit to search the tenant's default scope set."
        ),
    )
    scope_key: str | None = Field(
        default=None,
        description="Optional key for the chosen scope. Omit to use the ambient key.",
    )
    kind: str | None = Field(
        default=None,
        max_length=64,
        description="Optional filter by memory kind (e.g. 'decision').",
    )
    limit: int = Field(
        default=RECALL_DEFAULT_LIMIT,
        ge=1,
        le=RECALL_MAX_LIMIT,
        description=f"Max results (1-{RECALL_MAX_LIMIT}). Default {RECALL_DEFAULT_LIMIT}.",
    )

    @field_validator("kind")
    @classmethod
    def _coalesce_kind(cls, value: str | None) -> str | None:
        # : same normalisation as RememberInput so write and recall agree
        # on the empty-kind meaning. ``None`` (no filter) is preserved; a blank
        # string coalesces to the default kind (the bucket a blank-kind write
        # lands in) instead of being silently dropped to "no filter".
        return _normalise_kind(value)


class RecallTool(Tool):
    """``recall`` — ranked lexical/BM25 search over memory (also the search verb).

    Backed by :meth:`IMemory.search` (explicit, larger default page, lexical
    ranking — the path that finds exact identifiers a vector-only store misses).
    """

    name_: ClassVar[str] = RECALL_TOOL_NAME
    description_: ClassVar[str] = (
        "Search your long-term memory and return the most relevant saved facts, "
        "ranked. Keyword/lexical search: pass exact tokens (SKUs, order IDs, "
        "table/column names, file paths) for precise recall. Use this before "
        "re-deriving something you may already know or before re-querying a "
        "flaky data source."
    )
    #: Multilingual retrieval hint (EN+RU) for the ToolSearch corpus only.
    search_hint: ClassVar[str] = (
        "memory recall search retrieve lookup saved facts "
        "память вспомнить вспомни найти поиск сохранённые факты"
    )

    def __init__(self, store: IMemory, *, scanner: IMemoryContentScanner | None = None) -> None:
        self._store = store
        # : optional scanner applied to RECALLED rows (defense against a
        # poisoned-on-disk record the write-side scan never saw — e.g. a sister
        # session sharing a non-session scope). A flagged hit is replaced by a
        # [BLOCKED] placeholder rather than rendered verbatim back into context.
        self._scanner = scanner

    @property
    def name(self) -> str:
        return self.name_

    @property
    def definition(self) -> ToolDefinition:
        return _definition_from_model(self.name_, self.description_, RecallInput)

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        payload = RecallInput.model_validate(arguments)
        call_id = _call_id(context)
        if _memory_disabled(context):
            return _memory_disabled_error(call_id)

        # Scope fan-out: an explicit scope narrows to that one scope (validated
        # against the allow-list by ``_resolve_scope``); otherwise the store
        # reads the per-tenant default set, keyed by the ambient session/user/
        # project keys we resolve from context+metadata. The default fan-out is
        # CLIPPED to the per-tenant allow-list: a tenant restricted to
        # ``session`` must not recall ``global``/``user`` rows just because the
        # store's ``DEFAULT_RECALL_SCOPES`` includes them.
        kinds = [payload.kind] if payload.kind else None
        try:
            if payload.scope is not None:
                scope, scope_key = _resolve_scope(
                    context, payload.scope, payload.scope_key
                )
                hits = await self._store.search(
                    context.tenant_id,
                    payload.query,
                    scopes=[scope],
                    scope_keys={scope: scope_key},
                    kinds=kinds,
                    limit=payload.limit,
                )
            else:
                recall_scopes = _allowed_recall_scopes(context)
                # An EMPTY (not None) clipped list means the allow-list
                # permits no resolvable default scope — recall is a no-op.
                # We must NOT pass ``[]`` to the store (it reads ``[]`` as
                # "use DEFAULT_RECALL_SCOPES", silently widening past the
                # allow-list). ``None`` (no allow-list) still means store
                # defaults.
                if recall_scopes is not None and not recall_scopes:
                    return ToolResult(
                        tool_call_id=call_id,
                        content="No matching memories.",
                        metadata={"count": 0},
                    )
                hits = await self._store.search(
                    context.tenant_id,
                    payload.query,
                    scopes=recall_scopes,
                    scope_keys=_ambient_scope_keys(context),
                    kinds=kinds,
                    limit=payload.limit,
                )
        except MemoryStoreUnavailableError as exc:
            return _non_fatal_error(call_id, f"memory unavailable: {exc}")

        if not hits:
            return ToolResult(
                tool_call_id=call_id,
                content="No matching memories.",
                metadata={"count": 0},
            )
        # Each line surfaces the store id inline so the model can hand it
        # straight to ``forget`` — the display index alone is not addressable
        # (forget deletes by id, tenant-scoped). Same id ``remember`` already
        # prints, so this exposes nothing new.
        lines = [
            f"[{i + 1}] id={h.record.id} "
            f"({h.record.scope.value}/{self._safe_recall_kind(h.record.kind)}) "
            f"{self._safe_recall_text(h.record.text)}"
            for i, h in enumerate(hits)
        ]
        return ToolResult(
            tool_call_id=call_id,
            content="\n".join(lines),
            metadata={
                "count": len(hits),
                "memory_ids": [h.record.id for h in hits],
            },
        )

    def _safe_recall_text(self, text: str) -> str:
        """Return ``text`` for a recalled row, or a [BLOCKED] placeholder if it
 trips the injected scanner .

 Defense against a poisoned-on-disk record the write-side scan never saw.
 The raw text is replaced (never echoed) so the injection payload cannot
 re-enter context, while the model still sees that an entry was filtered.
 No scanner wired → text passes through unchanged.
 """
        if self._scanner is None:
            return text
        threat = self._scanner.scan(text)
        if threat:
            return blocked_memory_placeholder(text, reason=threat)
        return text

    def _safe_recall_kind(self, kind: str) -> str:
        """Return ``kind`` for a recalled row, or a [BLOCKED] placeholder if it
        trips the injected scanner.

        ``kind`` is rendered verbatim into the recall line, so a poisoned
        on-disk record whose KIND (not text) carries an injection must be
        neutralised here too — otherwise the same payload re-enters context
        through the kind field. The raw kind is replaced (never echoed). No
        scanner wired → kind passes through unchanged.
        """
        if self._scanner is None:
            return kind
        threat = self._scanner.scan(kind)
        if threat:
            return blocked_memory_placeholder(kind, reason=threat)
        return kind


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class ForgetInput(BaseModel):
    """``forget`` LLM-facing input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description=(
            "The id of the memory to delete. Copy the 'id=' value shown for "
            "the entry in a prior recall result (or the id a remember result "
            "returned)."
        ),
    )


class ForgetTool(Tool):
    """``forget`` — delete one memory by id. Backed by :meth:`IMemory.delete`."""

    name_: ClassVar[str] = FORGET_TOOL_NAME
    description_: ClassVar[str] = (
        "Delete a saved memory by its id (idempotent — already-gone is fine). "
        "Use to remove a fact that is now wrong or no longer useful."
    )
    #: Multilingual retrieval hint (EN+RU) for the ToolSearch corpus only.
    search_hint: ClassVar[str] = (
        "memory forget delete remove erase "
        "память забыть забудь удалить стереть"
    )

    def __init__(self, store: IMemory) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return self.name_

    @property
    def definition(self) -> ToolDefinition:
        return _definition_from_model(self.name_, self.description_, ForgetInput)

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        payload = ForgetInput.model_validate(arguments)
        call_id = _call_id(context)
        if _memory_disabled(context):
            return _memory_disabled_error(call_id)
        try:
            removed = await self._store.delete(context.tenant_id, payload.memory_id)
        except MemoryStoreUnavailableError as exc:
            return _non_fatal_error(call_id, f"memory unavailable: {exc}")
        return ToolResult(
            tool_call_id=call_id,
            content=(
                f"forgot memory {payload.memory_id}"
                if removed
                else f"memory {payload.memory_id} was already absent"
            ),
            metadata={"removed": removed, "memory_id": payload.memory_id},
        )


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _ambient_scope_keys(context: ToolContext) -> dict[MemoryScope, str]:
    """Best-effort ambient scope-key map for the default recall fan-out.

    We always know the session key (from the context). The dispatcher MAY also
    inject the resolved default scope+key (user/project) via metadata, and a
    full ``{scope: key}`` map under :data:`MEMORY_SCOPE_KEYS_CONTEXT_KEY`; we
    fold both in so a tenant using user/project/agent scope recalls its own
    bucket. A scope with no resolvable key is simply omitted (the store skips
    unaddressable non-global scopes), so no cross-key leak is possible.
    """
    keys: dict[MemoryScope, str] = {MemoryScope.session: context.session_id}
    md = context.metadata or {}
    # the per-scope key map (richest source).
    raw_map = md.get(MEMORY_SCOPE_KEYS_CONTEXT_KEY)
    if isinstance(raw_map, dict):
        for scope_value, scope_key in raw_map.items():
            if not scope_key:
                continue
            try:
                keys[MemoryScope(str(scope_value))] = str(scope_key)
            except ValueError:
                continue
    # the single default scope+key (back-compat / minimal injection).
    scope_str = md.get(MEMORY_SCOPE_CONTEXT_KEY)
    scope_key = md.get(MEMORY_SCOPE_KEY_CONTEXT_KEY)
    if scope_str and scope_key:
        try:
            keys[MemoryScope(str(scope_str))] = str(scope_key)
        except ValueError:
            pass
    return keys


def _allowed_recall_scopes(context: ToolContext) -> list[MemoryScope] | None:
    """The default recall fan-out clipped to the per-tenant allow-list.

    Returns ``None`` when no allow-list was supplied (the store then uses its
    own ``DEFAULT_RECALL_SCOPES``). When an allow-list IS present, the fan-out
    is the intersection of the store default scopes and the allow-list, further
    restricted to scopes whose key is resolvable from context/metadata
    (``global`` always resolves via its fixed empty-string key) — so a
    ``session``-only tenant never reads ``global``/``user``/``project`` rows on
    an unscoped recall.

    An EMPTY clipped result means "no allowed/resolvable scope" and is returned
    AS an empty list — the caller short-circuits to no-recall. We must NOT fall
    back to ``[session]`` here: a tenant whose allow-list excludes ``session``
    (e.g. ``["agent"]`` / ``["custom"]``) must not have ``session`` synthesised
    back in. Returning ``[]`` to the store would be read as "use defaults"
    (widening), so the recall callers treat the empty list as an explicit
    no-recall signal.
    """
    allowed = _allowed_scope_values(context)
    if allowed is None:
        return None
    resolvable = _ambient_scope_keys(context)
    clipped: list[MemoryScope] = []
    for scope in DEFAULT_RECALL_SCOPES:
        if scope.value not in allowed:
            continue
        # global is always addressable (fixed empty-string key); other scopes
        # need a concrete key, else the store cannot bucket them and they would
        # silently fall through to a default key.
        if scope is MemoryScope.global_ or scope in resolvable:
            clipped.append(scope)
    # Empty → no allowed+resolvable scope → explicit no-recall (NOT [session],
    # NOT the store defaults). The caller checks for this and skips the query.
    return clipped


def _memory_disabled_error(call_id: str) -> ToolResult:
    """Corrective tool result for a memory call made while memory is disabled.

    Defense-in-depth: the tool should never have been advertised, so this
    only fires on a direct/stale call. ``is_error=True`` so the model reads it
    as a corrective signal and stops re-issuing memory calls.
    """
    return ToolResult(
        tool_call_id=call_id,
        content=(
            "memory is disabled for this tenant; the Remember/Recall/Forget "
            "tools are unavailable. Do not call them again."
        ),
        is_error=True,
    )


def _blocked_write_error(call_id: str, threat: str) -> ToolResult:
    """Corrective tool result for a ``remember`` whose text tripped the scanner.

 : the untrusted memory text matched a prompt-injection / exfiltration
 pattern, so the write is refused (poison never reaches the store). ``threat``
 is the scanner's short descriptor (a pattern id / message), NOT the raw text.
 ``is_error=True`` so the model reads it as corrective and rewrites the fact.
 """
    return ToolResult(
        tool_call_id=call_id,
        content=(
            f"memory write blocked: the text matched a content-safety pattern "
            f"({threat}). Rewrite it as a plain factual note without "
            f"instruction-like or exfiltration content, then retry."
        ),
        is_error=True,
    )


def _call_id(context: ToolContext) -> str:
    """Stable per-invocation id for the ToolResult.

    The dispatcher matches results to calls by ``tool_call_id``; it injects the
    real id via ``metadata['tool_call_id']`` when available, else we fall back
    to the run id so the result is never unaddressable.
    """
    md = context.metadata or {}
    raw = md.get("tool_call_id")
    return str(raw) if raw else context.run_id


def _non_fatal_error(call_id: str, message: str) -> ToolResult:
    """Wrap a non-fatal memory failure as a readable error tool result.

    Memory must never abort the run; the model reads this and routes around it.
    """
    return ToolResult(tool_call_id=call_id, content=message, is_error=True)


def _definition_from_model(
    name: str,
    description: str,
    model: type[BaseModel],
) -> ToolDefinition:
    """Build a :class:`ToolDefinition` from a Pydantic input model's JSON schema
    (mirrors the ``AskUser`` auto-publish pattern)."""
    raw = model.model_json_schema()
    properties = raw.get("properties", {})
    required = raw.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []
    return ToolDefinition(
        name=name,
        description=description,
        parameters=ToolParameterSchema(properties=properties, required=required),
    )


def build_memory_tools(
    store: IMemory,
    *,
    scanner: IMemoryContentScanner | None = None,
) -> list[Tool]:
    """Construct the three memory tools bound to ``store``.

 The single wiring entrypoint the host calls (RC-gated by
 ``memory_enabled``). Returned in stable name order so the registry's
 KV-prefix-cache sort invariant is preserved.

 ``scanner`` is the optional untrusted-content scanner threaded
 into ``remember`` (scan-on-write) and ``recall`` (scan-on-read → [BLOCKED]
 placeholder). ``forget`` takes no text so it needs no scanner. When ``None``
 no scan runs (pure-core callers / tests).
 """
    return [
        ForgetTool(store),
        RecallTool(store, scanner=scanner),
        RememberTool(store, scanner=scanner),
    ]


__all__ = [
    "FORGET_TOOL_NAME",
    "MEMORY_ALLOWED_SCOPES_CONTEXT_KEY",
    "MEMORY_ENABLED_CONTEXT_KEY",
    "MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY",
    "MEMORY_SCOPE_CONTEXT_KEY",
    "MEMORY_SCOPE_KEYS_CONTEXT_KEY",
    "MEMORY_SCOPE_KEY_CONTEXT_KEY",
    "MEMORY_TOOL_NAMES",
    "MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY",
    "RECALL_DEFAULT_LIMIT",
    "RECALL_MAX_LIMIT",
    "RECALL_TOOL_NAME",
    "REMEMBER_TEXT_MAX_LENGTH",
    "REMEMBER_TOOL_NAME",
    "ForgetInput",
    "ForgetTool",
    "RecallInput",
    "RecallTool",
    "RememberInput",
    "RememberTool",
    "build_memory_tools",
]
