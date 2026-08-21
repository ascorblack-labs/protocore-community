"""Concrete :class:`ToolRegistry` — in-core implementation of :class:`IToolRegistry`.

Implements the 3-layer tool surface (policy → clipping →
progressive discovery) using BM25 retrieval from
:mod:`protocore.runtime.tool_retrieval`.

This is the **default** registry shipped by core. A host may substitute a
database-backed variant that adds tenant overrides and cross-process cache
invalidation; that adapter satisfies the same :class:`IToolRegistry` Protocol.

Thread / async safety
---------------------
* In-memory state is guarded by a single :class:`threading.RLock` — the
 registry is mutated from background tasks (tool installation hooks)
 AND queried from the per-turn ``query`` async generator; both must
 see a consistent catalogue.
* Snapshot reads (``list_all`` / ``compute_effective_surface``) take a
 copy under the lock and operate lock-free thereafter — keeps the hot
 per-turn path cheap.

Tenant scoping
--------------
* Single global namespace in core. The host adapter introduces
 per-tenant override semantics; this baseline simply honours the
 :class:`ToolVisibilityPolicy` whitelist passed by the loop.
* The ``filter_by_whitelist`` helper extends the Protocol by exposing
 registry-level resolution from a flat name list — used by subagent
 dispatch before the per-turn surface is computed.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

from protocore.contracts.tool_registry import (
    IToolRegistry,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolDefinition
from protocore.runtime.tool_retrieval import (
    ToolRetrievalCandidate,
    build_candidate,
    normalized_fallback_match,
    retrieve_tools,
)


def _search_hint(tool: Tool) -> str:
    """The tool's optional multilingual retrieval hint (``search_hint``).

    Read via ``getattr`` so plain core tools without the ClassVar stay
    supported. The hint joins the DISCOVERY corpus only (never the wire
    description) — see :func:`build_candidate`.
    """
    raw = getattr(tool, "search_hint", "")
    return raw if isinstance(raw, str) else ""


class ToolRegistry(IToolRegistry):
    """In-core, thread-safe, BM25-ranked tool catalogue.

 Implements :class:`IToolRegistry`. Lifecycle:

 * Constructed once per executor pod (or per test).
 * Tools registered at startup via :meth:`register` (idempotent on
 ``tool.name`` — re-registration overwrites the previous binding).
 * Looked up per-turn via :meth:`get` (O(1) dict access).
 * Surfaced per-turn via :meth:`compute_effective_surface` (BM25 over
 the policy-filtered subset).

 Sort invariant: every public list/search
 result is sorted by ``Tool.name`` ascending for KV-prefix-cache
 stability across turns.
 """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._lock = threading.RLock()
        self._tools: dict[str, Tool] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------------
    # IToolRegistry — register / get
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool; idempotent on :attr:`Tool.name`.

        Re-registering the same name overwrites the existing entry —
        used by tenant-override adapters that swap in a tenant-specific
        implementation at admission time.
        """
        with self._lock:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Idempotent — no error if absent."""
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Fetch tool by name; ``None`` if not registered."""
        with self._lock:
            return self._tools.get(name)

    # ------------------------------------------------------------------
    # IToolRegistry — listing / filtering
    # ------------------------------------------------------------------

    def list_all(self) -> Sequence[Tool]:
        """All registered tools, sorted by :attr:`Tool.name` ASC."""
        with self._lock:
            tools = list(self._tools.values())
        return sorted(tools, key=lambda t: t.name)

    def list_for_tenant(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> Sequence[Tool]:
        """List tools visible to a tenant after the visibility policy filter.

 Order: sorted by ``Tool.name`` ASC (sort invariant —
 cache-prefix stability).

 Note: ``tenant_id`` is accepted for Protocol compliance but
 unused in this baseline (single-namespace registry);
 the host PG-backed variant uses it for tenant scoping.
 """
        del tenant_id  # baseline: single namespace
        with self._lock:
            all_tools = list(self._tools.values())

        if policy.visible:
            filtered = [t for t in all_tools if t.name in policy.visible]
        else:
            filtered = list(all_tools)
        filtered = [t for t in filtered if t.name not in policy.blocked]
        return sorted(filtered, key=lambda t: t.name)

    def filter_by_whitelist(self, names: Sequence[str]) -> Sequence[Tool]:
        """Resolve a list of names to :class:`Tool` instances.

 Used by subagent dispatch before building the per-turn
 surface: the subagent's ``tool_whitelist_json`` is a flat name
 list — this method resolves it against the catalogue, dropping
 unknown names silently (registration is the source of truth).

 Result is sorted by name ASC (sort invariant).
 """
        with self._lock:
            tools: list[Tool] = [self._tools[n] for n in names if n in self._tools]
        return sorted(tools, key=lambda t: t.name)

    # ------------------------------------------------------------------
    # IToolRegistry — 3-layer surface (policy → clipping → retrieval)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str = "",
        whitelist: Sequence[str] | None = None,
        policy: ToolVisibilityPolicy | None = None,
    ) -> Sequence[Tool]:
        """BM25-ranked search across the policy-filtered subset.

        ``whitelist`` (if provided) narrows the candidate pool — used by
        the :class:`ToolSearch` tool to restrict matches to the
        subagent's allowed surface. ``policy`` (if provided) applies the
        per-run visibility contract (:func:`policy_admits` — ``blocked``
        always denied; a non-empty ``visible`` admits only
        ``visible | pinned | forced_pinned``) so discovery can never return
        a schema the dispatch gate would refuse (tools-initiative A2).

        The searchable corpus is ``name + description + search_hint`` —
        the optional per-tool multilingual hint makes RU discovery queries
        match without changing the LLM-visible description. When BM25 finds
        nothing for a non-empty query, a normalized substring/prefix
        fallback (:func:`normalized_fallback_match`) re-ranks the same
        candidates so inflected/partial queries still discover tools.

        Empty query returns the first ``top_k`` candidates by name
        order (deterministic). Sort invariant within ranked groups —
        ties broken by name ASC.
        """
        del tenant_id  # baseline: single namespace
        with self._lock:
            pool = list(self._tools.values())

        if whitelist is not None:
            allow = frozenset(whitelist)
            pool = [t for t in pool if t.name in allow]
        if policy is not None:
            pool = [t for t in pool if policy_admits(policy, t.name)]

        # Sort invariant: pool is name-ASC (cache-prefix
        # stability + cross-pod determinism). The BM25 stable sort in
        # :func:`retrieve_tools` breaks equal-score ties by candidate-list
        # order, so the candidate list MUST be name-ASC for the docstring's
        # "ties broken by name ASC" promise to hold regardless of the order
        # tools were registered in. without this, two pods that
        # registered the same tools in different orders could surface
        # different results for the same query at the top_k cut.
        pool.sort(key=lambda t: t.name)

        candidates = [
            build_candidate(t.name, t.definition.description, _search_hint(t))
            for t in pool
        ]
        name_to_tool = {t.name: t for t in pool}

        retrieved = retrieve_tools(query, candidates, top_k=top_k)
        if not retrieved and not query.strip():
            # Empty query: return first top_k by name (deterministic).
            return sorted(pool, key=lambda t: t.name)[:top_k]
        if not retrieved and query.strip():
            # Zero BM25 overlap (e.g. an inflected RU query) — normalized
            # substring/prefix fallback over the SAME policy-filtered pool.
            retrieved = normalized_fallback_match(query, candidates, top_k=top_k)
        # Final ordering: name ASC (cache-stable). The retrieval order is
        # informative for selection but the search result must stay
        # byte-deterministic per the sort invariant — same contract as
        # :meth:`compute_effective_surface`.
        chosen = [name_to_tool[c.name] for c in retrieved if c.name in name_to_tool]
        chosen.sort(key=lambda t: t.name)
        return chosen

    def compute_effective_surface(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
        *,
        query: str = "",
        top_k: int | None = None,
    ) -> Sequence[ToolDefinition]:
        """3-layer filter: policy → clipping → progressive discovery.

 Returns the ranked :class:`ToolDefinition` list ready to be
 embedded in the LLM context (system-prompt tools section).

 Implementation:

 1. Apply :class:`ToolVisibilityPolicy` (visible + blocked), then
 re-admit ``forced_pinned`` tools that ``blocked`` did NOT deny —
 the core tool-surface floor must survive a tenant ``visible``
 whitelist, not just the BM25 clip (see
 :meth:`_floored_visible_tools`).
 2. If ``top_k`` is ``None`` or list ≤ top_k: return floored set
 sorted by name (no retrieval).
 3. Otherwise: BM25 rank by ``query``; pinned (policy.pinned) +
 forced_pinned + ``always_load`` tools always included; top-K by
 score; ties broken by name.

 ``always_load`` (tools-initiative A2 — discovery resurrection): a
 tool whose class sets ``always_load = True`` (e.g. ``ToolSearch``)
 is ALWAYS part of the advertised surface, independent of BM25 score
 and ``top_k`` — pin semantics, but class-driven instead of
 policy-driven. Precedence: the visibility policy still wins —
 ``blocked`` denies an always-load tool outright, and a non-empty
 ``visible`` whitelist that omits it keeps it out (only
 ``forced_pinned`` re-admits past the whitelist). It survives ONLY
 the layer-2/3 clip.

 """
        visible_tools = self._floored_visible_tools(tenant_id, policy)

        # Layer 2 + 3: clipping + retrieval
        if top_k is None or len(visible_tools) <= top_k:
            return [t.definition for t in visible_tools]

        # ``_search_hint`` joins the per-turn clip corpus too (NOT just
        # :meth:`search` / ToolSearch) — a tool's RU+EN ``search_hint`` must
        # feed the BM25 candidate so a Russian prompt that scores ~0 against
        # the EN name/description still surfaces it in the advertised payload,
        # not only via progressive discovery. The hint never reaches the
        # wire ``description`` (see :func:`build_candidate`). Helps both
        # bundled and dynamic tools.
        candidates = [
            build_candidate(t.name, t.definition.description, _search_hint(t))
            for t in visible_tools
        ]
        name_to_tool = {t.name: t for t in visible_tools}
        # Layer-3 pins + the core floor. ``forced_pinned`` is already in the
        # pool (``_floored_visible_tools`` re-admitted it past the ``visible``
        # whitelist), but we ALSO pass it to retrieval so it bypasses the BM25
        # clip: a Russian prompt scores 0.0 against English tool names and
        # would otherwise drop the floor at the top-K cut. ``pinned``
        # is the per-session ToolSearch/progressive-discovery set. ``always_load``
        # names are the class-driven floor (policy-admitted only — derived from
        # the post-policy pool, so ``blocked``/whitelist still win). Union (not
        # replace) so all three survive the clip.
        always_load_names = frozenset(
            t.name for t in visible_tools if bool(getattr(t, "always_load", False))
        )
        pinned_names = (
            frozenset(policy.pinned) | policy.forced_pinned | always_load_names
        )
        retrieved = retrieve_tools(query, candidates, top_k=top_k, pinned=pinned_names)

        # Final ordering: name ASC (cache-stable). The retrieval order
        # is informative for selection but the LLM context must stay
        # byte-deterministic per cache invariant.
        chosen = [name_to_tool[c.name] for c in retrieved if c.name in name_to_tool]
        chosen.sort(key=lambda t: t.name)
        return [t.definition for t in chosen]

    def _floored_visible_tools(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> list[Tool]:
        """Policy-visible tools with the ``forced_pinned`` floor re-admitted.

 :meth:`list_for_tenant` applies the ``visible`` whitelist + ``blocked``
 deny. But ``forced_pinned`` (the core tool-surface floor) must be
 present *regardless of the whitelist* — a tenant ``visible`` set that
 omits the six core file tools would otherwise recreate the cause-#3
 collapse for that tenant, exactly what the floor exists to prevent.
 So we re-admit any
 registered ``forced_pinned`` tool that ``blocked`` did NOT explicitly
 deny — ``blocked`` still wins (an operator can hard-deny a tool even
 against the floor). Result is name-ASC (cache-prefix stable).

 ``policy.pinned`` (the ToolSearch / progressive-discovery pin set) is
 re-admitted the SAME way:
 the dispatch permission gate and :func:`policy_admits` treat the
 allowed set under a non-empty ``visible`` whitelist as
 ``visible | pinned | forced_pinned``, so the advertised surface must
 agree — otherwise a tool the model just pinned via ToolSearch would
 be dispatch-callable yet vanish from the next turn's schema.
 """
        visible = list(self.list_for_tenant(tenant_id, policy))
        readmittable = (frozenset(policy.pinned) | policy.forced_pinned)
        if not readmittable:
            return visible
        present = {t.name for t in visible}
        missing = readmittable - present - policy.blocked
        if not missing:
            return visible
        with self._lock:
            readmit = [
                self._tools[name] for name in missing if name in self._tools
            ]
        if not readmit:
            return visible
        merged = visible + readmit
        merged.sort(key=lambda t: t.name)
        return merged

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        with self._lock:
            return name in self._tools


__all__ = ["ToolRegistry", "ToolRetrievalCandidate"]
