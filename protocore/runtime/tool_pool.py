"""``assemble_tool_pool`` — context-time tool catalog assembly.

Builds the **filtered, retrieval-ranked tool
list** that the context manager embeds in the LLM system prompt.

Design summary
--------------
* The loop owns retrieval timing — it calls this once per turn, after
 the user message arrives.
* The function is **pure** w.r.t. registry state (no mutation) — it
 reads the registry under its own lock.
* Always-included (pinned) tools survive retrieval clipping; this
 protects the KV prefix cache.
* Result is sorted by ``Tool.name`` ASC for cache-prefix stability
 across turns.
"""
from __future__ import annotations

from collections.abc import Sequence

from protocore.contracts.tool_registry import IToolRegistry, ToolVisibilityPolicy
from protocore.contracts.tools import Tool
from protocore.runtime.tool_registry import ToolRegistry


def assemble_tool_pool(
    *,
    registry: IToolRegistry,
    tenant_id: str,
    visibility_policy: ToolVisibilityPolicy,
    context_query: str = "",
    max_tools_in_context: int,
    whitelist: Sequence[str] | None = None,
) -> list[Tool]:
    """Build the per-turn tool pool: top-K by retrieval, pinned first.

    Parameters
    ----------
    registry:
        :class:`IToolRegistry` instance (any concrete impl).
    tenant_id:
        Tenant scope for policy filter (Protocol-level; baseline core
        ignores). The host registry uses it for per-tenant overrides.
    visibility_policy:
        :class:`ToolVisibilityPolicy` controlling visible/blocked/pinned.
    context_query:
        Recent user message (or any free-text scoring hint). Empty means
        "no retrieval, return first-K by name".
    max_tools_in_context:
        Hard cap from ``rc.max_tools_in_context``. Above this, BM25
        prunes; below or equal, the entire policy-filtered set is
        returned.
    whitelist:
        Optional narrow scope (subagent dispatch). If provided +
        non-empty, only tools whose names are in this list AND pass the
        :class:`ToolVisibilityPolicy` are candidates.

    Returns
    -------
    list[Tool]
        Ordered, deduplicated tool list. Sort: name ASC (cache-stable).

    Notes
    -----
    The function is intentionally agnostic to the registry impl:

    * For the core :class:`ToolRegistry`, it reuses
      :meth:`ToolRegistry.compute_effective_surface` semantics via the
      same retrieval helpers — but returns *tools* rather than
      :class:`~protocore.contracts.types.ToolDefinition` because the
      caller may need both the JSON-Schema definition AND the
      concrete :class:`Tool` (for dispatch).
    * For a host registry, the same Protocol surface applies —
      ``list_for_tenant`` returns the policy-filtered set, then we
      apply BM25 here.
    """
    if max_tools_in_context <= 0:
        return []

    visible = list(registry.list_for_tenant(tenant_id, visibility_policy))

    if whitelist is not None:
        allow = frozenset(whitelist)
        if allow:
            visible = [t for t in visible if t.name in allow]

    # Fast path: catalog fits — sort by name and return.
    if len(visible) <= max_tools_in_context:
        return sorted(visible, key=lambda t: t.name)

    # Slow path: BM25 retrieval over the policy-filtered set. The
    # concrete ToolRegistry already ships a fully-correct
    # `compute_effective_surface` — but it returns ToolDefinition,
    # losing the Tool reference. So we re-rank here using the same
    # primitives.
    from protocore.runtime.tool_retrieval import (
        build_candidate,
        retrieve_tools,
    )

    candidates = [build_candidate(t.name, t.definition.description) for t in visible]
    name_to_tool = {t.name: t for t in visible}
    pinned_names = frozenset(visibility_policy.pinned)

    retrieved = retrieve_tools(
        context_query,
        candidates,
        top_k=max_tools_in_context,
        pinned=pinned_names,
    )

    chosen = [name_to_tool[c.name] for c in retrieved if c.name in name_to_tool]
    chosen.sort(key=lambda t: t.name)
    return chosen


def assemble_tool_pool_from_concrete(
    *,
    registry: ToolRegistry,
    tenant_id: str,
    visibility_policy: ToolVisibilityPolicy,
    context_query: str = "",
    max_tools_in_context: int,
    whitelist: Sequence[str] | None = None,
) -> list[Tool]:
    """Convenience wrapper when the concrete :class:`ToolRegistry` is in hand.

 Forwards directly to :func:`assemble_tool_pool`; exists primarily as
 a typed seam for the B integration (some callers know they
 have the concrete registry and want the precise type).
 """
    return assemble_tool_pool(
        registry=registry,
        tenant_id=tenant_id,
        visibility_policy=visibility_policy,
        context_query=context_query,
        max_tools_in_context=max_tools_in_context,
        whitelist=whitelist,
    )


__all__ = ["assemble_tool_pool", "assemble_tool_pool_from_concrete"]
