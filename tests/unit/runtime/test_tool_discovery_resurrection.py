"""Tools-initiative A2 — ToolSearch/discovery resurrection (core layer).

Covers the three core seams the initiative added:

1. ``always_load`` consumption — a tool whose class sets ``always_load=True``
   is ALWAYS on the computed surface (clip-proof, like a pin but
   class-driven); the visibility policy still wins (blocked / whitelist).
2. ``ToolRegistry.search`` honours the per-run :class:`ToolVisibilityPolicy`
   (blocked tools never leak through discovery) — and the dispatcher injects
   that policy into ``ToolContext.metadata`` for policy-aware tools.
3. RU-capable matching — per-tool ``search_hint`` (EN+RU) joins the BM25
   discovery corpus, and a normalized substring/prefix fallback catches
   inflected forms BM25 scores 0.0.
"""
from __future__ import annotations

from protocore.contracts.tool_registry import (
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.runtime.tool_registry import ToolRegistry
from protocore.runtime.tool_retrieval import (
    build_candidate,
    normalized_fallback_match,
)
from protocore.tools.memory import (
    MEMORY_TOOL_NAMES,
    RECALL_TOOL_NAME,
    REMEMBER_TOOL_NAME,
)

from ._tool_fixtures import MockTool

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _registry_with(*tools: MockTool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.register(tool)
    return reg


def _many_distinct_tools(n: int) -> list[MockTool]:
    """Tools whose descriptions all match a common query token."""
    return [
        MockTool(
            tool_name=f"Tool{chr(ord('A') + i)}",
            description=f"common keyword variant{i}",
        )
        for i in range(n)
    ]


# ----------------------------------------------------------------------
# 1. always_load — clip-proof, policy still wins
# ----------------------------------------------------------------------


def test_always_load_survives_bm25_clip() -> None:
    """An always_load tool with ZERO BM25 score stays on the surface."""
    discovery = MockTool(
        tool_name="ZZDiscovery",
        description="totally unrelated corpus entry",
        always_load=True,
    )
    reg = _registry_with(*_many_distinct_tools(6), discovery)
    defs = reg.compute_effective_surface(
        "tenant-1",
        ToolVisibilityPolicy(),
        query="common keyword",
        top_k=3,
    )
    names = [d.name for d in defs]
    assert "ZZDiscovery" in names, "always_load tool must survive the clip"


def test_always_load_blocked_by_policy_stays_blocked() -> None:
    """``blocked`` beats ``always_load`` — policy is authoritative."""
    discovery = MockTool(
        tool_name="ZZDiscovery",
        description="unrelated",
        always_load=True,
    )
    reg = _registry_with(*_many_distinct_tools(6), discovery)
    defs = reg.compute_effective_surface(
        "tenant-1",
        ToolVisibilityPolicy(blocked={"ZZDiscovery"}),
        query="common keyword",
        top_k=3,
    )
    assert "ZZDiscovery" not in [d.name for d in defs]


def test_always_load_not_readmitted_past_visible_whitelist() -> None:
    """Only ``forced_pinned`` re-admits past a tenant whitelist."""
    discovery = MockTool(
        tool_name="ZZDiscovery",
        description="unrelated",
        always_load=True,
    )
    others = _many_distinct_tools(4)
    reg = _registry_with(*others, discovery)
    defs = reg.compute_effective_surface(
        "tenant-1",
        ToolVisibilityPolicy(visible={t.tool_name for t in others}),
        query="common keyword",
        top_k=3,
    )
    assert "ZZDiscovery" not in [d.name for d in defs]


def test_always_load_included_without_clip() -> None:
    """No-clip path (pool <= top_k) trivially carries the tool."""
    discovery = MockTool(tool_name="ZZDiscovery", description="x", always_load=True)
    reg = _registry_with(discovery)
    defs = reg.compute_effective_surface(
        "tenant-1", ToolVisibilityPolicy(), query="anything", top_k=5
    )
    assert [d.name for d in defs] == ["ZZDiscovery"]


# ----------------------------------------------------------------------
# 2. search respects the visibility policy (info-leak fix)
# ----------------------------------------------------------------------


def test_search_drops_blocked_tools() -> None:
    memory = MockTool(
        tool_name=REMEMBER_TOOL_NAME,
        description="Save a durable fact to long-term memory",
    )
    other = MockTool(tool_name="Other", description="memory unrelated helper")
    reg = _registry_with(memory, other)
    policy = ToolVisibilityPolicy(blocked={REMEMBER_TOOL_NAME})
    hits = reg.search("memory", top_k=5, policy=policy)
    assert REMEMBER_TOOL_NAME not in [t.name for t in hits], (
        "a blocked tool's schema must not leak through discovery"
    )


def test_search_without_policy_keeps_legacy_pool() -> None:
    memory = MockTool(tool_name=REMEMBER_TOOL_NAME, description="long-term memory")
    reg = _registry_with(memory)
    hits = reg.search("memory", top_k=5)
    assert REMEMBER_TOOL_NAME in [t.name for t in hits]


def test_search_visible_whitelist_admits_pinned_and_forced() -> None:
    """Non-empty ``visible`` admits visible | pinned | forced_pinned only."""
    a = MockTool(tool_name="Alpha", description="alpha memory")
    b = MockTool(tool_name="Bravo", description="bravo memory")
    c = MockTool(tool_name="Charlie", description="charlie memory")
    reg = _registry_with(a, b, c)
    policy = ToolVisibilityPolicy(
        visible={"Alpha"},
        pinned={"Bravo"},
        forced_pinned=frozenset({"Charlie"}),
    )
    names = {t.name for t in reg.search("memory", top_k=5, policy=policy)}
    assert names == {"Alpha", "Bravo", "Charlie"}


def test_policy_admits_predicate() -> None:
    assert policy_admits(None, "X")
    pol = ToolVisibilityPolicy(visible={"A"}, blocked={"B"}, pinned={"P"})
    assert policy_admits(pol, "A")
    assert policy_admits(pol, "P")
    assert not policy_admits(pol, "B")
    assert not policy_admits(pol, "C")
    # blocked beats everything, even pinned/forced.
    pol2 = ToolVisibilityPolicy(blocked={"P"}, pinned={"P"}, forced_pinned=frozenset({"P"}))
    assert not policy_admits(pol2, "P")


def test_dispatcher_injects_policy_into_tool_context() -> None:
    """The dispatch path exposes the live policy under the metadata key."""
    import asyncio

    from protocore.contracts.types import ToolCall
    from protocore.runtime.tool_dispatch import DispatchOutcome, ToolDispatcher
    from protocore.runtime.tool_permission import ToolPermissionGate

    from ._tool_fixtures import make_default_ctx

    seen: dict[str, object] = {}

    async def _capture(args: dict[str, object]) -> None:
        del args

    tool = MockTool(tool_name="Probe", description="probe", on_invoke=_capture)

    real_invoke = tool.invoke

    async def _spy_invoke(context, arguments):  # type: ignore[no-untyped-def]
        seen["policy"] = (context.metadata or {}).get(
            TOOL_VISIBILITY_POLICY_METADATA_KEY
        )
        return await real_invoke(context, arguments)

    tool.invoke = _spy_invoke  # type: ignore[method-assign]
    reg = _registry_with(tool)
    dispatcher = ToolDispatcher(registry=reg, permission_gate=ToolPermissionGate())
    policy = ToolVisibilityPolicy(blocked={"SomethingElse"})

    async def _run() -> None:
        agen = dispatcher.dispatch(
            tool_call=ToolCall(id="tc-1", name="Probe", arguments={}),
            ctx=make_default_ctx(),
            visibility_policy=policy,
            timeout_seconds=5,
        )
        async for item in agen:
            if isinstance(item, DispatchOutcome):
                assert item.success

    asyncio.run(_run())
    assert seen["policy"] is policy


# ----------------------------------------------------------------------
# 3. RU-capable discovery (hints + normalized fallback)
# ----------------------------------------------------------------------


def test_ru_query_finds_memory_tools_via_hint() -> None:
    """The RU query "память" must surface the memory tools when admitted.

    Exercises the REAL core memory tools (their ``search_hint`` ClassVars
    carry the RU keywords) inside a registry that also holds EN-only tools.
    """
    from protocore.contracts.memory import IMemory
    from protocore.tools.memory import build_memory_tools

    class _NullStore(IMemory):  # type: ignore[misc]
        pass

    # Deliberately abstract: search never invokes the store, so the cheapest
    # stand-in is an instance that implements nothing.
    store: object = _NullStore.__new__(_NullStore)  # type: ignore[type-abstract]
    reg = ToolRegistry()
    for tool in build_memory_tools(store):  # type: ignore[arg-type]
        reg.register(tool)
    reg.register(MockTool(tool_name="Bash", description="Run a shell command"))

    hits = [t.name for t in reg.search("память", top_k=5)]
    assert REMEMBER_TOOL_NAME in hits
    assert RECALL_TOOL_NAME in hits
    assert "Bash" not in hits


def test_ru_query_blocked_memory_stays_hidden() -> None:
    """RU discovery still honours the policy (enabled-tenants only)."""
    from protocore.contracts.memory import IMemory
    from protocore.tools.memory import build_memory_tools

    class _NullStore(IMemory):  # type: ignore[misc]
        pass

    store: object = _NullStore.__new__(_NullStore)  # type: ignore[type-abstract]
    reg = ToolRegistry()
    for tool in build_memory_tools(store):  # type: ignore[arg-type]
        reg.register(tool)
    policy = ToolVisibilityPolicy(blocked=set(MEMORY_TOOL_NAMES))
    assert reg.search("память", top_k=5, policy=policy) == []


def test_inflected_ru_query_matches_via_fallback() -> None:
    """BM25 scores 0.0 for an inflected form; the prefix fallback catches it."""
    cand = build_candidate(
        "Remember",
        "Save a durable fact to long-term memory",
        hint="память запомнить сохранить",
    )
    other = build_candidate("Bash", "Run a shell command", hint="")
    hits = normalized_fallback_match("памяти", [cand, other], top_k=5)
    assert [c.name for c in hits] == ["Remember"]


def test_fallback_partial_tool_name_match() -> None:
    cand = build_candidate("List", "List workspace files", hint="")
    hits = normalized_fallback_match("workspace", [cand], top_k=5)
    assert [c.name for c in hits] == ["List"]


def test_fallback_no_overlap_returns_empty() -> None:
    cand = build_candidate("Bash", "Run a shell command", hint="")
    assert normalized_fallback_match("совершенно другое", [cand], top_k=5) == []


def test_fallback_deterministic_order() -> None:
    cands = [
        build_candidate("Bravo", "memory helper", hint=""),
        build_candidate("Alpha", "memory helper", hint=""),
    ]
    hits = normalized_fallback_match("memory", cands, top_k=5)
    assert [c.name for c in hits] == ["Alpha", "Bravo"]


def test_hint_never_reaches_description() -> None:
    cand = build_candidate("X", "desc", hint="секретный hint")
    assert "hint" not in cand.description
    assert "секретный" in cand.text


# ----------------------------------------------------------------------
# Surface/dispatch coherence for policy.pinned
# ----------------------------------------------------------------------


def test_pinned_readmitted_past_visible_whitelist() -> None:
    """A ToolSearch-pinned tool outside the ``visible`` whitelist stays on
    the surface — matching the dispatch gate's ``visible|pinned|forced``
    allowed set (otherwise the model could pin a tool, dispatch would allow
    it, yet its schema would vanish from the next turn)."""
    pinned_tool = MockTool(tool_name="ZZPinned", description="unrelated")
    others = _many_distinct_tools(3)
    reg = _registry_with(*others, pinned_tool)
    policy = ToolVisibilityPolicy(
        visible={t.tool_name for t in others},
        pinned={"ZZPinned"},
    )
    defs = reg.compute_effective_surface(
        "tenant-1", policy, query="common keyword", top_k=5
    )
    assert "ZZPinned" in [d.name for d in defs]


def test_pinned_blocked_stays_blocked() -> None:
    """``blocked`` beats ``pinned`` on the surface, mirroring dispatch."""
    pinned_tool = MockTool(tool_name="ZZPinned", description="unrelated")
    others = _many_distinct_tools(3)
    reg = _registry_with(*others, pinned_tool)
    policy = ToolVisibilityPolicy(
        visible={t.tool_name for t in others},
        pinned={"ZZPinned"},
        blocked={"ZZPinned"},
    )
    defs = reg.compute_effective_surface(
        "tenant-1", policy, query="common keyword", top_k=5
    )
    assert "ZZPinned" not in [d.name for d in defs]
