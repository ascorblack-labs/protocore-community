"""Tests for :func:`protocore.runtime.tool_pool.assemble_tool_pool`."""
from __future__ import annotations

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.runtime.tool_pool import (
    assemble_tool_pool,
    assemble_tool_pool_from_concrete,
)
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool

# ----------------------------------------------------------------------
# Basic shape + sort invariant
# ----------------------------------------------------------------------


def test_assemble_returns_sorted_by_name() -> None:
    reg = ToolRegistry()
    for name in ["Charlie", "Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=10,
    )
    assert [t.name for t in pool] == ["Alpha", "Bravo", "Charlie"]


def test_max_tools_zero_returns_empty() -> None:
    reg = ToolRegistry([MockTool(tool_name="Alpha")])
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=0,
    )
    assert pool == []


def test_max_tools_negative_returns_empty() -> None:
    reg = ToolRegistry([MockTool(tool_name="Alpha")])
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=-3,
    )
    assert pool == []


# ----------------------------------------------------------------------
# Policy filters
# ----------------------------------------------------------------------


def test_visible_set_filters_pool() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(visible={"Alpha", "Gamma"}),
        max_tools_in_context=10,
    )
    assert [t.name for t in pool] == ["Alpha", "Gamma"]


def test_blocked_set_subtracts() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(blocked={"Bravo"}),
        max_tools_in_context=10,
    )
    assert [t.name for t in pool] == ["Alpha", "Gamma"]


def test_subagent_whitelist_narrows() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma", "Delta"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=10,
        whitelist=["Gamma", "Alpha"],
    )
    assert [t.name for t in pool] == ["Alpha", "Gamma"]


def test_empty_subagent_whitelist_is_inclusive() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=10,
        whitelist=[],
    )
    assert [t.name for t in pool] == ["Alpha", "Bravo"]


# ----------------------------------------------------------------------
# Retrieval clipping
# ----------------------------------------------------------------------


def test_clipping_returns_top_k() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read a file"))
    reg.register(MockTool(tool_name="WriteFile", description="Write a file"))
    reg.register(MockTool(tool_name="GitDiff", description="Show git diff"))
    reg.register(MockTool(tool_name="Bash", description="Run shell"))
    reg.register(MockTool(tool_name="Cat", description="Echo cat picture"))

    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        context_query="read a text file",
        max_tools_in_context=2,
    )
    assert len(pool) <= 2
    names = [t.name for t in pool]
    # Most-relevant tool included.
    assert "ReadFile" in names
    # Sort invariant.
    assert names == sorted(names)


def test_clipping_pinned_always_included() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read a file"))
    reg.register(MockTool(tool_name="WriteFile", description="Write a file"))
    reg.register(MockTool(tool_name="Bash", description="Run shell"))
    reg.register(MockTool(tool_name="Grep", description="Search files"))
    reg.register(MockTool(tool_name="Cat", description="Picture of a cat"))

    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(pinned={"Bash"}),
        context_query="read text file",  # not bash-y
        max_tools_in_context=2,
    )
    names = [t.name for t in pool]
    assert "Bash" in names


def test_no_clipping_when_under_max() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        context_query="anything",
        max_tools_in_context=10,
    )
    # All tools returned regardless of query — no retrieval triggered.
    assert [t.name for t in pool] == ["Alpha", "Bravo"]


def test_empty_corpus() -> None:
    reg = ToolRegistry()
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=10,
    )
    assert pool == []


def test_typed_wrapper_forwards_correctly() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    pool = assemble_tool_pool_from_concrete(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        max_tools_in_context=10,
    )
    assert [t.name for t in pool] == ["Alpha", "Bravo"]


# ----------------------------------------------------------------------
# Sort stability across retrieval
# ----------------------------------------------------------------------


def test_clipped_pool_name_sort_invariant() -> None:
    """Output ordering MUST be by name ASC for prefix-cache stability."""
    reg = ToolRegistry()
    # 10 tools; clip to 5 — verify the 5 are name-sorted.
    for i in range(10):
        reg.register(
            MockTool(
                tool_name=f"Tool{i:02d}",
                description=f"description of tool number {i} which fetches data",
            )
        )
    pool = assemble_tool_pool(
        registry=reg,
        tenant_id="tnt",
        visibility_policy=ToolVisibilityPolicy(),
        context_query="fetches data",
        max_tools_in_context=5,
    )
    names = [t.name for t in pool]
    assert names == sorted(names)
