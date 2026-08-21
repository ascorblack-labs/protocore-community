"""Unit tests for ``mcp_sets_block`` — MCP tool sets are an ADDITIVE block.

The leader system prompt advertises a
tenant's MCP tool *sets* as DISCOVERABLE capabilities (PROP). The
contract mirrors ``agent_descriptions_block`` / ``persona_md``:

* ``mcp_sets`` empty/``None`` → the block renders NOTHING and the prompt is
 byte-identical to the no-MCP path every default tenant takes (the additive
 invariant — no surprise prompt drift for tenants without MCP);
* a populated ``mcp_sets`` → a ``## Available tool sets (MCP)`` header plus one
 ``- <Name>: <description>. Tools: t1, t2, …`` line per set, framed as
 capabilities reachable via ``ToolSearch`` (NOT a guaranteed-callable list);
* the macro is DEFENSIVE — a set dict missing ``description``/``tool_names``
 still renders without raising, and a set with many tools renders fine (the
 per-set name-count cap is the host side's job).

The macro signature takes the value; the leader template passes
``mcp_sets | default(none)`` so the pure-core templates stay renderable when a
caller omits the var (a host always supplies it, defaulting to ``None``).
"""

from __future__ import annotations

import pytest

from protocore.prompts import JinjaPromptTemplateProvider


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    """Default provider pointing at the bundled templates."""
    return JinjaPromptTemplateProvider()


def _base_ctx() -> dict[str, object]:
    """Minimal valid leader_system context — every optional var None."""
    return {
        "current_date": "2026-06-14",
        "persona_md": None,
        "agent_descriptions": None,
        "environment_capabilities": None,
        "capabilities": None,
        "finalization_contract_block": None,
    }


MCP_SETS: list[dict[str, object]] = [
    {
        "name": "GitHub",
        "description": "Read/write GitHub issues and pull requests",
        "tool_names": ["gh_pr_list", "gh_issue_create", "gh_repo_search"],
    },
    {
        "name": "Jira",
        "description": "Search and transition Jira tickets",
        "tool_names": ["jira_search", "jira_transition"],
    },
]


def test_mcp_sets_none_byte_identical_to_no_mcp_path(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The additive invariant: ``mcp_sets=None`` renders byte-for-byte the same
    prompt as a context that omits ``mcp_sets`` entirely (the no-MCP production
    path). No ``## Available tool sets`` header leaks onto a no-MCP tenant.
    """
    rendered_none = provider.render("leader_system", {**_base_ctx(), "mcp_sets": None})
    rendered_absent = provider.render("leader_system", _base_ctx())
    assert rendered_none == rendered_absent
    assert "## Available tool sets (MCP)" not in rendered_none
    # Empty list behaves exactly like None (falsy) — also byte-identical.
    rendered_empty = provider.render("leader_system", {**_base_ctx(), "mcp_sets": []})
    assert rendered_empty == rendered_none
    # Sanity: the always-on scaffolding is still present (we did not break it).
    assert "You are a Protocore agent." in rendered_none


def test_mcp_sets_none_additive_across_persona_and_agents(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """``mcp_sets=None`` is byte-identical to the absent-var render even when
    OTHER additive blocks (persona + subagents) are active — the block adds
    nothing on the no-MCP path regardless of surrounding context.
    """
    ctx = {
        **_base_ctx(),
        "persona_md": "# Aurora\nYou are warm.",
        "agent_descriptions": {"coder": "Writes code", "reviewer": "Reviews code"},
    }
    rendered_none = provider.render("leader_system", {**ctx, "mcp_sets": None})
    rendered_absent = provider.render("leader_system", ctx)
    assert rendered_none == rendered_absent
    # Subagent block still renders (proves placement did not displace it).
    assert "coder: Writes code" in rendered_none
    assert "## Available tool sets (MCP)" not in rendered_none


def test_mcp_sets_populated_renders_header_and_sets(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """A populated ``mcp_sets`` renders the header + one line per set with the
    exact ``- <Name>: <description>. Tools: t1, t2, …`` shape.
    """
    rendered = provider.render("leader_system", {**_base_ctx(), "mcp_sets": MCP_SETS})
    assert "## Available tool sets (MCP)" in rendered
    assert (
        "- GitHub: Read/write GitHub issues and pull requests. "
        "Tools: gh_pr_list, gh_issue_create, gh_repo_search" in rendered
    )
    assert (
        "- Jira: Search and transition Jira tickets. "
        "Tools: jira_search, jira_transition" in rendered
    )
    # Capabilities framing — reachable via ToolSearch, NOT guaranteed-callable.
    assert "ToolSearch" in rendered
    assert "NOT all in your visible set" in rendered


def test_mcp_sets_block_appears_after_subagent_block(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The MCP block renders AFTER the subagent (``agent_descriptions``) block,
    per the placement requirement.
    """
    ctx = {
        **_base_ctx(),
        "agent_descriptions": {"coder": "Writes code"},
        "mcp_sets": MCP_SETS,
    }
    rendered = provider.render("leader_system", ctx)
    subagent_idx = rendered.index("coder: Writes code")
    mcp_idx = rendered.index("## Available tool sets (MCP)")
    assert subagent_idx < mcp_idx


def test_mcp_sets_many_tools_does_not_break(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """A set with many tools renders fine — the per-set name-count cap is
    the host side's job; the macro must not break on a long list.
    """
    big = [
        {
            "name": "Bulk",
            "description": "many tools",
            "tool_names": [f"tool_{i}" for i in range(64)],
        }
    ]
    rendered = provider.render("leader_system", {**_base_ctx(), "mcp_sets": big})
    assert "- Bulk: many tools. Tools: tool_0, tool_1," in rendered
    assert "tool_63" in rendered


def test_mcp_sets_block_is_defensive_on_missing_keys(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Defensive: sets missing ``description`` / ``tool_names`` / ``name`` keys
    render without raising (the host side normalizes, but the macro must
    not crash on a partial dict).
    """
    partial: list[dict[str, object]] = [
        {"name": "OnlyName"},  # no description, no tool_names
        {"description": "no name set", "tool_names": ["x", "y"]},  # no name
        {"name": "Empty", "description": "", "tool_names": []},  # empty values
    ]
    rendered = provider.render("leader_system", {**_base_ctx(), "mcp_sets": partial})
    assert "## Available tool sets (MCP)" in rendered
    # Missing description falls back to a placeholder; missing tool_names → "".
    assert "- OnlyName: (no description). Tools: " in rendered
    assert "Tools: x, y" in rendered
    assert "- Empty: (no description). Tools: " in rendered


def test_mcp_sets_populated_keeps_always_on_scaffolding(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The MCP block is additive — the load-bearing always-on safeguards (the
    tool-use scaffolding + the anti-leak ordering rule) still render when
    ``mcp_sets`` is populated.

    The ordering rule is English only. The Russian mirror of the operating
    rules was removed from the scaffold on purpose: a full EN+RU copy doubled
    its length and biased the model toward answering in Russian, and reply
    language follows the user's latest message instead. Asserting its ABSENCE
    keeps a populated MCP block from being the thing that quietly reintroduces
    it — the same treatment the sibling scaffold tests were given.
    """
    rendered = provider.render("leader_system", {**_base_ctx(), "mcp_sets": MCP_SETS})
    assert "You are a Protocore agent." in rendered
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    assert "ТОЛЬКО ПОСЛЕ всех Write/Edit/Bash" not in rendered
