"""Force-pinned core tools survive the BM25 clip of a Russian query.

Reproduces the measured collapse: a **Russian**
prompt shares zero tokens with the English tool names/descriptions, so every
BM25 score is ``0.0`` and the per-turn surface collapses to ZERO core tools.
The fix (``ToolVisibilityPolicy.forced_pinned`` → merged into ``pinned`` inside
``ToolRegistry.compute_effective_surface``) keeps Agent plus the six core file
tools present for every prompt, RU or EN.

The fixture rebuilds the **real** 14-tool the host surface (names +
descriptions lifted verbatim from ``fixtures/tool_surface.json``) so the clip
reproduced here is the production one, not a toy.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.runtime.tool_registry import ToolRegistry

from ._tool_fixtures import MockTool

# Agent plus the six core file tools that MUST always be present (§A.1 default).
_CORE_TOOLS: frozenset[str] = frozenset(
    {"Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep"}
)

# Top-K used for the per-turn surface (matches RuntimeConstants.tool_retrieval_top_k).
_TOP_K = 12

# Real 14-tool surface (name → description) — verbatim from the stand's
# ``fixtures/tool_surface.json`` (truncated to the first descriptive sentence;
# the BM25 RU-vs-EN zero-score behaviour does not depend on the tail).
_REAL_SURFACE: dict[str, str] = {
    "Agent": (
        "Dispatch a registered subagent synchronously with an isolated "
        "tool/skill scope."
    ),
    "AppendFile": (
        "Append content to the end of an existing file. Use this for any file "
        "likely to exceed about 2 KB."
    ),
    "AskUser": (
        "Pause the agent loop and ask the user one or more questions, then "
        "return the user's answers as the tool result."
    ),
    "Bash": (
        "Execute a shell command in the sandbox. First call in a session "
        "spawns the sandbox pod."
    ),
    "Edit": (
        "Edit a file by replacing old_string with new_string (exact match, "
        "no regex)."
    ),
    "FinalizeFile": (
        "Mark a multi-chunk AppendFile sequence complete, optionally "
        "verifying the byte count."
    ),
    "Glob": (
        "Pattern match over workspace paths. Supports ``**`` for recursive "
        "descent."
    ),
    "Grep": (
        "Search workspace files for a regex pattern. Optional glob filter "
        "narrows the scan."
    ),
    "Read": (
        "Read a file from the sandbox workspace with line-numbered output. "
        "Requires the file to exist."
    ),
    "Skill": (
        "Invoke a registered skill by name (plain or plugin:name). Loads the "
        "skill body into context."
    ),
    "TodoWrite": (
        "Update the run's todo list. Replaces the entire list each call."
    ),
    "ToolSearch": (
        "Search for tools by keyword (BM25) or fetch by exact name (prefix "
        "the query with select:)."
    ),
    "WebFetch": (
        "Fetch a URL via HTTPS/HTTP. SSRF-guarded (private IPs blocked) and "
        "DNS-pinned."
    ),
    "Write": (
        "Write content to a file in the sandbox workspace (atomic rename). "
        "Creates parent directories."
    ),
}


@pytest.fixture
def chat_registry() -> ToolRegistry:
    """Registry holding the real 14-tool the host surface (RU-clip prone)."""
    return ToolRegistry(
        MockTool(tool_name=name, description=desc)
        for name, desc in _REAL_SURFACE.items()
    )


def test_russian_prompt_keeps_core_tools(chat_registry: ToolRegistry) -> None:
    """RU prompt + forced pins → all six core file tools survive the clip."""
    policy = ToolVisibilityPolicy(forced_pinned=_CORE_TOOLS)
    surface = {
        d.name
        for d in chat_registry.compute_effective_surface(
            "t",
            policy,
            query="напиши лендинг для кофейни",
            top_k=_TOP_K,
        )
    }
    # All six core tools must survive the Russian-query clip.
    assert _CORE_TOOLS <= surface


def test_empty_policy_russian_prompt_drops_core_tools(
    chat_registry: ToolRegistry,
) -> None:
    """Regression guard — proves the fix is load-bearing.

    Without ``forced_pinned`` the RU prompt collapses the surface so that
    NOT ALL six core tools survive (a zero BM25 score is clipped away).
    If this ever stopped holding, ``test_russian_prompt_keeps_core_tools``
    would pass trivially and the pin would not actually be doing any work.
    """
    policy = ToolVisibilityPolicy()  # no forced_pinned, no pinned
    surface = {
        d.name
        for d in chat_registry.compute_effective_surface(
            "t",
            policy,
            query="напиши лендинг для кофейни",
            top_k=_TOP_K,
        )
    }
    # The unpinned RU clip must NOT preserve the full core set.
    assert not (_CORE_TOOLS <= surface)


def test_forced_pinned_unions_with_pinned(chat_registry: ToolRegistry) -> None:
    """``forced_pinned`` augments (not replaces) the existing ``pinned`` set."""
    policy = ToolVisibilityPolicy(
        pinned={"WebFetch"},
        forced_pinned=_CORE_TOOLS,
    )
    surface = {
        d.name
        for d in chat_registry.compute_effective_surface(
            "t",
            policy,
            query="напиши лендинг для кофейни",
            top_k=_TOP_K,
        )
    }
    assert _CORE_TOOLS <= surface
    assert "WebFetch" in surface


def test_forced_pins_survive_a_visible_whitelist_that_omits_them(
    chat_registry: ToolRegistry,
) -> None:
    """A tenant ``visible`` whitelist must NOT drop the core floor.

    ``list_for_tenant`` applies ``visible`` first, so a leader whitelist of
    ``{WebFetch}`` would otherwise filter the six core file tools out BEFORE the
    forced-pin union could run — recreating the cause-#3 collapse for that
    tenant. The floor must be re-admitted past the whitelist. Covers both the
    BM25 path (top_k below the floored count) and the early-return path
    (top_k above it).
    """
    policy = ToolVisibilityPolicy(
        visible={"WebFetch"},
        forced_pinned=_CORE_TOOLS,
    )
    for top_k in (_TOP_K, 3):  # early-return path AND clipped path
        surface = {
            d.name
            for d in chat_registry.compute_effective_surface(
                "t",
                policy,
                query="напиши лендинг для кофейни",
                top_k=top_k,
            )
        }
        assert _CORE_TOOLS <= surface, f"core floor lost at top_k={top_k}"


def test_blocked_still_wins_over_forced_pins(chat_registry: ToolRegistry) -> None:
    """``blocked`` is an explicit operator deny and overrides the floor.

    Re-admitting ``forced_pinned`` past the ``visible`` whitelist must NOT
    resurrect a tool the operator hard-denied: ``blocked`` precedence is
    preserved. The other five core tools still survive.
    """
    policy = ToolVisibilityPolicy(
        blocked={"Bash"},
        forced_pinned=_CORE_TOOLS,
    )
    surface = {
        d.name
        for d in chat_registry.compute_effective_surface(
            "t",
            policy,
            query="напиши лендинг для кофейни",
            top_k=_TOP_K,
        )
    }
    assert "Bash" not in surface
    assert (_CORE_TOOLS - {"Bash"}) <= surface


def test_runtime_constants_default_pins_are_agent_plus_core_tools() -> None:
    """§A.1: the RC default exposes Agent plus the six core file tools."""
    rc = RuntimeConstants()
    assert set(rc.tool_surface_forced_pins) == _CORE_TOOLS


def test_runtime_constants_structured_output_default_on() -> None:
    """§A.1: ``structured_output_use_response_format`` exists, defaults True.

    The host vLLM client reads this RC by name (and a test constructs
    ``RuntimeConstants(structured_output_use_response_format=False)``); the
    frozen ``extra='forbid'`` model would reject that kwarg if the field were
    absent, so this guards the cross-unit §A.1 contract.
    """
    assert RuntimeConstants().structured_output_use_response_format is True
    assert (
        RuntimeConstants(
            structured_output_use_response_format=False
        ).structured_output_use_response_format
        is False
    )


def test_runtime_constants_pins_drive_policy(chat_registry: ToolRegistry) -> None:
    """The RC default, fed into the policy, restores core tools on a RU prompt.

    Proves the seam is RC-driven end to end: no inline magic list — the policy
    is built from ``rc.tool_surface_forced_pins`` and the core surface honours it.
    """
    rc = RuntimeConstants()
    policy = ToolVisibilityPolicy(forced_pinned=frozenset(rc.tool_surface_forced_pins))
    surface = {
        d.name
        for d in chat_registry.compute_effective_surface(
            "t",
            policy,
            query="сделай README для проекта",
            top_k=_TOP_K,
        )
    }
    assert _CORE_TOOLS <= surface
