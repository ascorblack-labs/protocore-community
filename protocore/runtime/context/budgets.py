"""Formula-derived token budgets per

Pure function. Same :class:`RuntimeConstants` snapshot yields the same
:class:`TokenBudgets` — cross-pod deterministic. No module-level cache.

All budgets are **derived** from a canonical input
(:attr:`RuntimeConstants.model_context_window`) and per-section ratios.
The dashboard surfaces ratios as canonical inputs and the derived values
as read-only computed fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.runtime_constants import RuntimeConstants


@dataclass(frozen=True, slots=True)
class TokenBudgets:
    """Per-layer derived token budgets for one turn.

 """

    max_context: int
    """Hard upper bound (provider's context window)."""

    compaction_trigger_tokens: int
    """When current_tokens > this, compaction runs before LLM call."""

    compaction_emergency_tokens: int
    """Emergency cliff: when current_tokens > this, a proactive
    ``force_compaction`` runs before the LLM call (both tiers, unconditional)
    instead of waiting for the provider to raise a context-window error.
    Derived from ``model_context_window * compaction_emergency_ratio`` —
    strictly above ``compaction_trigger_tokens`` (the RC validator enforces
    ``compaction_trigger_ratio < compaction_emergency_ratio``)."""

    tool_result_truncation_threshold: int
    """Tool results larger than this are blobbed (Tier 1)."""

    system_prompt_max_tokens: int
    skill_index_budget_tokens: int
    loaded_skills_budget_tokens: int
    tool_definitions_budget_tokens: int
    user_context_budget_tokens: int

    history_budget_tokens: int
    """Remainder available for conversation history after fixed overhead."""

    @property
    def fixed_overhead_tokens(self) -> int:
        """Sum of all fixed-overhead layers (system + skills + tools + user_ctx)."""
        return (
            self.system_prompt_max_tokens
            + self.skill_index_budget_tokens
            + self.loaded_skills_budget_tokens
            + self.tool_definitions_budget_tokens
            + self.user_context_budget_tokens
        )


def derive_budgets(rc: RuntimeConstants) -> TokenBudgets:
    """Compute :class:`TokenBudgets` from an RC snapshot.

    Pure function. The dashboard's RC editor surfaces ratios; this function
    is the single source of truth for derived values across every pod.
    """
    max_context = rc.model_context_window

    compaction_trigger = int(max_context * rc.compaction_trigger_ratio)
    compaction_emergency = int(max_context * rc.compaction_emergency_ratio)
    tool_result_threshold = int(max_context * rc.tool_result_truncation_ratio)
    system_prompt_max = int(max_context * rc.system_prompt_max_ratio)
    skill_index = int(max_context * rc.skill_index_budget_ratio)
    loaded_skills = int(max_context * rc.loaded_skills_ratio)
    tool_definitions = int(max_context * rc.tool_definitions_ratio)
    user_context = int(max_context * rc.user_context_ratio)

    fixed_overhead = (
        system_prompt_max
        + skill_index
        + loaded_skills
        + tool_definitions
        + user_context
    )
    history_budget = max_context - fixed_overhead

    return TokenBudgets(
        max_context=max_context,
        compaction_trigger_tokens=compaction_trigger,
        compaction_emergency_tokens=compaction_emergency,
        tool_result_truncation_threshold=tool_result_threshold,
        system_prompt_max_tokens=system_prompt_max,
        skill_index_budget_tokens=skill_index,
        loaded_skills_budget_tokens=loaded_skills,
        tool_definitions_budget_tokens=tool_definitions,
        user_context_budget_tokens=user_context,
        history_budget_tokens=history_budget,
    )


__all__ = ["TokenBudgets", "derive_budgets"]
