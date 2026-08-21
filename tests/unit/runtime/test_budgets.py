"""Tests for :mod:`protocore.runtime.context.budgets`."""
from __future__ import annotations

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.context.budgets import TokenBudgets, derive_budgets


def test_derive_budgets_pure() -> None:
    rc = RuntimeConstants(model_context_window=49_152)
    budgets1 = derive_budgets(rc)
    budgets2 = derive_budgets(rc)
    assert budgets1 == budgets2  # purity: same input → same output


def test_derive_budgets_scales_with_window() -> None:
    small = derive_budgets(RuntimeConstants(model_context_window=8_192))
    large = derive_budgets(RuntimeConstants(model_context_window=200_000))
    assert large.max_context > small.max_context
    assert large.history_budget_tokens > small.history_budget_tokens
    assert large.compaction_trigger_tokens > small.compaction_trigger_tokens


def test_fixed_overhead_leaves_history_room() -> None:
    rc = RuntimeConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.history_budget_tokens > 0
    assert budgets.history_budget_tokens + budgets.fixed_overhead_tokens == budgets.max_context


def test_compaction_trigger_below_max() -> None:
    rc = RuntimeConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.compaction_trigger_tokens < budgets.max_context


def test_tool_result_threshold_smaller_than_trigger() -> None:
    rc = RuntimeConstants(model_context_window=49_152)
    budgets = derive_budgets(rc)
    assert budgets.tool_result_truncation_threshold < budgets.compaction_trigger_tokens


def test_token_budgets_is_frozen_dataclass() -> None:
    rc = RuntimeConstants()
    budgets = derive_budgets(rc)
    assert isinstance(budgets, TokenBudgets)
    # FrozenInstanceError raised on mutation
    import dataclasses

    import pytest as _pytest

    with _pytest.raises(dataclasses.FrozenInstanceError):
        budgets.max_context = 0  # type: ignore[misc]
