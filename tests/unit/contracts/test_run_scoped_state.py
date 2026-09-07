"""The run's state is one object, shared by reference, and it says what persists."""
from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.run_state import (
    ConsecutiveErrorStreak,
    RunScopedState,
    SignatureStreak,
    ToolStreak,
)
from protocore.contracts.tools import ToolContext
from protocore.runtime.run_work_budget import RunWorkLedger
from protocore.runtime.subagent_budget import SubagentTreeBudget


class _Constants:
    max_subagent_runs_per_tree = 7
    max_total_tokens_per_tree = 900
    max_concurrent_subagents_per_tree = 3


def _context(state: RunScopedState | None) -> ToolContext:
    return ToolContext(
        tenant_id="scope", run_id="run", session_id="session", run_state=state
    )


def test_context_carries_the_same_object_not_a_copy() -> None:
    state = RunScopedState()
    ctx = _context(state)

    ctx.run_state.host["wired"] = True  # type: ignore[union-attr]

    assert ctx.run_state is state
    assert state.host == {"wired": True}


def test_a_context_built_without_state_carries_none() -> None:
    assert _context(None).run_state is None


def test_ledger_is_minted_once_and_then_returned() -> None:
    state = RunScopedState()

    first = state.ensure_run_work_ledger(_Constants())
    second = state.ensure_run_work_ledger(_Constants())

    assert first is second is state.run_work_ledger


def test_tree_budget_is_minted_only_at_the_first_fan_out() -> None:
    state = RunScopedState()
    assert state.subagent_tree_budget is None

    budget = state.ensure_subagent_tree_budget(4)

    assert budget.capacity == 4
    assert state.ensure_subagent_tree_budget(9) is budget


def test_clearing_streaks_leaves_the_tree_allowances_alone() -> None:
    state = RunScopedState(
        consecutive_error=ConsecutiveErrorStreak(tool_name="Write", count=3),
        transport_down=SignatureStreak(signature="Bash:TRANSPORT_DOWN", count=2),
        transport_down_injection_pending=True,
        string_type=ToolStreak(tool_name="Write", count=2),
    )
    lock = asyncio.Lock()
    state.tool_call_soft_cap_state.lock = lock
    state.tool_call_soft_cap_state.counts["Write"] = 5
    ledger = state.ensure_run_work_ledger(_Constants())
    budget = state.ensure_subagent_tree_budget(2)

    state.clear_run_scoped_streaks()

    assert state.consecutive_error is None
    assert state.transport_down is None
    assert state.transport_down_injection_pending is False
    assert state.string_type is None
    assert state.tool_call_soft_cap_state.counts == {}
    assert state.tool_call_soft_cap_state.lock is lock
    assert state.run_work_ledger is ledger
    assert state.subagent_tree_budget is budget


def test_transcript_state_is_a_copy_that_survives_the_batch() -> None:
    state = RunScopedState(
        consecutive_error=ConsecutiveErrorStreak(tool_name="Bash", count=1),
        satisfied_preconditions={"AppendFile:report"},
    )
    state.tool_call_soft_cap_state.counts["Bash"] = 1

    saved = state.transcript_state()

    state.consecutive_error = ConsecutiveErrorStreak(tool_name="Bash", count=9)
    state.satisfied_preconditions.add("AppendFile:other")
    state.tool_call_soft_cap_state.counts["Bash"] = 9
    state.transport_down_injection_pending = True

    state.restore_transcript_state(saved)

    assert state.consecutive_error == ConsecutiveErrorStreak(tool_name="Bash", count=1)
    assert state.satisfied_preconditions == {"AppendFile:report"}
    assert state.tool_call_soft_cap_state.counts == {"Bash": 1}
    assert state.transport_down_injection_pending is False


def test_snapshot_states_the_two_allowances_that_outlive_the_process() -> None:
    state = RunScopedState()
    ledger = state.ensure_run_work_ledger(_Constants())
    ledger.charge_tokens(input_tokens=10, output_tokens=5)
    state.ensure_subagent_tree_budget(3)

    payload = state.to_snapshot()

    assert payload["run_work_ledger"]["tokens_charged"] == 15
    assert payload["subagent_tree_budget"] == {"capacity": 3, "in_use": 0}


def test_a_resumed_run_gets_back_what_it_already_spent() -> None:
    spent = RunScopedState()
    ledger = spent.ensure_run_work_ledger(_Constants())
    ledger.charge_tokens(input_tokens=100, output_tokens=100)
    spent.ensure_subagent_tree_budget(2)

    resumed = RunScopedState()
    resumed.apply_snapshot(spent.to_snapshot())

    assert resumed.run_work_ledger is not None
    assert resumed.run_work_ledger.to_snapshot()["tokens_charged"] == 200
    assert resumed.subagent_tree_budget is not None
    assert resumed.subagent_tree_budget.capacity == 2
    # Occupancy is not restored: the slots were held by coroutines in a process
    # that is gone, and nothing here could ever release them.
    assert resumed.subagent_tree_budget.in_use == 0


def test_a_tree_that_never_fanned_out_is_not_handed_a_budget_on_resume() -> None:
    resumed = RunScopedState()
    resumed.apply_snapshot(RunScopedState().to_snapshot())

    assert resumed.subagent_tree_budget is None


def test_a_live_budget_is_not_replaced_by_the_persisted_pair() -> None:
    state = RunScopedState()
    live = state.ensure_subagent_tree_budget(5)

    state.apply_snapshot({"subagent_tree_budget": {"capacity": 2, "in_use": 1}})

    assert state.subagent_tree_budget is live
    assert state.subagent_tree_budget.capacity == 5


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not a mapping",
        {},
        {"run_work_ledger": None, "subagent_tree_budget": None},
        {"subagent_tree_budget": {"capacity": True, "in_use": 0}},
        {"subagent_tree_budget": {"capacity": -1, "in_use": 0}},
    ],
)
def test_a_payload_with_nothing_usable_leaves_the_run_as_it_was(payload: object) -> None:
    state = RunScopedState()

    state.apply_snapshot(payload)

    assert state.run_work_ledger is None
    assert state.subagent_tree_budget is None


def test_restored_objects_are_the_real_types() -> None:
    state = RunScopedState()
    state.apply_snapshot(
        {
            "run_work_ledger": {"max_child_runs": 2, "max_tokens": 3},
            "subagent_tree_budget": {"capacity": 1, "in_use": 0},
        }
    )

    assert isinstance(state.run_work_ledger, RunWorkLedger)
    assert isinstance(state.subagent_tree_budget, SubagentTreeBudget)
