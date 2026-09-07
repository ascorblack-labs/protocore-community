"""The tree's cumulative work budgets survive a cold re-drive.

``RunWorkLedger`` bounds the TOTAL work one run tree may do — how many child
runs it may start and how many tokens it may spend across every descendant.
It lives on the run's state and is inherited by reference, so before it became
durable a run that was picked up on another process started counting from zero:
"die, get re-driven, spend the whole cap again" had no upper bound.

``SubagentTreeBudget`` is a different animal and is treated differently here. It
is a live semaphore shared by reference, so there is no object to carry across a
process boundary; what is durable is the pair (capacity, slots taken), and a
resumed run rebuilds a semaphore from it.
"""
from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.snapshot import RUN_SCOPED_STATE_SNAPSHOT_KEY
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.run_work_budget import RunWorkLedger
from protocore.runtime.subagent_budget import SubagentTreeBudget


def test_ledger_is_in_the_snapshot(engine_factory) -> None:
    engine = engine_factory(
        rc=LoopConstants(
            model_context_window=4_096,
            max_subagent_runs_per_tree=5,
            max_total_tokens_per_tree=1_000,
        )
    )
    snapshot = engine.snapshot()

    assert snapshot[RUN_SCOPED_STATE_SNAPSHOT_KEY]["run_work_ledger"] == {
        "max_child_runs": 5,
        "max_tokens": 1_000,
        "child_runs_started": 0,
        "tokens_charged": 0,
        "charged_call_ids": [],
    }


def test_ledger_is_minted_even_when_nobody_composed_one(engine_factory) -> None:
    """A run whose state was never composed still snapshots a usable ledger."""
    snapshot = engine_factory().snapshot()

    assert isinstance(
        snapshot[RUN_SCOPED_STATE_SNAPSHOT_KEY]["run_work_ledger"], dict
    )


async def test_cold_resume_keeps_the_counters(engine_factory) -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        max_subagent_runs_per_tree=4,
        max_total_tokens_per_tree=10_000,
    )
    source = engine_factory(rc=rc)
    ledger = RunWorkLedger(max_child_runs=4, max_tokens=10_000)
    ledger.reserve_child_runs(3)
    ledger.charge_tokens(input_tokens=900, output_tokens=100)
    source.run_state.run_work_ledger = ledger
    snapshot = source.snapshot()

    resumed = engine_factory(rc=rc)
    await resumed.resume_from_snapshot(snapshot)

    restored = resumed.run_state.run_work_ledger
    assert isinstance(restored, RunWorkLedger)
    assert restored.child_runs_started == 3
    assert restored.tokens_charged == 1_000
    assert restored.max_child_runs == 4
    assert restored.max_tokens == 10_000


async def test_token_cap_still_refuses_after_a_restart(engine_factory) -> None:
    """``max_total_tokens_per_tree`` holds across a cold re-drive."""
    rc = LoopConstants(
        model_context_window=4_096,
        max_subagent_runs_per_tree=0,
        max_total_tokens_per_tree=500,
    )
    source = engine_factory(rc=rc)
    ledger = RunWorkLedger(max_child_runs=0, max_tokens=500)
    ledger.charge_tokens(input_tokens=500, output_tokens=0)
    source.run_state.run_work_ledger = ledger

    resumed = engine_factory(rc=rc)
    await resumed.resume_from_snapshot(source.snapshot())

    restored = resumed.run_state.run_work_ledger
    assert restored.delegation_refusal_reason() == "run_tree_token_budget_exhausted"


async def test_a_snapshot_without_a_ledger_is_refused(engine_factory) -> None:
    source = engine_factory()
    snapshot = source.snapshot()
    del snapshot[RUN_SCOPED_STATE_SNAPSHOT_KEY]

    resumed = engine_factory()
    resumed.history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="untouched")])
    ]
    with pytest.raises(ValueError, match=RUN_SCOPED_STATE_SNAPSHOT_KEY):
        await resumed.resume_from_snapshot(snapshot)

    # Refused BEFORE any mutation: the receiving engine is left as it was.
    assert resumed.history[0].content_blocks[0].text == "untouched"


def test_tree_budget_snapshots_capacity_and_occupancy(engine_factory) -> None:
    budget = SubagentTreeBudget(3)
    budget._in_use = 2
    engine = engine_factory()
    engine.run_state.subagent_tree_budget = budget

    assert engine.snapshot()[RUN_SCOPED_STATE_SNAPSHOT_KEY][
        "subagent_tree_budget"
    ] == {"capacity": 3, "in_use": 2}


def test_tree_budget_absent_when_the_tree_never_fanned_out(engine_factory) -> None:
    engine = engine_factory()

    assert (
        engine.snapshot()[RUN_SCOPED_STATE_SNAPSHOT_KEY]["subagent_tree_budget"]
        is None
    )


async def test_tree_budget_comes_back_with_its_capacity_and_no_holders(
    engine_factory,
) -> None:
    """Capacity is restored, occupancy is not — a dead process holds no slots.

    The permits the snapshot records were held by coroutines that no longer
    exist. Nothing in the new process can release them, so restoring them as
    taken withdraws that much capacity for the rest of the run's life.
    """
    source = engine_factory()
    budget = SubagentTreeBudget(3)
    await budget.acquire()
    await budget.acquire()
    source.run_state.subagent_tree_budget = budget

    resumed = engine_factory()
    await resumed.resume_from_snapshot(source.snapshot())

    rebuilt = resumed.run_state.subagent_tree_budget
    assert isinstance(rebuilt, SubagentTreeBudget)
    assert rebuilt.capacity == 3
    assert rebuilt.in_use == 0
    permit = await rebuilt.acquire()
    assert rebuilt.in_use == 1
    await permit.release()


async def test_a_run_resumed_at_full_occupancy_can_still_delegate(
    engine_factory,
) -> None:
    """The shape that used to hang: every slot recorded taken, none releasable."""
    source = engine_factory()
    budget = SubagentTreeBudget(2)
    await budget.acquire()
    await budget.acquire()
    source.run_state.subagent_tree_budget = budget

    resumed = engine_factory()
    await resumed.resume_from_snapshot(source.snapshot())

    rebuilt = resumed.run_state.subagent_tree_budget
    assert isinstance(rebuilt, SubagentTreeBudget)
    permit = await asyncio.wait_for(rebuilt.acquire(), 1.0)
    assert permit is not None


async def test_a_live_budget_with_holders_survives_a_self_resume(
    engine_factory,
) -> None:
    """A run resumed from its own snapshot keeps the object its children hold.

    Its descendants are still in flight on THIS budget; replacing it would
    strand their permits on an object nothing else can reach.
    """
    engine = engine_factory()
    budget = SubagentTreeBudget(3)
    permit = await budget.acquire()
    engine.run_state.subagent_tree_budget = budget

    await engine.resume_from_snapshot(engine.snapshot())

    assert engine.run_state.subagent_tree_budget is budget
    assert budget.in_use == 1
    await permit.release()
