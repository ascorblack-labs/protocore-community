"""Tree-wide subagent concurrency budget: release-while-awaiting-children.

Pins :class:`protocore.runtime.subagent_budget.SubagentTreeBudget` and its
per-child :class:`SubagentTreePermit`. The delegation fan-out branch in
:mod:`protocore.runtime.query` mints one per-turn semaphore per group, which
composes MULTIPLICATIVELY across depth; the tree budget bounds the ADDITIVE sum
of concurrently-executing children across every nested group.

The correctness crux is that the scheme is DEADLOCK-FREE where the naive "hold a
shared permit across the child's whole run" is not: a run awaiting its own
children RELEASES its tree slot for the duration of that wait and reacquires it
afterwards, so every slot holder is actively executing and none needs a further
slot to finish. These tests simulate a depth-2 delegation tree at the cap and
assert (a) it completes under a watchdog (no deadlock), (b) peak concurrent slot
holders never exceed the cap, and — as a load-bearing control — that the NAIVE
hold-across-children variant DOES deadlock at the same cap.
"""
from __future__ import annotations

import asyncio

import pytest

from protocore.runtime.subagent_budget import (
    SubagentTreeBudget,
    SubagentTreePermit,
)


class _PeakTracker:
    """Counts permits currently HELD-and-executing, recording the peak.

    Updated in lockstep with the held/released transitions of the simulated
    nodes: a slot is counted only while its owner is doing local work, never
    while it is parked in ``release_while_waiting`` awaiting descendants.
    """

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def acquired(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def released(self) -> None:
        self.current -= 1


async def _run_node(
    budget: SubagentTreeBudget,
    permit: SubagentTreePermit,
    *,
    width: int,
    depth: int,
    tracker: _PeakTracker,
) -> None:
    """Simulate one non-root run that already holds ``permit`` (tracker bumped).

    Mirrors the real delegation branch: do a slice of local work, then — if this
    run fans out its own group — release the tree slot BEFORE the gather and
    reacquire it AFTER (finally), so a run blocked on its children pins no slot.
    """
    await asyncio.sleep(0.005)  # local work while holding the slot
    if depth > 0:
        await permit.release_while_waiting()
        tracker.released()
        try:
            await asyncio.gather(
                *(
                    _dispatch_child(
                        budget, width=width, depth=depth - 1, tracker=tracker
                    )
                    for _ in range(width)
                )
            )
        finally:
            await permit.reacquire()
            tracker.acquired()
    await asyncio.sleep(0.005)  # more local work after children join


async def _dispatch_child(
    budget: SubagentTreeBudget,
    *,
    width: int,
    depth: int,
    tracker: _PeakTracker,
) -> None:
    """Acquire this child's own tree slot at the dispatch site, run it, release."""
    permit = await budget.acquire()
    tracker.acquired()
    try:
        await _run_node(budget, permit, width=width, depth=depth, tracker=tracker)
    finally:
        await permit.release()
        tracker.released()


async def _run_tree(budget: SubagentTreeBudget, *, width: int, depth: int) -> _PeakTracker:
    """Drive a root (holds no slot) fanning out a depth-``depth`` width-``width`` tree."""
    tracker = _PeakTracker()
    await asyncio.gather(
        *(
            _dispatch_child(budget, width=width, depth=depth - 1, tracker=tracker)
            for _ in range(width)
        )
    )
    return tracker


# ---------------------------------------------------------------------------
# Deadlock-free nested fan-out at the cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_fanout_at_cap_completes_without_deadlock() -> None:
    """cap=2, depth-2 tree (root width 2, each child fans out 2) ⇒ no deadlock."""
    budget = SubagentTreeBudget(2)
    # Watchdog: the naive hold-across-children scheme wedges here; the correct
    # one finishes in well under a second.
    tracker = await asyncio.wait_for(
        _run_tree(budget, width=2, depth=2), timeout=5.0
    )
    assert tracker.peak >= 1
    # After the whole tree drains, the budget is intact: cap fresh slots are
    # immediately available (no permit leaked a slot).
    permits = [await asyncio.wait_for(budget.acquire(), timeout=1.0) for _ in range(2)]
    for permit in permits:
        await permit.release()


@pytest.mark.asyncio
async def test_peak_concurrent_holders_never_exceeds_cap() -> None:
    """A wider/deeper tree still never holds more than ``cap`` slots at once."""
    cap = 3
    budget = SubagentTreeBudget(cap)
    tracker = await asyncio.wait_for(
        _run_tree(budget, width=3, depth=2), timeout=5.0
    )
    assert 1 <= tracker.peak <= cap


@pytest.mark.asyncio
async def test_unlimited_sentinel_is_a_noop() -> None:
    """cap=0 ⇒ unlimited: acquire never blocks and a deep tree still completes."""
    budget = SubagentTreeBudget(0)
    assert budget.unlimited is True
    # With no bound, more than any finite cap of holders may coexist; the tree
    # must still complete (and never wedge).
    tracker = await asyncio.wait_for(
        _run_tree(budget, width=3, depth=2), timeout=5.0
    )
    assert tracker.peak >= 1


# ---------------------------------------------------------------------------
# Load-bearing control: the naive scheme DEADLOCKS
# ---------------------------------------------------------------------------


async def _naive_run_node(
    budget: SubagentTreeBudget, *, width: int, depth: int
) -> None:
    """Acquire a slot and HOLD it across the child gather (the wrong way).

    This is the implementation the mint-site comment warns deadlocks: the permit
    is pinned for the run's entire blocking join, so at the cap the held permits
    starve the nested acquires.
    """
    permit = await budget.acquire()
    try:
        await asyncio.sleep(0.005)
        if depth > 0:
            await asyncio.gather(
                *(
                    _naive_run_node(budget, width=width, depth=depth - 1)
                    for _ in range(width)
                )
            )
    finally:
        await permit.release()


@pytest.mark.asyncio
async def test_naive_hold_across_children_deadlocks() -> None:
    """The naive variant wedges at cap=2 for a depth-2 width-2 tree (control)."""
    budget = SubagentTreeBudget(2)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            asyncio.gather(
                *(_naive_run_node(budget, width=2, depth=1) for _ in range(2))
            ),
            timeout=0.5,
        )


# ---------------------------------------------------------------------------
# Regression: a SERIAL delegation hop under a permit holder must also release
# ---------------------------------------------------------------------------
#
# Models the exact deadlock schedule from the review: a permit-holding run makes
# a SINGLE (serial-path) delegation call whose child then fans out in parallel.
# The serial join is where the earlier implementation kept holding the tree slot
# — so a parallel fan-out beneath any serial hop wedged the tree at the cap. The
# ``release_on_serial`` toggle represents the fix (True) vs the pre-fix bug
# (False): the same schedule must COMPLETE when the serial hop releases and
# DEADLOCK when it does not.


async def _serial_child_that_fans_out(
    budget: SubagentTreeBudget, *, width: int, tracker: _PeakTracker
) -> None:
    """A child dispatched SERIALLY (holds NO slot) that then fans out in parallel.

    Mirrors the real serial delegation child: ``permit_handle=None`` on the
    serial path, so it draws no tree slot of its own, but its OWN parallel group
    dispatches ``width`` grandchildren that each acquire one.
    """
    await asyncio.sleep(0.005)
    await asyncio.gather(
        *(_dispatch_child(budget, width=width, depth=0, tracker=tracker)
          for _ in range(width))
    )


async def _permit_holder_with_serial_hop(
    budget: SubagentTreeBudget,
    permit: SubagentTreePermit,
    *,
    width: int,
    tracker: _PeakTracker,
    release_on_serial: bool,
) -> None:
    """A run that holds ``permit`` and makes ONE serial delegation hop.

    With ``release_on_serial`` (the fix) it release-while-waits around the join,
    exactly as the serial-path code now does; without it (the pre-fix bug) it
    pins its slot across the child's whole run — the C1 deadlock.
    """
    await asyncio.sleep(0.005)
    if release_on_serial:
        await permit.release_while_waiting()
        tracker.released()
    try:
        await _serial_child_that_fans_out(budget, width=width, tracker=tracker)
    finally:
        if release_on_serial:
            await permit.reacquire()
            tracker.acquired()


async def _dispatch_permit_holder_with_serial_hop(
    budget: SubagentTreeBudget,
    *,
    width: int,
    tracker: _PeakTracker,
    release_on_serial: bool,
) -> None:
    permit = await budget.acquire()
    tracker.acquired()
    try:
        await _permit_holder_with_serial_hop(
            budget,
            permit,
            width=width,
            tracker=tracker,
            release_on_serial=release_on_serial,
        )
    finally:
        await permit.release()
        tracker.released()


async def _run_serial_hop_tree(
    budget: SubagentTreeBudget,
    *,
    breadth: int,
    width: int,
    release_on_serial: bool,
) -> _PeakTracker:
    """Root fans out ``breadth`` permit-holders, each doing a serial→parallel hop."""
    tracker = _PeakTracker()
    await asyncio.gather(
        *(
            _dispatch_permit_holder_with_serial_hop(
                budget,
                width=width,
                tracker=tracker,
                release_on_serial=release_on_serial,
            )
            for _ in range(breadth)
        )
    )
    return tracker


@pytest.mark.asyncio
async def test_serial_hop_under_permit_holder_completes_when_releasing() -> None:
    """C1 schedule, POST-fix: serial hop releases its slot ⇒ no deadlock at cap.

    cap=2; two permit-holders (2/2 slots) each make a single serial delegation
    hop whose child fans out two grandchildren. Because each holder releases its
    slot while awaiting its serial child, the grandchildren acquire the freed
    slots and the tree drains.
    """
    budget = SubagentTreeBudget(2)
    tracker = await asyncio.wait_for(
        _run_serial_hop_tree(budget, breadth=2, width=2, release_on_serial=True),
        timeout=5.0,
    )
    assert 1 <= tracker.peak <= 2
    # No slot leaked: cap fresh slots are immediately available afterwards.
    permits = [await asyncio.wait_for(budget.acquire(), timeout=1.0) for _ in range(2)]
    for permit in permits:
        await permit.release()


@pytest.mark.asyncio
async def test_serial_hop_without_release_deadlocks() -> None:
    """C1 schedule, PRE-fix control: serial hop pins its slot ⇒ the tree wedges.

    The exact same schedule as above with ``release_on_serial=False`` (the
    behaviour before the serial path released its permit): the two holders pin
    both slots while blocked on their serial children, whose parallel fan-out
    then starves on ``budget.acquire()``. Proves the model captures C1 and that
    the release on the serial path is load-bearing.
    """
    budget = SubagentTreeBudget(2)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            _run_serial_hop_tree(budget, breadth=2, width=2, release_on_serial=False),
            timeout=0.5,
        )


# ---------------------------------------------------------------------------
# Permit handle idempotency + finally-path correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permit_reacquired_after_child_gather_raises() -> None:
    """A raising nested gather still reacquires the slot in ``finally``."""
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()

    async def failing_children() -> None:
        await permit.release_while_waiting()
        try:
            raise RuntimeError("nested group blew up")
        finally:
            await permit.reacquire()

    with pytest.raises(RuntimeError, match="nested group blew up"):
        await failing_children()

    # The slot was reacquired, so releasing it once returns the budget to full
    # capacity — a second acquire must not block.
    await permit.release()
    again = await asyncio.wait_for(budget.acquire(), timeout=1.0)
    await again.release()


@pytest.mark.asyncio
async def test_permit_release_is_idempotent() -> None:
    """Double release (temporary + final) never over-returns a slot."""
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()
    await permit.release_while_waiting()
    await permit.release()  # already released: no-op
    await permit.release()  # still a no-op
    # Exactly one slot exists; we can hold it once and no more concurrently.
    held = await asyncio.wait_for(budget.acquire(), timeout=1.0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(budget.acquire(), timeout=0.2)
    await held.release()


@pytest.mark.asyncio
async def test_reacquire_without_prior_release_is_noop() -> None:
    """Reacquiring a still-held permit does not consume a second slot."""
    budget = SubagentTreeBudget(1)
    permit = await budget.acquire()
    await permit.reacquire()  # still held ⇒ no-op, must not deadlock on its own slot
    await permit.release()
    fresh = await asyncio.wait_for(budget.acquire(), timeout=1.0)
    await fresh.release()
