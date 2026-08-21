"""Tree-wide concurrency budget for parallel subagent delegation.

The concurrent-delegation branch in :mod:`protocore.runtime.query` mints a fresh
:class:`asyncio.Semaphore` PER leader turn to bound the WIDTH of that one fan-out
group (``max_concurrent_subagents``). That per-turn semaphore is correct for a
single group but composes MULTIPLICATIVELY across depth: a depth-2 tree where the
root fans out W children and each child fans out W grandchildren runs up to
``W * W`` children concurrently, because each nested group carries its own,
independent per-turn semaphore. On a shared sandbox / token budget that
multiplicative blow-up is the thing tenants actually hit.

A NAIVE tree-wide cap — one shared semaphore, acquired around each child dispatch
and held for the child's whole run — DEADLOCKS. The delegation join is blocking:
a parent holds its permit for the child's ENTIRE run while that child needs
permits for its OWN grandchildren. At the cap the held permits starve the nested
acquires and the tree wedges.

This module implements the deadlock-free alternative: **release-while-awaiting-
children** (leaf-counting). A permit is held only by a run that is ACTIVELY
executing; a run that is *awaiting its own children* (blocked in an
``asyncio.gather`` join) releases its permit for the duration of that wait and
reacquires it afterwards. The invariant this maintains:

    every permit holder is a run doing local work (not blocked on descendants)

so at most ``cap`` runs execute at once, and — crucially — none of those holders
needs to acquire a *further* permit to make progress on its current slice, so the
budget can never form an acquisition cycle. Leaf runs (which never fan out) hold
their permit to completion and then release it, which is exactly what lets a
waiting parent reacquire and resume. Hence no deadlock.

The budget object is minted lazily at the FIRST parallel fan-out (the first run
that dispatches a concurrent delegation group with no budget yet in its helper
bag) and shared by reference from there down — across the maximal
parallel-dispatched subtree, threaded through the helper bag (see
:mod:`protocore.runtime.tool_dispatch` helper keys). That first fan-out need not
be the process root: if the root only ever does serial single-call delegations
the budget is minted deeper. This is not a concurrency gap — any two runs that
execute concurrently necessarily branched at a common parallel-fan-out ancestor,
which mints and shares the single budget BEFORE dispatching them, so concurrent
runs always draw from one budget. Each dispatched child receives its OWN
:class:`SubagentTreePermit` handle — the object it uses to release-while-waiting
around its nested gather (or a single serial delegation hop) and to be released
once by its parent when the child completes.
"""
from __future__ import annotations

import asyncio
from typing import Final

#: Sentinel: a per-tree cap of ``0`` means UNLIMITED — the budget is a no-op and
#: behaviour is byte-identical to the pre-budget world (per-turn semaphores only).
#: Mirrors the ``parallel_read_tools_max_fanout`` sentinel convention.
_UNLIMITED_TREE_CAP: Final[int] = 0


class SubagentTreePermit:
    """One tree-budget permit, owned by a single executing child run.

    Lifecycle (all calls come from the SAME child coroutine, in order, so no
    intra-permit locking is needed — a permit is never shared between concurrent
    coroutines):

    1. Created HELD by :meth:`SubagentTreeBudget.acquire` at the dispatch site
       (the parent acquires on the child's behalf).
    2. The child, when it fans out its OWN parallel delegation group, calls
       :meth:`release_while_waiting` BEFORE its ``asyncio.gather`` and
       :meth:`reacquire` AFTER (in a ``finally``, so it reacquires even if the
       gather raises). While awaiting descendants it holds NO permit.
    3. The parent calls :meth:`release` once, in its ``finally``, after the child
       run returns.

    All three mutators are idempotent with respect to the held/released state:
    :meth:`release` / :meth:`release_while_waiting` release only if currently
    held, and :meth:`reacquire` acquires only if currently released. A permit
    therefore never double-releases (which would hand a semaphore slot it does
    not own to some other run) and never double-acquires.
    """

    __slots__ = ("_budget", "_held")

    def __init__(self, budget: SubagentTreeBudget, *, held: bool) -> None:
        self._budget = budget
        self._held = held

    async def release_while_waiting(self) -> None:
        """Release the permit before awaiting this run's own children.

        Idempotent: a no-op if the permit is already released. This is the half
        of the scheme that makes it deadlock-free — a run blocked in its nested
        gather join must NOT pin a tree slot, or a full tree at the cap could
        never let its descendants acquire.
        """
        self._release_if_held()

    async def reacquire(self) -> None:
        """Reacquire the permit after this run's children have joined.

        Idempotent: a no-op if the permit is already held. Awaits a free slot the
        same way the initial acquire did, so a resuming parent is subject to the
        same tree cap as a freshly-dispatched child.
        """
        if not self._held:
            await self._budget._acquire_slot()
            self._held = True

    async def release(self) -> None:
        """Final release once the owning child run has completed.

        Idempotent: a no-op if the permit is already released (e.g. the child was
        cancelled mid-gather, before its ``finally`` reacquire ran). One permit,
        at most one net slot returned to the budget.
        """
        self._release_if_held()

    def _release_if_held(self) -> None:
        if self._held:
            self._budget._release_slot()
            self._held = False


class SubagentTreeBudget:
    """Shared tree-wide semaphore bounding the SUM of concurrently-executing
    parallel-dispatched subagent runs across the whole run tree.

    Constructed once at the first parallel fan-out (from
    ``rc.max_concurrent_subagents_per_tree``, read from the minting run's scope)
    and threaded by reference to every descendant of that maximal
    parallel-dispatched subtree via the helper bag. The cap is CAPTURED at mint
    time: an in-flight tree does not resize to a mid-flight RC edit — only trees
    minted afterward pick up the new value. ``cap == 0`` selects the UNLIMITED
    no-op mode:
    no semaphore is created and every permit is a free-floating token, so the
    tree-wide bound is inert and only the per-turn ``max_concurrent_subagents``
    width caps apply (identical to the pre-budget behaviour).

    The per-GROUP width is still bounded independently by the per-turn semaphore
    in the delegation branch; this budget bounds the additive total ACROSS nested
    groups so depth no longer multiplies the effective concurrency.
    """

    __slots__ = ("_cap", "_semaphore")

    def __init__(self, cap: int) -> None:
        self._cap = cap
        # A cap of 0 (or any non-positive value, defensively) is the unlimited
        # sentinel: no semaphore, so acquire/release are pure bookkeeping and the
        # tree bound never blocks.
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(cap) if cap > _UNLIMITED_TREE_CAP else None
        )

    @property
    def unlimited(self) -> bool:
        """True when this budget imposes no tree-wide bound (cap==0 sentinel)."""
        return self._semaphore is None

    async def acquire(self) -> SubagentTreePermit:
        """Acquire one tree slot and return the owning child's permit handle.

        Blocks until a slot frees when the tree is at the cap. Under the unlimited
        sentinel returns immediately with a permit that owns no real slot.
        """
        await self._acquire_slot()
        return SubagentTreePermit(self, held=True)

    async def _acquire_slot(self) -> None:
        if self._semaphore is not None:
            await self._semaphore.acquire()

    def _release_slot(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()
