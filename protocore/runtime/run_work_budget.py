"""Cumulative total-work budget for one root run and everything it spawns.

A run tree is bounded today in two dimensions and neither of them bounds the
TOTAL:

* WIDTH — ``max_concurrent_subagents`` (one leader turn) and
  ``max_concurrent_subagents_per_tree`` (the additive sum across nested groups)
  bound how many child runs execute AT ONCE.
* DEPTH — ``max_subagent_depth`` bounds how far delegation may nest.

Both are instantaneous. A leader may dispatch a legal-width group, wait for it,
dispatch another, and repeat: every wave passes both checks and the cumulative
number of child runs — and the cumulative token spend behind them — is bounded
only by wall-clock. This module supplies the missing third dimension: how much
work the whole tree may do over its LIFETIME.

Two cumulative quantities are tracked, both per ROOT RUN and both counted across
every descendant at every depth and every wave:

1. ``max_subagent_runs_per_tree`` — how many child runs the tree may START.
2. ``max_total_tokens_per_tree`` — input + output tokens summed over every LLM
   call made by the root run and all of its descendants.

Why this is a SIBLING of :class:`~protocore.runtime.subagent_budget.SubagentTreeBudget`
and not an extension of it
--------------------------------------------------------------------------

The two objects answer different questions and have incompatible shapes.

``SubagentTreeBudget`` models PERMITS. A permit is borrowed and given back; the
whole scheme turns on a holder RELEASING while it awaits its descendants, which
is what keeps the tree from deadlocking at its cap. Its counter must go down.

A cumulative total must never go down. Nothing is given back when a child
finishes — the run happened, the tokens were spent. Folding a monotonic counter
into the permit object would put a decrementing and a non-decrementing quantity
behind one release path, where every future edit to the permit lifecycle is a
chance to hand back work that was already done.

The two also differ in WHEN they must exist. ``SubagentTreeBudget`` is minted
lazily at the first PARALLEL fan-out, which is sound for a concurrency bound:
two runs can only be concurrent if they branched at a common fan-out ancestor,
and that ancestor mints the budget before dispatching either. A cumulative
budget cannot be minted there. A leader that emits ONE delegation call per turn
never fans out, so a lazily-minted ledger would never exist for exactly the
serial wave-after-wave pattern this bound is for, and the token count would miss
every call made before the first fan-out. The ledger is therefore minted for the
ROOT run when its helper bag is composed, and inherited by reference from there
down — the same dict-copy propagation ``cancel_event`` / ``root_run_id`` /
``subagent_tree_budget`` already ride on. :func:`resolve_run_work_ledger` mints
on demand as a fallback so a caller that never composed a bag (focused unit
fixtures, degenerate wiring) still gets a working ledger for its own subtree.

Exhaustion refuses DELEGATION, never the run
--------------------------------------------

An exhausted ledger stops new child runs from starting. It does not terminate
the run, does not abort children already in flight, and does not stand between
the leader and its answer. That asymmetry is deliberate: the unbounded quantity
is the TREE (each child is a fresh engine with a fresh per-run budget, so the
existing per-run bounds do not compose over delegation), whereas a leader left
to work alone is already bounded by ``max_turns_per_run`` and
``run_max_output_tokens_budget``. Refusing delegation removes the unbounded term
and cannot wedge the run, and the leader still gets to finalise with what it has
— which is the whole point, because a bounded run that cannot answer is worse
than the unbounded run it replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

#: Sentinel: a cap of ``0`` means UNLIMITED for both budgets — the ledger still
#: COUNTS (its totals stay available for diagnostics) but never refuses.
#: Mirrors the ``max_concurrent_subagents_per_tree`` / ``run_max_output_tokens_budget``
#: sentinel convention.
_UNLIMITED: Final[int] = 0

#: Machine-readable reason tokens. Short, snake_case and stable — they are
#: echoed to the model in the refusal text and carried in the structured error,
#: so they double as the grep handle in production logs.
SUBAGENT_RUN_BUDGET_EXHAUSTED: Final[str] = "subagent_run_budget_exhausted"
RUN_TREE_TOKEN_BUDGET_EXHAUSTED: Final[str] = "run_tree_token_budget_exhausted"


@dataclass(frozen=True, slots=True)
class ChildRunGrant:
    """The answer to "may this call start ``requested`` child runs?".

    ``granted`` is how many the ledger reserved — possibly FEWER than requested
    and possibly zero. A partial grant is a real answer, not a failure: a batch
    of five with two left in the budget runs two and reports three refused, which
    tells the leader both that some work was done and that the door is now shut.

    ``reason`` is empty exactly when ``granted == requested``. Otherwise it is
    one of the module's reason tokens, naming WHICH budget bound the call.
    """

    requested: int
    granted: int
    reason: str

    @property
    def refused(self) -> int:
        """How many of the requested runs the ledger would not pay for."""
        return self.requested - self.granted

    @property
    def fully_granted(self) -> bool:
        return self.granted == self.requested


class RunWorkLedger:
    """Monotonic record of the total work done under one root run.

    Shared BY REFERENCE across the whole run tree: one instance per root run,
    reached by every descendant through the inherited helper bag. Counters only
    ever increase — a completed child run is not refunded, and neither are its
    tokens.

    Both caps are CAPTURED at construction, matching ``SubagentTreeBudget``: a
    run already in flight does not resize to a mid-flight constants edit; runs
    started afterwards pick the new values up.

    Concurrency: every mutator is SYNCHRONOUS and contains no ``await``. That is
    what makes it race-free without a lock — the whole tree runs on one event
    loop, so a check-then-increment with no suspension point in between cannot
    interleave with another child's. Do not make these coroutines.
    """

    __slots__ = (
        "_child_runs_started",
        "_max_child_runs",
        "_max_tokens",
        "_tokens_charged",
    )

    def __init__(self, *, max_child_runs: int, max_tokens: int) -> None:
        # Defensive ``max(0, ...)``: a negative cap is meaningless and the
        # sentinel for "no bound" is 0, so any non-positive value reads as
        # unlimited rather than as a budget that refuses everything.
        self._max_child_runs = max(_UNLIMITED, int(max_child_runs))
        self._max_tokens = max(_UNLIMITED, int(max_tokens))
        self._child_runs_started = 0
        self._tokens_charged = 0

    # ----- state -----

    @property
    def max_child_runs(self) -> int:
        return self._max_child_runs

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def child_runs_started(self) -> int:
        return self._child_runs_started

    @property
    def tokens_charged(self) -> int:
        return self._tokens_charged

    @property
    def unlimited(self) -> bool:
        """True when neither budget can ever refuse a delegation."""
        return self._max_child_runs == _UNLIMITED and self._max_tokens == _UNLIMITED

    # ----- accounting -----

    def charge_tokens(self, *, input_tokens: int, output_tokens: int) -> None:
        """Add one LLM call's input+output to the tree total.

        Called from every engine in the tree — the root's and every
        descendant's — against the SAME ledger, which is what makes the total a
        tree total rather than a per-run one. Negative values are ignored rather
        than subtracted: a provider that reports nonsense must not be able to buy
        the tree more budget.
        """
        self._tokens_charged += max(0, int(input_tokens)) + max(0, int(output_tokens))

    def reserve_child_runs(self, requested: int) -> ChildRunGrant:
        """Reserve up to ``requested`` child-run slots, charging what is granted.

        Grants a PREFIX when only some fit. The token budget is all-or-nothing by
        contrast — it is a statement about the tree's total spend, and once it is
        gone no further delegation is worth starting, so it grants zero rather
        than a fraction.

        Reserved slots are charged immediately and never released: the ledger
        counts runs STARTED, and a child that fails still consumed the work.
        """
        wanted = max(0, int(requested))
        if wanted == 0:
            return ChildRunGrant(requested=0, granted=0, reason="")
        if self._max_tokens != _UNLIMITED and self._tokens_charged >= self._max_tokens:
            return ChildRunGrant(
                requested=wanted,
                granted=0,
                reason=RUN_TREE_TOKEN_BUDGET_EXHAUSTED,
            )
        if self._max_child_runs == _UNLIMITED:
            self._child_runs_started += wanted
            return ChildRunGrant(requested=wanted, granted=wanted, reason="")
        remaining = self._max_child_runs - self._child_runs_started
        granted = max(0, min(wanted, remaining))
        self._child_runs_started += granted
        return ChildRunGrant(
            requested=wanted,
            granted=granted,
            reason="" if granted == wanted else SUBAGENT_RUN_BUDGET_EXHAUSTED,
        )

    def delegation_refusal_reason(self) -> str:
        """Which budget, if any, is already spent — WITHOUT reserving anything.

        Empty when at least one more child run could be started. Used by the
        pre-dispatch gate to refuse a delegation call before it reaches the
        tool, so an exhausted tree spends nothing to learn it is exhausted.
        """
        if self._max_tokens != _UNLIMITED and self._tokens_charged >= self._max_tokens:
            return RUN_TREE_TOKEN_BUDGET_EXHAUSTED
        if (
            self._max_child_runs != _UNLIMITED
            and self._child_runs_started >= self._max_child_runs
        ):
            return SUBAGENT_RUN_BUDGET_EXHAUSTED
        return ""

    def spent_summary(self) -> str:
        """One-line ``spent/cap`` rendering for logs and model-facing refusals.

        Names both budgets whichever one tripped, because the leader's next
        decision depends on how much room is left in the OTHER one.
        """
        runs_cap = "unlimited" if self._max_child_runs == _UNLIMITED else str(
            self._max_child_runs
        )
        tokens_cap = "unlimited" if self._max_tokens == _UNLIMITED else str(
            self._max_tokens
        )
        return (
            f"subagent runs {self._child_runs_started}/{runs_cap}, "
            f"tokens {self._tokens_charged}/{tokens_cap}"
        )


#: Helper-bag key under which the ROOT run's ledger lives. Duplicated as a
#: literal in :mod:`protocore.runtime.tool_dispatch` alongside the other helper
#: keys; kept here too so this module stands alone.
RUN_WORK_LEDGER_HELPER_KEY: Final[str] = "run_work_ledger"


def resolve_run_work_ledger(helpers: Any, rc: Any) -> RunWorkLedger:
    """Return the tree's ledger from ``helpers``, minting it there if absent.

    ``helpers`` is the per-run helper bag and ``rc`` the runtime-constants
    snapshot the minting run resolved (already narrowed by the caller's access
    plan by the time a live run reaches here). Both are duck-typed so this can be
    called from core and from the host tool layer without either importing
    the other's concrete types.

    A bag that is not a mutable mapping — focused unit fixtures, degenerate
    wiring — gets a fresh ledger that bounds only the caller. That is strictly
    better than no bound and cannot be mistaken for a tree-wide one, because
    nothing else can reach it.
    """
    if isinstance(helpers, dict):
        existing = helpers.get(RUN_WORK_LEDGER_HELPER_KEY)
        if isinstance(existing, RunWorkLedger):
            return existing
        ledger = _mint(rc)
        helpers[RUN_WORK_LEDGER_HELPER_KEY] = ledger
        return ledger
    return _mint(rc)


def run_work_ledger_from(helpers: Any) -> RunWorkLedger | None:
    """Read the ledger out of a helper bag WITHOUT minting one.

    For readers that must not create budget state as a side effect of looking —
    diagnostics, and any caller for which "no ledger" is a meaningful answer.
    """
    if isinstance(helpers, dict):
        ledger = helpers.get(RUN_WORK_LEDGER_HELPER_KEY)
        if isinstance(ledger, RunWorkLedger):
            return ledger
    return None


def _mint(rc: Any) -> RunWorkLedger:
    """Build a ledger from a constants snapshot, unlimited when it has none."""
    return RunWorkLedger(
        max_child_runs=int(getattr(rc, "max_subagent_runs_per_tree", _UNLIMITED)),
        max_tokens=int(getattr(rc, "max_total_tokens_per_tree", _UNLIMITED)),
    )
