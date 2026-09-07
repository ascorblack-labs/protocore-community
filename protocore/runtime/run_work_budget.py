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
ROOT run when its state is composed, and inherited by reference from there down
— the same propagation ``cancel_event`` / ``root_run_id`` /
``subagent_tree_budget`` already ride on.
:meth:`~protocore.contracts.run_state.RunScopedState.ensure_run_work_ledger`
mints on demand as a fallback so a run composed without one (focused unit
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

from collections.abc import Mapping
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
#: The tree can still start child runs, just fewer than THIS call asked for.
#: A different answer from exhaustion and it needs a different word, because a
#: smaller call would be admitted and telling the model to give up would throw
#: away work the budget can still pay for.
SUBAGENT_RUN_BUDGET_SHORT: Final[str] = "subagent_run_budget_short"
RUN_TREE_TOKEN_BUDGET_EXHAUSTED: Final[str] = "run_tree_token_budget_exhausted"


@dataclass(frozen=True, slots=True)
class ChildRunGrant:
    """The answer to "may this call start ``requested`` child runs?".

    Admission is ALL-OR-NOTHING: ``granted`` is either ``requested`` or zero.
    A prefix grant would have to be sliced by whoever knows how the call maps to
    child runs, and nothing downstream of the reservation can do that for every
    delegation shape there is; a grant that was handed out and then thrown away
    is worse than no grant, because the tree pays for runs that never start.
    So a call the tree cannot pay for in full is refused in full, and
    ``remaining`` tells the caller how large a call WOULD fit.

    ``reason`` is empty exactly when ``granted == requested``. Otherwise it is
    one of the module's reason tokens, naming WHICH budget bound the call.
    """

    requested: int
    granted: int
    reason: str
    remaining: int = 0

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
    reached by every descendant as ``RunScopedState.run_work_ledger``. Counters only
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
        "_charged_call_ids",
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
        # Ids of the calls already charged, so one call cannot be charged twice.
        # A delegation call can reach the charge point more than once — an
        # approval-gated call is dispatched, paused, and dispatched again when
        # the approval lands — and the second pass starts no second child run.
        # Durable, because the pause can outlive the process.
        self._charged_call_ids: set[str] = set()

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
    def remaining_child_runs(self) -> int:
        """How many more child runs the tree may start, or ``-1`` for unlimited."""
        if self._max_child_runs == _UNLIMITED:
            return -1
        return max(0, self._max_child_runs - self._child_runs_started)

    def has_charged(self, call_id: str) -> bool:
        """Whether ``call_id`` has already been charged against this ledger."""
        return call_id in self._charged_call_ids

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

    def reserve_child_runs(
        self, requested: int, *, call_id: str | None = None
    ) -> ChildRunGrant:
        """Reserve ``requested`` child-run slots, all of them or none.

        Reserved slots are charged immediately and never released: the ledger
        counts runs STARTED, and a child that fails still consumed the work.

        ``call_id`` makes the charge idempotent. A call that has already been
        charged is granted again for free — it is the same work, reaching this
        point a second time because it was paused and resumed, not a second
        batch of child runs.
        """
        wanted = max(0, int(requested))
        if wanted == 0:
            return ChildRunGrant(
                requested=0, granted=0, reason="", remaining=self.remaining_child_runs
            )
        if call_id is not None and call_id in self._charged_call_ids:
            return ChildRunGrant(
                requested=wanted,
                granted=wanted,
                reason="",
                remaining=self.remaining_child_runs,
            )
        reason = self.delegation_refusal_reason(wanted)
        if reason:
            return ChildRunGrant(
                requested=wanted,
                granted=0,
                reason=reason,
                remaining=self.remaining_child_runs,
            )
        self._child_runs_started += wanted
        if call_id is not None:
            self._charged_call_ids.add(call_id)
        return ChildRunGrant(
            requested=wanted,
            granted=wanted,
            reason="",
            remaining=self.remaining_child_runs,
        )

    def delegation_refusal_reason(self, requested: int = 1) -> str:
        """Which budget, if any, refuses ``requested`` child runs — reserving nothing.

        Empty when the tree can pay for all of them. ``requested`` defaults to
        one, which is the "is there anything left at all?" question.
        """
        wanted = max(1, int(requested))
        if self._max_tokens != _UNLIMITED and self._tokens_charged >= self._max_tokens:
            return RUN_TREE_TOKEN_BUDGET_EXHAUSTED
        if self._max_child_runs == _UNLIMITED:
            return ""
        remaining = self._max_child_runs - self._child_runs_started
        if remaining <= 0:
            return SUBAGENT_RUN_BUDGET_EXHAUSTED
        if wanted > remaining:
            return SUBAGENT_RUN_BUDGET_SHORT
        return ""

    # ----- durability -----

    def to_snapshot(self) -> dict[str, Any]:
        """Serialise the ledger for the run's durable state.

        Both the captured caps and the running totals go out. The caps are part
        of the ledger's identity — they were fixed when the tree started and a
        resume must not silently re-read them from a constants set that has
        changed underneath an in-flight tree.
        """
        return {
            "max_child_runs": self._max_child_runs,
            "max_tokens": self._max_tokens,
            "child_runs_started": self._child_runs_started,
            "tokens_charged": self._tokens_charged,
            "charged_call_ids": sorted(self._charged_call_ids),
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> RunWorkLedger:
        """Rebuild a ledger from :meth:`to_snapshot`.

        Totals are coerced to non-negative integers: a cumulative counter that
        came back smaller than it went out would hand the tree budget it has
        already spent, which is the exact failure this ledger exists to remove.
        """
        ledger = cls(
            max_child_runs=_as_count(data.get("max_child_runs")),
            max_tokens=_as_count(data.get("max_tokens")),
        )
        ledger._child_runs_started = _as_count(data.get("child_runs_started"))
        ledger._tokens_charged = _as_count(data.get("tokens_charged"))
        charged = data.get("charged_call_ids")
        if isinstance(charged, list):
            ledger._charged_call_ids = {
                item for item in charged if isinstance(item, str)
            }
        return ledger

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


def _as_count(value: Any) -> int:
    """Coerce a persisted counter to a non-negative ``int``, or ``0``.

    ``bool`` is rejected explicitly (it is an ``int`` subclass) so a malformed
    ``true`` does not restore as a count of one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, int(value))


def mint_run_work_ledger(rc: Any) -> RunWorkLedger:
    """Build a ledger from a constants snapshot, unlimited when it has none."""
    return RunWorkLedger(
        max_child_runs=int(getattr(rc, "max_subagent_runs_per_tree", _UNLIMITED)),
        max_tokens=int(getattr(rc, "max_total_tokens_per_tree", _UNLIMITED)),
    )
