"""Cumulative total-work budget for one root run and its whole delegation tree.

Pins :class:`protocore.runtime.run_work_budget.RunWorkLedger` and the two places
core enforces it: the deferred parallel-delegation dispatch and the serial one.

What these tests are FOR. A run tree was bounded only in width
(``max_concurrent_subagents``, ``max_concurrent_subagents_per_tree``) and depth
(``max_subagent_depth``), and both are instantaneous. A leader that dispatches a
legal group, waits for it, and dispatches another passes every check on every
wave, so the cumulative number of child runs — and the tokens behind them — had
no bound at all. The properties below are the ones that make the cumulative
bound real rather than nominal:

* it accumulates ACROSS waves and never resets;
* a descendant at depth counts into the SAME ledger as the root;
* exhaustion produces a refusal the model can act on, with no child run started;
* the run can still FINALISE after exhaustion.

The last one is the point of the whole design. A bound that stops a runaway by
leaving the leader unable to answer has replaced one failure with a worse one.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _dispatch_tool,
    _drain_dispatch_tool_deferred,
    _resolve_run_work_ledger,
)
from protocore.runtime.run_work_budget import (
    RUN_TREE_TOKEN_BUDGET_EXHAUSTED,
    RUN_WORK_LEDGER_HELPER_KEY,
    SUBAGENT_RUN_BUDGET_EXHAUSTED,
    RunWorkLedger,
    resolve_run_work_ledger,
    run_work_ledger_from,
)
from protocore.runtime.tool_dispatch import HELPER_RUN_WORK_LEDGER_KEY

from .test_query_parallel_safe_tools import _queue_multi_tool_stream
from .test_query_parallel_subagents import (
    _register_delegation_tool,
    _ScriptedDelegationTool,
)


def _rc(*, runs: int = 0, tokens: int = 0) -> RuntimeConstants:
    """A constants snapshot with only the two total-work caps set."""
    return RuntimeConstants(
        model_context_window=4_096,
        max_subagent_runs_per_tree=runs,
        max_total_tokens_per_tree=tokens,
    )


class _ReservingDelegationTool(_ScriptedDelegationTool):
    """A delegation fake that RESERVES from the ledger, as the real tool does.

    Core cannot count child runs on its own and this fake is where that shows.
    One delegation tool call carries a LIST of tasks, and the list lives inside
    the tool's own argument schema — which core, by design, knows nothing about.
    So the division is: the delegation tool reserves (it is the only layer that
    knows how many runs a call will start), and core refuses (it is where a
    refusal becomes a tool result in the transcript, on both dispatch paths).

    This fake reserves one run per call, which is what the real tool does for a
    one-element batch. Without it a core-level end-to-end test would exercise the
    refusal against a ledger nothing ever charges.
    """

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
        helpers = context.metadata.get("protocore.helpers")
        ledger = run_work_ledger_from(helpers)
        if ledger is not None:
            ledger.reserve_child_runs(1)
        return await super().invoke(context, arguments)


def _tool_result_texts(engine: Any) -> list[str]:
    return [
        block.content
        for msg in engine.history
        for block in msg.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


# ---------------------------------------------------------------------------
# Ledger arithmetic
# ---------------------------------------------------------------------------


def test_child_run_reservations_accumulate_and_never_reset() -> None:
    """Successive reservations draw down ONE running total.

    This is the wave-after-wave property stated as arithmetic: three separate
    reservations of 2 against a cap of 5 must not each be judged against the
    full cap. The third is the one that proves it — under a per-wave cap it
    would be granted in full.
    """
    ledger = RunWorkLedger(max_child_runs=5, max_tokens=0)

    first = ledger.reserve_child_runs(2)
    second = ledger.reserve_child_runs(2)
    third = ledger.reserve_child_runs(2)

    assert (first.granted, first.reason) == (2, "")
    assert (second.granted, second.reason) == (2, "")
    assert third.granted == 1, "the third wave must see only the remaining slot"
    assert third.reason == SUBAGENT_RUN_BUDGET_EXHAUSTED
    assert third.refused == 1
    assert ledger.child_runs_started == 5
    assert ledger.reserve_child_runs(1).granted == 0


def test_grant_is_a_prefix_not_all_or_nothing() -> None:
    """A batch bigger than the remainder runs the part that fits."""
    ledger = RunWorkLedger(max_child_runs=3, max_tokens=0)
    grant = ledger.reserve_child_runs(10)
    assert (grant.requested, grant.granted, grant.refused) == (10, 3, 7)
    assert grant.fully_granted is False


def test_zero_caps_are_the_unlimited_sentinel() -> None:
    """0 means UNLIMITED for both budgets — the ledger counts but never refuses."""
    ledger = RunWorkLedger(max_child_runs=0, max_tokens=0)
    assert ledger.unlimited is True
    grant = ledger.reserve_child_runs(1_000)
    assert grant.fully_granted is True
    ledger.charge_tokens(input_tokens=10**9, output_tokens=10**9)
    assert ledger.delegation_refusal_reason() == ""
    # Counting continues regardless, so the totals stay usable as diagnostics.
    assert ledger.child_runs_started == 1_000
    assert ledger.tokens_charged == 2 * 10**9


def test_tokens_are_monotonic_and_ignore_negative_reports() -> None:
    """A provider reporting a negative usage figure cannot buy back budget."""
    ledger = RunWorkLedger(max_child_runs=0, max_tokens=100)
    ledger.charge_tokens(input_tokens=60, output_tokens=10)
    ledger.charge_tokens(input_tokens=-500, output_tokens=-500)
    assert ledger.tokens_charged == 70
    assert ledger.delegation_refusal_reason() == ""
    ledger.charge_tokens(input_tokens=30, output_tokens=0)
    assert ledger.tokens_charged == 100
    assert ledger.delegation_refusal_reason() == RUN_TREE_TOKEN_BUDGET_EXHAUSTED


def test_token_exhaustion_refuses_the_whole_batch_not_a_prefix() -> None:
    """The token budget is all-or-nothing where the run budget is a prefix.

    Once the tree's total spend is gone, no fraction of a further batch is worth
    starting — unlike a run count, where the remaining slots are real capacity.
    """
    ledger = RunWorkLedger(max_child_runs=100, max_tokens=10)
    ledger.charge_tokens(input_tokens=10, output_tokens=0)
    grant = ledger.reserve_child_runs(4)
    assert (grant.granted, grant.reason) == (0, RUN_TREE_TOKEN_BUDGET_EXHAUSTED)
    assert ledger.child_runs_started == 0, "a refused batch must not be charged"


def test_token_budget_outranks_the_run_budget_in_the_reason() -> None:
    """Both spent ⇒ the token budget is named, because it is the harder wall.

    A leader told "no runs left" might reasonably ask for one more; told the
    token budget is gone, there is nothing to negotiate.
    """
    ledger = RunWorkLedger(max_child_runs=1, max_tokens=10)
    ledger.reserve_child_runs(1)
    ledger.charge_tokens(input_tokens=10, output_tokens=0)
    assert ledger.delegation_refusal_reason() == RUN_TREE_TOKEN_BUDGET_EXHAUSTED


def test_spent_summary_names_both_budgets() -> None:
    """The refusal text has to say how much of EACH budget is left."""
    ledger = RunWorkLedger(max_child_runs=4, max_tokens=0)
    ledger.reserve_child_runs(3)
    ledger.charge_tokens(input_tokens=7, output_tokens=3)
    assert ledger.spent_summary() == "subagent runs 3/4, tokens 10/unlimited"


# ---------------------------------------------------------------------------
# Sharing: one ledger per tree, inherited by reference
# ---------------------------------------------------------------------------


def test_resolve_mints_into_the_bag_and_returns_the_same_object() -> None:
    helpers: dict[str, Any] = {}
    first = resolve_run_work_ledger(helpers, _rc(runs=3))
    second = resolve_run_work_ledger(helpers, _rc(runs=99))
    assert first is second, "a second resolve must not mint a second ledger"
    assert first.max_child_runs == 3, "the cap is captured at mint, not re-read"
    assert helpers[RUN_WORK_LEDGER_HELPER_KEY] is first
    assert run_work_ledger_from(helpers) is first


def test_helper_key_is_one_string_across_modules() -> None:
    """``tool_dispatch`` re-exports the owning module's key rather than restating it."""
    assert HELPER_RUN_WORK_LEDGER_KEY == RUN_WORK_LEDGER_HELPER_KEY


def test_descendants_at_depth_count_into_the_root_ledger() -> None:
    """A grandchild's work lands in the ROOT's ledger, not a fresh one.

    Simulates exactly what the host subagent runner does to build a child
    bag: ``dict(parent_helpers)``. The copy is shallow, so the ledger travels by
    reference — which is the mechanism that makes "counted across the whole
    tree" true at every depth rather than only for direct children.
    """
    root: dict[str, Any] = {}
    ledger = resolve_run_work_ledger(root, _rc(runs=6))

    child = dict(root)
    grandchild = dict(child)
    great_grandchild = dict(grandchild)

    for bag in (child, grandchild, great_grandchild):
        resolve_run_work_ledger(bag, _rc(runs=6)).reserve_child_runs(2)
        resolve_run_work_ledger(bag, _rc(runs=6)).charge_tokens(
            input_tokens=5, output_tokens=5
        )

    assert ledger.child_runs_started == 6
    assert ledger.tokens_charged == 30
    assert ledger.delegation_refusal_reason() == SUBAGENT_RUN_BUDGET_EXHAUSTED
    assert run_work_ledger_from(great_grandchild) is ledger


def test_bagless_caller_gets_its_own_ledger() -> None:
    """No mutable bag ⇒ a local ledger that bounds only the caller.

    Better than no bound, and unmistakable for a tree-wide one: nothing else can
    reach it.
    """
    first = resolve_run_work_ledger(None, _rc(runs=2))
    second = resolve_run_work_ledger(None, _rc(runs=2))
    assert first is not second


def test_missing_constants_read_as_unlimited() -> None:
    """A constants object without the fields must not refuse everything."""

    class _Bare:
        pass

    assert resolve_run_work_ledger(None, _Bare()).unlimited is True


# ---------------------------------------------------------------------------
# Core enforcement: the deferred (parallel) dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausted_budget_refuses_delegation_without_invoking_the_tool(
    engine_factory, in_memory_runtime
) -> None:
    """The refusal happens BEFORE dispatch — no child run, no tool invocation."""
    engine = engine_factory(rc=_rc(runs=1))
    setattr(engine, "_helpers", {})  # noqa: B010 — stand in for the executor bag
    tool = _register_delegation_tool(in_memory_runtime)
    _resolve_run_work_ledger(engine).reserve_child_runs(1)  # spend the only slot

    events, outcome = await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Agent", arguments={"path": "a"})
    )

    assert tool.calls == [], "an exhausted tree must not start the child run"
    assert outcome is not None
    assert outcome.is_error is True
    assert len(events) == 1
    assert events[0].payload["tool_call_id"] == "c-1"


@pytest.mark.asyncio
async def test_refusal_is_actionable_not_generic(
    engine_factory, in_memory_runtime
) -> None:
    """The refusal says the budget is spent, permanent, and what to do instead.

    Pins the wording because the wording IS the mechanism. A leader that reads a
    refusal as transient re-issues the call, is refused again, and burns the
    turns it needed to write an answer — strictly worse than the unbounded
    delegation this bound replaces. The three clauses asserted here are the ones
    that close off that reading.
    """
    engine = engine_factory(rc=_rc(runs=1))
    setattr(engine, "_helpers", {})  # noqa: B010
    _register_delegation_tool(in_memory_runtime)
    _resolve_run_work_ledger(engine).reserve_child_runs(1)

    _events, outcome = await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Agent", arguments={"path": "a"})
    )

    assert outcome is not None
    content = outcome.content
    assert SUBAGENT_RUN_BUDGET_EXHAUSTED in content
    assert "does NOT refill" in content
    assert "retrying this call will fail identically" in content
    assert "Finalize your answer now" in content
    # And the machine-readable half, which is what earns the loop's finalize hint.
    assert outcome.metadata is not None
    structured = outcome.metadata["structured_error"]
    assert structured["finalization_recommended"] is True
    assert structured["reason"] == SUBAGENT_RUN_BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_token_exhaustion_refuses_delegation_by_name(
    engine_factory, in_memory_runtime
) -> None:
    """The token budget refuses delegation, and says which budget it was."""
    engine = engine_factory(rc=_rc(tokens=50))
    setattr(engine, "_helpers", {})  # noqa: B010
    tool = _register_delegation_tool(in_memory_runtime)
    _resolve_run_work_ledger(engine).charge_tokens(input_tokens=40, output_tokens=10)

    _events, outcome = await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Agent", arguments={"path": "a"})
    )

    assert tool.calls == []
    assert outcome is not None
    assert RUN_TREE_TOKEN_BUDGET_EXHAUSTED in outcome.content


@pytest.mark.asyncio
async def test_non_delegation_tools_are_untouched_by_an_exhausted_budget(
    engine_factory, in_memory_runtime
) -> None:
    """The budget bounds DELEGATION only.

    Load-bearing: the leader's own tools are how it assembles the answer it is
    being told to finalize. Refusing those would turn the bound into the thing
    it exists to prevent.
    """
    engine = engine_factory(rc=_rc(runs=1))
    setattr(engine, "_helpers", {})  # noqa: B010
    _register_delegation_tool(in_memory_runtime)
    plain = _ScriptedDelegationTool(tool_name="Read", description="read")
    plain.is_parallel_delegation = False
    in_memory_runtime["tools"].register(plain)
    _resolve_run_work_ledger(engine).reserve_child_runs(1)

    _events, outcome = await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-2", name="Read", arguments={"path": "x"})
    )

    assert outcome is not None
    assert outcome.is_error is False
    assert plain.calls == [{"path": "x"}]


@pytest.mark.asyncio
async def test_budget_with_room_left_dispatches_normally(
    engine_factory, in_memory_runtime
) -> None:
    """Control: an unexhausted budget changes nothing about the dispatch."""
    engine = engine_factory(rc=_rc(runs=5))
    setattr(engine, "_helpers", {})  # noqa: B010
    tool = _register_delegation_tool(in_memory_runtime)

    _events, outcome = await _drain_dispatch_tool_deferred(
        engine, ToolCall(id="c-1", name="Agent", arguments={"path": "a"})
    )

    assert outcome is not None
    assert outcome.is_error is False
    assert tool.calls == [{"path": "a"}]


# ---------------------------------------------------------------------------
# Core enforcement: the serial dispatch path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serial_dispatch_refuses_too(engine_factory, in_memory_runtime) -> None:
    """The serial path must refuse as well, or the bound has a hole.

    A single delegation call, a hook-gated one, or a scope with
    ``parallel_subagents_enabled`` off all take the serial route — and one call
    per turn, repeated, is precisely the wave-after-wave shape the cumulative
    bound exists for. Enforcing only on the fan-out branch would leave the
    easiest way to exceed the budget wide open.
    """
    engine = engine_factory(rc=_rc(runs=1))
    setattr(engine, "_helpers", {})  # noqa: B010
    tool = _register_delegation_tool(in_memory_runtime)
    _resolve_run_work_ledger(engine).reserve_child_runs(1)

    call = ToolCall(id="c-1", name="Agent", arguments={"path": "a"})
    engine.remember_tool_name(call.id, call.name)
    events = [evt async for evt in _dispatch_tool(engine, call)]

    assert tool.calls == []
    assert len(events) == 1
    texts = _tool_result_texts(engine)
    assert len(texts) == 1
    assert SUBAGENT_RUN_BUDGET_EXHAUSTED in texts[0]
    # The loop's finalize hint rides along on the serial path too.
    assert "[finalization-recommended]" in texts[0]


# ---------------------------------------------------------------------------
# End to end: waves, and finishing after exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_holds_across_waves_and_the_run_still_finalises(
    engine_factory, in_memory_runtime
) -> None:
    """Two waves of two against a cap of 3: three children run, the fourth is
    refused, and the leader still produces its answer.

    The measured numbers below are what the implementation does, not a target:
    wave one is dispatched in full (2 of 3), wave two is admitted one child deep
    and refused on the second, and the run reaches a normal text completion
    afterwards. If the second wave were judged against a fresh cap it would run
    both, which is the unbounded behaviour this replaces.
    """
    engine = engine_factory(rc=_rc(runs=3))
    setattr(engine, "_helpers", {})  # noqa: B010
    tool = _ReservingDelegationTool(tool_name="Agent", description="delegate")
    in_memory_runtime["tools"].register(tool)

    for wave in ("a", "b"):
        _queue_multi_tool_stream(
            in_memory_runtime["llm"],
            tool_calls=[
                (f"call-{wave}1", "Agent", {"path": f"{wave}1"}),
                (f"call-{wave}2", "Agent", {"path": f"{wave}2"}),
            ],
        )
    in_memory_runtime["llm"].queue_response(text="final answer")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    dispatched = [call["path"] for call in tool.calls]
    assert dispatched == ["a1", "a2", "b1"], (
        "the cap must carry across waves: wave two sees one slot, not a fresh 3"
    )
    ledger = _resolve_run_work_ledger(engine)
    assert ledger.child_runs_started == 3

    texts = _tool_result_texts(engine)
    assert len(texts) == 4, "every requested call still gets a result"
    assert SUBAGENT_RUN_BUDGET_EXHAUSTED in texts[3]

    # The run finished normally after exhaustion — the budget refused the
    # delegation, not the answer. COMPLETED and specifically not FAILED: the
    # existing per-run token budget terminates the run when it trips, and this
    # one deliberately does not, because a leader that cannot write its answer
    # is a worse outcome than the runaway.
    assert engine.state is LoopState.COMPLETED
    final_text = "".join(
        block.text
        for msg in engine.history
        if msg.role is MessageRole.assistant
        for block in msg.content_blocks
        if isinstance(block, TextBlock)
    )
    assert "final answer" in final_text


@pytest.mark.asyncio
async def test_llm_usage_charges_the_tree_ledger(
    engine_factory, in_memory_runtime
) -> None:
    """Every LLM call the engine makes lands in the cumulative token total.

    Charged alongside ``engine.total_usage`` rather than derived from it: that
    counter belongs to ONE engine and every delegated child gets a fresh one, so
    it can never answer how much the whole tree has spent.

    Driven through ``queue_tool_call_response(usage_input_tokens=...)``, which is
    the fake provider's only route that emits a real ``usage`` STREAM delta —
    measured, not assumed: ``queue_response(input_tokens=...)`` sets usage on the
    response envelope and reaches neither ``total_usage`` nor the ledger, so a
    test written against it would have asserted 0 == 0 and pinned nothing.
    """
    engine = engine_factory(rc=_rc(tokens=0))
    setattr(engine, "_helpers", {})  # noqa: B010
    _register_delegation_tool(in_memory_runtime)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c-1",
        tool_name="Agent",
        tool_input={"path": "a"},
        usage_input_tokens=777,
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _evt in engine.run(user_msg):
        pass

    ledger = _resolve_run_work_ledger(engine)
    assert ledger.tokens_charged == 777
    assert ledger.tokens_charged == (
        engine.total_usage.input_tokens + engine.total_usage.output_tokens
    )
