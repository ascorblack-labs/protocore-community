"""Repeated-tool-error circuit breaker (core).

When a tool fails ``RuntimeConstants.max_consecutive_tool_errors`` times in a row
with the SAME error class, the core loop HARD-STOPS it for the rest of the run
(removed from the advertised surface AND denied at dispatch via
``effective_tool_policy.blocked``) and injects ONE bounded corrective
convergence turn so the model answers/finalises instead of storming. The
forensic trigger is the ``/project`` Read/Grep/Glob/List tools raising a hard
``ToolInvocationError`` on a non-project session — a tool that can NEVER succeed,
so the softer ``tool_dispatch_consecutive_error_cap`` ("try a different
tool/argument") never converges and the model loops 20-33 calls.

These tests drive :func:`protocore.runtime.query._dispatch_tool` end-to-end
against a real :class:`ToolDispatcher` (no the host plumbing) with a failing
``MockTool`` and assert the breaker trips at the cap, NOT before, resets on a
different error / success, fires the corrective turn at most once, and survives a
snapshot round-trip (cross-pod resume keeps the tool blocked without
re-injecting the message).
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import ToolInvocationError
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_CIRCUIT_BREAKER,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    Message,
    MessageRole,
    ToolCall,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query import (
    _apply_deferred_tool_history,
    _dispatch_tool,
    _resolve_max_consecutive_tool_errors,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.tool_dispatch import DispatchErrorKind, DispatchOutcome
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool

FAILING = "Read"  # mirror the /project tool family that storms


def _engine(
    *,
    rc: RuntimeConstants | None = None,
    tools: list[MockTool] | None = None,
    expected_terminal_tool: str | None = None,
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for tool in tools or []:
        registry.register(tool)
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-cb",
            tenant_id="tenant-cb",
            session_id="sess-cb",
            model_name="qwen3.6-35b-a3b",
            rc=rc or RuntimeConstants(model_context_window=4_096),
            expected_terminal_tool=expected_terminal_tool,
        ),
        llm_provider=object(),  # type: ignore[arg-type]  # never streamed in these tests
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    # The per-run helper bag is wired by the executor in production; here we
    # attach an empty one (the dispatcher reads adapters from it). The breaker's
    # in-flight streak lives on the ENGINE (``_circuit_breaker_streak``), not the
    # bag, so it is snapshot-persisted across resume.
    engine._helpers = {}  # type: ignore[attr-defined]
    return engine


def _failing_tool(exc: BaseException | None = None) -> MockTool:
    return MockTool(
        tool_name=FAILING,
        raise_exception=exc or ToolInvocationError(
            "project knowledge is not attached to this session; do not retry"
        ),
    )


def _call(call_id: str, *, name: str = FAILING) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"v": "x"})


async def _dispatch(engine: QueryEngine, call: ToolCall) -> list[TurnEvent]:
    return [evt async for evt in _dispatch_tool(engine, call)]


def _corrective_turns(engine: QueryEngine) -> list[Message]:
    return [
        m
        for m in engine.history
        if m.role is MessageRole.user
        and m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_CIRCUIT_BREAKER
    ]


@pytest.mark.asyncio
async def test_breaker_trips_after_cap_consecutive_same_tool_same_error() -> None:
    """Cap (default 3) consecutive identical failures → tool blocked + ONE
    corrective convergence turn injected on the trip dispatch."""
    engine = _engine(tools=[_failing_tool()])
    cap = engine.config.rc.max_consecutive_tool_errors
    assert cap == 3

    # Failures 1..cap-1: streak builds, breaker does NOT trip.
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"t{i}"))
        assert FAILING not in engine._circuit_broken_tools
        assert _corrective_turns(engine) == []

    # Failure #cap: trips.
    await _dispatch(engine, _call(f"t{cap}"))
    assert FAILING in engine._circuit_broken_tools
    corr = _corrective_turns(engine)
    assert len(corr) == 1
    text = corr[0].content_blocks[0].text  # type: ignore[union-attr]
    assert FAILING in text
    # Bilingual + forces convergence (answer from the conversation, don't retry).
    assert "disabled" in text and "do NOT call it again" in text
    assert "отключён" in text


@pytest.mark.asyncio
async def test_breaker_does_not_trip_below_cap() -> None:
    """cap-1 identical failures must NOT trip the breaker (single/normal errors
    keep flowing as tool_result(is_error=True))."""
    engine = _engine(tools=[_failing_tool()])
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap - 1):
        events = await _dispatch(engine, _call(f"t{i}"))
        # The dispatch still surfaces a normal error tool_result.
        assert any(e.type is EventType.TOOL_RESULT for e in events)
    assert FAILING not in engine._circuit_broken_tools
    assert _corrective_turns(engine) == []
    # The tool is still offered + dispatchable.
    assert FAILING not in engine.effective_tool_policy.blocked


@pytest.mark.asyncio
async def test_breaker_blocks_tool_from_surface_and_dispatch_after_trip() -> None:
    """Once tripped, the tool is unioned into ``effective_tool_policy.blocked``
    so it vanishes from the surface AND a further dispatch is denied by the
    permission gate (never reaching the tool body again)."""
    tool = _failing_tool()
    engine = _engine(tools=[tool])
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap):
        await _dispatch(engine, _call(f"t{i}"))
    assert FAILING in engine.effective_tool_policy.blocked
    # Surface no longer advertises it.
    surface = engine.tools.compute_effective_surface(
        tenant_id=engine.config.tenant_id, policy=engine.effective_tool_policy
    )
    assert FAILING not in {d.name for d in surface}

    invocations_before = len(tool.calls)
    events = await _dispatch(engine, _call("t-after"))
    # The gate denied it: the tool body did NOT run again.
    assert len(tool.calls) == invocations_before
    # A denial tool_result is still appended (pairing stays valid) ...
    assert any(e.type is EventType.TOOL_RESULT for e in events)
    # ... and the corrective turn is NOT re-injected.
    assert len(_corrective_turns(engine)) == 1


@pytest.mark.asyncio
async def test_breaker_does_not_trip_on_mixed_errors() -> None:
    """A DIFFERENT error class on the same tool resets the streak — the breaker
    must only fire on N CONSECUTIVE same-class failures."""
    # ToolInvocationError → DispatchErrorKind.execution; ToolPolicyDenied →
    # DispatchErrorKind.permission (a different class), so alternating them never
    # accumulates ``cap`` of the SAME class in a row.
    from protocore.contracts.tools import ToolPolicyDenied

    perm = MockTool(tool_name=FAILING)  # reused below by swapping the exception
    engine = _engine(tools=[perm])
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap * 2):
        # Alternate the error CLASS every call.
        if i % 2 == 0:
            perm.raise_exception = ToolInvocationError("exec-style failure")
        else:
            perm.raise_exception = ToolPolicyDenied("permission-style failure")
        await _dispatch(engine, _call(f"t{i}"))
    assert FAILING not in engine._circuit_broken_tools
    assert _corrective_turns(engine) == []


@pytest.mark.asyncio
async def test_success_resets_streak() -> None:
    """A successful call clears the streak so a later error storm restarts at
    1 — a tool that recovers must not stay one error from the breaker."""
    tool = MockTool(tool_name=FAILING)
    engine = _engine(tools=[tool])
    cap = engine.config.rc.max_consecutive_tool_errors

    # cap-1 failures ...
    tool.raise_exception = ToolInvocationError("transient")
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"f{i}"))
    assert FAILING not in engine._circuit_broken_tools

    # ... then a SUCCESS resets ...
    tool.raise_exception = None
    await _dispatch(engine, _call("ok"))
    assert FAILING not in engine._circuit_broken_tools

    # ... so the next cap-1 failures STILL do not trip (streak restarted at 1).
    tool.raise_exception = ToolInvocationError("transient")
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"g{i}"))
    assert FAILING not in engine._circuit_broken_tools
    assert _corrective_turns(engine) == []


@pytest.mark.asyncio
async def test_breaker_state_survives_snapshot_round_trip() -> None:
    """The broken-tool set + the at-most-once notify latch are snapshot-persisted
    so a cross-pod resume keeps the tool blocked and does NOT re-inject the
    corrective turn."""
    engine = _engine(tools=[_failing_tool()])
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap):
        await _dispatch(engine, _call(f"t{i}"))
    assert FAILING in engine._circuit_broken_tools

    snapshot = engine.snapshot()
    assert snapshot["circuit_broken_tools"] == [FAILING]
    assert snapshot["circuit_breaker_notified_tools"] == [FAILING]

    # Fresh engine (new pod) rehydrated from the snapshot. A fresh tool
    # instance lets us prove the gate denies dispatch WITHOUT the body running.
    resumed_tool = _failing_tool()
    resumed = _engine(tools=[resumed_tool])
    await resumed.resume_from_snapshot(snapshot)
    assert resumed._circuit_broken_tools == {FAILING}
    assert resumed._circuit_breaker_notified_tools == {FAILING}
    # Still blocked on the resumed engine ...
    assert FAILING in resumed.effective_tool_policy.blocked
    # The snapshot history carries the ORIGINAL corrective turn (it lives in the
    # live engine history; only the durable-persistence filter strips it), so
    # exactly one is present after resume.
    assert len(_corrective_turns(resumed)) == 1
    # ... and a further call is GATE-DENIED (tool body never runs again) and
    # does NOT inject a SECOND corrective turn (the notify latch survived).
    await _dispatch(resumed, _call("t-resumed"))
    assert resumed_tool.calls == []  # denied before the body
    assert len(_corrective_turns(resumed)) == 1


@pytest.mark.asyncio
async def test_default_cap_value_is_three() -> None:
    """Lock the documented default (trips on the 3rd identical failure)."""
    assert RuntimeConstants().max_consecutive_tool_errors == 3
    assert _resolve_max_consecutive_tool_errors(_engine()) == 3


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_ineligible_soft_errors_do_not_trip_breaker() -> None:
    """A tool result that opted OUT of consecutive-error capping
    (``consecutive_error_cap_eligible=False``, e.g. a benign Bash nonzero exit
    used as DATA: ``grep -q`` no-match, ``test`` false) must NOT advance the hard
    breaker. Repeated such soft errors must never block the tool nor inject the
    corrective turn."""
    bash = MockTool(
        tool_name="Bash",
        response_is_error=True,  # soft error result (no raise)
        response_metadata={TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY: False},
    )
    engine = _engine(tools=[bash])
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap * 3):  # well past the cap
        events = await _dispatch(engine, _call(f"b{i}", name="Bash"))
        # The soft error still surfaces as a tool_result(is_error=True).
        assert any(e.type is EventType.TOOL_RESULT for e in events)
    assert "Bash" not in engine._circuit_broken_tools
    assert "Bash" not in engine.effective_tool_policy.blocked
    assert _corrective_turns(engine) == []
    # The in-flight streak was never advanced for the ineligible errors.
    assert engine._circuit_breaker_streak is None


@pytest.mark.asyncio
async def test_success_of_different_tool_resets_streak() -> None:
    """A SUCCESS of ANY tool breaks the consecutive chain:
    ``Read(err) → List(ok) → Read(err) → Read(err)`` must NOT trip at cap 3,
    because the Read failures were not consecutive."""
    read = _failing_tool()
    other = MockTool(tool_name="List")  # always succeeds
    engine = _engine(tools=[read, other])
    cap = engine.config.rc.max_consecutive_tool_errors
    assert cap == 3

    # Read fails cap-1 times ...
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"r{i}"))
    assert FAILING not in engine._circuit_broken_tools

    # ... a DIFFERENT tool succeeds (resets the streak) ...
    await _dispatch(engine, _call("ok", name="List"))
    assert engine._circuit_breaker_streak is None

    # ... then cap-1 more Read failures STILL do not trip (restarted at 1).
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"r2-{i}"))
    assert FAILING not in engine._circuit_broken_tools
    assert _corrective_turns(engine) == []


@pytest.mark.asyncio
async def test_in_flight_streak_survives_snapshot_resume_and_trips() -> None:
    """The pre-trip streak is snapshot-persisted, so a cross-pod
    resume at cap-1 failures keeps the count: ONE more failure on the resumed
    engine trips. (Before the fix a fresh helper bag reset the count and the run
    could exceed the cap without tripping.)"""
    engine = _engine(tools=[_failing_tool()])
    cap = engine.config.rc.max_consecutive_tool_errors

    # cap-1 failures — under the cap, no trip yet.
    for i in range(cap - 1):
        await _dispatch(engine, _call(f"t{i}"))
    assert FAILING not in engine._circuit_broken_tools
    # The in-flight streak is captured in the snapshot (NOT just the helper bag).
    snapshot = engine.snapshot()
    assert snapshot["circuit_breaker_streak"] == {
        "tool_name": FAILING,
        "error_class": DispatchErrorKind.execution.value,
        "count": cap - 1,
    }

    # Resume on a fresh pod (fresh helper bag) ...
    resumed = _engine(tools=[_failing_tool()])
    await resumed.resume_from_snapshot(snapshot)
    assert resumed._circuit_breaker_streak == {
        "tool_name": FAILING,
        "error_class": DispatchErrorKind.execution.value,
        "count": cap - 1,
    }
    assert FAILING not in resumed._circuit_broken_tools

    # ... ONE more failure trips (count reaches cap).
    await _dispatch(resumed, _call("t-final"))
    assert FAILING in resumed._circuit_broken_tools
    assert len(_corrective_turns(resumed)) == 1


@pytest.mark.asyncio
async def test_streak_restore_rejects_bool_count() -> None:
    """``bool`` is an ``int`` subclass, so a malformed snapshot
    ``count: True`` must degrade to an EMPTY streak (not restore a non-empty
    one). Also reject a non-dict / negative / wrong-typed shape."""
    base = _engine().snapshot()

    async def _resume(streak: object) -> QueryEngine:
        resumed = _engine()
        snap = dict(base)
        snap["circuit_breaker_streak"] = streak
        await resumed.resume_from_snapshot(snap)
        return resumed

    # bool count → empty (the crux).
    r = await _resume({"tool_name": "Read", "error_class": "execution", "count": True})
    assert r._circuit_breaker_streak is None
    # negative count / missing keys / non-dict → empty.
    assert (
        await _resume(
            {"tool_name": "Read", "error_class": "execution", "count": -1}
        )
    )._circuit_breaker_streak is None
    assert (await _resume({"tool_name": "Read"}))._circuit_breaker_streak is None
    assert (await _resume("nonsense"))._circuit_breaker_streak is None
    # A well-shaped int count is still restored.
    ok = await _resume(
        {"tool_name": "Read", "error_class": "execution", "count": 2}
    )
    assert ok._circuit_breaker_streak == {
        "tool_name": "Read",
        "error_class": "execution",
        "count": 2,
    }


@pytest.mark.asyncio
async def test_terminal_only_deferred_history_does_not_track_breaker() -> None:
    """The parallel terminal-only finalize-gate veto routes a
    SYNTHETIC is_error outcome through ``_apply_deferred_tool_history`` with
    ``track_circuit_breaker=False``; it must NOT advance the breaker or inject a
    corrective turn (mirrors the serial path returning before breaker tracking),
    so the finalize-background gate is never disrupted."""
    engine = _engine(expected_terminal_tool="Finalize")
    cap = engine.config.rc.max_consecutive_tool_errors
    for i in range(cap * 2):
        synthetic = DispatchOutcome(
            tool_call=_call(f"x{i}", name="Read"),
            success=False,
            content="terminal_only: write your final answer then call Finalize",
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            metadata={},
        )
        _apply_deferred_tool_history(
            engine,
            _call(f"x{i}", name="Read"),
            synthetic,
            track_circuit_breaker=False,
        )
    assert "Read" not in engine._circuit_broken_tools
    assert engine._circuit_breaker_streak is None
    assert _corrective_turns(engine) == []

    # Sanity: with tracking ON (the normal parallel path), the SAME outcomes DO
    # advance the breaker — proving the flag is the gate, not some other guard.
    engine2 = _engine()
    for i in range(cap):
        outcome = DispatchOutcome(
            tool_call=_call(f"y{i}", name="Read"),
            success=False,
            content="boom",
            is_error=True,
            error_kind=DispatchErrorKind.execution,
            metadata={},
        )
        _apply_deferred_tool_history(engine2, _call(f"y{i}", name="Read"), outcome)
    assert "Read" in engine2._circuit_broken_tools
    assert len(_corrective_turns(engine2)) == 1
