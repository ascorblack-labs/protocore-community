"""Terminal candidate-answer preservation.

ENGAGEMENT-TEST PATTERN — READ BEFORE EDITING
==============================================
Engagement tests drive the REAL dispatched path and assert the engagement
DIAG/behaviour fires there. A helper-in-isolation test (calling e.g.
``_resolve_terminal_candidate_corrective`` directly) is NOT sufficient: that is
exactly the gap that can let tests pass green while the real
``_dispatch_tool`` path is unverified — the pre-dispatch terminal-verify gate
can run SILENTLY on its no-veto path and the silence causes mis-diagnoses. Any
new feature/gate ships a test that goes through ``_dispatch_tool`` (via
``_drive_dispatch`` below) and asserts the engagement DIAG actually fires, on
both the firing and the non-firing branch. See the heartbeat tests at the foot
of this file for the canonical shape.

The repair-turn cause-fix. When the pre-dispatch terminal verify seam withholds
the model's first SUBSTANTIVE terminal answer, that draft must not be silently
lost if the repair turn regresses to an empty / 1-char body. These tests pin
``_resolve_terminal_candidate_corrective`` plus the durable engine state it
writes:

* default-off (``terminal_candidate_preserve_enabled=False``) is bit-identical
  to today — the candidate is discarded, the verdict is returned unchanged;
* enabled — the first substantive draft is preserved durably across the veto
  and survives a ``snapshot()`` / ``resume_from_snapshot()`` round-trip
  (horizontal / cross-pod safe);
* the repair-turn regression (a substantive first answer followed by a 1-char
  repair body) is re-vetoed exactly once, then allowed through once the one-shot
  repair credit is spent.

Pure engine-state / pure-function unit tests — no live provider, no executor
pod (mirrors ``test_terminal_finalize.py``).
"""
from __future__ import annotations

import asyncio
import logging

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import MessageRole, ToolCall
from protocore.runtime.query import (
    _resolve_terminal_candidate_corrective,
    _terminal_candidate_hash,
    _terminal_candidate_message,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

SUBSTANTIVE = "Here is a complete, substantive answer to the user's question."
CORRECTIVE = "fix the cited refs before resubmitting"


def _engine(
    *,
    rc_kwargs: dict | None = None,
    expected_terminal_tool: str | None = "final_answer",
) -> QueryEngine:
    """Real :class:`QueryEngine` wired to in-memory adapters (mirrors the
    ``_build_terminal_engine`` helper in ``test_terminal_finalize.py``).
    """
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-u6",
            tenant_id="tenant-u6",
            session_id="session-u6",
            model_name="qwen3.6-35b-a3b",
            #  — prose-gate stays at its DEFAULT: ``final_answer``
            # is unregistered in this in-memory runtime, so the schema-conditioned
            # gate treats it as unknown-schema → EXEMPT (no RC override needed).
            rc=RuntimeConstants(**(rc_kwargs or {})),
            expected_terminal_tool=expected_terminal_tool,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _call(message: str, name: str = "final_answer") -> ToolCall:
    return ToolCall(name=name, arguments={"message": message})


# ---------------------------------------------------------------------------
# message extraction helper
# ---------------------------------------------------------------------------

def test_message_extraction_handles_non_string_and_missing():
    assert _terminal_candidate_message(_call("  hi  ")) == "hi"
    assert _terminal_candidate_message(ToolCall(name="final_answer", arguments={})) == ""
    bad = ToolCall(name="final_answer", arguments={"message": 123})
    assert _terminal_candidate_message(bad) == ""


def test_hash_is_stable_and_content_bound():
    assert _terminal_candidate_hash("abc") == _terminal_candidate_hash("abc")
    assert _terminal_candidate_hash("abc") != _terminal_candidate_hash("abd")


# ---------------------------------------------------------------------------
# default-off: bit-identical to today (candidate discarded)
# ---------------------------------------------------------------------------

def test_default_off_returns_corrective_unchanged_and_persists_nothing():
    eng = _engine()  # preserve flag defaults False
    out = _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    assert out == CORRECTIVE  # verdict untouched
    assert eng._terminal_candidate is None
    assert eng._terminal_candidate_reveto_used is False


def test_default_off_passthrough_none_verdict():
    eng = _engine()
    assert _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), None) is None
    assert eng._terminal_candidate is None


# ---------------------------------------------------------------------------
# enabled: first substantive draft preserved across the veto
# ---------------------------------------------------------------------------

def test_enabled_preserves_first_substantive_candidate_on_veto():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    out = _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    assert out == CORRECTIVE  # veto still fires
    saved = eng._terminal_candidate
    assert isinstance(saved, dict)
    assert saved["substantive"] is True
    assert saved["tool"] == "final_answer"
    assert saved["veto_reason"] == "pre_dispatch_terminal_verify"
    assert saved["message_chars"] == len(SUBSTANTIVE)
    assert saved["candidate_hash"] == _terminal_candidate_hash(SUBSTANTIVE)
    assert saved["args"] == {"message": SUBSTANTIVE}


def test_enabled_does_not_preserve_when_not_vetoed():
    # A substantive body that passes verification (corrective None) is NOT a
    # withheld draft — there is nothing to preserve.
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    assert _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), None) is None
    assert eng._terminal_candidate is None


def test_enabled_does_not_preserve_non_substantive_body():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    # 5-char body, below the 20-char floor -> not substantive -> not preserved.
    out = _resolve_terminal_candidate_corrective(eng, _call("short"), CORRECTIVE)
    assert out == CORRECTIVE
    assert eng._terminal_candidate is None


def test_min_chars_zero_treats_any_nonempty_as_substantive():
    eng = _engine(rc_kwargs={"terminal_candidate_preserve_enabled": True})  # floor 0 -> 1
    out = _resolve_terminal_candidate_corrective(eng, _call("x"), CORRECTIVE)
    assert out == CORRECTIVE
    assert eng._terminal_candidate is not None
    assert eng._terminal_candidate["message_chars"] == 1


def test_substantive_candidate_not_clobbered_by_later_veto():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    first_hash = eng._terminal_candidate["candidate_hash"]
    # A second, different substantive body that is also vetoed must NOT
    # overwrite the first preserved draft.
    _resolve_terminal_candidate_corrective(
        eng, _call(SUBSTANTIVE + " variation two"), CORRECTIVE
    )
    assert eng._terminal_candidate["candidate_hash"] == first_hash


# ---------------------------------------------------------------------------
# snapshot round-trip (durable / cross-pod safe)
# ---------------------------------------------------------------------------

def test_preserved_candidate_and_latch_survive_snapshot_roundtrip():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    # also trip the re-veto latch via a regression
    _resolve_terminal_candidate_corrective(eng, _call("x"), CORRECTIVE)
    assert eng._terminal_candidate_reveto_used is True

    snap = eng.snapshot()
    assert snap["terminal_candidate"]["substantive"] is True
    assert snap["terminal_candidate_reveto_used"] is True

    eng2 = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    asyncio.run(eng2.resume_from_snapshot(snap))
    assert isinstance(eng2._terminal_candidate, dict)
    assert eng2._terminal_candidate["candidate_hash"] == (
        eng._terminal_candidate["candidate_hash"]
    )
    assert eng2._terminal_candidate_reveto_used is True


def test_snapshot_roundtrip_defaults_when_absent():
    # An engine that never preserved a candidate round-trips to None / False.
    eng = _engine()
    eng2 = _engine()
    asyncio.run(eng2.resume_from_snapshot(eng.snapshot()))
    assert eng2._terminal_candidate is None
    assert eng2._terminal_candidate_reveto_used is False


# ---------------------------------------------------------------------------
# the repair-turn reproduction: substantive draft -> 1-char repair -> regression
# ---------------------------------------------------------------------------

def test_regression_to_one_char_is_revetoed_once_then_allowed():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })

    # Turn 1: substantive answer, withheld by the veto -> preserved.
    out1 = _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    assert out1 == CORRECTIVE
    assert eng._terminal_candidate["substantive"] is True

    # Turn 2: model regresses to a 1-char body. Even if the host trigger
    # would now let it through (corrective None), preservation re-vetoes once.
    out2 = _resolve_terminal_candidate_corrective(eng, _call("x"), None)
    assert out2 is not None  # forced re-veto, reusing the fallback wording
    assert eng._terminal_candidate_reveto_used is True

    # Turn 3: still regressed, repair credit spent -> allow through.
    out3 = _resolve_terminal_candidate_corrective(eng, _call("x"), CORRECTIVE)
    assert out3 is None  # allow-through overrides any corrective
    assert eng._terminal_candidate_reveto_used is True


def test_regression_reveto_prefers_existing_corrective_text():
    # When the trigger DID return a corrective for the regressed turn, the
    # re-veto reuses THAT text (persona frozen) rather than the fallback.
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    out = _resolve_terminal_candidate_corrective(eng, _call(""), CORRECTIVE)
    assert out == CORRECTIVE


def test_empty_repair_body_is_a_regression():
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    out = _resolve_terminal_candidate_corrective(eng, _call(""), None)
    assert out is not None  # empty body re-vetoed
    assert eng._terminal_candidate_reveto_used is True


def test_substantive_followup_after_preserve_is_not_a_regression():
    # If the repair turn is itself substantive, it must NOT be treated as a
    # regression — the normal verdict applies (here: passes -> None).
    eng = _engine(rc_kwargs={
        "terminal_candidate_preserve_enabled": True,
        "terminal_answer_min_message_chars": 20,
    })
    _resolve_terminal_candidate_corrective(eng, _call(SUBSTANTIVE), CORRECTIVE)
    out = _resolve_terminal_candidate_corrective(
        eng, _call("A different but equally complete answer here."), None
    )
    assert out is None


# ---------------------------------------------------------------------------
# Integration test THROUGH ``_dispatch_tool`` (NOT the helper).
#
# The helper-only tests above call ``_resolve_terminal_candidate_corrective``
# directly and PASS — but they never proved the real dispatch path runs the
# regression check on the REPAIR turn. The pre-dispatch terminal veto is
# fire-at-most-once (it latches ``_pre_dispatch_terminal_verify_used`` on the
# first veto), so the candidate-regression check WIRED INSIDE that gate never
# re-runs on the model's corrected re-submission. The regression: the first
# substantive ``final_answer`` is withheld, a draft is preserved, then a 1-char
# repair body dispatches UNGUARDED. These tests drive the actual
# ``_dispatch_tool`` async generator end-to-end to pin the fix.
# ---------------------------------------------------------------------------

def _veto_engine() -> QueryEngine:
    """Engine with the pre-dispatch veto seam ARMED and candidate preservation
    on (the per-tenant shape). The trigger always vetoes so the first
    terminal answer is withheld and preserved; the floor is 20 chars so a
    1-char repair body is a regression.
    """

    # ``QueryEngineConfig`` is a frozen dataclass, so the always-veto trigger
    # must be wired at construction time (not assigned after).
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-u6",
            tenant_id="tenant-u6",
            session_id="session-u6",
            model_name="qwen3.6-35b-a3b",
            #  — prose-gate at DEFAULT: unregistered
            # ``final_answer`` ⟹ unknown-schema ⟹ EXEMPT (no RC override needed).
            rc=RuntimeConstants(
                pre_dispatch_terminal_verify_enabled=True,
                pre_terminal_self_verify_max_extra_turns=1,
                terminal_candidate_preserve_enabled=True,
                terminal_answer_min_message_chars=20,
            ),
            expected_terminal_tool="final_answer",
            # Host-supplied trigger: always veto (refs "wrong").
            pre_dispatch_terminal_verify_trigger=lambda _engine, _tool_call: CORRECTIVE,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _drive_dispatch(eng: QueryEngine, tool_call: ToolCall) -> list:
    """Run the real ``_dispatch_tool`` async generator, return its events."""

    from protocore.runtime.query import _dispatch_tool

    async def _run() -> list:
        return [evt async for evt in _dispatch_tool(eng, tool_call)]

    return asyncio.run(_run())


def _result_error_kind(events: list) -> str | None:
    """The ``error.kind`` of the single TOOL_RESULT event, or None."""

    from protocore.runtime.events import EventType

    results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(results) == 1, f"expected exactly one TOOL_RESULT, got {len(results)}"
    return (results[0].payload.get("error") or {}).get("kind")


def test_dispatch_tool_repair_turn_is_revetoed_after_first_veto():
    """The repair-turn regression, through the real dispatch path.

    Turn 1 (substantive answer) is vetoed by the pre-dispatch seam and the
    draft is preserved; the pre-dispatch one-shot latch closes. Turn 2 emits a
    1-char repair body — the pre-dispatch gate is now CLOSED, so before the fix
    the 1-char body dispatched unguarded (reaching the dispatcher). With the
    fix the independent repair seam re-vetoes it exactly once.
    """

    eng = _veto_engine()

    # --- Turn 1: substantive answer -> vetoed, preserved, pre-dispatch latch set.
    events1 = _drive_dispatch(eng, _call(SUBSTANTIVE))
    assert _result_error_kind(events1) == "pre_dispatch_terminal_verify"
    assert eng._pre_dispatch_terminal_verify_used is True
    assert isinstance(eng._terminal_candidate, dict)
    assert eng._terminal_candidate["substantive"] is True
    # The corrective user turn was injected after the veto tool_result.
    assert eng.history[-1].role is MessageRole.user
    assert eng._terminal_candidate_reveto_used is False

    # --- Turn 2: 1-char repair body. The pre-dispatch gate is CLOSED now
    #     (latch set on turn 1) AND the self-verify budget is spent, so the OLD
    #     code never re-ran the regression check here and the 1-char answer
    #     dispatched. The fix re-vetoes via the INDEPENDENT repair seam.
    from protocore.runtime.query import _pre_dispatch_terminal_verify_applies

    one_char = _call("x")
    assert _pre_dispatch_terminal_verify_applies(eng, one_char) is False  # gate closed
    events2 = _drive_dispatch(eng, one_char)
    assert _result_error_kind(events2) == "terminal_candidate_repair_reveto"
    assert eng._terminal_candidate_reveto_used is True  # one-shot consumed
    # The repair seam injected its own corrective user turn (the model is told
    # to re-send its full answer); the terminal tool was NOT dispatched.
    assert eng.history[-1].role is MessageRole.user

    # --- Turn 3: still 1-char, repair credit spent -> allow-through. The seam
    #     no longer re-vetoes; the body dispatches normally (in this in-memory
    #     runtime ``final_answer`` is unregistered, so it surfaces as a
    #     dispatcher error — but crucially NOT the repair-reveto kind).
    events3 = _drive_dispatch(eng, _call("y"))
    assert _result_error_kind(events3) != "terminal_candidate_repair_reveto"


def test_dispatch_tool_substantive_repair_is_not_revetoed():
    """A correct substantive re-answer after the first veto MUST pass through.

    The repair seam keys regression on body length, not on identity, so a
    genuinely substantive corrected answer is allowed to dispatch (no
    re-veto), and the one-shot repair latch is NOT spent.
    """

    eng = _veto_engine()

    # Turn 1: substantive -> veto + preserve.
    _drive_dispatch(eng, _call(SUBSTANTIVE))
    assert eng._terminal_candidate["substantive"] is True

    # Turn 2: a DIFFERENT substantive body (>= 20 chars). Not a regression, so
    # the repair seam does not re-veto and does not burn the one-shot latch.
    events2 = _drive_dispatch(
        eng, _call("A second, fully substantive corrected answer body.")
    )
    assert _result_error_kind(events2) != "terminal_candidate_repair_reveto"
    assert eng._terminal_candidate_reveto_used is False


def test_dispatch_tool_repair_seam_inert_when_preserve_disabled():
    """Default-off: the repair seam never fires (bit-identical to the
    pre-preservation behaviour).

    With ``terminal_candidate_preserve_enabled=False`` (and the pre-dispatch
    veto also off) a 1-char terminal answer dispatches with no candidate seam
    interference at all — proving the new branch is fully gated.
    """

    eng = _engine()  # all candidate-preserve flags default off
    events = _drive_dispatch(eng, _call("x"))
    assert _result_error_kind(events) != "terminal_candidate_repair_reveto"
    assert eng._terminal_candidate is None
    assert eng._terminal_candidate_reveto_used is False


# ---------------------------------------------------------------------------
# HEARTBEAT DIAG on the gate's True-but-no-veto path.
#
# These are ENGAGEMENT tests (see the module docstring): they drive the REAL
# ``_dispatch_tool`` path and assert the unconditional heartbeat DIAG
# ``query.pre_dispatch_terminal_verify.applied`` fires on BOTH a veto and a
# no-veto terminal dispatch. The night of 2026-06-01 failed precisely because
# the gate's no-veto path was silent and only a helper-in-isolation test
# existed — so the gate's engagement was unobservable both in tests and in
# prod. A regression here means the heartbeat (the linchpin that makes every
# later fix verifiable) has gone silent again.
# ---------------------------------------------------------------------------

# Uses the runtime-facing observed-state helper-bag key so the test exercises
# the same opaque-bag read the heartbeat performs in production.
_LEDGER_HELPER_KEY = "terminal_answer_observed_refs"


def _no_veto_engine() -> QueryEngine:
    """Engine with the pre-dispatch verify gate ARMED but the trigger declining.

    The gate APPLIES (``pre_dispatch_terminal_verify_enabled=True`` +
    ``expected_terminal_tool`` matches + budget open + trigger wired), but the
    host-supplied trigger returns ``None`` (every cited ref grounded), so
    the terminal dispatch is ALLOWED. ``terminal_candidate_preserve_enabled`` is
    left OFF so the candidate-repair layer cannot turn the ``None`` verdict into
    a re-veto — the heartbeat must report ``verdict=no_veto``.
    """

    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-hb-noveto",
            tenant_id="tenant-u6",
            session_id="session-u6",
            model_name="qwen3.6-35b-a3b",
            #  — prose-gate at DEFAULT: unregistered
            # ``final_answer`` ⟹ unknown-schema ⟹ EXEMPT (no RC override needed).
            rc=RuntimeConstants(
                pre_dispatch_terminal_verify_enabled=True,
                pre_terminal_self_verify_max_extra_turns=1,
            ),
            expected_terminal_tool="final_answer",
            # Host-supplied trigger: never veto (all cited refs grounded).
            pre_dispatch_terminal_verify_trigger=lambda _engine, _tool_call: None,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _heartbeat_records(caplog) -> list:
    """Heartbeat DIAG records captured this test (``...applied`` only)."""

    return [
        r
        for r in caplog.records
        if "query.pre_dispatch_terminal_verify.applied" in r.getMessage()
    ]


def test_heartbeat_diag_fires_on_no_veto_dispatch(caplog):
    """The gate ran and DECLINED to veto -> exactly one heartbeat, verdict=no_veto.

    This is THE path that was silent on 2026-06-01: the gate applied, the
    trigger found every cited ref grounded, dispatch proceeded, and NOTHING was
    logged. The heartbeat now fires here. Driven through the real
    ``_dispatch_tool`` async generator (not the helper in isolation).
    """

    eng = _no_veto_engine()
    # Seed the opaque helper bag with an observed-ref ledger so the heartbeat's
    # best-effort ``observed=M`` read off ``engine._helpers`` is exercised.
    eng._helpers = {_LEDGER_HELPER_KEY: {"/proc/a.json", "/proc/b.json"}}
    call = ToolCall(
        name="final_answer",
        arguments={"message": SUBSTANTIVE, "refs": ["/proc/a.json"]},
    )

    with caplog.at_level(logging.WARNING, logger="protocore.runtime.query"):
        events = _drive_dispatch(eng, call)

    # The gate did NOT veto -> the dispatch proceeded to the dispatcher (where
    # ``final_answer`` is unregistered in this in-memory runtime), so the result
    # is NOT a pre-dispatch-verify withholding.
    assert _result_error_kind(events) != "pre_dispatch_terminal_verify"
    assert eng._pre_dispatch_terminal_verify_used is False  # no veto -> no latch

    records = _heartbeat_records(caplog)
    assert len(records) == 1, f"expected one heartbeat, got {len(records)}"
    msg = records[0].getMessage()
    assert "verdict=no_veto" in msg
    assert "run=run-hb-noveto" in msg
    assert "cited=1" in msg  # one ref on the call
    assert "observed=2" in msg  # ledger seeded with two paths


def test_heartbeat_diag_fires_on_veto_dispatch(caplog):
    """The gate ran and VETOED -> exactly one heartbeat, verdict=veto.

    Uses the always-veto ``_veto_engine`` (the per-tenant shape). The heartbeat
    fires on the SAME application as the existing
    ``pre_dispatch_terminal_verify.vetoed`` DIAG, but unconditionally and
    BEFORE it. Driven through the real ``_dispatch_tool`` path.
    """

    eng = _veto_engine()
    call = ToolCall(
        name="final_answer",
        arguments={"message": SUBSTANTIVE, "refs": ["/proc/x.json", "/proc/y.json"]},
    )

    with caplog.at_level(logging.WARNING, logger="protocore.runtime.query"):
        events = _drive_dispatch(eng, call)

    # The veto fired (submission withheld).
    assert _result_error_kind(events) == "pre_dispatch_terminal_verify"

    records = _heartbeat_records(caplog)
    assert len(records) == 1, f"expected one heartbeat, got {len(records)}"
    msg = records[0].getMessage()
    assert "verdict=veto" in msg
    assert "run=run-u6" in msg
    assert "cited=2" in msg  # two refs on the call
    # No ledger seeded on the veto engine -> best-effort read reports the
    # "unavailable" sentinel rather than a misleading 0.
    assert "observed=-1" in msg


def test_heartbeat_diag_silent_when_gate_does_not_apply(caplog):
    """Default-off tenant: the gate never applies -> NO heartbeat at all.

    Proves the heartbeat is fully gated by the gate's own preconditions
    (here ``pre_dispatch_terminal_verify_enabled`` is unset), so a
    default-off tenant's executor log is bit-identical / silent.
    """

    eng = _engine()  # all gate flags default off
    with caplog.at_level(logging.WARNING, logger="protocore.runtime.query"):
        _drive_dispatch(eng, _call("x"))
    assert _heartbeat_records(caplog) == []
