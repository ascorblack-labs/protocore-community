"""Unit tests — terminal payload normalize, deadline early finalize predicate,
pre-terminal self-verify latch, parallel-read RC defaults, and the generic
read-dedup tool key.

These cover the pure-core primitives + the engine-facing helpers in isolation.
The host apply-points (``final_answer`` normalize call, the self-verify
trigger implementation, the ledger lock acquisition, the read-dedup tool
wiring) are out of core scope and covered host-side.
"""

from __future__ import annotations

import time

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.read_dedup_cache import ReadDedupCache
from protocore.runtime.terminal_payload_normalize import normalize_terminal_text


# --------------------------------------------------------------------------
# terminal payload entity normalization
# --------------------------------------------------------------------------
def test_entity_unescape_marker() -> None:
    assert (
        normalize_terminal_text("&lt;YES&gt; FST-APSRIZJW", entity_unescape=True)
        == "<YES> FST-APSRIZJW"
    )


def test_entity_named_and_numeric_refs() -> None:
    assert normalize_terminal_text("a &amp; b", entity_unescape=True) == "a & b"
    assert (
        normalize_terminal_text("x &#60;NO&#62; y", entity_unescape=True)
        == "x <NO> y"
    )


def test_entity_unescape_disabled_is_noop() -> None:
    assert (
        normalize_terminal_text("&lt;YES&gt;", entity_unescape=False) == "&lt;YES&gt;"
    )


def test_entity_none_and_empty_passthrough() -> None:
    assert normalize_terminal_text(None, entity_unescape=True) is None
    assert normalize_terminal_text("", entity_unescape=True) == ""


def test_entity_single_pass_does_not_overunescape() -> None:
    # Double-escaped input unescapes exactly one level (not looped) so a
    # message legitimately containing an entity-looking substring is safe.
    assert normalize_terminal_text("&amp;lt;", entity_unescape=True) == "&lt;"


def test_entity_sentinels_non_mutating_today() -> None:
    assert (
        normalize_terminal_text(
            "&lt;YES&gt;", entity_unescape=True, sentinels=("<YES>", "<NO>")
        )
        == "<YES>"
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_entity_unescape_gate_param(enabled: bool) -> None:
    out = normalize_terminal_text("&lt;X&gt;", entity_unescape=enabled)
    assert out == ("<X>" if enabled else "&lt;X&gt;")


# --------------------------------------------------------------------------
# RC defaults reproduce today
# --------------------------------------------------------------------------
def test_rc_defaults_reproduce_today() -> None:
    rc = RuntimeConstants()
    # deadline-finalize disabled by default
    assert rc.agent_max_seconds == 0.0
    assert rc.agent_deadline_finalize_slack_seconds >= 0.0
    # post-dispatch self-verify disabled by default
    assert rc.pre_terminal_self_verify_enabled is False
    assert rc.pre_terminal_self_verify_max_extra_turns == 1
    # pre-dispatch terminal verify off by default
    assert rc.pre_dispatch_terminal_verify_enabled is False
    # entity-normalize is NOT byte-preserving → default OFF;
    # a tenant opts in via a per-tenant override.
    assert rc.terminal_answer_entity_normalize_enabled is False
    assert rc.terminal_answer_sentinels == []
    # parallel reads on; fanout default is the value-preserving sentinel 0
    # (== UNLIMITED == unbounded gather); only a positive override chunks.
    assert rc.parallel_read_tools_enabled is True
    assert rc.parallel_read_tools_max_fanout == 0
    # generic dedup off by default (workspace dedup unaffected)
    assert rc.read_dedup_enabled is False
    assert rc.read_dedup_ttl_seconds == 300
    assert rc.read_dedup_max_entries == 256


# --------------------------------------------------------------------------
# Minimal engine doubles for the deadline / self-verify helpers
# --------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, rc: RuntimeConstants) -> None:
        self.rc = rc
        self.expected_terminal_tool = "final_answer"
        self.pre_terminal_self_verify_trigger = None
        self.run_id = "run-test"
        self.tenant_id = "tenant-test"


class _FakeEngine:
    def __init__(self, rc: RuntimeConstants, *, started_monotonic: float) -> None:
        self.config = _FakeConfig(rc)
        self._run_started_monotonic = started_monotonic
        self._pre_terminal_self_verify_used = False
        self._self_verify_extra_turns_used = 0
        self._pre_dispatch_terminal_verify_used = False
        self.history: list = []
        # ``_maybe_inject_pre_terminal_self_verify`` persists the snapshot on
        # injection; count the calls so a test can assert it fired.
        self.persist_calls = 0

    def turn_id(self) -> str:
        return "turn-test"

    async def _persist_snapshot(self) -> None:
        self.persist_calls += 1


# --------------------------------------------------------------------------
# deadline reached predicate
# --------------------------------------------------------------------------
def _deadline_fn():
    import sys

    return sys.modules["protocore.runtime.query"]._terminal_deadline_reached


def _self_verify_fn():
    import sys

    return sys.modules["protocore.runtime.query"]._maybe_inject_pre_terminal_self_verify


def _fixed_monotonic(monkeypatch: pytest.MonkeyPatch, value: float = 1_000.0) -> float:
    """Make elapsed-time assertions independent of runner uptime."""
    monkeypatch.setattr(time, "monotonic", lambda: value)
    return value


def test_deadline_helper_is_module_level() -> None:
    import sys

    from protocore.runtime import query as _q_pkg_attr  # noqa: F401

    mod = sys.modules["protocore.runtime.query"]
    assert hasattr(mod, "_terminal_deadline_reached")
    assert hasattr(mod, "_maybe_inject_pre_terminal_self_verify")


def test_deadline_disabled_when_budget_zero() -> None:
    fn = _deadline_fn()
    eng = _FakeEngine(RuntimeConstants(), started_monotonic=1.0)
    assert fn(eng) is False  # agent_max_seconds == 0


def test_deadline_not_reached_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = _deadline_fn()
    now = _fixed_monotonic(monkeypatch)
    rc = RuntimeConstants(
        agent_max_seconds=600.0, agent_deadline_finalize_slack_seconds=45.0
    )
    eng = _FakeEngine(rc, started_monotonic=now)  # 0s elapsed
    assert fn(eng) is False


def test_deadline_reached_within_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    fn = _deadline_fn()
    now = _fixed_monotonic(monkeypatch)
    rc = RuntimeConstants(
        agent_max_seconds=600.0, agent_deadline_finalize_slack_seconds=45.0
    )
    # elapsed = 600 - 45 + 1 = 556 -> past threshold (555)
    eng = _FakeEngine(rc, started_monotonic=now - 556.0)
    assert fn(eng) is True


def test_deadline_noop_when_clock_unstamped() -> None:
    fn = _deadline_fn()
    rc = RuntimeConstants(agent_max_seconds=600.0)
    eng = _FakeEngine(rc, started_monotonic=0.0)
    assert fn(eng) is False


def test_deadline_slack_ge_budget_finalises_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _deadline_fn()
    now = _fixed_monotonic(monkeypatch)
    # slack > budget -> threshold floored at 0 -> any elapsed finalises.
    rc = RuntimeConstants(
        agent_max_seconds=10.0, agent_deadline_finalize_slack_seconds=999.0
    )
    eng = _FakeEngine(rc, started_monotonic=now - 0.5)
    assert fn(eng) is True


# --------------------------------------------------------------------------
# The deadline accounts pre-``run()`` (e.g. auto-grounding) time.
# A run-clock stamped in the PAST (the host prelude stamps it BEFORE
# QueryEngine.run so ``agent_max_seconds`` spans grounding) makes the
# deadline-finalize fire sooner — i.e. the budget includes that elapsed time.
# --------------------------------------------------------------------------
def test_deadline_accounts_pre_run_grounding_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _deadline_fn()
    now = _fixed_monotonic(monkeypatch)
    rc = RuntimeConstants(
        agent_max_seconds=300.0, agent_deadline_finalize_slack_seconds=90.0
    )
    # Example values: finalize threshold = 300 - 90 = 210s.
    # A FRESH clock (0s elapsed) is NOT yet at the threshold.
    fresh = _FakeEngine(rc, started_monotonic=now)
    assert fn(fresh) is False
    # But a run whose clock was stamped 211s ago — e.g. because the executor
    # stamped it BEFORE a long auto-grounding prelude — is already past the
    # finalize point, so the deadline fires (the budget SPANS grounding).
    grounded = _FakeEngine(rc, started_monotonic=now - 211.0)
    assert fn(grounded) is True


def test_deadline_low_uptime_keeps_positive_stamp_distinct_from_unstamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _deadline_fn()
    now = _fixed_monotonic(monkeypatch, 100.0)
    rc = RuntimeConstants(agent_max_seconds=10.0)
    stamped = _FakeEngine(rc, started_monotonic=1.0)
    unstamped = _FakeEngine(rc, started_monotonic=0.0)

    assert now < 211.0
    assert stamped._run_started_monotonic > 0.0
    assert fn(stamped) is True
    assert fn(unstamped) is False


def test_pre_run_stamp_inert_when_budget_disabled() -> None:
    # Universality guard: a valid pre-stamp is INERT for a tenant
    # with ``agent_max_seconds=0`` (deadline disabled) — the deadline predicate
    # returns False regardless of when the clock was stamped, so a Part-A
    # pre-stamp can never change deadline-disabled behaviour.
    fn = _deadline_fn()
    rc = RuntimeConstants()  # agent_max_seconds defaults to 0.0
    eng = _FakeEngine(rc, started_monotonic=1.0)
    assert fn(eng) is False


# --------------------------------------------------------------------------
# pre-terminal self-verify latch
# --------------------------------------------------------------------------
async def test_self_verify_disabled_by_default_no_injection() -> None:
    fn = _self_verify_fn()
    rc = RuntimeConstants()  # pre_terminal_self_verify_enabled == False
    eng = _FakeEngine(rc, started_monotonic=time.monotonic())
    eng.config.pre_terminal_self_verify_trigger = lambda _e: "fix it"
    assert await fn(eng) is False
    assert eng.history == []
    # disabled path must NOT persist.
    assert eng.persist_calls == 0


async def test_self_verify_injects_once_then_latches() -> None:
    fn = _self_verify_fn()
    rc = RuntimeConstants(
        pre_terminal_self_verify_enabled=True,
        pre_terminal_self_verify_max_extra_turns=1,
    )
    eng = _FakeEngine(rc, started_monotonic=time.monotonic())
    eng.config.pre_terminal_self_verify_trigger = (
        lambda _e: "you cited an unobserved ref"
    )
    # First call injects.
    assert await fn(eng) is True
    assert len(eng.history) == 1
    assert eng._pre_terminal_self_verify_used is True
    assert eng._self_verify_extra_turns_used == 1
    # The corrective turn + latch are persisted at injection time so a
    # crash/resume between here and the next LLM call does not lose them.
    assert eng.persist_calls == 1
    # Second call latched off (fire-at-most-once); no extra persist.
    assert await fn(eng) is False
    assert len(eng.history) == 1
    assert eng.persist_calls == 1


async def test_self_verify_trigger_none_declines() -> None:
    fn = _self_verify_fn()
    rc = RuntimeConstants(pre_terminal_self_verify_enabled=True)
    eng = _FakeEngine(rc, started_monotonic=time.monotonic())
    eng.config.pre_terminal_self_verify_trigger = lambda _e: None
    assert await fn(eng) is False
    assert eng.history == []
    assert eng.persist_calls == 0


async def test_self_verify_trigger_exception_is_swallowed() -> None:
    fn = _self_verify_fn()

    def _boom(_e: object) -> str:
        raise RuntimeError("trigger blew up")

    rc = RuntimeConstants(pre_terminal_self_verify_enabled=True)
    eng = _FakeEngine(rc, started_monotonic=time.monotonic())
    eng.config.pre_terminal_self_verify_trigger = _boom
    # Must never break finalisation -- declines gracefully.
    assert await fn(eng) is False


async def test_self_verify_max_extra_turns_zero_blocks_injection() -> None:
    fn = _self_verify_fn()
    rc = RuntimeConstants(
        pre_terminal_self_verify_enabled=True,
        pre_terminal_self_verify_max_extra_turns=0,
    )
    eng = _FakeEngine(rc, started_monotonic=time.monotonic())
    eng.config.pre_terminal_self_verify_trigger = lambda _e: "fix"
    assert await fn(eng) is False


# --------------------------------------------------------------------------
# generic read-dedup tool key
# --------------------------------------------------------------------------
def test_read_dedup_tool_key_order_independent() -> None:
    assert ReadDedupCache.tool_key("t", {"a": 1, "b": 2}) == ReadDedupCache.tool_key(
        "t", {"b": 2, "a": 1}
    )


def test_read_dedup_tool_key_prefix_and_empty() -> None:
    assert ReadDedupCache.tool_key("remote_read", {"path": "/p"}).startswith(
        "remote_read:"
    )
    assert ReadDedupCache.tool_key("t", None) == "t:{}"


def test_read_dedup_tool_key_non_serialisable_falls_back() -> None:
    class _X:
        pass

    # default=repr keeps it from raising; key is stable.
    key = ReadDedupCache.tool_key("t", {"obj": _X()})
    assert key.startswith("t:")


# ==========================================================================
# Durable latches + the PRE-DISPATCH terminal-tool verify seam
# ==========================================================================
import sys as _sys  # noqa: E402

from protocore.contracts.types import (  # noqa: E402
    MessageRole,
    ToolCall,
    ToolResultBlock,
)
from protocore.runtime.events import EventType  # noqa: E402
from protocore.runtime.query_engine import (  # noqa: E402
    QueryEngine,
    QueryEngineConfig,
)
from protocore.tests_support.adapters import (  # noqa: E402
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool  # noqa: E402


def _query_mod():
    # The ``query`` function shadows the module name on direct import; reach
    # the module object through ``sys.modules`` (same trick the deadline /
    # self-verify helpers use above).
    from protocore.runtime import query as _q  # noqa: F401

    return _sys.modules["protocore.runtime.query"]


def _build_terminal_engine(
    *,
    rc: RuntimeConstants,
    tools: InMemoryToolRegistry | None = None,
    expected_terminal_tool: str | None = "final_answer",
    pre_dispatch_trigger=None,
):
    """Real :class:`QueryEngine` wired to in-memory adapters with a declared
    terminal tool + optional pre-dispatch verify trigger."""
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-r8",
            tenant_id="tenant-r8",
            session_id="sess-r8",
            model_name="qwen3.6-35b-a3b",
            #  — these tests register ``final_answer`` as
            # a bare MockTool whose default schema has NO message/answer/text
            # field, so the schema-conditioned prose-gate would treat it as a
            # BACKGROUND terminal and fire BEFORE the pre-dispatch-verify seam
            # under test. Disable the prose-gate here so it does not intercept
            # the pre-dispatch-verify dispatches (the gate's own coverage lives
            # in test_finalize_terminal_gate.py).
            rc=rc.model_copy(update={"finalize_prose_gate_enabled": False}),
            expected_terminal_tool=expected_terminal_tool,
            pre_dispatch_terminal_verify_trigger=pre_dispatch_trigger,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=tools or InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


# --------------------------------------------------------------------------
# The wind-down survives snapshot/resume
# --------------------------------------------------------------------------
def test_wind_down_absent_from_a_fresh_snapshot() -> None:
    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    snap = eng.snapshot()
    assert snap["soft_stop_cause"] is None
    assert snap["soft_stop_stage"] == ""


@pytest.mark.asyncio
async def test_wind_down_roundtrips_through_a_snapshot() -> None:
    """A run resumed after its tools were withdrawn stays wound down.

    Without this the re-drive re-advertises everything the stop removed, and
    the run carries on working past the bound it hit.
    """
    from protocore.runtime import soft_stop as _soft_stop

    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    _soft_stop.enter(eng, cause_name=_soft_stop.CAUSE_DEADLINE)
    snap = eng.snapshot()
    assert snap["soft_stop_cause"] == _soft_stop.CAUSE_DEADLINE
    assert snap["soft_stop_stage"] == _soft_stop.STAGE_WITHDRAWN

    other = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    await other.resume_from_snapshot(snap)
    assert _soft_stop.is_armed(other) is True
    assert _soft_stop.tools_withdrawn(other) is True
    # And the resumed run cannot re-enter the wind-down a second time.
    assert _soft_stop.enter(other, cause_name=_soft_stop.CAUSE_MAX_TURNS) == []


@pytest.mark.asyncio
async def test_a_snapshot_without_the_wind_down_resumes_without_one() -> None:
    from protocore.runtime import soft_stop as _soft_stop

    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    snap = eng.snapshot()
    snap.pop("soft_stop_cause", None)
    snap.pop("soft_stop_stage", None)
    await eng.resume_from_snapshot(snap)
    assert _soft_stop.is_armed(eng) is False


@pytest.mark.parametrize(
    ("budget_seconds", "expected_reached"),
    [(150.0, True), (300.0, False)],
)
@pytest.mark.asyncio
async def test_snapshot_resume_deadline_survives_lower_monotonic_uptime(
    monkeypatch: pytest.MonkeyPatch,
    budget_seconds: float,
    expected_reached: bool,
) -> None:
    rc = RuntimeConstants(
        model_context_window=4_096,
        agent_max_seconds=budget_seconds,
        agent_deadline_finalize_slack_seconds=0.0,
    )
    source = _build_terminal_engine(rc=rc)
    snapshot = source.snapshot()
    snapshot["run_started_epoch"] = 800.0

    _fixed_monotonic(monkeypatch, 50.0)
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    resumed = _build_terminal_engine(rc=rc)
    await resumed.resume_from_snapshot(snapshot)

    # Wall elapsed is 200s while the new process has only 50s of uptime. The
    # negative synthetic start is a valid re-anchor, not the zero sentinel.
    assert resumed._run_started_monotonic == -150.0
    assert _deadline_fn()(resumed) is expected_reached


@pytest.mark.asyncio
async def test_run_preserves_negative_resume_reanchor_but_stamps_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from protocore.contracts.types import Message, TextBlock

    rc = RuntimeConstants(model_context_window=4_096)
    source = _build_terminal_engine(rc=rc, expected_terminal_tool=None)
    resumed_snapshot = source.snapshot()
    resumed_snapshot["run_started_epoch"] = 800.0
    unstamped_snapshot = source.snapshot()
    unstamped_snapshot["run_started_epoch"] = 0.0

    _fixed_monotonic(monkeypatch, 50.0)
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])

    resumed = _build_terminal_engine(rc=rc, expected_terminal_tool=None)
    await resumed.resume_from_snapshot(resumed_snapshot)
    assert resumed._run_started_monotonic == -150.0
    resumed.llm.queue_response(text="done")  # type: ignore[attr-defined]
    async for _evt in resumed.run(msg):
        pass
    assert resumed._run_started_monotonic == -150.0
    assert resumed._run_started_epoch == 800.0

    unstamped = _build_terminal_engine(rc=rc, expected_terminal_tool=None)
    await unstamped.resume_from_snapshot(unstamped_snapshot)
    assert unstamped._run_started_monotonic == 0.0
    unstamped.llm.queue_response(text="done")  # type: ignore[attr-defined]
    async for _evt in unstamped.run(msg):
        pass
    assert unstamped._run_started_monotonic == 50.0
    assert unstamped._run_started_epoch == 1_000.0


@pytest.mark.asyncio
async def test_future_epoch_repeated_resume_keeps_one_consumed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted epoch ahead of wall clock must not restart elapsed.

    Two resumes while wall time stays below the epoch, with more than the
    budget consumed in real monotonic time, must expire. Driving only
    ``max(0, now - epoch)`` would keep the deadline unreached.
    """
    from protocore.runtime.deadline_clock import restore_deadline_clock

    budget_seconds = 150.0
    persisted_epoch = 1_200.0
    rc = RuntimeConstants(
        model_context_window=4_096,
        agent_max_seconds=budget_seconds,
        agent_deadline_finalize_slack_seconds=0.0,
    )
    source = _build_terminal_engine(rc=rc)
    first_snapshot = source.snapshot()
    first_snapshot["run_started_epoch"] = persisted_epoch
    first_snapshot.pop("run_deadline_elapsed_seconds", None)

    _fixed_monotonic(monkeypatch, 50.0)
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    first = _build_terminal_engine(rc=rc)
    await first.resume_from_snapshot(first_snapshot)
    assert first._run_started_epoch == persisted_epoch
    assert first._run_started_monotonic == 50.0
    assert _deadline_fn()(first) is False

    _fixed_monotonic(monkeypatch, 150.0)
    monkeypatch.setattr(time, "time", lambda: 1_100.0)
    mid_snapshot = first.snapshot()
    assert mid_snapshot["run_started_epoch"] == persisted_epoch
    assert mid_snapshot["run_deadline_elapsed_seconds"] == 100.0
    assert _deadline_fn()(first) is False

    _fixed_monotonic(monkeypatch, 10.0)
    second = _build_terminal_engine(rc=rc)
    await second.resume_from_snapshot(mid_snapshot)
    assert second._run_started_epoch == persisted_epoch
    assert second._run_started_monotonic == -90.0
    assert _deadline_fn()(second) is False

    _fixed_monotonic(monkeypatch, 109.0)
    monkeypatch.setattr(time, "time", lambda: 1_199.0)
    assert (109.0 - (-90.0)) > budget_seconds
    assert time.time() < persisted_epoch
    assert _deadline_fn()(second) is True

    # The shipped restorer, not a local reimplementation, is what resume used.
    restored = restore_deadline_clock(
        persisted_epoch=mid_snapshot["run_started_epoch"],
        persisted_elapsed=mid_snapshot["run_deadline_elapsed_seconds"],
        now_wall=1_100.0,
        now_monotonic=10.0,
    )
    assert restored.elapsed_seconds == 100.0
    assert restored.monotonic_anchor == -90.0


@pytest.mark.parametrize(
    "bad_epoch",
    [float("nan"), float("inf"), float("-inf"), -1.0, "bad", None, True],
)
@pytest.mark.asyncio
async def test_malformed_epoch_rejected_before_deadline_state_mutates(
    bad_epoch: object,
) -> None:
    from protocore.runtime.deadline_clock import InvalidDeadlineSnapshot

    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    eng._run_started_epoch = 123.0
    eng._run_started_monotonic = 45.0
    snapshot = eng.snapshot()
    snapshot["run_started_epoch"] = bad_epoch
    with pytest.raises(InvalidDeadlineSnapshot, match="run_started_epoch"):
        await eng.resume_from_snapshot(snapshot)
    assert eng._run_started_epoch == 123.0
    assert eng._run_started_monotonic == 45.0
    assert eng.history == []


@pytest.mark.parametrize(
    "bad_elapsed",
    [float("nan"), float("inf"), float("-inf"), -1.0, "bad", None, True],
)
@pytest.mark.asyncio
async def test_malformed_elapsed_rejected_before_deadline_state_mutates(
    bad_elapsed: object,
) -> None:
    from protocore.runtime.deadline_clock import InvalidDeadlineSnapshot

    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    eng._run_started_epoch = 123.0
    eng._run_started_monotonic = 45.0
    snapshot = eng.snapshot()
    snapshot["run_started_epoch"] = 800.0
    snapshot["run_deadline_elapsed_seconds"] = bad_elapsed
    with pytest.raises(InvalidDeadlineSnapshot, match="run_deadline_elapsed_seconds"):
        await eng.resume_from_snapshot(snapshot)
    assert eng._run_started_epoch == 123.0
    assert eng._run_started_monotonic == 45.0


# --------------------------------------------------------------------------
# ``run()`` PRESERVES a pre-set ``_run_started_monotonic`` so
# the host prelude (auto-grounding) that stamps the run-clock
# BEFORE ``run()`` makes ``agent_max_seconds`` span that pre-``run()`` time.
# Each stamp is preserved independently (a caller may pre-stamp only one).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_preserves_preset_monotonic_stamp() -> None:
    from protocore.contracts.types import Message, TextBlock

    eng = _build_terminal_engine(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool=None,
    )
    # Simulate the host prelude setting a valid run-clock before run().
    preset_monotonic = 1.0
    preset_epoch = time.time() - 123.0
    eng._run_started_monotonic = preset_monotonic
    eng._run_started_epoch = preset_epoch

    # A plain end-turn response so ``run()`` drives one turn and stops.
    eng.llm.queue_response(text="done")  # type: ignore[attr-defined]
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _evt in eng.run(msg):
        pass

    # run() must NOT have re-stamped a fresh clock; the pre-set values stand
    # (so the deadline budget includes the pre-run grounding elapsed).
    assert eng._run_started_monotonic == preset_monotonic
    assert eng._run_started_epoch == preset_epoch


@pytest.mark.asyncio
async def test_run_stamps_epoch_independently_of_monotonic() -> None:
    # A caller that pre-stamps ONLY the monotonic clock (leaving epoch 0.0)
    # must still get a real wall-clock epoch stamped by ``run()`` — the
    # snapshot needs it to re-anchor on a cross-pod re-drive. The two guards
    # are independent (the pre-run stamp-preservation core fix).
    from protocore.contracts.types import Message, TextBlock

    eng = _build_terminal_engine(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool=None,
    )
    preset_monotonic = 1.0
    eng._run_started_monotonic = preset_monotonic
    assert eng._run_started_epoch == 0.0  # left unset by the caller

    eng.llm.queue_response(text="done")  # type: ignore[attr-defined]
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _evt in eng.run(msg):
        pass

    # Monotonic preserved; epoch now stamped (non-zero) so resume can anchor.
    assert eng._run_started_monotonic == preset_monotonic
    assert eng._run_started_epoch > 0.0


@pytest.mark.asyncio
async def test_run_stamps_both_when_nothing_preset() -> None:
    # Guard against regressing the original behaviour: with NOTHING pre-stamped,
    # ``run()`` stamps both clocks fresh.
    from protocore.contracts.types import Message, TextBlock

    eng = _build_terminal_engine(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool=None,
    )
    assert eng._run_started_monotonic == 0.0
    assert eng._run_started_epoch == 0.0

    eng.llm.queue_response(text="done")  # type: ignore[attr-defined]
    msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    async for _evt in eng.run(msg):
        pass

    assert eng._run_started_monotonic > 0.0
    assert eng._run_started_epoch > 0.0


# --------------------------------------------------------------------------
# PRE-DISPATCH terminal-tool verify: gate predicate
# --------------------------------------------------------------------------
def _applies_fn():
    return _query_mod()._pre_dispatch_terminal_verify_applies


def test_pre_dispatch_verify_gate_off_by_default() -> None:
    fn = _applies_fn()
    eng = _build_terminal_engine(
        rc=RuntimeConstants(model_context_window=4_096),
        pre_dispatch_trigger=lambda _e, _tc: "bad refs",
    )
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["x"]})
    # pre_dispatch_terminal_verify_enabled defaults False → never intercepts.
    assert fn(eng, call) is False


def test_pre_dispatch_verify_gate_only_terminal_tool() -> None:
    fn = _applies_fn()
    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(
        rc=rc, pre_dispatch_trigger=lambda _e, _tc: "bad refs"
    )
    # A non-terminal tool is never intercepted, even with the gate on.
    non_terminal = ToolCall(id="t-1", name="remote_read", arguments={})
    assert fn(eng, non_terminal) is False
    # The declared terminal tool IS in scope.
    terminal = ToolCall(id="t-2", name="final_answer", arguments={"refs": ["x"]})
    assert fn(eng, terminal) is True


def test_pre_dispatch_verify_gate_requires_trigger() -> None:
    fn = _applies_fn()
    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(rc=rc, pre_dispatch_trigger=None)
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["x"]})
    # Gate on but no the host trigger wired → no interception.
    assert fn(eng, call) is False


def test_pre_dispatch_verify_gate_latched_off_after_fire() -> None:
    fn = _applies_fn()
    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(
        rc=rc, pre_dispatch_trigger=lambda _e, _tc: "bad refs"
    )
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["x"]})
    assert fn(eng, call) is True
    # Durable latch set → gate closes (fire-at-most-once).
    eng._pre_dispatch_terminal_verify_used = True
    assert fn(eng, call) is False


def test_pre_dispatch_verify_gate_respects_shared_budget() -> None:
    fn = _applies_fn()
    rc = RuntimeConstants(
        model_context_window=4_096,
        pre_dispatch_terminal_verify_enabled=True,
        pre_terminal_self_verify_max_extra_turns=1,
    )
    eng = _build_terminal_engine(
        rc=rc, pre_dispatch_trigger=lambda _e, _tc: "bad refs"
    )
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["x"]})
    # Shared budget already spent (e.g. the post-dispatch self-verify fired) → no veto.
    eng._self_verify_extra_turns_used = 1
    assert fn(eng, call) is False


# --------------------------------------------------------------------------
# PRE-DISPATCH veto behaviour through the real ``_dispatch_tool``
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pre_dispatch_veto_blocks_terminal_dispatch() -> None:
    """A vetoing trigger must prevent the terminal tool from ever running
    (no external RPC), append a non-terminal error tool_result + a corrective
    user turn, latch durably, and debit the shared budget."""
    mod = _query_mod()
    tools = InMemoryToolRegistry()

    class _TerminalTool(MockTool):
        is_destructive = True  # terminal tools serialise (never parallel)

    terminal = _TerminalTool(tool_name="final_answer", description="submit answer")
    tools.register(terminal)

    captured = {}

    def _trigger(engine, tool_call):
        captured["args"] = dict(tool_call.arguments)
        return "You cited ref /catalog/999 which you never observed. Fix it."

    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(
        rc=rc, tools=tools, pre_dispatch_trigger=_trigger
    )
    eng.transition_to(__import__("protocore.runtime.loop_state", fromlist=["LoopState"]).LoopState.RUNNING)
    call = ToolCall(
        id="t-1", name="final_answer", arguments={"refs": ["/catalog/999"]}
    )
    eng.remember_tool_name(call.id, call.name)

    events = [evt async for evt in mod._dispatch_tool(eng, call)]

    # The terminal tool's run() was NEVER invoked (the external side effect
    # did not fire) — this is the core of the pre-dispatch veto.
    assert terminal.calls == []
    # The trigger saw the un-submitted refs.
    assert captured["args"] == {"refs": ["/catalog/999"]}
    # A single error TOOL_RESULT was emitted for the vetoed call.
    result_events = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert len(result_events) == 1
    assert result_events[0].payload["success"] is False
    assert result_events[0].payload["error"]["kind"] == "pre_dispatch_terminal_verify"
    # History: a non-terminal (is_error) tool_result, then the corrective user
    # turn naming the fix.
    tool_msgs = [m for m in eng.history if m.role is MessageRole.tool]
    assert len(tool_msgs) == 1
    block = tool_msgs[0].content_blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert block.is_error is True
    user_msgs = [m for m in eng.history if m.role is MessageRole.user]
    assert len(user_msgs) == 1
    assert "Fix it" in user_msgs[0].text
    # Latch set durably + shared budget debited.
    assert eng._pre_dispatch_terminal_verify_used is True
    assert eng._self_verify_extra_turns_used == 1
    # The vetoed result is NOT terminal → the loop will not finalise.
    assert mod._history_tool_result_is_terminal(eng, call.id) is False


@pytest.mark.asyncio
async def test_pre_dispatch_no_veto_when_trigger_declines_dispatches_normally() -> None:
    """When the trigger returns None the terminal tool dispatches normally
    (the seam is transparent)."""
    mod = _query_mod()
    tools = InMemoryToolRegistry()

    class _TerminalTool(MockTool):
        is_destructive = True

    terminal = _TerminalTool(
        tool_name="final_answer", description="submit", response_content="submitted"
    )
    tools.register(terminal)

    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(
        rc=rc, tools=tools, pre_dispatch_trigger=lambda _e, _tc: None
    )
    from protocore.runtime.loop_state import LoopState

    eng.transition_to(LoopState.RUNNING)
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["/ok"]})
    eng.remember_tool_name(call.id, call.name)

    _ = [evt async for evt in mod._dispatch_tool(eng, call)]

    # Trigger declined → the terminal tool actually ran.
    assert terminal.calls == [{"refs": ["/ok"]}]
    assert eng._pre_dispatch_terminal_verify_used is False
    assert eng._self_verify_extra_turns_used == 0


@pytest.mark.asyncio
async def test_pre_dispatch_trigger_exception_does_not_block_dispatch() -> None:
    """A trigger that raises must NEVER break dispatch — treated as no-veto."""
    mod = _query_mod()
    tools = InMemoryToolRegistry()

    class _TerminalTool(MockTool):
        is_destructive = True

    terminal = _TerminalTool(tool_name="final_answer", description="submit")
    tools.register(terminal)

    def _boom(_e, _tc):
        raise RuntimeError("trigger blew up")

    rc = RuntimeConstants(
        model_context_window=4_096, pre_dispatch_terminal_verify_enabled=True
    )
    eng = _build_terminal_engine(rc=rc, tools=tools, pre_dispatch_trigger=_boom)
    from protocore.runtime.loop_state import LoopState

    eng.transition_to(LoopState.RUNNING)
    call = ToolCall(id="t-1", name="final_answer", arguments={"refs": ["/ok"]})
    eng.remember_tool_name(call.id, call.name)

    _ = [evt async for evt in mod._dispatch_tool(eng, call)]

    # Exception swallowed → no veto → terminal tool ran, latch untouched.
    assert terminal.calls == [{"refs": ["/ok"]}]
    assert eng._pre_dispatch_terminal_verify_used is False


# --------------------------------------------------------------------------
# pre-dispatch verify durable latch survives snapshot/resume
# --------------------------------------------------------------------------
def test_pre_dispatch_verify_latch_in_snapshot_default_false() -> None:
    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    assert eng.snapshot()["pre_dispatch_terminal_verify_used"] is False


@pytest.mark.asyncio
async def test_pre_dispatch_verify_latch_roundtrips_true() -> None:
    eng = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    eng._pre_dispatch_terminal_verify_used = True
    snap = eng.snapshot()
    other = _build_terminal_engine(rc=RuntimeConstants(model_context_window=4_096))
    await other.resume_from_snapshot(snap)
    # Durable latch survives resume → a re-driven run will not re-veto the
    # model's corrected re-submission (which would loop).
    assert other._pre_dispatch_terminal_verify_used is True


# ==========================================================================
# Final-turn output-token reserve.
#
# ``_apply_terminal_synthesis_output_reserve(engine, max_output_tokens,
# output_cap)`` floors the per-message output budget on the terminal /
# forced-final turn, bounded by the global cap. Default-off is bit-identical.
# It is a pure helper, so the tests call it directly with explicit token
# numbers on a real engine wired through ``_build_terminal_engine``.
# ==========================================================================
def _reserve_fn():
    return _query_mod()._apply_terminal_synthesis_output_reserve


def test_output_reserve_default_off_is_noop() -> None:
    # reserve defaults to 0 -> the floor never changes the budget, even on the
    # actual final turn (the terminal-only latch is set).
    fn = _reserve_fn()
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096, terminal_tool_nudge_enabled=True
        )
    )
    eng._terminal_only_active = True  # actual forced-final turn
    assert fn(eng, 100, 8000) == 100


def test_output_reserve_noop_when_not_final_turn() -> None:
    # Reserve set + nudge enabled, but the terminal-only latch has NOT
    # fired (the nudge was never actually emitted) and the deadline is disabled
    # -> this is NOT the forced-final turn, so the budget is unchanged. Before
    # the fix the reserve fired on every such turn (nudge-required was True);
    # now it correctly defers until the genuine final turn.
    fn = _reserve_fn()
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096,
            terminal_tool_nudge_enabled=True,
            terminal_synthesis_output_reserve_tokens=4096,
        )
    )
    # nudge-required would be True (the old, too-broad gate) ...
    assert _query_mod()._terminal_tool_nudge_required(eng) is True
    # ... but the actual final-turn latch + deadline are both unset.
    assert getattr(eng, "_terminal_only_active", False) is False
    assert _query_mod()._terminal_deadline_reached(eng) is False
    assert fn(eng, 100, 8000) == 100


def test_output_reserve_floors_on_final_turn() -> None:
    fn = _reserve_fn()
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096,
            terminal_tool_nudge_enabled=True,
            terminal_synthesis_output_reserve_tokens=4096,
        )
    )
    eng._terminal_only_active = True  # the nudge fired -> forced-final turn
    # post-band budget (100) is below the 4096 reserve and the 8000 cap -> the
    # final turn reserves 4096.
    assert fn(eng, 100, 8000) == 4096


def test_output_reserve_does_not_lower_a_larger_budget() -> None:
    fn = _reserve_fn()
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096,
            terminal_tool_nudge_enabled=True,
            terminal_synthesis_output_reserve_tokens=4096,
        )
    )
    eng._terminal_only_active = True
    # budget already above the reserve -> untouched (it is a floor, not a cap).
    assert fn(eng, 5000, 8000) == 5000


def test_output_reserve_never_raises_global_cap() -> None:
    fn = _reserve_fn()
    # reserve (10000) exceeds the global cap (3000) -> the floor is clamped to
    # the cap, so it NEVER raises above max_context*ratio.
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096,
            terminal_tool_nudge_enabled=True,
            terminal_synthesis_output_reserve_tokens=10000,
        )
    )
    eng._terminal_only_active = True
    assert fn(eng, 100, 3000) == 3000
    # and it still never lowers a budget already at the cap
    assert fn(eng, 3000, 3000) == 3000


def test_output_reserve_fires_on_deadline_backstop_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = _reserve_fn()
    now = _fixed_monotonic(monkeypatch)
    # nudge disabled, but the deadline backstop is active -> still a final turn.
    eng = _build_terminal_engine(
        rc=RuntimeConstants(
            model_context_window=4_096,
            agent_max_seconds=30.0,
            terminal_synthesis_output_reserve_tokens=4096,
        )
    )
    eng._run_started_monotonic = now - 31.0
    assert _query_mod()._terminal_tool_nudge_required(eng) is False
    assert _query_mod()._terminal_deadline_reached(eng) is True
    assert fn(eng, 100, 8000) == 4096
