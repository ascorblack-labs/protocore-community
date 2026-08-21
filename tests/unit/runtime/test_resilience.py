"""Runtime tests for the universal resilience layer (D6).

Covers the active behaviour: neutral classification, decorrelated-jitter
backoff, the async token-bucket budget, the deadline reserve, the
classify-then-act policy (incl. classify-don't-retry-mutations), and the
universal :func:`resilient_transport_call` wrapper (budget exhaustion,
deadline give-up, success deposit, behaviour-preservation).
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from protocore.contracts.resilience import (
    ResilienceAction,
    ResilienceErrorClass,
    ToolTransportError,
    ToolTransportRetryBudgetExhausted,
    ToolTransportTimeout,
    TransportCallSpec,
)
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.resilience import (
    ResiliencePolicy,
    TokenBucketRetryBudget,
    classify_transport_error,
    deadline_finalization_reserve_ok,
    decorrelated_jitter_backoff,
    reason_to_error_class,
    resilient_transport_call,
)

# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


class TestClassifyTransportError:
    def test_timeout_subclass_is_transient(self) -> None:
        assert (
            classify_transport_error(ToolTransportTimeout("x"))
            is ResilienceErrorClass.transient_retryable
        )

    def test_budget_exhausted_maps_unknown_so_policy_finalizes(self) -> None:
        assert (
            classify_transport_error(ToolTransportRetryBudgetExhausted())
            is ResilienceErrorClass.unknown
        )

    def test_bare_transport_error_with_timeout_message_is_transient(self) -> None:
        assert (
            classify_transport_error(ToolTransportError("deadline exceeded"))
            is ResilienceErrorClass.transient_retryable
        )

    def test_bare_transport_error_structural_is_abort(self) -> None:
        # An unrecognised structural transport error must NOT be masked as
        # transient (05-transport-resilience-SYNTHESIS).
        assert (
            classify_transport_error(ToolTransportError("permission denied"))
            is ResilienceErrorClass.deterministic_abort
        )

    def test_stdlib_timeout_is_transient(self) -> None:
        assert (
            classify_transport_error(TimeoutError())
            is ResilienceErrorClass.transient_retryable
        )

    def test_stdlib_connection_reset_is_timeout_rebuild(self) -> None:
        # ECONNRESET/EPIPE class means the pooled connection is (likely) dead;
        # retrying on the SAME socket reproduces the failure. A fresh client
        # must be forced before retrying, so this class must route to
        # timeout_rebuild (-> rebuild_and_retry), NOT plain transient_retryable.
        assert (
            classify_transport_error(ConnectionResetError())
            is ResilienceErrorClass.timeout_rebuild
        )
        assert (
            classify_transport_error(BrokenPipeError())
            is ResilienceErrorClass.timeout_rebuild
        )
        # Any other ConnectionError subclass is treated the same (stale conn).
        assert (
            classify_transport_error(ConnectionAbortedError())
            is ResilienceErrorClass.timeout_rebuild
        )

    def test_connection_reset_message_marker_is_timeout_rebuild(self) -> None:
        # The message-marker fallback (non-ConnectionError exception whose text
        # names a reset/broken-pipe) must also route to timeout_rebuild so a
        # backend that surfaces a reset as a plain string still rebuilds.
        assert (
            classify_transport_error(RuntimeError("connection reset by peer"))
            is ResilienceErrorClass.timeout_rebuild
        )
        assert (
            classify_transport_error(RuntimeError("[Errno 32] broken pipe"))
            is ResilienceErrorClass.timeout_rebuild
        )

    def test_bare_transport_error_reset_message_is_timeout_rebuild(self) -> None:
        # A bare ToolTransportError naming a reset routes to rebuild too (not
        # the generic transient branch), so the wrapper rebuilds before retry.
        assert (
            classify_transport_error(ToolTransportError("connection reset"))
            is ResilienceErrorClass.timeout_rebuild
        )

    def test_plain_timeout_message_stays_transient(self) -> None:
        # A pure deadline/timeout (no reset/broken-pipe) is NOT a stale-socket
        # signal — it stays transient_retryable (retry same connection).
        assert (
            classify_transport_error(RuntimeError("read timed out"))
            is ResilienceErrorClass.transient_retryable
        )
        assert (
            classify_transport_error(ToolTransportError("deadline exceeded"))
            is ResilienceErrorClass.transient_retryable
        )

    def test_rate_limit_message_classifies_rate_limited(self) -> None:
        assert (
            classify_transport_error(RuntimeError("HTTP 429 rate limit exceeded"))
            is ResilienceErrorClass.rate_limited
        )

    def test_attached_classifier_verdict_is_honoured(self) -> None:
        # Backend attaches a verdict via the dynamic `classified` attribute
        # (mirrors classified_to_exception). reason collapses to neutral.
        class _Verdict:
            reason = "context_overflow"

        exc = RuntimeError("opaque")
        exc.classified = _Verdict()  # type: ignore[attr-defined]
        assert (
            classify_transport_error(exc) is ResilienceErrorClass.context_overflow
        )

    def test_unknown_message_with_no_markers_is_abort(self) -> None:
        assert (
            classify_transport_error(ValueError("totally opaque failure"))
            is ResilienceErrorClass.deterministic_abort
        )


class TestReasonToErrorClass:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("rate_limit", ResilienceErrorClass.rate_limited),
            ("context_overflow", ResilienceErrorClass.context_overflow),
            ("payload_too_large", ResilienceErrorClass.payload_too_large),
            ("timeout", ResilienceErrorClass.timeout_rebuild),
            ("server_error", ResilienceErrorClass.transient_retryable),
            ("auth", ResilienceErrorClass.deterministic_abort),
            ("provider_policy_blocked", ResilienceErrorClass.deterministic_abort),
            ("model_not_found", ResilienceErrorClass.deterministic_abort),
            ("unknown", ResilienceErrorClass.unknown),
            ("some_unmapped_reason", ResilienceErrorClass.unknown),
        ],
    )
    def test_reason_mapping(
        self, reason: str, expected: ResilienceErrorClass
    ) -> None:
        assert reason_to_error_class(reason) is expected

    def test_strenum_reason_reads_value(self) -> None:
        from enum import StrEnum

        class _R(StrEnum):
            rate_limit = "rate_limit"

        assert reason_to_error_class(_R.rate_limit) is ResilienceErrorClass.rate_limited


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------


class TestDecorrelatedJitterBackoff:
    def test_zero_base_is_immediate(self) -> None:
        assert (
            decorrelated_jitter_backoff(
                attempt=3, base_seconds=0.0, max_seconds=10.0
            )
            == 0.0
        )

    def test_first_attempt_never_sleeps(self) -> None:
        assert (
            decorrelated_jitter_backoff(
                attempt=0, base_seconds=1.0, max_seconds=10.0
            )
            == 0.0
        )

    def test_respects_ceiling(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            delay = decorrelated_jitter_backoff(
                attempt=5,
                base_seconds=1.0,
                max_seconds=4.0,
                previous_delay=100.0,
                rng=rng,
            )
            assert 0.0 <= delay <= 4.0

    def test_at_least_base_on_first_retry(self) -> None:
        rng = random.Random(1)
        delay = decorrelated_jitter_backoff(
            attempt=1, base_seconds=2.0, max_seconds=10.0, previous_delay=0.0, rng=rng
        )
        # lower bound = min(base, upper) where upper>=base => >= base
        assert delay >= 2.0

    def test_first_retry_is_jittered_not_collapsed_to_base(self) -> None:
        # Defect: previously ``previous_delay=0.0`` collapsed the
        # first retry to EXACTLY ``base`` (uniform(base, base)), defeating the
        # whole point of decorrelated jitter on the first (most common) retry —
        # every concurrent retrier woke in lockstep at ``base``. The first
        # retry must draw from a real window ``[base, min(base*3, ceiling)]``,
        # seeding the decorrelation from ``base`` (AWS: uniform(base, prev*3)
        # with prev seeded to base on the first call).
        rng = random.Random(20260603)
        seen = {
            decorrelated_jitter_backoff(
                attempt=1,
                base_seconds=1.0,
                max_seconds=10.0,
                previous_delay=0.0,
                rng=rng,
            )
            for _ in range(400)
        }
        # A jittered first retry yields many distinct values, all in
        # [base, base*3]; a collapsed one yields the single value {1.0}.
        assert len(seen) > 5
        assert all(1.0 <= d <= 3.0 for d in seen)
        assert max(seen) > 1.0  # genuinely above base, not pinned to it

    def test_first_retry_jitter_respects_ceiling_below_base_times_three(
        self,
    ) -> None:
        # When the ceiling is below base*3 the first-retry window is clamped to
        # [base, ceiling] (never above the configured max).
        rng = random.Random(7)
        for _ in range(200):
            delay = decorrelated_jitter_backoff(
                attempt=1,
                base_seconds=2.0,
                max_seconds=3.0,
                previous_delay=0.0,
                rng=rng,
            )
            assert 2.0 <= delay <= 3.0


# --------------------------------------------------------------------------
# Deadline reserve
# --------------------------------------------------------------------------


class TestDeadlineReserve:
    def test_untracked_none_is_inert(self) -> None:
        # #147 — ``None`` means NO wall-clock budget tracked → inert (allow).
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=None,
                reserve_seconds=90.0,
                next_attempt_cost_seconds=27.0,
            )
            is True
        )

    def test_expired_zero_refuses_with_reserve(self) -> None:
        # #147 — a TRACKED-but-EXPIRED budget arrives as 0.0; the reserve gate
        # must REFUSE the retry (preserve the terminal window) rather than the
        # pre-#147 fail-open. Refuses with a positive reserve...
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=0.0,
                reserve_seconds=90.0,
                next_attempt_cost_seconds=5.0,
            )
            is False
        )

    def test_expired_zero_refuses_with_zero_reserve_but_positive_cost(self) -> None:
        # ...and refuses even with a 0 reserve when the next attempt costs >0
        # (a tracked-expired budget can never afford another RPC).
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=0.0,
                reserve_seconds=0.0,
                next_attempt_cost_seconds=5.0,
            )
            is False
        )

    def test_retry_allowed_when_reserve_fits(self) -> None:
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=100.0, reserve_seconds=30.0, next_attempt_cost_seconds=10.0
            )
            is True
        )

    def test_retry_refused_when_reserve_would_be_eaten(self) -> None:
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=35.0, reserve_seconds=30.0, next_attempt_cost_seconds=10.0
            )
            is False
        )

    def test_reserve90_cost27_115_refuses(self) -> None:
        # remaining 115, reserve 90, cost 27 → 115-27=88 < 90.
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=115.0,
                reserve_seconds=90.0,
                next_attempt_cost_seconds=27.0,
            )
            is False
        )

    def test_reserve90_cost27_118_allows(self) -> None:
        # remaining 118, reserve 90, cost 27 → 118-27=91 >= 90.
        assert (
            deadline_finalization_reserve_ok(
                remaining_seconds=118.0,
                reserve_seconds=90.0,
                next_attempt_cost_seconds=27.0,
            )
            is True
        )


# --------------------------------------------------------------------------
# Token bucket
# --------------------------------------------------------------------------


class TestTokenBucketRetryBudget:
    @pytest.mark.asyncio
    async def test_consume_until_suppressed(self) -> None:
        bucket = TokenBucketRetryBudget(
            max_tokens=4.0, token_ratio=0.1, suppress_below_ratio=0.5
        )
        # threshold = 4 * 0.5 = 2.0; start at 4.0.
        # consume -> 3.0 (not suppressed), -> 2.0 ... wait: suppress checks
        # BEFORE consuming, suppress when tokens <= 2.0.
        assert await bucket.consume_or_suppress("h") is False  # 4 -> 3
        assert await bucket.consume_or_suppress("h") is False  # 3 -> 2
        # now tokens == 2.0 <= threshold => suppressed (no consume)
        assert await bucket.consume_or_suppress("h") is True
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 2.0

    @pytest.mark.asyncio
    async def test_success_deposit_refills(self) -> None:
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=2.0, suppress_below_ratio=0.5
        )
        await bucket.consume_or_suppress("h")  # 10 -> 9
        await bucket.deposit_success("h")  # 9 -> 10 (capped)
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 10.0

    @pytest.mark.asyncio
    async def test_per_host_isolation(self) -> None:
        bucket = TokenBucketRetryBudget(
            max_tokens=2.0, token_ratio=0.1, suppress_below_ratio=0.0
        )
        await bucket.consume_or_suppress("a")
        a = bucket.snapshot("a")
        b = bucket.snapshot("b")
        assert a is not None and a.tokens == 1.0
        assert b is None  # never touched

    @pytest.mark.asyncio
    async def test_injected_lock_is_used(self) -> None:
        lock = asyncio.Lock()
        bucket = TokenBucketRetryBudget(
            max_tokens=5.0, token_ratio=0.1, suppress_below_ratio=0.0, lock=lock
        )
        # concurrent consumers share one budget under the lock
        await asyncio.gather(*(bucket.consume_or_suppress("h") for _ in range(5)))
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 0.0


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def _policy(**overrides: Any) -> ResiliencePolicy:
    rc = RuntimeConstants(**overrides)
    return ResiliencePolicy(rc=rc)


class TestResiliencePolicy:
    def test_deterministic_abort_never_retries(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.deterministic_abort,
            idempotent=True,
            attempt=1,
            max_attempts=3,
        )
        assert d.action is ResilienceAction.abort
        assert d.retryable is False

    def test_mutation_is_classified_but_not_retried(self) -> None:
        # classify-don't-retry-mutations (D6): a non-idempotent transient
        # failure finalizes (read-back is the caller's job), never re-issues.
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=False,
            attempt=1,
            max_attempts=3,
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now
        assert d.finalization_recommended is True
        assert d.reason_detail == "non_idempotent_no_retry"

    def test_transient_idempotent_retries_with_backoff(self) -> None:
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=4.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            rng=random.Random(3),
        )
        assert d.retryable is True
        assert d.action is ResilienceAction.backoff_and_retry
        assert d.backoff_seconds >= 1.0

    def test_attempts_exhausted_finalizes(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=3,
            max_attempts=3,
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now
        assert d.reason_detail == "attempts_exhausted"

    def test_budget_suppressed_finalizes(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            budget_suppressed=True,
        )
        assert d.retryable is False
        assert d.reason_detail == "retry_budget_exhausted"

    def test_deadline_reserve_blocks_retry(self) -> None:
        d = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=35.0,
            next_call_timeout_seconds=10.0,
        )
        assert d.retryable is False
        assert d.reason_detail == "deadline_reserve"

    def test_deadline_reserve_counts_next_rpc_timeout_not_previous_sleep(
        self,
    ) -> None:
        # The reserve must account for the PROJECTED next RPC cost (its backoff
        # + its call timeout), not the previous sleep (which is 0.0 on the first
        # retry and would let a retry eat the finalization slice). Here remaining
        # (40s) fits the reserve alone (30s) AND a zero-backoff retry by the OLD
        # math (40-0 >= 30), but NOT the reserve PLUS the next 15s RPC timeout
        # (40-15 = 25 < 30). With the next-call timeout threaded in, the policy
        # must give up.
        d = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
            # base 0.0 → projected backoff 0.0, so the ONLY thing that can
            # push the cost past the reserve is the next-call timeout.
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=40.0,
            next_call_timeout_seconds=15.0,
        )
        assert d.retryable is False
        assert d.reason_detail == "deadline_reserve"

    def test_deadline_reserve_allows_retry_when_reserve_plus_rpc_fits(
        self,
    ) -> None:
        # The same shape but with ample remaining: reserve (30) + next RPC (15)
        # = 45 <= remaining (100), so the retry is permitted.
        d = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=100.0,
            next_call_timeout_seconds=15.0,
        )
        assert d.retryable is True

    def test_expired_tracked_budget_finalizes_via_decide(self) -> None:
        # #147 — a TRACKED-but-EXPIRED budget (remaining 0.0) within attempt +
        # budget allowance must finalize on the deadline reserve, NOT fail-open
        # into another retry. Mirrors the executor closure returning 0.0 when
        # ``agent_max_seconds`` is spent.
        d = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=90.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=0.0,
            next_call_timeout_seconds=25.0,
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now
        assert d.reason_detail == "deadline_reserve"

    def test_untracked_none_budget_keeps_reserve_inert_via_decide(self) -> None:
        # #147 — ``remaining_seconds=None`` (untracked) leaves the reserve gate
        # inert so an otherwise-retryable transient still retries.
        d = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=90.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=None,
            next_call_timeout_seconds=25.0,
        )
        assert d.retryable is True

    def test_timeout_rebuild_action(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.timeout_rebuild,
            idempotent=True,
            attempt=1,
            max_attempts=3,
        )
        assert d.action is ResilienceAction.rebuild_and_retry
        assert d.retryable is True

    def test_context_overflow_is_compress_not_transport_retry(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.context_overflow,
            idempotent=True,
            attempt=1,
            max_attempts=3,
        )
        assert d.action is ResilienceAction.compress_and_retry
        assert d.retryable is True

    def test_payload_too_large_is_shrink(self) -> None:
        d = _policy(resilience_enabled=True).decide(
            error_class=ResilienceErrorClass.payload_too_large,
            idempotent=True,
            attempt=1,
            max_attempts=3,
        )
        assert d.action is ResilienceAction.shrink_and_retry

    def test_disabled_policy_never_retries_a_retry_class(self) -> None:
        # ResiliencePolicy stores `resilience_enabled` but decide() must
        # ENFORCE it: when disabled, a candidate-retryable class (here
        # transient, idempotent, attempts remaining) must finalize instead of
        # retrying. Defensive even though live callers gate before constructing
        # the policy — the primitive itself must be safe.
        d = _policy(resilience_enabled=False).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=5,
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now
        assert d.finalization_recommended is True
        assert d.reason_detail == "resilience_disabled"

    def test_disabled_policy_still_aborts_structural(self) -> None:
        # When disabled, a deterministic structural failure must still ABORT
        # (surface unchanged), not be converted to a finalize — the disabled
        # gate only suppresses RETRIES, it does not mask structural errors.
        d = _policy(resilience_enabled=False).decide(
            error_class=ResilienceErrorClass.deterministic_abort,
            idempotent=True,
            attempt=1,
            max_attempts=5,
        )
        assert d.action is ResilienceAction.abort
        assert d.retryable is False

    def test_disabled_policy_finalizes_mutation(self) -> None:
        # A non-idempotent (mutating) failure under a disabled policy must
        # still finalize (classify-don't-retry), never re-issue.
        d = _policy(resilience_enabled=False).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=False,
            attempt=1,
            max_attempts=5,
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now

    # -- : honour a server-stated rate-limit reset ------------------

    def test_rate_limited_honours_server_reset_over_jittered_backoff(
        self,
    ) -> None:
        # The contract promises rate_limited backoff honours a server-stated
        # reset (honouring Retry-After). With a tiny jitter base (1s ceiling 2s)
        # but a 30s server reset, the policy must prefer the server reset:
        # backoff = max(jitter, min(server_reset, ceiling)) and the ceiling here
        # is high enough to honour 30s.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=60.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=30.0,
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.action is ResilienceAction.backoff_and_retry
        # The 30s server reset dominates the <=3s jitter window.
        assert d.backoff_seconds == pytest.approx(30.0)

    def test_rate_limited_server_reset_is_honoured_verbatim_not_ceiling_clamped(
        self,
    ) -> None:
        # The server reset is honoured VERBATIM; it is NOT clamped to the
        # generic backoff ceiling. A 300s reset with a 10s backoff ceiling must
        # wait the full 300s (sleeping only 10s would re-issue into a
        # still-closed window). The deadline-reserve gate (not the backoff cap)
        # is what bounds a large reset — and here no deadline is tracked, so the
        # retry proceeds after the full reset.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=10.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=300.0,
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds == pytest.approx(300.0)  # verbatim, not 10.0

    def test_rate_limited_server_reset_deadline_reserve_bounds_large_reset(
        self,
    ) -> None:
        # The deadline-reserve gate (NOT the backoff cap) is what protects the
        # finalization slice when an honoured reset is large: a 300s reset with
        # only 60s of wall-clock left and a 30s reserve must FINALIZE rather
        # than sleep past the deadline.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=10.0,
            resilience_deadline_reserve_seconds=30.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            remaining_seconds=60.0,
            next_call_timeout_seconds=5.0,
            server_reset_seconds=300.0,
            rng=random.Random(1),
        )
        assert d.retryable is False
        assert d.action is ResilienceAction.finalize_now
        assert d.reason_detail == "deadline_reserve"

    def test_rate_limited_garbage_reset_is_capped_by_absolute_sanity_bound(
        self,
    ) -> None:
        # A hostile/garbage multi-day "reset" must not pin an unbounded in-loop
        # sleep: it is capped by the absolute sanity bound (1h), well above any
        # legitimate window. No deadline tracked → retry after the capped value.
        from protocore.runtime.resilience import (
            _MAX_HONOURED_RATE_LIMIT_RESET_SECONDS,
        )

        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=10.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=5_000_000.0,  # ~58 days
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds == pytest.approx(
            _MAX_HONOURED_RATE_LIMIT_RESET_SECONDS
        )

    def test_rate_limited_prefers_jitter_when_reset_is_smaller(self) -> None:
        # max(server_reset, jittered_backoff): a tiny/expired reset must NOT
        # shrink a larger configured backoff below what the jitter would give.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=5.0,
            resilience_backoff_max_seconds=20.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=0.5,  # smaller than the jitter floor (base=5)
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds >= 5.0  # jitter floor wins over the tiny reset

    def test_server_reset_honoured_even_with_backoff_disabled(self) -> None:
        # Even with the generic jittered backoff disabled (base 0.0,
        # immediate-retry mode), an explicit server reset is authoritative:
        # re-issuing immediately into a known-closed window is exactly the bug.
        # So a 30s reset must wait 30s despite base 0.0.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=0.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=30.0,
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds == pytest.approx(30.0)

    def test_no_server_reset_with_backoff_disabled_is_immediate(self) -> None:
        # Behaviour preservation: WITHOUT a reset hint, base 0.0 stays the
        # immediate-retry path (0.0 backoff) — the reset honoring does not
        # change the no-hint default.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=0.0,
        ).decide(
            error_class=ResilienceErrorClass.rate_limited,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds == 0.0

    def test_server_reset_only_applies_to_rate_limited_class(self) -> None:
        # A transient (non-rate-limit) failure carrying an incidental reset
        # hint keeps the plain jittered backoff — the reset semantics are a
        # rate-limit affordance, not a universal override.
        d = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=60.0,
        ).decide(
            error_class=ResilienceErrorClass.transient_retryable,
            idempotent=True,
            attempt=1,
            max_attempts=3,
            server_reset_seconds=30.0,
            rng=random.Random(1),
        )
        assert d.retryable is True
        assert d.backoff_seconds <= 3.0  # plain jitter, NOT the 30s reset


# --------------------------------------------------------------------------
# resilient_transport_call wrapper
# --------------------------------------------------------------------------


class _ScriptedTransport:
    """A transport whose invoke() replays a scripted list of outcomes.

    Each entry is either an Exception (raised) or a value (returned).
    Records how many times invoke fired (to assert no blind re-issue).
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0

    async def invoke(self, spec: TransportCallSpec) -> Any:
        self.calls += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
class TestResilientTransportCall:
    async def test_success_first_try_returns(self) -> None:
        t = _ScriptedTransport(["ok"])
        policy = _policy(resilience_enabled=True)
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 1

    async def test_transient_then_success_retries(self) -> None:
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2

    async def test_mutation_never_reissued(self) -> None:
        # D6 — a non-idempotent (mutating) spec must NOT be blind-retried;
        # the original transient error surfaces as a give-up so the caller
        # reads back the commit state.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=False, method_class="mutation"),
                policy=policy,
                max_attempts=3,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # NOT re-issued

    async def test_structural_error_surfaces_immediately(self) -> None:
        t = _ScriptedTransport([ToolTransportError("permission denied"), "ok"])
        policy = _policy(resilience_enabled=True)
        with pytest.raises(ToolTransportError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # not retried — structural

    async def test_attempts_exhausted_raises_budget_exhausted(self) -> None:
        t = _ScriptedTransport(
            [ToolTransportTimeout("t1"), ToolTransportTimeout("t2")]
        )
        policy = _policy(resilience_enabled=True)
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=2,
                sleep=_no_sleep,
            )
        assert t.calls == 2

    async def test_budget_suppression_gives_up_early(self) -> None:
        # 5 attempts allowed, but the bucket suppresses after draining.
        t = _ScriptedTransport([ToolTransportTimeout(f"t{i}") for i in range(5)])
        policy = _policy(resilience_enabled=True)
        bucket = TokenBucketRetryBudget(
            max_tokens=2.0, token_ratio=0.0, suppress_below_ratio=0.5
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=5,
                budget=bucket,
                sleep=_no_sleep,
            )
        # threshold = 2*0.5 = 1.0; start 2.0. call1 fail -> consume (2->1) +
        # retry; call2 fail -> tokens 1.0 <= 1.0 => suppress => give up. So
        # exactly TWO invocations were issued before the budget cut it off.
        assert t.calls == 2

    async def test_success_deposits_token(self) -> None:
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=2.0, suppress_below_ratio=0.5
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            budget=bucket,
            sleep=_no_sleep,
        )
        assert out == "ok"
        snap = bucket.snapshot("h")
        # one retry consumed (10->9), one success deposited (9->10 capped)
        assert snap is not None and snap.tokens == 10.0

    async def test_last_attempt_does_not_consume_budget_token(self) -> None:
        # A fully-failing 2-attempt call spends exactly ONE
        # token (the single retry), not two.
        t = _ScriptedTransport(
            [ToolTransportTimeout("t1"), ToolTransportTimeout("t2")]
        )
        policy = _policy(resilience_enabled=True)
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=2,
                budget=bucket,
                sleep=_no_sleep,
            )
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 9.0  # exactly one consumed

    async def test_single_attempt_behaviour_is_single_shot(self) -> None:
        # Default RC: resilience_transport_max_attempts=1 → single shot, the
        # error surfaces (finalize_now via attempts_exhausted). Behaviour
        # preservation: a tenant that does not opt in never retries.
        t = _ScriptedTransport([ToolTransportTimeout("t1")])
        policy = _policy(resilience_enabled=False)
        with pytest.raises(ToolTransportError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=1,
                sleep=_no_sleep,
            )
        assert t.calls == 1

    async def test_on_attempt_callback_observes_decisions(self) -> None:
        seen: list[tuple[int, ResilienceErrorClass, ResilienceAction]] = []
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)

        def _cb(attempt: int, ec: ResilienceErrorClass, decision: Any) -> None:
            seen.append((attempt, ec, decision.action))

        await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            on_attempt=_cb,
            sleep=_no_sleep,
        )
        assert seen == [
            (1, ResilienceErrorClass.transient_retryable, ResilienceAction.backoff_and_retry)
        ]

    async def test_deadline_reserve_uses_next_rpc_timeout_from_spec(self) -> None:
        # End-to-end through the wrapper: the spec's ``timeout_ms`` is threaded
        # into the projected next-attempt cost so a retry that would fit the
        # reserve alone but NOT reserve + the next RPC's timeout is refused.
        # remaining=40s, reserve=30s, spec timeout 15000ms → 40 - 15 = 25 < 30
        # → give up after the FIRST attempt (no second RPC), surfacing the
        # budget-exhausted finalize.
        t = _ScriptedTransport(
            [ToolTransportTimeout("t1"), ToolTransportTimeout("t2")]
        )
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(
                    host_key="h", idempotent=True, timeout_ms=15_000
                ),
                policy=policy,
                max_attempts=3,
                remaining_seconds_fn=lambda: 40.0,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # reserve protected; the next RPC was never issued

    async def test_deadline_reserve_allows_when_remaining_ample(self) -> None:
        # Same spec timeout, but ample remaining (200s): reserve (30) + next
        # RPC (15) fits, so the retry proceeds and the second attempt succeeds.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True, timeout_ms=15_000),
            policy=policy,
            max_attempts=3,
            remaining_seconds_fn=lambda: 200.0,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2

    async def test_expired_remaining_fn_finalizes_no_reissue(self) -> None:
        # #147 — the remaining-fn returns 0.0 (a TRACKED-but-EXPIRED budget,
        # exactly what the executor closure now returns at the deadline). The
        # reserve gate must REFUSE the retry → exactly ONE RPC, then the
        # budget-exhausted finalize (NOT a doomed post-deadline second RPC).
        t = _ScriptedTransport(
            [ToolTransportTimeout("t1"), ToolTransportTimeout("t2")]
        )
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=90.0,
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(
                    host_key="h", idempotent=True, timeout_ms=25_000
                ),
                policy=policy,
                max_attempts=3,
                remaining_seconds_fn=lambda: 0.0,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # expired budget → no second RPC

    async def test_untracked_remaining_fn_none_keeps_reserve_inert(self) -> None:
        # #147 — a remaining-fn that returns None (untracked) leaves the reserve
        # inert, so the transient retry proceeds and the second attempt wins.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=90.0,
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True, timeout_ms=25_000),
            policy=policy,
            max_attempts=3,
            remaining_seconds_fn=lambda: None,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2

    async def test_cancellation_propagates(self) -> None:
        class _Cancel:
            calls = 0

            async def invoke(self, spec: TransportCallSpec) -> Any:
                self.calls += 1
                raise asyncio.CancelledError()

        c = _Cancel()
        policy = _policy(resilience_enabled=True)
        with pytest.raises(asyncio.CancelledError):
            await resilient_transport_call(
                c,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                sleep=_no_sleep,
            )
        assert c.calls == 1  # not retried on cancel

    async def test_on_attempt_observer_exception_does_not_abort_the_call(
        self,
    ) -> None:
        # Defect: a throwing on_attempt observer must NOT abort the
        # transport call (telemetry is best-effort). The first attempt fails
        # transiently, the observer raises, and the call must STILL retry and
        # return the second attempt's success.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)
        calls = 0

        def _boom(attempt: int, ec: ResilienceErrorClass, decision: Any) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("observer blew up")

        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            on_attempt=_boom,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2  # retried despite the observer raising
        assert calls == 1  # observer was invoked (and its raise was swallowed)

    async def test_timeout_rebuild_invokes_transport_rebuild_before_retry(
        self,
    ) -> None:
        # A stale-connection class (-> timeout_rebuild -> rebuild_and_retry)
        # must invoke the transport's optional rebuild() hook BEFORE the retry,
        # forcing a fresh client on a stale connection. Here the first attempt
        # raises ConnectionReset, which classifies timeout_rebuild; the wrapper
        # must call rebuild() exactly once, then retry and win.
        class _RebuildTransport:
            def __init__(self) -> None:
                self.calls = 0
                self.rebuilds = 0
                self._script: list[Any] = [ConnectionResetError("ECONNRESET"), "ok"]

            async def invoke(self, spec: TransportCallSpec) -> Any:
                self.calls += 1
                outcome = self._script.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            async def rebuild(self) -> None:
                self.rebuilds += 1

        t = _RebuildTransport()
        policy = _policy(resilience_enabled=True)
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2
        assert t.rebuilds == 1  # rebuilt before the retry

    async def test_timeout_rebuild_without_rebuild_hook_still_retries(
        self,
    ) -> None:
        # A transport that does NOT expose rebuild() must still be retried on a
        # timeout_rebuild class (the rebuild hook is optional; absence is a
        # graceful no-op, never an error).
        t = _ScriptedTransport([ConnectionResetError("ECONNRESET"), "ok"])
        policy = _policy(resilience_enabled=True)
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2

    async def test_rebuild_hook_failure_does_not_abort_the_retry(self) -> None:
        # A rebuild() that itself raises must not abort the call — the retry
        # proceeds on the existing transport (best-effort rebuild). The first
        # attempt is a stale-connection error; rebuild() raises; the wrapper
        # still issues the retry, which succeeds.
        class _BadRebuild:
            def __init__(self) -> None:
                self.calls = 0
                self._script: list[Any] = [ConnectionResetError("ECONNRESET"), "ok"]

            async def invoke(self, spec: TransportCallSpec) -> Any:
                self.calls += 1
                outcome = self._script.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            async def rebuild(self) -> None:
                raise RuntimeError("rebuild failed")

        t = _BadRebuild()
        policy = _policy(resilience_enabled=True)
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2

    async def test_rebuild_hook_cancelled_error_propagates(self) -> None:
        # A cancel that lands inside rebuild() (orchestrator teardown) must
        # propagate, never be swallowed as a best-effort rebuild failure.
        class _CancelRebuild:
            def __init__(self) -> None:
                self.calls = 0

            async def invoke(self, spec: TransportCallSpec) -> Any:
                self.calls += 1
                raise ConnectionResetError("ECONNRESET")

            async def rebuild(self) -> None:
                raise asyncio.CancelledError()

        t = _CancelRebuild()
        policy = _policy(resilience_enabled=True)
        with pytest.raises(asyncio.CancelledError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # no retry after cancel during rebuild

    async def test_rate_limit_reset_hint_on_exception_is_honoured(self) -> None:
        # end-to-end — a backend attaches a reset hint on the
        # ToolTransportError (``retry_after_seconds``); the wrapper extracts it
        # and threads it into the decision so the honoured backoff is the
        # server reset (slept via the injected sleep). We capture the slept
        # value to prove the reset (not the tiny jitter) drove the sleep.
        slept: list[float] = []

        async def _record_sleep(seconds: float) -> None:
            slept.append(seconds)

        err = ToolTransportError("429 rate limit", retry_after_seconds=12.0)
        t = _ScriptedTransport([err, "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=60.0,
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_record_sleep,
            rng=random.Random(1),
        )
        assert out == "ok"
        assert t.calls == 2
        assert slept == [pytest.approx(12.0)]  # the server reset, not jitter

    async def test_rate_limit_reset_hint_from_classified_verdict(self) -> None:
        # The reset hint may also ride on the attached classifier verdict
        # (``exc.classified.retry_after_seconds``) so a backend that classifies
        # structurally (not via subclass) still honours the reset.
        slept: list[float] = []

        async def _record_sleep(seconds: float) -> None:
            slept.append(seconds)

        class _Verdict:
            reason = "rate_limit"
            retry_after_seconds = 8.0

        exc = RuntimeError("opaque pushback")
        exc.classified = _Verdict()  # type: ignore[attr-defined]
        t = _ScriptedTransport([exc, "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=60.0,
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            sleep=_record_sleep,
            rng=random.Random(1),
        )
        assert out == "ok"
        assert slept == [pytest.approx(8.0)]

    async def test_cancellation_during_backoff_sleep_propagates_no_reissue(
        self,
    ) -> None:
        # A cancel that arrives WHILE the loop is sleeping the backoff between
        # attempts must propagate out of the sleep, NOT be swallowed; the loop
        # must NOT issue a second transport RPC after the run was cancelled. The
        # first attempt fails transiently (so a backoff sleep is scheduled), the
        # injected sleep raises CancelledError, and we assert exactly ONE invoke
        # happened.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=4.0,
        )

        async def _cancel_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                sleep=_cancel_sleep,
                rng=random.Random(1),
            )
        assert t.calls == 1  # NO second RPC after the cancel during sleep


@pytest.mark.asyncio
class TestRetryBudgetAccountingOnlyChargesRealRetries:
    """The retry-budget token is spent ONLY when a retry is actually issued.

    The per-host budget token was debited BEFORE
    :meth:`ResiliencePolicy.decide` determined whether a retry would actually
    be issued. On every finalize path (policy disabled, deadline-reserve gate,
    a server rate-limit reset that finalizes instead of retrying) a token was
    consumed with NO retry issued, corrupting the
    :class:`TokenBucketRetryBudget` accounting. The fix consults the budget
    read-only for the suppress verdict, then charges the token only after
    ``decide()`` returns a real RETRY action.
    """

    async def test_disabled_policy_finalize_does_not_consume_budget(self) -> None:
        # resilience_enabled=False ⇒ the policy finalizes on the FIRST failure
        # (reason=resilience_disabled) WITHOUT a retry. A budget passed in must
        # therefore NOT lose a token.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), ToolTransportTimeout("t2")])
        policy = _policy(resilience_enabled=False)
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )
        with pytest.raises(ToolTransportError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                budget=bucket,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # finalized — no retry issued
        snap = bucket.snapshot("h")
        # No retry was issued, so the bucket MUST be untouched (10.0, not 9.0).
        assert snap is not None and snap.tokens == 10.0

    async def test_deadline_reserve_finalize_does_not_consume_budget(self) -> None:
        # The reserve gate refuses the retry (remaining 0.0 = tracked-expired)
        # so the policy finalizes after the FIRST attempt — no retry. The token
        # must NOT be charged for the un-issued retry.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), ToolTransportTimeout("t2")])
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=90.0,
        )
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True, timeout_ms=25_000),
                policy=policy,
                max_attempts=3,
                budget=bucket,
                remaining_seconds_fn=lambda: 0.0,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # reserve refused → no second RPC
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 10.0  # token NOT consumed

    async def test_honored_reset_finalize_does_not_consume_budget(self) -> None:
        # A server-stated rate-limit reset larger than the remaining deadline
        # window: decide() honours the reset verbatim, then the deadline-reserve
        # gate refuses (next-attempt cost > remaining) and finalizes. No retry
        # is issued, so the budget token must NOT be spent.
        rl = ToolTransportError("rate limit exceeded")
        rl.retry_after_seconds = 120.0  # type: ignore[attr-defined]
        t = _ScriptedTransport([rl, "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_deadline_reserve_seconds=30.0,
        )
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )
        with pytest.raises(ToolTransportRetryBudgetExhausted):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True, timeout_ms=5_000),
                policy=policy,
                max_attempts=3,
                budget=bucket,
                # remaining 40s, honoured reset 120s + 5s call ⇒ 40 - 125 < 30
                # reserve ⇒ reserve gate refuses → finalize, no retry.
                remaining_seconds_fn=lambda: 40.0,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # honoured-reset finalize → no retry
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 10.0  # token NOT consumed

    async def test_real_retry_consumes_exactly_one_token(self) -> None:
        # Regression guard — a genuine transport retry (transient → backoff →
        # retry) MUST still debit exactly one token. The first attempt fails
        # transiently, the second succeeds.
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(resilience_enabled=True)
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )
        out = await resilient_transport_call(
            t,
            TransportCallSpec(host_key="h", idempotent=True),
            policy=policy,
            max_attempts=3,
            budget=bucket,
            sleep=_no_sleep,
        )
        assert out == "ok"
        assert t.calls == 2  # one retry issued
        snap = bucket.snapshot("h")
        # Exactly one token consumed for the single issued retry (token_ratio=0
        # so the success deposit is a no-op and the count is unambiguous).
        assert snap is not None and snap.tokens == 9.0

    async def test_cancel_during_backoff_does_not_charge_budget(self) -> None:
        # The budget token must be debited only AFTER the backoff/rebuild awaits
        # complete, immediately before the actual transport invoke. If a
        # CancelledError lands DURING the backoff sleep, the retry
        # transport-invoke is NEVER issued, so NO token may be spent. The prior
        # fix charged before the backoff await, so a cancel in that window
        # leaked one token from the per-host budget.
        #
        # The cancel MUST also still propagate — a CancelledError during the
        # backoff sleep means the run was cancelled and must not be swallowed
        # (no post-cancel RPC).
        t = _ScriptedTransport([ToolTransportTimeout("t1"), "ok"])
        policy = _policy(
            resilience_enabled=True,
            resilience_backoff_base_seconds=1.0,
            resilience_backoff_max_seconds=4.0,
        )
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )

        async def _cancel_sleep(_seconds: float) -> None:
            # Stand in for an orchestrator teardown / deadline kill arriving
            # while the loop is sleeping the advisory backoff between attempts.
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                budget=bucket,
                sleep=_cancel_sleep,
            )
        # Only the first attempt was issued; the cancelled retry never ran.
        assert t.calls == 1
        snap = bucket.snapshot("h")
        # No retry was issued (cancel hit the backoff sleep) ⇒ the token MUST
        # NOT be consumed.
        assert snap is not None and snap.tokens == 10.0

    async def test_cancel_during_rebuild_does_not_charge_budget(self) -> None:
        # The same guarantee for the rebuild_and_retry class: the token is
        # charged only after the rebuild await completes. A cancel raised inside
        # transport.rebuild() (the await BEFORE the retry invoke) must NOT spend
        # a token and must propagate (rebuild cancellation is explicitly
        # re-raised by ``_maybe_rebuild_transport``).
        class _CancelOnRebuildTransport(_ScriptedTransport):
            async def rebuild(self) -> None:
                raise asyncio.CancelledError

        # A connection-reset error routes to timeout_rebuild → rebuild_and_retry
        # so the rebuild hook fires before the retry.
        t = _CancelOnRebuildTransport([ConnectionResetError("reset"), "ok"])
        policy = _policy(resilience_enabled=True)
        bucket = TokenBucketRetryBudget(
            max_tokens=10.0, token_ratio=0.0, suppress_below_ratio=0.0
        )

        with pytest.raises(asyncio.CancelledError):
            await resilient_transport_call(
                t,
                TransportCallSpec(host_key="h", idempotent=True),
                policy=policy,
                max_attempts=3,
                budget=bucket,
                sleep=_no_sleep,
            )
        assert t.calls == 1  # retry invoke never issued (cancel in rebuild)
        snap = bucket.snapshot("h")
        assert snap is not None and snap.tokens == 10.0  # token NOT consumed
