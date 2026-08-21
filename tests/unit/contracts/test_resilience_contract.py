"""Contract tests for the universal resilience data types (D6).

Pure-data invariants: the neutral taxonomy, the token-bucket model, the
transport-call spec, and the neutral error hierarchy. No transport / LLM
machinery here — see ``tests/unit/runtime/test_resilience.py`` for the
active policy + wrapper behaviour.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from protocore.contracts.resilience import (
    RETRYABLE_ERROR_CLASSES,
    IToolTransport,
    ResilienceAction,
    ResilienceDecision,
    ResilienceError,
    ResilienceErrorClass,
    RetryBudgetState,
    ToolTransportError,
    ToolTransportRetryBudgetExhausted,
    ToolTransportTimeout,
    TransportCallSpec,
)


class TestErrorTaxonomy:
    def test_error_classes_are_neutral_and_complete(self) -> None:
        # The neutral taxonomy must cover every recovery-relevant distinction
        # the design calls out (transient/retryable, deterministic/abort,
        # context-overflow->compress, payload->shrink, timeout->rebuild,
        # rate-limited, unknown).
        names = {c.value for c in ResilienceErrorClass}
        assert names == {
            "transient_retryable",
            "rate_limited",
            "timeout_rebuild",
            "context_overflow",
            "payload_too_large",
            "deterministic_abort",
            "unknown",
        }

    def test_retryable_set_excludes_compress_shrink_abort(self) -> None:
        # context_overflow (compress) / payload_too_large (shrink) /
        # deterministic_abort are NOT same-request transport retries.
        assert ResilienceErrorClass.transient_retryable in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.rate_limited in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.timeout_rebuild in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.unknown in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.context_overflow not in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.payload_too_large not in RETRYABLE_ERROR_CLASSES
        assert ResilienceErrorClass.deterministic_abort not in RETRYABLE_ERROR_CLASSES

    def test_action_set_is_small_and_named(self) -> None:
        names = {a.value for a in ResilienceAction}
        assert names == {
            "retry",
            "backoff_and_retry",
            "rebuild_and_retry",
            "compress_and_retry",
            "shrink_and_retry",
            "finalize_now",
            "abort",
        }


class TestNeutralErrors:
    def test_transport_errors_inherit_resilience_error(self) -> None:
        assert issubclass(ToolTransportError, ResilienceError)
        assert issubclass(ToolTransportTimeout, ToolTransportError)
        assert issubclass(ToolTransportRetryBudgetExhausted, ToolTransportError)

    def test_budget_exhausted_recommends_finalization(self) -> None:
        exc = ToolTransportRetryBudgetExhausted(
            host_key="h1", method_class="essential_read"
        )
        assert exc.finalization_recommended is True
        assert exc.host_key == "h1"
        assert exc.method_class == "essential_read"

    def test_transport_error_carries_optional_retry_after_hint(self) -> None:
        # a backend may attach a server-stated reset/retry-after on
        # the transport error so the universal layer honours it for the
        # rate_limited class. Default is None (no hint).
        assert ToolTransportError("boom").retry_after_seconds is None
        exc = ToolTransportError("429 rate limit", retry_after_seconds=12.0)
        assert exc.retry_after_seconds == 12.0
        # The hint is inherited by the typed subclasses too.
        timeout = ToolTransportTimeout("slow", retry_after_seconds=3.0)
        assert timeout.retry_after_seconds == 3.0


class TestResilienceDecision:
    def test_decision_is_frozen(self) -> None:
        d = ResilienceDecision(
            error_class=ResilienceErrorClass.transient_retryable,
            action=ResilienceAction.backoff_and_retry,
        )
        with pytest.raises(ValidationError):
            d.retryable = True  # type: ignore[misc]

    def test_backoff_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ResilienceDecision(
                error_class=ResilienceErrorClass.transient_retryable,
                action=ResilienceAction.backoff_and_retry,
                backoff_seconds=-1.0,
            )


class TestRetryBudgetState:
    def test_full_starts_at_capacity(self) -> None:
        b = RetryBudgetState.full(
            max_tokens=100.0, token_ratio=0.1, suppress_below_ratio=0.5
        )
        assert b.tokens == 100.0
        assert b.suppress_threshold == 50.0
        assert b.should_suppress_retry() is False

    def test_consume_retry_floors_at_zero(self) -> None:
        b = RetryBudgetState(
            tokens=0.5, max_tokens=10.0, token_ratio=0.1, suppress_below_ratio=0.0
        )
        b.consume_retry()
        assert b.tokens == 0.0
        b.consume_retry()
        assert b.tokens == 0.0  # floored

    def test_deposit_success_caps_at_max(self) -> None:
        b = RetryBudgetState(
            tokens=9.95, max_tokens=10.0, token_ratio=0.1, suppress_below_ratio=0.5
        )
        b.deposit_success()
        assert b.tokens == 10.0  # capped, not 10.05

    def test_suppress_when_at_or_below_threshold(self) -> None:
        b = RetryBudgetState(
            tokens=50.0, max_tokens=100.0, token_ratio=0.1, suppress_below_ratio=0.5
        )
        # exactly at threshold => suppress (<=)
        assert b.should_suppress_retry() is True
        b.tokens = 50.01
        assert b.should_suppress_retry() is False

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RetryBudgetState(
                tokens=1.0, max_tokens=0.0, token_ratio=0.1, suppress_below_ratio=0.5
            )


class TestTransportCallSpec:
    def test_spec_is_frozen_and_defaults_idempotent(self) -> None:
        spec = TransportCallSpec(host_key="h", tool="read", method="Read")
        assert spec.idempotent is True  # default-safe for reads
        assert spec.method_class == "essential_read"
        with pytest.raises(ValidationError):
            spec.idempotent = False  # type: ignore[misc]

    def test_mutation_spec_marks_non_idempotent(self) -> None:
        spec = TransportCallSpec(
            host_key="h", tool="write", method="Write", idempotent=False,
            method_class="mutation",
        )
        assert spec.idempotent is False

    def test_timeout_ms_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            TransportCallSpec(timeout_ms=-1)


class TestIToolTransportProtocol:
    def test_runtime_checkable_accepts_conforming_impl(self) -> None:
        class _T:
            async def invoke(self, spec: TransportCallSpec) -> str:
                return "ok"

        assert isinstance(_T(), IToolTransport)

    def test_runtime_checkable_rejects_non_conforming(self) -> None:
        class _NotT:
            def something_else(self) -> None: ...

        assert not isinstance(_NotT(), IToolTransport)

    def test_optional_rebuild_hook_is_accepted(self) -> None:
        # a transport MAY expose an optional ``rebuild`` coroutine
        # (stale-connection -> fresh client). Its presence must not break the
        # runtime-checkable invoke() protocol.
        class _T:
            async def invoke(self, spec: TransportCallSpec) -> str:
                return "ok"

            async def rebuild(self) -> None:
                return None

        assert isinstance(_T(), IToolTransport)
