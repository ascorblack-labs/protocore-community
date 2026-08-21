"""Universal resilience contracts (pure core, backend-agnostic).

A UNIVERSAL classify-then-act layer that covers BOTH the LLM provider calls
AND the tool/VM transport, generalising transport-stability work that was
previously scoped to a single backend.

Combines three independently-validated shapes:

* a ``resilientProvider``-style decorator — a classified-retryable status
  set, min-interval pacing, errors-as-observations;
* a rich error→strategy taxonomy + a jittered, thread-safe backoff;
* carry-forward primitives — a token-bucket failure-rate retry budget, a
  deadline-aware finalization reserve, classify-don't-retry, no-hedge on a
  single backend instance, never-raise-the-deadline, timeout-class symmetry.

Universal-core invariants:

* **No backend/wire symbols.** The taxonomy + policy here are domain-neutral.
  The host side maps its wire-format-specific LLM classifier and its
  transport classifier onto these neutral classes.
* **Behaviour-preserving by default.** Every knob defaults to a value that
  reproduces current behaviour; resilience is opt-in per tenant via
  ``RuntimeConstants.resilience_*``.
* **Horizontal-scale-safe.** The token bucket here is a pure data model;
  the runtime helper that mutates it (``protocore.runtime.resilience``)
  takes an injected lock so a pod keeps its own non-amplification budget
  without module-level state. Correctness-affecting state lives per-run.
* **Import boundary.** Pure ``pydantic`` / stdlib; core never imports
  the host. Guard: ``protocore/tests/test_core_import_boundary.py``.

The contracts split:

* :class:`ResilienceErrorClass` — the small neutral failure taxonomy.
* :class:`ResilienceAction` — the small neutral recovery-strategy set.
* :class:`ResilienceDecision` — one classify-then-act verdict.
* :class:`RetryBudgetState` — the pure token-bucket model.
* :class:`IToolTransport` — the injectable tool/VM transport hook the
  universal layer wraps (the host binds its concrete transport to it;
  core only owns the shape).
* :class:`ToolTransportError` / :class:`ToolTransportTimeout` /
  :class:`ToolTransportRetryBudgetExhausted` — neutral transport errors.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ResilienceError(Exception):
    """Base for the universal resilience layer's own errors."""


class ToolTransportError(ResilienceError):
    """A tool/VM transport call failed (structural OR transient).

    Backend adapters raise a subclass (or attach a classified verdict)
    so the universal layer can decide retry-vs-abort without importing
    any wire-format symbols.

    ``retry_after_seconds`` carries an OPTIONAL server-stated reset/
    retry-after hint (seconds from now): when a host signals a rate-limit
    reset window (e.g. an HTTP ``Retry-After`` header, a unified-reset
    header, or a "limit resets at: ..." message the backend parsed), the
    adapter sets it here so the universal :class:`ResiliencePolicy` can
    prefer the server-stated reset over a generic jittered backoff for the
    :attr:`ResilienceErrorClass.rate_limited` class. ``None``
    when no reset is known — the policy then uses its normal backoff.
    """

    def __init__(
        self,
        message: str = "tool transport error",
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ToolTransportTimeout(ToolTransportError):
    """A tool/VM transport call exceeded its per-call deadline.

    The neutral analogue of a backend ``DEADLINE_EXCEEDED`` /
    ``UNAVAILABLE "...timed out"`` class. Classified as
    :attr:`ResilienceErrorClass.transient_retryable` by the default
    transport classifier — but only retried when the call is idempotent
    (classify-don't-retry-mutations).
    """


class ToolTransportRetryBudgetExhausted(ToolTransportError):
    """The failure-rate retry budget for this host suppressed a retry.

    The structured give-up emitted when the per-host budget runs dry.
    Carries ``finalization_recommended=True`` so the loop finalises on best
    evidence instead of choosing "one more read" against a failing host.
    """

    finalization_recommended: bool = True

    def __init__(
        self,
        message: str = "transport retry budget exhausted",
        *,
        host_key: str = "",
        method_class: str = "",
    ) -> None:
        super().__init__(message)
        self.host_key = host_key
        self.method_class = method_class


class ResilienceErrorClass(StrEnum):
    """Backend-agnostic failure taxonomy (the *what kind* of failure).

    Deliberately small — the union of the recovery-relevant distinctions
    both reference agents converged on, with NO provider/wire specifics.
    The host classifiers (LLM ``FailoverReason``, the transport
    timeout predicate) collapse their rich, wire-format-aware verdicts
    onto exactly one of these neutral classes.
    """

    transient_retryable = "transient_retryable"
    """Transient/flaky failure that may clear on its own — network blip,
    5xx, single-VM hiccup, transport read-timeout. Retryable subject to
    the budget + deadline reserve + idempotency stance."""

    rate_limited = "rate_limited"
    """Provider/host signalled a rate limit (429 / quota / pushback).
    Retryable, and the backoff HONOURS a server-stated reset when one is
    known: a backend attaches the reset window as
    :attr:`ToolTransportError.retry_after_seconds` (or on the classified
    verdict), and the policy prefers ``max(server_reset, jittered_backoff)``
    with the server reset taken verbatim. Sleeping a generic short backoff
    while the host's window is still closed only re-issues into the same
    limit, so the server reset is authoritative when present and the
    deadline-reserve gate (not the generic backoff cap) decides whether a
    large reset still leaves room to retry."""

    timeout_rebuild = "timeout_rebuild"
    """A failure where the right move is to REBUILD the client/connection
    before retrying, not re-issue on the same (likely dead) socket. Covers
    a timeout that warrants a fresh client AND a reset/broken-pipe class
    (``ECONNRESET`` / ``EPIPE`` — a stale keep-alive socket). Retryable; the
    action is :attr:`ResilienceAction.rebuild_and_retry`, and the universal
    wrapper invokes the transport's optional ``rebuild()`` hook before the
    retry."""

    context_overflow = "context_overflow"
    """The request exceeded the model/context window — COMPRESS, do not
    failover (``context_overflow -> COMPRESS``). Not a transport retry;
    handled by the loop's compaction path."""

    payload_too_large = "payload_too_large"
    """The request/argument body itself is too large — SHRINK it (drop
    blocks / chunk the tool call) then retry
    (``payload_too_large -> shrink+retry``)."""

    deterministic_abort = "deterministic_abort"
    """A deterministic NO — auth/billing/policy-blocked/model-not-found /
    a malformed deterministic 4xx. Retrying unchanged hits the same wall;
    abort (or hand to a fallback model where one is configured)."""

    unknown = "unknown"
    """Unclassifiable. Default-safe: retry once with backoff under the
    budget, then abort (the conservative ``unknown -> retry with backoff``
    strategy plus the ``classify-don't-retry`` bias toward giving up on a
    failing host)."""


# Classes for which a (budget/deadline/idempotency-gated) retry is even
# considered. context_overflow / payload_too_large are handled by COMPRESS /
# SHRINK paths, NOT a same-request transport retry; deterministic_abort is a
# hard NO. Exposed so both the policy and tests share one source of truth.
RETRYABLE_ERROR_CLASSES: frozenset[ResilienceErrorClass] = frozenset(
    {
        ResilienceErrorClass.transient_retryable,
        ResilienceErrorClass.rate_limited,
        ResilienceErrorClass.timeout_rebuild,
        ResilienceErrorClass.unknown,
    }
)
"""Error classes that are candidate-retryable (still gated by budget,
deadline reserve, and the per-call idempotency stance)."""


class ResilienceAction(StrEnum):
    """The neutral recovery strategy chosen for a classified failure.

    Small by construction — adding one requires a corresponding handler
    in whichever loop consumes the decision. Mirrors the host
    ``RecoveryAction`` set but stays in pure core so the *policy* (which
    class → which action) is universal and testable without the host.
    """

    retry = "retry"
    """Re-attempt the same request immediately (transient, first retries)."""

    backoff_and_retry = "backoff_and_retry"
    """Wait a (jittered, decorrelated) backoff, then retry. Used for
    rate-limit / repeated-transient / unknown."""

    rebuild_and_retry = "rebuild_and_retry"
    """Rebuild the client/connection, then retry (timeout_rebuild)."""

    compress_and_retry = "compress_and_retry"
    """Compress/compact the request (context_overflow) then retry."""

    shrink_and_retry = "shrink_and_retry"
    """Shrink the request/argument body (payload_too_large) then retry."""

    finalize_now = "finalize_now"
    """Stop retrying and finalise on best evidence — the structured
    give-up emitted on retry-budget exhaustion / deadline-reserve
    pressure (``finalization_recommended``)."""

    abort = "abort"
    """No recovery available — surface the failure as terminal."""


@runtime_checkable
class ClassifiedLike(Protocol):
    """Structural shape of an attached classifier verdict.

    Both the LLM ``ClassifiedError`` and any transport verdict expose a
    ``reason`` and the boolean hints; the universal layer reads them
    structurally (``getattr``) so core never imports the host
    classifier type. ``reason`` may be a ``StrEnum`` or a plain string.
    """

    reason: Any
    retryable: bool


class ResilienceDecision(BaseModel):
    """One classify-then-act verdict produced by :class:`ResiliencePolicy`.

    Pure data — the policy decides; the caller (LLM loop / transport
    wrapper) executes. ``backoff_seconds`` is advisory (the caller may
    apply its own per-attempt schedule); ``finalization_recommended``
    is the handshake into the loop's guaranteed-terminal path.
    """

    model_config = ConfigDict(frozen=True)

    error_class: ResilienceErrorClass
    action: ResilienceAction
    retryable: bool = False
    """Whether a retry is permitted AFTER honouring budget + deadline +
    idempotency. ``False`` for ``abort`` / ``finalize_now`` / a
    non-idempotent transient."""

    backoff_seconds: float = Field(default=0.0, ge=0.0)
    """Advisory backoff before the retry. ``0.0`` for immediate-retry /
    non-retry actions."""

    finalization_recommended: bool = False
    """When ``True`` the caller should finalise on best evidence rather
    than retry (budget exhaustion / deadline pressure)."""

    reason_detail: str = ""
    """Short telemetry string — the underlying wire reason when known
    (mirrors the host classifier ``reason`` for audit)."""


class RetryBudgetState(BaseModel):
    """Pure token-bucket model for the failure-rate retry budget.

    The token-bucket retry pattern ("fixing retries with token buckets"):
    gate the *retry*, never the first attempt. Each retry consumes one
    token; each success deposits ``token_ratio`` tokens (capped at
    ``max_tokens``). When the bucket falls to
    ``max_tokens * suppress_below_ratio`` a retry is SUPPRESSED — the host
    is failing enough that re-issuing just feeds the storm; give up on best
    evidence instead.

    This is the per-(host, run) state the runtime helper mutates under an
    injected lock. It is intentionally a serialisable model so a caller
    MAY persist/inspect it; the *authority* is process-local-per-pod for
    non-amplification — a pod limiting its own multiplication is sufficient;
    only correctness-affecting state must be per-run durable.
    """

    model_config = ConfigDict(frozen=False)

    tokens: float
    max_tokens: float = Field(gt=0.0)
    token_ratio: float = Field(ge=0.0)
    suppress_below_ratio: float = Field(ge=0.0, le=1.0)

    @property
    def suppress_threshold(self) -> float:
        """Token level at/below which a retry is suppressed."""
        return self.max_tokens * self.suppress_below_ratio

    def should_suppress_retry(self) -> bool:
        """Whether the next retry must be suppressed (bucket too low)."""
        return self.tokens <= self.suppress_threshold

    def consume_retry(self) -> None:
        """Spend one token on an issued retry (floored at 0)."""
        self.tokens = max(0.0, self.tokens - 1.0)

    def deposit_success(self) -> None:
        """Deposit ``token_ratio`` tokens on a success (capped)."""
        self.tokens = min(self.max_tokens, self.tokens + self.token_ratio)

    @classmethod
    def full(
        cls,
        *,
        max_tokens: float,
        token_ratio: float,
        suppress_below_ratio: float,
    ) -> RetryBudgetState:
        """Construct a fresh, full bucket."""
        return cls(
            tokens=max_tokens,
            max_tokens=max_tokens,
            token_ratio=token_ratio,
            suppress_below_ratio=suppress_below_ratio,
        )


class TransportCallSpec(BaseModel):
    """Neutral description of one tool/VM transport call.

    Passed to :meth:`IToolTransport.invoke` so the universal layer can
    classify-then-act WITHOUT knowing the wire protocol. ``idempotent``
    is the load-bearing field: a non-idempotent (mutating) call is NEVER
    blind-retried (when a backend has no idempotency key,
    classify-don't-retry + read-back is the substitute).
    ``method_class`` lets the budget protect essential reads / mutations /
    terminal from cheap broad discovery draining first.
    """

    model_config = ConfigDict(frozen=True)

    host_key: str = ""
    """Stable per-host identifier (hashed URL) for budget bucketing.
    Empty when budgeting is off."""

    tool: str = ""
    """Agent-facing tool/verb name (telemetry only)."""

    method: str = ""
    """Backend method name (telemetry only)."""

    method_class: str = "essential_read"
    """Coarse value class — ``optional_discovery`` | ``essential_read`` |
    ``mutation`` | ``terminal``. Drains cheap discovery first; protects
    the rest (budget method-keying)."""

    idempotent: bool = True
    """Whether re-issuing the SAME request is safe. ``False`` for
    mutations — the universal layer then classifies-but-does-not-retry
    (read-back is the caller's job)."""

    timeout_ms: int = Field(default=0, ge=0)
    """Per-call deadline hint (0 = backend default). The universal layer
    only ever SHRINKS this to fit a finalization reserve — never widens
    it (never-raise-the-deadline)."""

    extra: Mapping[str, Any] = Field(default_factory=dict)
    """Backend-specific passthrough (request payload handle, etc.)."""


@runtime_checkable
class IToolTransport(Protocol):
    """Injectable tool/VM transport the universal resilience layer wraps.

    Core owns ONLY this shape (concrete impls live in the host).
    The host side binds its concrete transport (and any other backend) to
    it; the universal
    :func:`protocore.runtime.resilience.resilient_transport_call` drives it
    with classify-then-act + budget + backoff + deadline reserve, so
    transport resilience is identical across every backend.

    Implementations:

    * raise :class:`ToolTransportTimeout` (or attach a classified verdict
      whose ``error_class`` is ``transient_retryable``/``timeout_rebuild``)
      for transient deadline/connection failures;
    * raise a structural :class:`ToolTransportError` (or a
      ``deterministic_abort`` verdict) for non-transient failures that
      must surface immediately;
    * are responsible for NOT mutating shared state on a re-issue — the
      universal layer guarantees it only re-issues ``idempotent`` specs;
    * MAY expose an optional ``async def rebuild(self) -> None`` hook. When
      a failure classifies :attr:`ResilienceErrorClass.timeout_rebuild`
      (e.g. ``ECONNRESET`` / ``EPIPE`` — a stale keep-alive socket), the
      universal wrapper calls ``rebuild()`` (when present) BEFORE the retry
      so the next attempt runs on a fresh client/connection rather than the
      dead one. The hook is OPTIONAL: a transport without it is
      simply retried as-is (graceful no-op), and a ``rebuild()`` that raises
      is treated as best-effort (the retry still proceeds on the existing
      transport). It is therefore NOT part of the runtime-checkable surface.
    """

    async def invoke(self, spec: TransportCallSpec) -> Any:
        """Execute one transport call described by ``spec``; return the
        backend response, or raise a :class:`ToolTransportError` subclass.
        """
        ...


__all__ = [
    "RETRYABLE_ERROR_CLASSES",
    "ClassifiedLike",
    "IToolTransport",
    "ResilienceAction",
    "ResilienceDecision",
    "ResilienceError",
    "ResilienceErrorClass",
    "RetryBudgetState",
    "ToolTransportError",
    "ToolTransportRetryBudgetExhausted",
    "ToolTransportTimeout",
    "TransportCallSpec",
]
