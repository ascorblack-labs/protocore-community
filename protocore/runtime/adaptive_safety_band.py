"""Adaptive safety band for provider context window budgeting.

The local prompt-token estimator is never exact: it cannot apply the chat
template, it misjudges Cyrillic-in-JSON-escape inflation, and it cannot account
for provider-side serialization overhead. Production traces showed
this with a 1.3% drift that ate the entire fixed 512-token safety margin.

The adaptive safety band closes that gap by **learning** the actual drift
between local estimate and provider-reported prompt_tokens. The band's value
is subtracted from the available output budget; if the band is too small the
adapter observes a provider 400 ContextWindowExceededError, doubles the band,
and persists the new value so the next call uses the wider band.

Operationally the band is:
  - per (provider, model) — different tokenizers drift differently;
  - initialised conservatively (default 1024) — better to under-output than
    blow the window during the calibration phase;
  - bounded — [min_band, max_band] to prevent runaway shrinkage/expansion;
  - EMA-smoothed with 95th-percentile observations — so a single noisy
    request does not collapse the band, and a single legitimate widening
    sticks.

For persistence across pods / restarts a separate ``AdaptiveBandStore``
protocol is defined. The default implementation is in-process (no
persistence); a host implementation can back it onto Redis. Persisting
is intentionally a side effect of ``observe_400`` so calibrated values
survive restarts of the orchestrator that just discovered drift.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Final, Protocol

_DEFAULT_INITIAL: Final[int] = 1024
_DEFAULT_MIN_BAND: Final[int] = 64
_DEFAULT_MAX_BAND: Final[int] = 4096
_DEFAULT_HISTORY: Final[int] = 64
# Fixed extra headroom (tokens) the calibrated band must keep BEYOND the
# observed drift so a borderline-fitting request still clears the provider
# window. This is the documented ``safety_overhead`` of ``observe_400``'s
# invariant and replaces the historical fixed 512-token margin. Surfaced as
# ``RuntimeConstants.adaptive_safety_band_overhead``
# so it is dashboard-configurable; this Final is only the in-process default
# for callers (tests, leader engines) that construct the band without RC.
_DEFAULT_SAFETY_OVERHEAD: Final[int] = 512
_EMA_WEIGHT_PRIOR: Final[float] = 0.7
_EMA_WEIGHT_NEW: Final[float] = 0.3


@dataclass(frozen=True, slots=True)
class AdaptiveBandSnapshot:
    """Serializable snapshot of an adaptive band's state for persistence."""

    provider: str
    model: str
    ema: int
    observations: tuple[int, ...]
    min_band: int
    max_band: int


class AdaptiveBandStore(Protocol):
    """Optional persistence backend for adaptive bands.

    A no-op implementation is used by default. The host wires this onto
    Redis so band calibration survives orchestrator restarts and is shared
    across pods.
    """

    async def get(
        self, *, provider: str, model: str
    ) -> AdaptiveBandSnapshot | None: ...

    async def put(self, snapshot: AdaptiveBandSnapshot) -> None: ...


class NullAdaptiveBandStore:
    """No-op store; bands live in-process only."""

    async def get(self, *, provider: str, model: str) -> AdaptiveBandSnapshot | None:
        return None

    async def put(self, snapshot: AdaptiveBandSnapshot) -> None:
        return None


class AdaptiveSafetyBand:
    """Per-(provider, model) safety band that learns from estimator drift.

    Thread-safe via a single instance lock — operations are O(history) at
    worst and contention is negligible.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        initial: int = _DEFAULT_INITIAL,
        min_band: int = _DEFAULT_MIN_BAND,
        max_band: int = _DEFAULT_MAX_BAND,
        history: int = _DEFAULT_HISTORY,
        safety_overhead: int = _DEFAULT_SAFETY_OVERHEAD,
        store: AdaptiveBandStore | None = None,
        persist_enabled: bool = True,
    ) -> None:
        if min_band < 0:
            raise ValueError("min_band must be >= 0")
        if max_band < min_band:
            raise ValueError("max_band must be >= min_band")
        if history < 4:
            raise ValueError("history must be >= 4")
        if safety_overhead < 0:
            raise ValueError("safety_overhead must be >= 0")
        self._provider = provider
        self._model = model
        self._min = min_band
        self._max = max_band
        self._safety_overhead = safety_overhead
        self._ema = self._clamp(initial)
        self._observations: deque[int] = deque(maxlen=history)
        self._store: AdaptiveBandStore = store or NullAdaptiveBandStore()
        self._persist_enabled = persist_enabled
        self._lock = threading.Lock()

    def current(self) -> int:
        with self._lock:
            return self._ema

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def min_band(self) -> int:
        return self._min

    @property
    def max_band(self) -> int:
        return self._max

    @property
    def safety_overhead(self) -> int:
        return self._safety_overhead

    def observe_actual(self, *, estimated: int, actual: int) -> None:
        """Record an estimator-vs-actual observation from a successful call.

        ``estimated`` is what the local counter said the prompt would be;
        ``actual`` is the provider-reported ``usage.input_tokens``.

        We clamp the gap at 0 because the safety band only needs to absorb
        UNDER-counting (estimator under, provider over). When the estimator
        over-counts (gap < 0), the band is already protective and we record
        a zero observation — over time those pull the EMA down via
        ``_compute_target``'s 95th percentile, narrowing the band when the
        worst-case under-count is consistently small.
        """
        gap = max(0, actual - estimated)
        with self._lock:
            self._observations.append(gap)
            self._ema = self._clamp(self._compute_target())

    def observe_400(
        self,
        *,
        requested_max_tokens: int,
        estimated_prompt: int,
        provider_reported_total: int,
        context_window: int,
    ) -> None:
        """Widen the band immediately after a provider 400.

        Math: provider said the request totalled ``provider_reported_total``
        tokens against a window of ``context_window``. Our local plan
        allocated ``estimated_prompt + requested_max_tokens + current_band``
        which the provider considered overshot. The new band must be wide
        enough that ``estimated_prompt + requested_max_tokens + new_band ≥
        provider_reported_total + safety_overhead``. Solving for the band,
        the shortfall the band must cover is::

            shortfall = max(
                0,
                provider_reported_total + safety_overhead
                - estimated_prompt - requested_max_tokens,
            )

        The prior body computed ``shortfall =
        max(0, provider_reported_total - estimated_prompt)`` and never read
        ``requested_max_tokens`` (or ``safety_overhead``), which over-counted
        the drift by the FULL requested output budget. For a 8k-prompt /
        2k-output request that fit but reported a few hundred tokens of
        drift, the old math treated the entire 2k output as additional
        drift and over-widened the band, needlessly shrinking the per-call
        output budget on every subsequent call. The band must absorb only
        the gap between the provider total (plus the fixed safety overhead)
        and what our plan already accounted for (prompt + max_tokens).

        ``context_window`` caps the result: the band can never need to be so
        wide that ``estimated_prompt + requested_max_tokens + new_band`` would
        exceed the window — that allocation is the very thing the band is
        budgeted against, so widening past the window is meaningless.

        Previously this used ``max(doubled, shortfall*2)``
        which jumps straight to max_band on a moderate shortfall. We (a) clamp
        ``shortfall`` to ≥0 so over-counting / already-budgeted 400s don't
        pathologically inflate; (b) bound widening per-step to ``current*4``
        so a single 400 multiplies but does not saturate; (c) seed an
        observation with the actual shortfall so subsequent successful
        observations naturally pull the EMA back down.
        """
        with self._lock:
            shortfall = max(
                0,
                provider_reported_total
                + self._safety_overhead
                - estimated_prompt
                - requested_max_tokens,
            )
            # The band is budgeted on top of ``estimated_prompt +
            # requested_max_tokens``; it can never usefully exceed the room
            # the window leaves above that allocation.
            window_headroom = max(
                0, context_window - estimated_prompt - requested_max_tokens
            )
            shortfall = min(shortfall, window_headroom)
            doubled = self._ema * 2
            step_cap = self._ema * 4 if self._ema > 0 else self._max
            target = max(doubled, min(step_cap, max(shortfall, self._min)))
            target = min(target, window_headroom) if window_headroom > 0 else target
            self._ema = self._clamp(int(target))
            # Seed the observation so the recovery path can narrow when
            # successful calls follow.
            if shortfall > 0:
                self._observations.append(shortfall)

    def snapshot(self) -> AdaptiveBandSnapshot:
        with self._lock:
            return AdaptiveBandSnapshot(
                provider=self._provider,
                model=self._model,
                ema=self._ema,
                observations=tuple(self._observations),
                min_band=self._min,
                max_band=self._max,
            )

    async def persist(self) -> None:
        """Push the current snapshot to the store.

        No-op when ``persist_enabled`` is False — the constructor argument is
        the runtime kill-switch backing
        ``RuntimeConstants.adaptive_safety_band_persist``. Previously the
        constant existed but was never consulted, so flipping the operator
        switch off had no behaviour change.
        """
        if not self._persist_enabled:
            return
        await self._store.put(self.snapshot())

    @classmethod
    async def load_or_create(
        cls,
        *,
        provider: str,
        model: str,
        store: AdaptiveBandStore,
        initial: int = _DEFAULT_INITIAL,
        min_band: int = _DEFAULT_MIN_BAND,
        max_band: int = _DEFAULT_MAX_BAND,
        history: int = _DEFAULT_HISTORY,
        safety_overhead: int = _DEFAULT_SAFETY_OVERHEAD,
        persist_enabled: bool = True,
    ) -> AdaptiveSafetyBand:
        snapshot = await store.get(provider=provider, model=model) if persist_enabled else None
        band = cls(
            provider=provider,
            model=model,
            initial=snapshot.ema if snapshot else initial,
            min_band=min_band,
            max_band=max_band,
            history=history,
            safety_overhead=safety_overhead,
            store=store,
            persist_enabled=persist_enabled,
        )
        if snapshot:
            for obs in snapshot.observations:
                band._observations.append(obs)
        return band

    def _clamp(self, value: int) -> int:
        if value < self._min:
            return self._min
        if value > self._max:
            return self._max
        return value

    def _compute_target(self) -> int:
        if not self._observations:
            return self._ema
        ordered = sorted(self._observations)
        # 95th percentile so the band tracks worst case rather than average.
        index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        p95 = ordered[index]
        # EMA towards p95 — smooths single-call noise.
        smoothed = (_EMA_WEIGHT_PRIOR * self._ema) + (_EMA_WEIGHT_NEW * p95)
        return int(smoothed)


__all__ = [
    "AdaptiveBandSnapshot",
    "AdaptiveBandStore",
    "AdaptiveSafetyBand",
    "NullAdaptiveBandStore",
]
