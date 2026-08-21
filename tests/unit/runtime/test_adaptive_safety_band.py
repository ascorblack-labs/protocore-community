"""Unit tests for AdaptiveSafetyBand estimator drift.

Covers:
 - Initial band value clamping
 - observe_400 math: B1 ceiling-jump fix (step_cap=current*4, shortfall-seeded)
 - observe_400 negative-shortfall guard (over-counting case)
 - observe_actual narrowing from successful calls (EMA + 95th-percentile)
 - min/max band clamps
 - B4 persist gating: persist is a no-op when persist_enabled=False

Ported from side-branch commit 23452d0 onto canonical
the canonical core.
"""

from __future__ import annotations

import pytest

from protocore.runtime.adaptive_safety_band import (
    AdaptiveBandSnapshot,
    AdaptiveSafetyBand,
)


def test_adaptive_band_initial() -> None:
    b = AdaptiveSafetyBand(provider="vllm", model="Qwen", initial=1024)
    assert b.current() == 1024


def test_adaptive_band_observe_400_doubles_on_small_shortfall() -> None:
    """B1: small shortfall doubles the band, doesn't jump to ceiling."""
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=512, max_band=8192
    )
    b.observe_400(
        requested_max_tokens=2000,
        estimated_prompt=8000,
        provider_reported_total=8200,
        context_window=10000,
    )
    # Math: shortfall = max(0, 8200 + 512 - 8000 - 2000) = 0
    # (the reported total fit inside prompt+max_tokens budget), so the
    # floor doubling dominates: target = max(1024, min(2048, 64)) = 1024.
    assert b.current() == 1024


def test_adaptive_band_observe_400_clamps_negative_shortfall() -> None:
    """B1: over-counting case (shortfall<0) should not widen pathologically."""
    b = AdaptiveSafetyBand(provider="vllm", model="Qwen", initial=512)
    b.observe_400(
        requested_max_tokens=1000,
        estimated_prompt=10000,
        provider_reported_total=9000,
        context_window=20000,
    )
    # Math: shortfall = max(0, 9000 + 512 - 10000 - 1000) = 0,
    # target = max(1024, min(2048, max(0, 64))) = 1024
    assert b.current() == 1024


def test_adaptive_band_observe_400_seeds_observation() -> None:
    """B1: observe_400 seeds observations so subsequent successes can narrow.

    The seeded value is the *real* shortfall the band must cover,
    i.e. ``provider_reported_total + safety_overhead - estimated_prompt -
    requested_max_tokens`` (NOT the old ``total - estimated_prompt`` which
    ignored the output budget).
    """
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=512, safety_overhead=512
    )
    b.observe_400(
        requested_max_tokens=2000,
        estimated_prompt=8000,
        provider_reported_total=12000,
        context_window=20000,
    )
    snap = b.snapshot()
    # shortfall = 12000 + 512 - 8000 - 2000 = 2512 (window headroom 10000)
    assert 2512 in snap.observations


def test_adaptive_band_observe_400_does_not_overwiden_by_output_budget() -> None:
    """Regression: observe_400 must honor the documented invariant
    ``estimated_prompt + requested_max_tokens + new_band >=
    provider_reported_total + safety_overhead`` — it must NOT widen the band
    by the full requested output budget.

    Scenario: the local estimate (10_000) was nearly right — the provider
    reported only ~300 tokens of drift above prompt+max_tokens, well within
    the fixed safety_overhead, so there is essentially no genuine drift to
    absorb and the band should stay at its calibration floor.

    Under the OLD (buggy) math ``shortfall = total - estimated_prompt`` the
    requested output budget (40_000) would be mis-counted as drift, seeding
    an enormous observation (~40_300) and widening the band toward its
    ceiling. The fixed math subtracts requested_max_tokens, so the band
    stays at the doubling floor and seeds no spurious observation.
    """
    initial = 512
    overhead = 512
    estimated_prompt = 10_000
    requested_max_tokens = 40_000
    # Provider total sits just above prompt+max_tokens but within overhead,
    # so the real shortfall is 0 once the output budget is accounted for.
    provider_reported_total = estimated_prompt + requested_max_tokens + 300
    b = AdaptiveSafetyBand(
        provider="vllm",
        model="Qwen",
        initial=initial,
        max_band=32_768,
        safety_overhead=overhead,
    )
    b.observe_400(
        requested_max_tokens=requested_max_tokens,
        estimated_prompt=estimated_prompt,
        provider_reported_total=provider_reported_total,
        context_window=131_072,
    )
    # Correct band: the genuine shortfall is
    #   max(0, total + overhead - est - max_tokens)
    #   = max(0, 300 + 512) ... minus the budget already covered = 812
    # which is below the doubling floor (2 * 512 = 1024), so the band must
    # land at exactly the floor and never approach the ceiling.
    assert b.current() == initial * 2
    # And it must NOT have seeded an observation anywhere near the output
    # budget — that would prove the budget leaked into the drift estimate.
    snap = b.snapshot()
    assert all(obs < requested_max_tokens for obs in snap.observations)
    # The seeded shortfall (if any) must equal the invariant-derived value,
    # not the raw provider_total - estimated_prompt over-count.
    over_count = provider_reported_total - estimated_prompt  # old buggy value
    assert over_count not in snap.observations


def test_adaptive_band_observe_400_output_budget_not_counted_as_drift() -> None:
    """Regression (default-overhead, no new ctor kwarg).

    This test deliberately uses ONLY the pre-existing constructor surface
    (``initial`` / ``max_band``) and the default safety overhead so it runs
    against the unfixed module too. On the buggy ``shortfall = total -
    estimated_prompt`` math the large requested output budget is counted as
    drift, widening the band to the doubling-step ceiling (2048) and seeding
    the over-count (~40_300). The fixed math accounts for the output budget,
    leaving the band at the floor (1024) and seeding only the real residual.
    """
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=512, max_band=32_768
    )
    b.observe_400(
        requested_max_tokens=40_000,
        estimated_prompt=10_000,
        provider_reported_total=50_300,  # only 300 over prompt+max_tokens
        context_window=131_072,
    )
    # Fixed: band stays at the doubling floor; old buggy math hit 2048.
    assert b.current() == 1024
    # The output budget must never leak into the seeded drift observation.
    assert 40_300 not in b.snapshot().observations
    assert all(obs < 40_000 for obs in b.snapshot().observations)


def test_adaptive_band_observe_400_clamped_by_context_window() -> None:
    """The band can never push prompt+max_tokens+band past the
    provider context window — context_window caps the widening."""
    estimated_prompt = 7_000
    requested_max_tokens = 2_000
    context_window = 10_000
    # Huge reported total would demand a giant band, but only 1000 tokens of
    # window headroom remain above prompt+max_tokens, so the band is capped.
    b = AdaptiveSafetyBand(
        provider="vllm",
        model="Qwen",
        initial=512,
        max_band=8192,
        safety_overhead=512,
    )
    b.observe_400(
        requested_max_tokens=requested_max_tokens,
        estimated_prompt=estimated_prompt,
        provider_reported_total=50_000,
        context_window=context_window,
    )
    window_headroom = context_window - estimated_prompt - requested_max_tokens
    assert b.current() <= window_headroom


def test_adaptive_band_observe_400_genuine_drift_widens() -> None:
    """Sanity: a real under-count (drift exceeds prompt+max_tokens
    budget even after accounting for the output budget) still widens the
    band — the fix must not suppress legitimate widening."""
    estimated_prompt = 8_000
    requested_max_tokens = 1_000
    overhead = 512
    # Provider reports 5000 tokens above what we budgeted — genuine drift.
    provider_reported_total = estimated_prompt + requested_max_tokens + 5_000
    b = AdaptiveSafetyBand(
        provider="vllm",
        model="Qwen",
        initial=512,
        max_band=32_768,
        safety_overhead=overhead,
    )
    b.observe_400(
        requested_max_tokens=requested_max_tokens,
        estimated_prompt=estimated_prompt,
        provider_reported_total=provider_reported_total,
        context_window=131_072,
    )
    expected_shortfall = (
        provider_reported_total
        + overhead
        - estimated_prompt
        - requested_max_tokens
    )
    assert b.current() > 512  # widened beyond initial
    assert expected_shortfall in b.snapshot().observations


def test_adaptive_band_narrows_after_successful_observations() -> None:
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=1024, history=10
    )
    for _ in range(20):
        b.observe_actual(estimated=1000, actual=1010)
    assert b.current() < 1024


def test_adaptive_band_clamps_to_max() -> None:
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=1024, max_band=4096
    )
    b.observe_400(
        requested_max_tokens=10000,
        estimated_prompt=20000,
        provider_reported_total=100000,
        context_window=50000,
    )
    assert b.current() == 4096


def test_adaptive_band_clamps_to_min() -> None:
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", initial=64, min_band=64
    )
    for _ in range(50):
        b.observe_actual(estimated=1000, actual=1000)
    assert b.current() == 64


@pytest.mark.asyncio
async def test_adaptive_band_persist_no_op_when_disabled() -> None:
    """B4 fix: persist_enabled=False should make persist() a no-op."""

    class _CountingStore:
        def __init__(self) -> None:
            self.puts = 0

        async def get(
            self, *, provider: str, model: str
        ) -> AdaptiveBandSnapshot | None:
            return None

        async def put(self, snap: AdaptiveBandSnapshot) -> None:
            self.puts += 1

    store = _CountingStore()
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", store=store, persist_enabled=False
    )
    await b.persist()
    assert store.puts == 0


@pytest.mark.asyncio
async def test_adaptive_band_persist_called_when_enabled() -> None:
    class _CountingStore:
        def __init__(self) -> None:
            self.puts = 0

        async def get(
            self, *, provider: str, model: str
        ) -> AdaptiveBandSnapshot | None:
            return None

        async def put(self, snap: AdaptiveBandSnapshot) -> None:
            self.puts += 1

    store = _CountingStore()
    b = AdaptiveSafetyBand(
        provider="vllm", model="Qwen", store=store, persist_enabled=True
    )
    await b.persist()
    assert store.puts == 1
