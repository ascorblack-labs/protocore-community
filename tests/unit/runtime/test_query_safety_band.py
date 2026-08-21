"""AdaptiveSafetyBand resolver in query.

Covers ``_resolve_safety_band_value`` — the helper-bag-aware band
lookup that ``_drive_one_stream`` consults to reduce
``LLMRequest.max_tokens`` by the calibrated drift margin.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.adaptive_safety_band import AdaptiveSafetyBand
from protocore.runtime.query import _resolve_safety_band_value


def _engine_stub(*, rc: RuntimeConstants, helpers: object | None) -> object:
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.rc = rc
    if helpers is None:
        del engine._helpers
    else:
        engine._helpers = helpers
    return engine


def test_resolver_returns_zero_when_killswitch_off() -> None:
    band = AdaptiveSafetyBand(provider="acme", model="acme-model-1", initial=512)
    rc = RuntimeConstants(adaptive_safety_band_enabled=False)
    engine = _engine_stub(rc=rc, helpers={"adaptive_safety_band": band})
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_returns_zero_when_helpers_missing() -> None:
    rc = RuntimeConstants()  # kill-switch defaults to enabled
    engine = _engine_stub(rc=rc, helpers=None)
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_returns_zero_when_band_missing() -> None:
    rc = RuntimeConstants()
    engine = _engine_stub(rc=rc, helpers={})
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_reads_band_current() -> None:
    band = AdaptiveSafetyBand(provider="vllm", model="qwen-7b", initial=768)
    rc = RuntimeConstants()
    engine = _engine_stub(rc=rc, helpers={"adaptive_safety_band": band})
    assert _resolve_safety_band_value(engine) == 768


def test_resolver_negative_band_clamped_to_zero() -> None:
    """Defensive: a band that returns a negative current shouldn't subtract."""

    class _BrokenBand:
        def current(self) -> int:
            return -42

    rc = RuntimeConstants()
    engine = _engine_stub(rc=rc, helpers={"adaptive_safety_band": _BrokenBand()})
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_swallows_exception() -> None:
    """A band whose current() raises must not break the loop."""

    class _RaisingBand:
        def current(self) -> int:
            raise RuntimeError("oh no")

    rc = RuntimeConstants()
    engine = _engine_stub(rc=rc, helpers={"adaptive_safety_band": _RaisingBand()})
    assert _resolve_safety_band_value(engine) == 0
