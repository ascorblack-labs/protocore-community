"""The safety-band resolver in query.

Covers ``_resolve_safety_band_value`` — the band lookup that
``_drive_one_stream`` consults to reduce ``LLMRequest.max_tokens`` by the
calibrated drift margin. The band itself is whatever the host wired onto the
run's state; all the loop asks of it is ``current()``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from protocore.contracts.run_state import RunScopedState
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.query import _resolve_safety_band_value


class _Band:
    """The whole of what the loop asks a safety band for."""

    def __init__(self, value: int) -> None:
        self._value = value

    def current(self) -> int:
        return self._value


def _engine_stub(*, rc: LoopConstants, band: object | None) -> object:
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.rc = rc
    engine.run_state = RunScopedState(adaptive_safety_band=band)
    return engine


def test_resolver_returns_zero_when_killswitch_off() -> None:
    band = _Band(512)
    rc = LoopConstants(adaptive_safety_band_enabled=False)
    engine = _engine_stub(rc=rc, band=band)
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_returns_zero_when_the_run_carries_no_band() -> None:
    rc = LoopConstants()  # kill-switch defaults to enabled
    engine = _engine_stub(rc=rc, band=None)
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_reads_band_current() -> None:
    band = _Band(768)
    rc = LoopConstants()
    engine = _engine_stub(rc=rc, band=band)
    assert _resolve_safety_band_value(engine) == 768


def test_resolver_negative_band_clamped_to_zero() -> None:
    """Defensive: a band that returns a negative current shouldn't subtract."""

    class _BrokenBand:
        def current(self) -> int:
            return -42

    rc = LoopConstants()
    engine = _engine_stub(rc=rc, band=_BrokenBand())
    assert _resolve_safety_band_value(engine) == 0


def test_resolver_swallows_exception() -> None:
    """A band whose current() raises must not break the loop."""

    class _RaisingBand:
        def current(self) -> int:
            raise RuntimeError("oh no")

    rc = LoopConstants()
    engine = _engine_stub(rc=rc, band=_RaisingBand())
    assert _resolve_safety_band_value(engine) == 0
