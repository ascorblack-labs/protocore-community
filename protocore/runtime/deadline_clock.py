"""Restore one deadline budget across snapshot resume.

Wall-clock epoch alone cannot preserve consumed time: a persisted epoch ahead
of the recovering node (backward clock correction, VM restore, cross-node
skew) makes ``max(0, now - epoch)`` restart elapsed at zero on every re-drive.
This module keeps a monotonic-independent consumed duration next to the epoch
and re-anchors from the larger of that duration and observed wall elapsed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class InvalidDeadlineSnapshot(ValueError):
    """Persisted deadline clock is not a finite nonnegative number."""


@dataclass(frozen=True, slots=True)
class DeadlineClockRestore:
    """Validated epoch and the monotonic anchor that preserves consumed time."""

    epoch: float
    monotonic_anchor: float
    elapsed_seconds: float


def parse_deadline_seconds(value: object, *, field_name: str) -> float:
    """Return a finite nonnegative duration/epoch, or raise before any restore."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDeadlineSnapshot(f"{field_name} must be a finite nonnegative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise InvalidDeadlineSnapshot(f"{field_name} must be a finite nonnegative number")
    return parsed


def consumed_deadline_seconds(*, started_monotonic: float, now_monotonic: float) -> float:
    """Seconds already charged to the budget, or 0.0 when the clock is unstamped."""
    if started_monotonic == 0.0:
        return 0.0
    elapsed = now_monotonic - started_monotonic
    if elapsed < 0.0:
        return 0.0
    return elapsed


def restore_deadline_clock(
    *,
    persisted_epoch: object = 0.0,
    persisted_elapsed: object = 0.0,
    now_wall: float,
    now_monotonic: float,
) -> DeadlineClockRestore:
    """Validate snapshot fields and re-anchor without restarting elapsed.

    A future-relative epoch contributes no wall elapsed. Already-consumed time
    still binds the next resume so a second re-drive cannot reset the budget.
    Exact ``0.0`` remains the unstamped sentinel for both fields.
    """
    epoch = parse_deadline_seconds(persisted_epoch, field_name="run_started_epoch")
    stored_elapsed = parse_deadline_seconds(persisted_elapsed, field_name="run_deadline_elapsed_seconds")
    wall_elapsed = 0.0
    if epoch > 0.0 and now_wall >= epoch:
        wall_elapsed = now_wall - epoch
    elapsed = stored_elapsed if stored_elapsed > wall_elapsed else wall_elapsed
    if epoch == 0.0 and elapsed == 0.0:
        return DeadlineClockRestore(epoch=0.0, monotonic_anchor=0.0, elapsed_seconds=0.0)
    return DeadlineClockRestore(
        epoch=epoch,
        monotonic_anchor=now_monotonic - elapsed,
        elapsed_seconds=elapsed,
    )
