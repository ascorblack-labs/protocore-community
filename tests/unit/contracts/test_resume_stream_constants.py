"""Runtime bounds for the durable resume-command consumer."""

import pytest
from pydantic import ValidationError

from protocore import RuntimeConstants


def test_resume_stream_defaults_keep_heartbeat_inside_reclaim_window() -> None:
    constants = RuntimeConstants()

    assert constants.resume_stream_read_block_ms == 1_000
    assert constants.resume_stream_reclaim_idle_ms == 30_000
    assert constants.resume_stream_heartbeat_ms == 5_000
    assert constants.resume_stream_heartbeat_ms < constants.resume_stream_reclaim_idle_ms


def test_resume_stream_heartbeat_requires_reclaim_margin() -> None:
    with pytest.raises(ValidationError, match=r"resume_stream_heartbeat_ms \* 3 must be"):
        RuntimeConstants(
            resume_stream_heartbeat_ms=10_001,
            resume_stream_reclaim_idle_ms=30_000,
        )


def test_resume_stream_heartbeat_accepts_exact_reclaim_margin() -> None:
    constants = RuntimeConstants(
        resume_stream_heartbeat_ms=10_000,
        resume_stream_reclaim_idle_ms=30_000,
    )

    assert constants.resume_stream_heartbeat_ms == 10_000
