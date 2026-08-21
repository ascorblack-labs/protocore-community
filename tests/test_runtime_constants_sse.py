from __future__ import annotations

import pytest
from pydantic import ValidationError

from protocore.contracts.runtime_constants import RuntimeConstants


def test_sse_runtime_constants_defaults_round_trip() -> None:
    rc = RuntimeConstants()
    dumped = rc.model_dump()

    assert dumped["sse_heartbeat_interval_seconds"] == 15.0
    assert dumped["sse_reconnect_retry_ms"] == 2000
    assert dumped["sse_subscribe_block_ms"] == 5000
    assert dumped["sse_replay_batch_count"] == 1000
    assert RuntimeConstants.model_validate(dumped) == rc


def test_sse_runtime_constants_override_round_trip() -> None:
    rc = RuntimeConstants.model_validate(
        {
            "sse_heartbeat_interval_seconds": 7.5,
            "sse_reconnect_retry_ms": 750,
            "sse_subscribe_block_ms": 250,
            "sse_replay_batch_count": 25,
        }
    )

    assert rc.sse_heartbeat_interval_seconds == 7.5
    assert rc.sse_reconnect_retry_ms == 750
    assert rc.sse_subscribe_block_ms == 250
    assert rc.sse_replay_batch_count == 25
    assert RuntimeConstants.model_validate(rc.model_dump()) == rc


@pytest.mark.parametrize(
    "field",
    [
        "sse_heartbeat_interval_seconds",
        "sse_reconnect_retry_ms",
        "sse_subscribe_block_ms",
        "sse_replay_batch_count",
    ],
)
def test_sse_runtime_constants_require_positive_values(field: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeConstants.model_validate({field: 0})
