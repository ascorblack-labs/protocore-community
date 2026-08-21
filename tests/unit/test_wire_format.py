"""Tests for :mod:`protocore.runtime.wire_format`."""
from __future__ import annotations

from protocore.contracts.types import CompactionSourceRef
from protocore.runtime.wire_format import (
    is_compacted_placeholder,
    parse_compacted_placeholder,
    render_compacted_placeholder,
)


def _ref() -> CompactionSourceRef:
    return CompactionSourceRef(
        blob_ref="tenant1/abc",
        sha256="deadbeef" * 8,
        original_tokens=12345,
        label="tool_result",
    )


def test_render_byte_deterministic() -> None:
    ref = _ref()
    a = render_compacted_placeholder(ref, "SNAPSHOT")
    b = render_compacted_placeholder(ref, "SNAPSHOT")
    assert a == b


def test_round_trip_snapshot() -> None:
    ref = _ref()
    text = render_compacted_placeholder(ref, "SNAPSHOT")
    parsed = parse_compacted_placeholder(text)
    assert parsed is not None
    rebuilt_ref, variant = parsed
    assert rebuilt_ref == ref
    assert variant == "SNAPSHOT"


def test_round_trip_rerun() -> None:
    ref = _ref()
    text = render_compacted_placeholder(ref, "RERUN")
    parsed = parse_compacted_placeholder(text)
    assert parsed is not None
    _, variant = parsed
    assert variant == "RERUN"


def test_is_compacted_placeholder() -> None:
    ref = _ref()
    text = render_compacted_placeholder(ref, "SNAPSHOT")
    assert is_compacted_placeholder(text)
    assert not is_compacted_placeholder("regular content")
    assert not is_compacted_placeholder("")


def test_parse_invalid_returns_none() -> None:
    assert parse_compacted_placeholder("garbage") is None
    assert parse_compacted_placeholder("PROTOCOL_OTHER_V1:SNAPSHOT|x|y|1|z") is None
