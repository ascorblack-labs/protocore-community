"""Tests for :mod:`protocore.ingress`."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from protocore.constants import MAX_ENVELOPE_PAYLOAD_CHARS
from protocore.contracts.types import AgentEnvelope, EnvelopeKind
from protocore.ingress import EnvelopeParseError, parse_envelope, serialize_envelope


def test_round_trip() -> None:
    original = AgentEnvelope(kind=EnvelopeKind.task, payload="hello")
    raw = serialize_envelope(original)
    rebuilt = parse_envelope(raw)
    assert rebuilt.kind is EnvelopeKind.task
    assert rebuilt.payload == "hello"


def test_round_trip_cyrillic() -> None:
    original = AgentEnvelope(kind=EnvelopeKind.task, payload="Тестовое сообщение")
    raw = serialize_envelope(original)
    rebuilt = parse_envelope(raw)
    assert rebuilt.payload == "Тестовое сообщение"


def test_parse_invalid_json() -> None:
    with pytest.raises(EnvelopeParseError):
        parse_envelope("not json")


def test_parse_schema_mismatch() -> None:
    with pytest.raises(EnvelopeParseError):
        parse_envelope('{"foo": "bar"}')  # missing kind/payload


def test_parse_bytes_input() -> None:
    original = AgentEnvelope(kind=EnvelopeKind.result, payload="done")
    raw = serialize_envelope(original).encode("utf-8")
    rebuilt = parse_envelope(raw)
    assert rebuilt.payload == "done"


def test_parse_non_utf8_bytes_raises_envelope_error() -> None:
    # Contract: parse_envelope raises EnvelopeParseError on ANY failure,
    # including a bytes payload that is not valid UTF-8 (regression: this
    # previously leaked a raw UnicodeDecodeError from the decode step).
    with pytest.raises(EnvelopeParseError):
        parse_envelope(b"\xff\xfe{invalid}")


def test_oversize_payload_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        AgentEnvelope(
            kind=EnvelopeKind.task,
            payload="x" * (MAX_ENVELOPE_PAYLOAD_CHARS + 1),
        )
