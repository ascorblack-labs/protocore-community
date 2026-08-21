"""Single ingress contract — :func:`parse_envelope`.

Defensive parse for cross-component messaging. Enforces :data:`MAX_ENVELOPE_PAYLOAD_CHARS`.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from protocore.contracts.types import AgentEnvelope
from protocore.json_utils import OutputParserException, parse_complete_json


class EnvelopeParseError(Exception):
    """Failed to parse an :class:`AgentEnvelope` from input."""


def parse_envelope(raw: str | bytes) -> AgentEnvelope:
    """Parse a JSON envelope to an :class:`AgentEnvelope`.

    Raises :class:`EnvelopeParseError` on any failure. Multilingual safe
    (Cyrillic and CJK in payload pass through unchanged).
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise EnvelopeParseError(f"envelope is not valid UTF-8: {e}") from e
    else:
        text = raw
    try:
        data = parse_complete_json(text)
    except OutputParserException as e:
        raise EnvelopeParseError(f"invalid JSON envelope: {e}") from e
    try:
        return AgentEnvelope.model_validate(data)
    except ValidationError as e:
        raise EnvelopeParseError(f"envelope schema mismatch: {e}") from e


def serialize_envelope(envelope: AgentEnvelope) -> str:
    """Serialize an :class:`AgentEnvelope` to JSON (UTF-8, ensure_ascii=False)."""
    return json.dumps(
        envelope.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = ["EnvelopeParseError", "parse_envelope", "serialize_envelope"]
