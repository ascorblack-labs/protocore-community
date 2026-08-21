"""Multilingual token counting heuristic.

:class:`LanguageProfile` enum with **cyrillic_in_json_escape** profile
because UTF-8 JSON-escape of Cyrillic chars doubles the byte cost
(the cef6e778 root cause).

This is a **heuristic** estimator — provider-exact counts come via
:meth:`~protocore.contracts.llm.ILLMProvider.count_tokens`. Used for
pre-flight budget checks where exact token counts aren't yet available.
"""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from protocore.contracts.runtime_constants import RuntimeConstants


class LanguageProfile(StrEnum):
    """Per-text language profile for chars-per-token heuristic."""

    latin_prose = "latin_prose"
    cyrillic_prose = "cyrillic_prose"
    cyrillic_in_json_escape = "cyrillic_in_json_escape"
    cjk = "cjk"
    json_struct = "json_struct"


# Cyrillic-in-JSON-escape detector: \uNNNN where NNNN is in [0400, 04FF].
_CYRILLIC_JSON_ESCAPE_RE = re.compile(r"\\u04[0-9a-fA-F]{2}")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_CJK_RE = re.compile(r"[　-鿿]")
_JSON_STRUCT_CHARS: Final[frozenset[str]] = frozenset('{}[]",:')


def detect_profile(text: str) -> LanguageProfile:
    """Classify ``text`` into a :class:`LanguageProfile`.

    Heuristic:
        1. ``\\u04xx`` escapes present → ``cyrillic_in_json_escape``
        2. Cyrillic chars present → ``cyrillic_prose``
        3. CJK chars present → ``cjk``
        4. ≥30% structural chars → ``json_struct``
        5. otherwise → ``latin_prose``
    """
    if not text:
        return LanguageProfile.latin_prose

    if _CYRILLIC_JSON_ESCAPE_RE.search(text):
        return LanguageProfile.cyrillic_in_json_escape
    if _CYRILLIC_RE.search(text):
        return LanguageProfile.cyrillic_prose
    if _CJK_RE.search(text):
        return LanguageProfile.cjk

    struct_count = sum(1 for ch in text if ch in _JSON_STRUCT_CHARS)
    if struct_count > 0 and (struct_count / len(text)) >= 0.30:
        return LanguageProfile.json_struct
    return LanguageProfile.latin_prose


def chars_per_token(profile: LanguageProfile, rc: RuntimeConstants) -> float:
    """Return the chars-per-token ratio for a profile from RC."""
    if profile is LanguageProfile.latin_prose:
        return rc.token_count_chars_per_token_latin
    if profile is LanguageProfile.cyrillic_prose:
        return rc.token_count_chars_per_token_cyrillic
    if profile is LanguageProfile.cyrillic_in_json_escape:
        return rc.token_count_chars_per_token_cyrillic_json_escape
    if profile is LanguageProfile.cjk:
        return rc.token_count_chars_per_token_cjk
    return rc.token_count_chars_per_token_json_struct


def estimate_tokens(text: str, rc: RuntimeConstants) -> int:
    """Estimate token count of ``text`` by partitioning it per profile.

    Each character class is costed at its own chars-per-token ratio rather
    than classifying the whole blob by the first matching rule. Whole-text
    first-match classification (the :func:`detect_profile` heuristic) badly
    over-counts mixed content: a single ``\\u04xx`` escape or one Cyrillic
    char anywhere forced 40K of Latin into the ``cyrillic_in_json_escape``
    (cpt 1.2) / ``cyrillic_prose`` (cpt 2.5) bucket — a 3.3x / 1.6x
    over-count that falsely tripped compaction on healthy history.

    Partition (each char counted exactly once, highest-cost class wins):
        1. ``\\u04xx`` escape sequences → ``cyrillic_in_json_escape`` cpt
        2. remaining Cyrillic chars → ``cyrillic_prose`` cpt
        3. CJK chars → ``cjk`` cpt
        4. JSON structural chars → ``json_struct`` cpt
        5. everything else → ``latin_prose`` cpt
    """
    if not text:
        return 0

    escape_chars = 0
    for match in _CYRILLIC_JSON_ESCAPE_RE.finditer(text):
        escape_chars += match.end() - match.start()

    cyrillic_chars = 0
    cjk_chars = 0
    struct_chars = 0
    for ch in text:
        if _CYRILLIC_RE.match(ch):
            cyrillic_chars += 1
        elif _CJK_RE.match(ch):
            cjk_chars += 1
        elif ch in _JSON_STRUCT_CHARS:
            struct_chars += 1

    # `\uNNNN` escapes are pure ASCII; their `\u04` Cyrillic-hex bytes are not
    # matched by `_CYRILLIC_RE`, so escape_chars and cyrillic_chars don't
    # overlap. Remaining chars fall through to the Latin bucket.
    latin_chars = len(text) - escape_chars - cyrillic_chars - cjk_chars - struct_chars

    total = (
        escape_chars / rc.token_count_chars_per_token_cyrillic_json_escape
        + cyrillic_chars / rc.token_count_chars_per_token_cyrillic
        + cjk_chars / rc.token_count_chars_per_token_cjk
        + struct_chars / rc.token_count_chars_per_token_json_struct
        + latin_chars / rc.token_count_chars_per_token_latin
    )
    return max(1, round(total))


__all__ = [
    "LanguageProfile",
    "chars_per_token",
    "detect_profile",
    "estimate_tokens",
]
