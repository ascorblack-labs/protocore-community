"""Tests for :mod:`protocore.runtime.token_counting`."""
from __future__ import annotations

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.runtime.token_counting import (
    LanguageProfile,
    chars_per_token,
    detect_profile,
    estimate_tokens,
)


def test_detect_latin() -> None:
    assert detect_profile("hello world") is LanguageProfile.latin_prose


def test_detect_cyrillic() -> None:
    assert detect_profile("Привет, мир") is LanguageProfile.cyrillic_prose


def test_detect_cyrillic_in_json_escape() -> None:
    """The cef6e778 root-cause profile."""
    escaped = '"\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442"'
    assert detect_profile(escaped) is LanguageProfile.cyrillic_in_json_escape


def test_detect_cjk() -> None:
    assert detect_profile("你好世界") is LanguageProfile.cjk


def test_detect_json_struct() -> None:
    assert detect_profile('{"k": [1, 2, 3], "v": null}') is LanguageProfile.json_struct


def test_detect_empty_defaults_latin() -> None:
    assert detect_profile("") is LanguageProfile.latin_prose


def test_chars_per_token_distinct_for_cyrillic_escape() -> None:
    rc = RuntimeConstants()
    # Cyrillic-in-JSON-escape MUST be lower cpt than cyrillic-prose (more
    # tokens per character due to UTF-8 escape doubling cost).
    cyrillic_cpt = chars_per_token(LanguageProfile.cyrillic_prose, rc)
    escape_cpt = chars_per_token(LanguageProfile.cyrillic_in_json_escape, rc)
    assert escape_cpt < cyrillic_cpt


def test_estimate_tokens_nonzero() -> None:
    rc = RuntimeConstants()
    assert estimate_tokens("hello", rc) >= 1


def test_estimate_tokens_empty() -> None:
    rc = RuntimeConstants()
    assert estimate_tokens("", rc) == 0


def test_estimate_tokens_scales_with_length() -> None:
    rc = RuntimeConstants()
    short = estimate_tokens("a" * 100, rc)
    longer = estimate_tokens("a" * 1000, rc)
    assert longer > short


def test_estimate_tokens_mixed_content_not_poisoned_by_one_escape() -> None:
    """Regression: one ``\\u04xx`` escape in mostly-Latin text must NOT force
    the whole blob into the cyrillic_in_json_escape (cpt 1.2) bucket.

    Before the per-class partition fix, 40K Latin chars + one escaped
    Cyrillic char estimated ~33.3K tokens vs ~10K for the same Latin text —
    a 3.3x over-count that falsely tripped per-iteration compaction.
    """
    rc = RuntimeConstants()
    latin = "a" * 40_000
    pure = estimate_tokens(latin, rc)
    # "\\u0440" is the 6 literal ASCII chars backslash-u-0-4-4-0, i.e. a JSON
    # ensure_ascii escape of a Cyrillic char — NOT a real Cyrillic codepoint.
    with_one_escape = estimate_tokens(latin + "\\u0440", rc)
    # The single escape adds at most a handful of tokens; it must not multiply
    # the estimate. Allow a tiny absolute delta for the escape itself.
    assert with_one_escape <= pure + 20
    # And it stays far below the old over-counted value.
    assert with_one_escape < pure * 2


def test_estimate_tokens_mixed_content_not_poisoned_by_one_cyrillic() -> None:
    """Regression: one raw Cyrillic char in mostly-Latin text must NOT force
    the whole blob into the cyrillic_prose (cpt 2.5) bucket (was a 1.6x
    over-count)."""
    rc = RuntimeConstants()
    latin = "a" * 40_000
    pure = estimate_tokens(latin, rc)
    with_one_cyrillic = estimate_tokens(latin + "я", rc)
    assert with_one_cyrillic <= pure + 20
    assert with_one_cyrillic < pure * 2


def test_estimate_tokens_pure_class_matches_single_profile() -> None:
    """The per-class partition must not change pure single-class estimates."""
    rc = RuntimeConstants()
    # Pure Latin: 1000 / 4.0 = 250.
    assert estimate_tokens("a" * 1000, rc) == 250
    # Pure Cyrillic prose: 1000 / 2.5 = 400.
    assert estimate_tokens("я" * 1000, rc) == 400
    # Pure CJK: round(1000 / 1.5) = 667.
    assert estimate_tokens("你" * 1000, rc) == 667


def test_estimate_tokens_escaped_cyrillic_still_costed_high() -> None:
    """Escaped Cyrillic chars must still be costed at the high (escape) cpt —
    the fix must not UNDER-count them, only stop them poisoning the rest."""
    rc = RuntimeConstants()
    escaped = "\\u0440" * 1000  # 6000 chars of pure \\u04xx escape sequences.
    estimate = estimate_tokens(escaped, rc)
    # 6000 / 1.2 = 5000; escapes cost far more than the same char count of
    # Latin (6000 / 4.0 = 1500).
    assert estimate == 5000
