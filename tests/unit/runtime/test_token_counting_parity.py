"""The optional native estimator must answer exactly what Python answers.

The token estimate feeds every budget in the core: compaction thresholds, the
context window check, session-memory sizing. A native implementation that is
merely close would move those thresholds by a token here and there, and the
symptom would be a run that compacts on one machine and not on another. So the
bar is equality, not approximation — the arithmetic is the same partition, the
same divisions in the same order, the same ties-to-even rounding, and ``f64``
is Python's ``float``.

The suite runs whether or not the extension is installed. Without it the Python
implementation is still exercised directly and the selection logic is still
checked; with it, every case is answered twice and compared.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime import token_counting
from protocore.runtime.token_counting import (
    DISABLE_NATIVE_ENV_VAR,
    NATIVE_ACTIVE,
    estimate_tokens,
    estimate_tokens_python,
)

native_required = pytest.mark.skipif(
    token_counting._native_estimate_tokens is None,
    reason="the native extension is an optional extra and is not installed",
)

# Deliberately mixed: each case is chosen to land characters in a different
# bucket of the partition, and several straddle two of them.
CORPUS: tuple[str, ...] = (
    "",
    "a",
    "hello world",
    "a" * 10_000,
    "Привет, мир",
    "я" * 1_000,
    "日本語のテキストです",
    "你好" * 500,
    '{"key": ["value", 1, null], "other": {"nested": true}}',
    r'{"text": "русский"}',
    r"Ѐӿӿҫ",
    # Escape-shaped text that is not an escape: one hex digit short, out of
    # range, or interrupted. The partition must not count these.
    r"\u04z9 Ԁ \u04 \\u0430",
    "Mixed Привет 日本語 {\"k\": 1} and \\u0435 all at once",
    "tab\tand\nnewlines\r\n",
    "emoji \U0001f600 and \U0001f680",
    "ӿ" * 100 + "a" * 100 + "、" * 100,
    " " * 500,
    "\\" * 200,
    "́combininǵ",
)

# Tunings that move each ratio independently, so a term swapped between two
# buckets in one implementation shows up as a disagreement.
TUNINGS: tuple[dict[str, float], ...] = (
    {},
    {"token_count_chars_per_token_latin": 1.0},
    {"token_count_chars_per_token_cyrillic": 7.5},
    {"token_count_chars_per_token_cyrillic_json_escape": 0.25},
    {"token_count_chars_per_token_cjk": 9.0},
    {"token_count_chars_per_token_json_struct": 0.5},
    {
        "token_count_chars_per_token_latin": 3.3,
        "token_count_chars_per_token_cyrillic": 2.1,
        "token_count_chars_per_token_cyrillic_json_escape": 1.7,
        "token_count_chars_per_token_cjk": 1.1,
        "token_count_chars_per_token_json_struct": 2.9,
    },
)


def _constants(tuning: dict[str, float]) -> LoopConstants:
    return LoopConstants().model_copy(update=dict(tuning))


@native_required
@pytest.mark.parametrize("text", CORPUS)
@pytest.mark.parametrize("tuning", TUNINGS)
def test_native_and_python_agree(text: str, tuning: dict[str, float]) -> None:
    rc = _constants(tuning)
    native = token_counting._native_estimate_tokens
    assert native is not None
    assert native(
        text,
        rc.token_count_chars_per_token_latin,
        rc.token_count_chars_per_token_cyrillic,
        rc.token_count_chars_per_token_cyrillic_json_escape,
        rc.token_count_chars_per_token_cjk,
        rc.token_count_chars_per_token_json_struct,
    ) == estimate_tokens_python(text, rc)


@native_required
@pytest.mark.parametrize("length", [1, 7, 64, 999, 5_000])
def test_native_and_python_agree_on_generated_mixtures(length: int) -> None:
    """Every rotation of a mixed alphabet, at several lengths."""
    alphabet = 'aZ9 Пя日、{}",:\\u0440\t\n'
    rc = LoopConstants()
    for offset in range(len(alphabet)):
        rotated = alphabet[offset:] + alphabet[:offset]
        text = (rotated * (length // len(rotated) + 1))[:length]
        native = token_counting._native_estimate_tokens
        assert native is not None
        assert native(
            text,
            rc.token_count_chars_per_token_latin,
            rc.token_count_chars_per_token_cyrillic,
            rc.token_count_chars_per_token_cyrillic_json_escape,
            rc.token_count_chars_per_token_cjk,
            rc.token_count_chars_per_token_json_struct,
        ) == estimate_tokens_python(text, rc), f"disagreed on {text!r}"


@pytest.mark.parametrize("text", CORPUS)
def test_the_selected_implementation_answers_the_python_result(text: str) -> None:
    """Whichever implementation is selected, the answer is the Python one."""
    rc = LoopConstants()
    assert estimate_tokens(text, rc) == estimate_tokens_python(text, rc)


def test_native_active_reports_which_implementation_is_selected() -> None:
    installed = token_counting._native_estimate_tokens is not None
    disabled = bool(os.environ.get(DISABLE_NATIVE_ENV_VAR))
    assert NATIVE_ACTIVE is (installed and not disabled)


@native_required
def test_the_native_estimator_can_be_switched_off_from_the_environment() -> None:
    """With the extension installed, the environment still selects Python.

    A fresh interpreter, because the selection is made once at import: this is
    a packaging question, not something that should be re-answered per call.
    """
    script = "import protocore.runtime.token_counting as t; print(t.NATIVE_ACTIVE)"

    def report(value: str | None) -> str:
        environment = dict(os.environ)
        if value is None:
            environment.pop(DISABLE_NATIVE_ENV_VAR, None)
        else:
            environment[DISABLE_NATIVE_ENV_VAR] = value
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            text=True,
            env=environment,
        ).stdout.strip()

    assert report(None) == "True"
    assert report("1") == "False"
