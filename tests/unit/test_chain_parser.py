"""Tests for :mod:`protocore.runtime.chain_parser`."""
from __future__ import annotations

from protocore.runtime.chain_parser import ChainOperator, parse_chain


def test_single_command() -> None:
    chain = parse_chain("ls -la")
    assert len(chain) == 1
    assert chain[0].raw == "ls -la"
    assert chain[0].leading_op is None


def test_pipe() -> None:
    chain = parse_chain("ls | grep foo")
    assert len(chain) == 2
    assert chain[0].raw == "ls"
    assert chain[1].raw == "grep foo"
    assert chain[1].leading_op is ChainOperator.PIPE


def test_and_or() -> None:
    chain = parse_chain("foo && bar || baz")
    assert len(chain) == 3
    assert chain[1].leading_op is ChainOperator.AND
    assert chain[2].leading_op is ChainOperator.OR


def test_semicolon() -> None:
    chain = parse_chain("a; b; c")
    assert len(chain) == 3
    assert chain[1].leading_op is ChainOperator.SEMI


def test_dollar_paren_substitution_collected() -> None:
    chain = parse_chain("echo $(id)")
    assert len(chain) == 1
    assert "id" in chain[0].substitutions


def test_backtick_substitution_collected() -> None:
    chain = parse_chain("echo `whoami`")
    assert len(chain) == 1
    assert "whoami" in chain[0].substitutions


def test_quoted_pipe_not_split() -> None:
    chain = parse_chain('echo "a | b"')
    assert len(chain) == 1
    assert chain[0].raw == 'echo "a | b"'


def test_single_quoted_pipe_not_split() -> None:
    chain = parse_chain("echo 'a | b'")
    assert len(chain) == 1


def test_empty_input() -> None:
    assert parse_chain("") == []
    assert parse_chain("   ") == []


# ---------------------------------------------------------------------------
# Collect ``$`` / backtick substitutions even inside a DOUBLE-quoted span
# (bash DOES execute ``$(...)`` inside double quotes), so
# DefaultShellSafetyPolicy re-arms every per-substitution deny pattern. Single
# quotes stay literal (no execution → no collection). ``extractQuotedContent``
# (withDoubleQuotes preserves double-quoted content for the substitution-pattern
# scan).
# ---------------------------------------------------------------------------


def test_dollar_paren_substitution_collected_inside_double_quotes() -> None:
    """A ``$()`` inside double quotes MUST be collected (bash executes it)."""
    chain = parse_chain('cat "$(rm -rf /)"')
    assert len(chain) == 1
    assert "rm -rf /" in chain[0].substitutions


def test_backtick_substitution_collected_inside_double_quotes() -> None:
    """A backtick substitution inside double quotes MUST be collected."""
    chain = parse_chain('echo "`whoami`"')
    assert len(chain) == 1
    assert "whoami" in chain[0].substitutions


def test_substitution_not_collected_inside_single_quotes() -> None:
    """Single quotes are literal — ``$()`` there is NOT a substitution."""
    chain = parse_chain("echo '$(rm -rf /)'")
    assert len(chain) == 1
    assert chain[0].substitutions == ()


def test_double_quote_state_restored_after_inner_substitution() -> None:
    """Quote/escape state must survive an inner double-quoted substitution.

    The ``|`` after the substitution is still inside the double-quoted span,
    so the command must NOT split on it.
    """
    chain = parse_chain('echo "$(id) | x"')
    assert len(chain) == 1
    assert "id" in chain[0].substitutions
