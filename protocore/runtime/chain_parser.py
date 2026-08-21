"""Shell-grammar parser — proper quote/escape state machine.

Splits a shell command line into sub-commands by ``|``, ``&&``, ``||``,
``;``, surfacing ``$()`` / backtick substitutions for policy checks.

NOT regex — actual quote/escape state machine. Powers the Bash tool
(via :class:`DefaultShellSafetyPolicy`) to enforce policy on every
sub-command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class ChainOperator(StrEnum):
    """Inter-command operator surfaced by :func:`parse_chain`."""

    PIPE = "|"
    AND = "&&"
    OR = "||"
    SEMI = ";"


@dataclass(frozen=True, slots=True)
class CommandSegment:
    """A single sub-command in a chain."""

    raw: str
    """Verbatim sub-command text (unescaped)."""

    substitutions: tuple[str, ...] = field(default_factory=tuple)
    """``$()`` / backtick substitution bodies inside this segment."""

    leading_op: ChainOperator | None = None
    """Operator that connected this segment to its predecessor (``None`` for first)."""


_TWO_CHAR_OPS: Final[frozenset[str]] = frozenset(["&&", "||"])
_ONE_CHAR_OPS: Final[frozenset[str]] = frozenset(["|", ";"])


def parse_chain(command: str) -> list[CommandSegment]:
    """Parse a shell command line into a chain of :class:`CommandSegment`.

    Honors single / double quote spans (``'…'`` keeps everything literal;
    ``"…"`` allows ``\\`` escape). Collects ``$()`` and backtick
    substitutions so the caller can apply policy to substitution bodies.

    Empty / whitespace-only input → empty list.
    """
    if not command or not command.strip():
        return []

    segments: list[CommandSegment] = []
    buf: list[str] = []
    subs: list[str] = []
    leading: ChainOperator | None = None

    i = 0
    n = len(command)
    in_single = False
    in_double = False
    escape = False

    paren_depth = 0
    paren_buf: list[str] = []
    in_paren_sub = False

    backtick = False
    backtick_buf: list[str] = []

    def _flush_segment(next_op: ChainOperator | None) -> None:
        text = "".join(buf).strip()
        if text:
            segments.append(
                CommandSegment(
                    raw=text,
                    substitutions=tuple(subs),
                    leading_op=leading,
                )
            )
        buf.clear()
        subs.clear()

    while i < n:
        ch = command[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            buf.append(ch)
            escape = True
            i += 1
            continue

        # Inside $() — collect until matching )
        if in_paren_sub:
            paren_buf.append(ch)
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    in_paren_sub = False
                    # ``$(`` was consumed; ``)`` is included in paren_buf
                    sub_text = "".join(paren_buf)
                    subs.append(sub_text[2:-1])  # strip $( and final )
                    buf.append(sub_text)
                    paren_buf.clear()
            i += 1
            continue

        # Inside backtick — collect until closing `
        if backtick:
            backtick_buf.append(ch)
            if ch == "`":
                backtick = False
                sub_text = "".join(backtick_buf)
                subs.append(sub_text[1:-1])  # strip backticks
                buf.append(sub_text)
                backtick_buf.clear()
            i += 1
            continue

        # Quote tracking
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue
        # Single quotes are literal — nothing inside them is a substitution
        # or an operator (bash does not expand ``$()`` inside ``'…'``).
        if in_single:
            buf.append(ch)
            i += 1
            continue

        # ``$(`` substitution opener — collected EVEN inside a double-quoted
        # span, because bash DOES execute ``$(...)`` inside double quotes.
        # The substitution machinery uses its own buffer + depth counter, so
        # the surrounding ``in_double`` state is preserved across the body and
        # restored on the closing ``)`` (DefaultShellSafetyPolicy re-arms every
        # per-substitution deny pattern on the collected body).
        if ch == "$" and i + 1 < n and command[i + 1] == "(":
            in_paren_sub = True
            paren_depth = 1
            paren_buf.extend(["$", "("])
            i += 2
            continue

        # Backtick substitution opener (also active inside double quotes).
        if ch == "`":
            backtick = True
            backtick_buf.append("`")
            i += 1
            continue

        # Any other character inside a double-quoted span is literal — no
        # operator splitting (e.g. a ``|`` inside ``"…"`` is part of the word).
        if in_double:
            buf.append(ch)
            i += 1
            continue

        # Two-char operators (must check before single char)
        if i + 1 < n:
            two = command[i : i + 2]
            if two in _TWO_CHAR_OPS:
                op = ChainOperator(two)
                _flush_segment(op)
                leading = op
                i += 2
                continue

        # Single-char operators
        if ch in _ONE_CHAR_OPS:
            op = ChainOperator(ch)
            _flush_segment(op)
            leading = op
            i += 1
            continue

        buf.append(ch)
        i += 1

    _flush_segment(None)
    return segments


__all__ = ["ChainOperator", "CommandSegment", "parse_chain"]
