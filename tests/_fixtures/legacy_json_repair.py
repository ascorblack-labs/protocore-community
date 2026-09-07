"""The stream-JSON repair as it stood before it was made incremental.

Kept verbatim so the property test has an independent reference to compare
against. Comparing the rewritten repair with itself proves only that it is
self-consistent; the question a rewrite has to answer is whether it still
recovers everything the previous implementation recovered, and only the
previous implementation can answer it.

Frozen on purpose: nothing here is maintained alongside the live module, and
nothing outside the comparison test may import it.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Final

from protocore.constants import MAX_DATA_NESTING_DEPTH, MAX_STRUCTURED_JSON_CHARS


class OutputParserException(ValueError):
    """Raised when defensive parsing cannot recover any JSON value."""


class JSONNestingDepthExceeded(OutputParserException):
    """Raised when a payload nests deeper than the parser is willing to walk.

    Distinct from an ordinary parse failure, and deliberately raised BEFORE the
    payload reaches :func:`json.loads`. CPython's JSON scanner recurses once per
    nesting level and raises ``RecursionError`` on a deep enough document; that
    error is not a ``JSONDecodeError``, so it escapes every ``except
    json.JSONDecodeError`` in this module, unwinds through the run loop, and
    arrives at the catch-all as a bare "maximum recursion depth exceeded" with
    no indication that a tool-call argument blob was what produced it. Refusing
    the document up front turns that into a named, catchable condition that says
    which limit was crossed.

    Subclasses :class:`OutputParserException` so a caller that already treats
    unparseable output as recoverable keeps working; callers that want to tell
    "too deep" from "malformed" catch this first.
    """


def _exceeds_nesting_depth(text: str, max_depth: int) -> bool:
    """True iff ``text`` opens more than ``max_depth`` nested containers.

    A single string-aware pass over the characters — no recursion, no parse, no
    allocation per level. Escapes and quoted brackets are honoured so a payload
    whose STRINGS contain braces is not mistaken for a deep structure. Cheap
    enough to run ahead of every parse: the loop is O(len(text)) and stops at
    the first level past the bound.
    """
    depth = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth > max_depth:
                return True
        elif char in "}]":
            if depth > 0:
                depth -= 1
    return False


def _reject_deep_nesting(text: str, max_depth: int) -> None:
    """Raise :class:`JSONNestingDepthExceeded` when ``text`` is nested too deep."""
    if _exceeds_nesting_depth(text, max_depth):
        raise JSONNestingDepthExceeded(
            f"json nesting exceeds {max_depth} levels — refusing to parse; "
            "a document this deep would exhaust the interpreter stack in the "
            "JSON scanner rather than fail as invalid input"
        )


# Thinking-tag stripping ---------------------------------------------------

# ``<think>...</think>`` / ``<thinking>...</thinking>`` (case-insensitive).
_THINK_TAG_RE: Final[re.Pattern[str]] = re.compile(
    r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>",
    re.IGNORECASE,
)
# Unclosed ``<think>...`` — happens when ``max_tokens`` cuts off mid-reasoning.
_THINK_TAG_UNCLOSED_RE: Final[re.Pattern[str]] = re.compile(
    r"<think(?:ing)?>[\s\S]*$",
    re.IGNORECASE,
)
# Some models emit "Thinking Process:\n\n{...}" prefix instead of XML tags.
_THINKING_PROCESS_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Thinking Process:[\s\S]*?\n\s*\n(?=[\[{])",
    re.IGNORECASE,
)
# Markdown JSON fence (```json ... ```, ```js ... ```, or bare ``` ... ```).
_JSON_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json|javascript|js)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove ``<thinking>`` / ``<think>`` spans + ``Thinking Process:`` prefix.

    Handles complete tags, unclosed tags (truncated by ``max_tokens``), and
    the markdown-style ``Thinking Process:`` preamble that some local models
    emit instead of XML tags.
    """
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_TAG_UNCLOSED_RE.sub("", text)
    return _THINKING_PROCESS_PREFIX_RE.sub("", text)


# Backward-compat alias matching v1 name.
strip_thinking_tokens = strip_thinking


# Generation coercion (used by JsonOutputParser.parse_result) ---------------

def _parse_any(
    candidate: str,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
) -> Any:
    """Parse a JSON value (object, array, scalar). Falls back to ``ast.literal_eval``.

    The ``ast.literal_eval`` fallback handles single-quoted dict/list literals
    that some local models emit when their grammar grammar is loose. Tuples
    are normalized to lists. Sets and other non-JSON types are rejected.

    The nesting bound is checked BEFORE either parser sees the text: both
    recurse per level, and both signal exhaustion with ``RecursionError``
    rather than a parse error, so neither can be relied on to fail cleanly on a
    pathological payload.
    """
    _reject_deep_nesting(candidate, max_depth)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        stripped = candidate.strip()
        if not stripped.startswith(("{", "[")) or not stripped.endswith(("}", "]")):
            raise OutputParserException("invalid_json_output") from None
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            raise OutputParserException("invalid_json_output") from exc
        if isinstance(parsed, tuple):
            return list(parsed)
        if isinstance(parsed, set) or not isinstance(
            parsed,
            (dict, list, str, int, float, bool, type(None)),
        ):
            raise OutputParserException("invalid_json_output") from None
        return parsed


def _extract_json_slice(
    text: str,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
) -> str:
    """Extract the first balanced JSON object/array slice from mixed text.

    Used by :func:`parse_complete_json_any` and :meth:`PartialJSONParser.repair`.
    """
    stripped = text.strip()
    if not stripped:
        raise OutputParserException("empty_output")
    try:
        _parse_any(stripped, max_depth=max_depth)
    except JSONNestingDepthExceeded:
        # A depth refusal is about the document, not about this slice being the
        # wrong one — re-slicing cannot make it shallower, so it propagates.
        raise
    except OutputParserException:
        pass
    else:
        return stripped

    start_obj = stripped.find("{")
    start_arr = stripped.find("[")
    starts = [idx for idx in (start_obj, start_arr) if idx != -1]
    if not starts:
        raise OutputParserException("no_json_found")
    start = min(starts)
    candidate = stripped[start:]

    in_string = False
    escape = False
    stack: list[str] = []
    for idx, char in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
                if not stack:
                    return candidate[: idx + 1]
    return candidate


def parse_complete_json_any(
    text: str,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
) -> Any:
    """Parse a JSON value (object, array, scalar) — lenient extraction.

    Strips thinking tokens; tolerates mixed text where the first balanced
    JSON object/array is embedded. Used by tool-call surfaces that emit
    non-strict JSON (Hermes XML wrapper, ReAct, raw chat with embedded JSON).

    Raises :class:`OutputParserException` on failure.
    """
    if len(text) > MAX_STRUCTURED_JSON_CHARS:
        raise OutputParserException(
            f"input exceeds MAX_STRUCTURED_JSON_CHARS ({MAX_STRUCTURED_JSON_CHARS})",
        )
    prepared = strip_thinking(text).strip()
    if not prepared:
        raise OutputParserException("empty_output")
    candidate = _extract_json_slice(prepared, max_depth=max_depth)
    return _parse_any(candidate, max_depth=max_depth)


# Strict parsing (object only) — used by ingress/envelope flows --------------

class PartialJSONParser:
    """Best-effort parser for partially generated JSON."""

    def __init__(self, *, max_depth: int = MAX_DATA_NESTING_DEPTH) -> None:
        self.max_depth = max_depth

    _dangling_key_before_closer_re: Final[re.Pattern[str]] = re.compile(
        r'(,\s*)?"[^"\\]*(?:\\.[^"\\]*)*"\s*:\s*(?=[}\]])',
    )
    _dangling_key_at_end_re: Final[re.Pattern[str]] = re.compile(
        r'(,\s*)?"[^"\\]*(?:\\.[^"\\]*)*"\s*:\s*$',
    )
    _trailing_comma_re: Final[re.Pattern[str]] = re.compile(r",\s*([}\]])")

    def parse(self, text: str) -> Any | None:
        """Parse complete JSON or repair an incomplete JSON prefix.

        Returns the parsed value, or ``None`` if the text cannot be parsed
        even after repair. Use :meth:`parse_with_flag` when callers need to
        distinguish between complete and repaired output.
        """
        result, _ = self.parse_with_flag(text)
        return result

    def parse_with_flag(self, text: str) -> tuple[Any | None, bool]:
        """Parse complete JSON or repair an incomplete JSON prefix.

        Returns a ``(value, was_repaired)`` tuple so callers can detect that
        the returned data may be truncated or incomplete:

            - ``(parsed_value, False)`` — input was valid, complete JSON.
            - ``(repaired_value, True)`` — input was incomplete; result was
              reconstructed by closing open brackets/quotes. Downstream
              consumers should treat this data as potentially partial.
            - ``(None, False)`` — input could not be parsed even after repair.
        """
        prepared = strip_thinking(text).strip()
        if not prepared:
            return None, False
        # A document past the depth bound is refused outright rather than routed
        # into repair: repair closes open brackets, so it can only ever make an
        # over-deep payload deeper, and the point of the bound is that nothing
        # downstream walks the structure at all.
        _reject_deep_nesting(prepared, self.max_depth)
        try:
            return parse_complete_json_any(prepared, max_depth=self.max_depth), False
        except OutputParserException:
            try:
                repaired = self.repair(prepared)
            except OutputParserException:
                return None, False
            try:
                return _parse_any(repaired, max_depth=self.max_depth), True
            except OutputParserException:
                return None, False

    def repair(self, text: str) -> str:
        """Repair an incomplete JSON payload to a parseable state."""
        source = _extract_json_slice(
            strip_thinking(text), max_depth=self.max_depth
        )
        started = False
        in_string = False
        escape = False
        stack: list[str] = []
        out: list[str] = []

        for char in source:
            if not started:
                if char not in "{[":
                    continue
                started = True

            out.append(char)

            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]":
                if stack and stack[-1] == char:
                    stack.pop()

        repaired = "".join(out).strip()
        repaired = self._cleanup_dangling_tokens(repaired)

        if in_string:
            repaired += '"'
        while stack:
            repaired += stack.pop()

        return self._cleanup_dangling_tokens(repaired)

    def _cleanup_dangling_tokens(self, candidate: str) -> str:
        cleaned = candidate
        previous: str | None = None
        while previous != cleaned:
            previous = cleaned
            cleaned = self._trailing_comma_re.sub(r"\1", cleaned)
            cleaned = self._dangling_key_before_closer_re.sub("", cleaned)
            cleaned = self._dangling_key_at_end_re.sub("", cleaned).rstrip()
        return cleaned


