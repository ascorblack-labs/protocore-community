"""Defensive JSON parsing + streaming utilities.

Critical for the small-model harness. Provides:

    - :func:`strip_thinking` / :func:`strip_thinking_tokens`: remove
      ``<thinking>`` / ``<think>`` blocks (incl. unclosed tags from
      ``max_tokens`` truncation) and ``Thinking Process:`` preambles.
    - :func:`parse_complete_json`: strict-dict parse for ingress / envelope
      flows (rejects non-object payloads).
    - :func:`parse_complete_json_any`: lenient parse for non-OpenAI tool-call
      surfaces (Hermes XML, ReAct, raw chat) — accepts any JSON value.
    - :func:`is_strict_json_text`: quick predicate for already-clean JSON.
    - :func:`structured_json_candidates`: extract candidate JSON object slices
      from mixed text output (markdown-fenced + brace-scanned + nested-aware).
    - :class:`PartialJSONParser`: repair-and-parse for truncated JSON; closes
      stacks, drops dangling keys, removes trailing commas.
    - :class:`StreamingJSONParser`: depth-balanced char-by-char accumulator
      that emits the first complete JSON value.
    - :class:`RobustStreamingJSONParser`: streaming with partial-repair
      emission for live UI updates during generation.
    - :class:`JsonOutputParser` ``[TModel]``: Pydantic-aware extractor with
      ``parse``, ``parse_result``, ``parse_stream``, ``parse_stream_final``.

Pure-stdlib + Pydantic. No third-party JSON deps; cross-pod deterministic.
"""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any, Final

from pydantic import BaseModel, ValidationError

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


def _coerce_generation_text(generation: Any) -> str:
    """Extract text from a generation-shaped object (str, .text attr, or dict)."""
    if isinstance(generation, str):
        return generation
    text_attr = getattr(generation, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(generation, dict):
        value = generation.get("text")
        if isinstance(value, str):
            return value
    return str(generation)


# Lenient parsing (any JSON value) ------------------------------------------


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
        return _normalise_literal(parsed)


_JSON_SCALARS: Final[tuple[type, ...]] = (str, int, float, bool, type(None))


def _normalise_literal(value: Any) -> Any:
    """Coerce an ``ast.literal_eval`` result to JSON-expressible types.

    Tuples become lists; anything else Python-only — a set above all — is
    refused. The check reaches every level, not just the top one: repairing a
    truncated object key produces text like ``{"partial"}``, which Python reads
    as a set literal, and a set nested one list deep used to travel back to the
    caller of a JSON parser.
    """
    if isinstance(value, dict):
        return {key: _normalise_literal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_literal(item) for item in value]
    if isinstance(value, _JSON_SCALARS):
        return value
    raise OutputParserException("invalid_json_output") from None


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


def parse_complete_json(
    text: str,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
) -> dict[str, Any]:
    """Parse a string that should be a complete JSON object.

    Strips thinking tokens; tolerates leading/trailing whitespace. Rejects
    non-object payloads (use :func:`parse_complete_json_any` for those).
    Raises :class:`OutputParserException` on failure.
    """
    if len(text) > MAX_STRUCTURED_JSON_CHARS:
        raise OutputParserException(
            f"input exceeds MAX_STRUCTURED_JSON_CHARS ({MAX_STRUCTURED_JSON_CHARS})",
        )
    cleaned = strip_thinking(text).strip()
    _reject_deep_nesting(cleaned, max_depth)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"failed to parse JSON: {e}") from e
    if not isinstance(result, dict):
        raise OutputParserException(
            f"expected JSON object, got {type(result).__name__}",
        )
    return result


def is_strict_json_text(
    text: str,
    *,
    max_depth: int = MAX_DATA_NESTING_DEPTH,
) -> bool:
    """Return ``True`` when ``text`` is already a standalone strict JSON value.

    A payload past the nesting bound answers ``False``: it is not text this
    module will hand to a downstream parser, and answering the predicate
    honestly here means the caller routes it to whatever it does with
    non-JSON rather than to a parse that would fail on the stack.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _exceeds_nesting_depth(stripped, max_depth):
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


# Candidate extraction (for mixed text) -------------------------------------


def structured_json_candidates(text: str) -> list[dict[str, Any]]:
    """Extract candidate JSON objects embedded in larger text output.

    Returns all top-level JSON object spans found; empty list on no
    candidates. Does NOT raise — caller chooses how to handle missing
    candidates. Used by tool-call adapters that need to find embedded
    JSON in ReAct / chat-with-JSON output.
    """
    if len(text) > MAX_STRUCTURED_JSON_CHARS:
        return []
    cleaned = strip_thinking(text)
    candidates: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = cleaned[start : i + 1]
                if _exceeds_nesting_depth(fragment, MAX_DATA_NESTING_DEPTH):
                    start = None
                    continue
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)
                start = None
    return candidates


def structured_json_strings(raw_text: str) -> list[str]:
    """Generate best-effort raw JSON candidate strings from mixed text.

    v1-style helper: returns the strings themselves (deduplicated) so callers
    can attempt multiple parse strategies. Combines:

        - the cleaned text itself
        - markdown-fenced JSON blocks (``json``/``js``/bare)
        - the outermost ``{…}`` slice
        - the outermost ``[…]`` slice
        - a partial-repair JSON dump (from :class:`PartialJSONParser`)
    """
    if len(raw_text) > MAX_STRUCTURED_JSON_CHARS:
        return []
    text = strip_thinking(raw_text).strip()
    candidates: list[str] = []
    if text:
        candidates.append(text)

    for match in _JSON_FENCE_RE.finditer(text):
        inner = match.group(1).strip()
        if inner:
            candidates.append(inner)

    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        first_newline = inner.find("\n")
        if first_newline != -1:
            language = inner[:first_newline].strip().lower()
            if language in {"json", "javascript", "js"}:
                inner = inner[first_newline + 1 :].strip()
        elif inner.lower().startswith("json"):
            inner = inner[4:].strip()
        if inner:
            candidates.append(inner)

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1].strip())

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
        candidates.append(text[first_arr : last_arr + 1].strip())

    partial_parsed = PartialJSONParser().parse(text)
    if partial_parsed is not None:
        candidates.append(json.dumps(partial_parsed, ensure_ascii=True))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


# Partial / streaming parsers -----------------------------------------------


_ATOM_STRING: Final[str] = "string"
_ATOM_SPACE: Final[str] = "space"
_ATOM_CHAR: Final[str] = "char"

_JSON_WHITESPACE: Final[str] = " \t\n\r"


class _Atom:
    """One string literal, one whitespace run, or one other character."""

    __slots__ = ("closed", "kind", "text")

    def __init__(self, kind: str, text: str, *, closed: bool = True) -> None:
        self.kind = kind
        self.text = text
        self.closed = closed


def _atomise(text: str) -> list[_Atom]:
    """Split ``text`` into string-aware atoms in a single left-to-right pass.

    Quoted runs become one atom each, so a later scan can tell a structural
    brace from one that merely sits inside a string without re-deciding that
    question at every offset.
    """
    atoms: list[_Atom] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == '"':
            cursor = index + 1
            closed = False
            while cursor < length:
                current = text[cursor]
                if current == "\\":
                    cursor += 2
                    continue
                cursor += 1
                if current == '"':
                    closed = True
                    break
            cursor = min(cursor, length)
            atoms.append(_Atom(_ATOM_STRING, text[index:cursor], closed=closed))
            index = cursor
            continue
        if char in _JSON_WHITESPACE:
            cursor = index
            while cursor < length and text[cursor] in _JSON_WHITESPACE:
                cursor += 1
            atoms.append(_Atom(_ATOM_SPACE, text[index:cursor]))
            index = cursor
            continue
        atoms.append(_Atom(_ATOM_CHAR, char))
        index += 1
    return atoms


class PartialJSONParser:
    """Best-effort parser for partially generated JSON."""

    def __init__(self, *, max_depth: int = MAX_DATA_NESTING_DEPTH) -> None:
        self.max_depth = max_depth


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
        """Drop trailing commas and value-less keys that precede a closer.

        A single right-to-left pass over string-aware atoms. The obvious
        spelling — three regexes applied until the text stops changing — is
        quadratic on exactly the payloads this parser exists for: a pattern
        that matches a quoted run is retried at every quote in the buffer, and
        an unterminated string makes each of those attempts walk to the end of
        the input. A 64 KiB truncated string with escapes took eighteen seconds
        that way. Scanning once, from the right, costs one pass and cannot
        mistake a brace inside a string for a structural one.

        A key with no colon yet is dropped as well, not only a key whose colon
        has no value. That is the ordinary shape of a body cut off by an output
        limit — ``{"a": 1, "te`` — and leaving the fragment in place produces
        ``{"a": 1, "te"}``, which is a Python set literal rather than an
        object, so the whole document is refused and every fact already
        complete in it is lost. Dropped, it repairs to ``{"a": 1}`` and the
        complete part survives.
        """
        atoms = _atomise(candidate)
        dropped = [False] * len(atoms)

        def previous_significant(index: int) -> int:
            cursor = index - 1
            while cursor >= 0 and (dropped[cursor] or atoms[cursor].kind == _ATOM_SPACE):
                cursor -= 1
            return cursor

        def is_value_less_key(index: int) -> bool:
            """Whether the string atom at ``index`` is a key awaiting a colon.

            Only what stands to its left can tell a key from a value, because
            the caller has already established that nothing significant stands
            to its right before the object's closer. A string preceded by
            ``:`` is that colon's value; one preceded by ``{`` or ``,`` is a
            key whose colon never arrived.
            """
            if atoms[index].kind != _ATOM_STRING or not atoms[index].closed:
                return False
            before = previous_significant(index)
            if before < 0 or atoms[before].kind != _ATOM_CHAR:
                return False
            return atoms[before].text in "{,"

        def cascade(boundary: int, *, closer: str | None = None) -> None:
            """Strip dangling constructs immediately left of ``boundary``."""
            while True:
                previous = previous_significant(boundary)
                if previous < 0:
                    return
                atom = atoms[previous]
                if atom.kind == _ATOM_CHAR and atom.text == ",":
                    for cursor in range(previous, boundary):
                        dropped[cursor] = True
                    boundary = previous
                    continue
                if closer == "}" and is_value_less_key(previous):
                    for cursor in range(previous, boundary):
                        dropped[cursor] = True
                    boundary = previous
                    continue
                if not (atom.kind == _ATOM_CHAR and atom.text == ":"):
                    return
                key = previous_significant(previous)
                if key < 0 or atoms[key].kind != _ATOM_STRING or not atoms[key].closed:
                    return
                for cursor in range(key, boundary):
                    dropped[cursor] = True
                boundary = key

        cascade(len(atoms))
        for index in range(len(atoms) - 1, -1, -1):
            if dropped[index]:
                continue
            atom = atoms[index]
            if atom.kind == _ATOM_CHAR and atom.text in "}]":
                cascade(index, closer=atom.text)

        return "".join(
            atom.text for index, atom in enumerate(atoms) if not dropped[index]
        ).rstrip()


class StreamingJSONParser:
    """Stateful parser that consumes char chunks and emits the first complete JSON value.

    Depth-balanced char-by-char accumulator; emits ``None`` until the
    outermost ``{...}`` / ``[...]`` is balanced, then returns the parsed
    value and resets state. Cross-pod safe (no module state).
    """

    def __init__(self, *, max_depth: int = MAX_DATA_NESTING_DEPTH) -> None:
        self.max_depth = max_depth
        self._buffer: list[str] = []
        self._depth: int = 0
        self._in_string: bool = False
        self._escape: bool = False
        self._started: bool = False

    def reset(self) -> None:
        """Reset parser state so a new payload can be consumed."""
        self._buffer = []
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._started = False

    @property
    def buffer_text(self) -> str:
        """Current accumulated buffer (for partial-repair fallback)."""
        return "".join(self._buffer)

    def consume(self, chunk: str) -> Any | None:
        """Feed one chunk; return the parsed value when fully balanced, else ``None``."""
        for char in chunk:
            parsed = self._consume_char(char)
            if parsed is not None:
                return parsed
        return None

    def _consume_char(self, char: str) -> Any | None:
        if not self._started:
            if char not in "{[":
                return None
            self._started = True

        self._buffer.append(char)

        if self._in_string:
            if self._escape:
                self._escape = False
                return None
            if char == "\\":
                self._escape = True
                return None
            if char == '"':
                self._in_string = False
            return None

        if char == '"':
            self._in_string = True
            return None
        if char in "{[":
            self._depth += 1
            # The accumulator already counts levels, so the depth bound costs
            # one comparison here and spares the eventual ``json.loads`` a
            # document it would answer with ``RecursionError``.
            if self._depth > self.max_depth:
                self.reset()
                raise JSONNestingDepthExceeded(
                    f"json nesting exceeds {self.max_depth} levels in the "
                    "streamed payload — parser state reset"
                )
            return None
        if char in "}]":
            self._depth -= 1
            if self._depth == 0:
                candidate = self.buffer_text.strip()
                try:
                    parsed = _parse_any(candidate, max_depth=self.max_depth)
                except OutputParserException:
                    return None
                self.reset()
                return parsed
        return None


# Incremental partial view ---------------------------------------------------

# Longest tag prefix ``strip_thinking`` can react to, kept across chunk
# boundaries so a tag split between two chunks is still noticed.
_THINK_TAG_LOOKBEHIND: Final[int] = len("<thinking")

_ST_BEFORE: Final[int] = 0
_ST_VALUE: Final[int] = 1
_ST_OBJ_KEY: Final[int] = 2
_ST_OBJ_COLON: Final[int] = 3
_ST_AFTER: Final[int] = 4
_ST_STRING: Final[int] = 5
_ST_TOKEN: Final[int] = 6
_ST_DONE: Final[int] = 7

_ESC_NONE: Final[int] = 0
_ESC_BACKSLASH: Final[int] = 1
_ESC_HEX: Final[int] = 2

_SIMPLE_ESCAPES: Final[dict[str, str]] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdefABCDEF")
_TOKEN_CHARS: Final[frozenset[str]] = frozenset(
    "+-.0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
)
_SURROGATE_HIGH_FIRST: Final[int] = 0xD800
_SURROGATE_HIGH_LAST: Final[int] = 0xDBFF
_SURROGATE_LOW_FIRST: Final[int] = 0xDC00
_SURROGATE_LOW_LAST: Final[int] = 0xDFFF
_SURROGATE_BASE: Final[int] = 0x10000
_SURROGATE_SHIFT: Final[int] = 10
_UNICODE_ESCAPE_DIGITS: Final[int] = 4
_UNSET: Final[Any] = object()
_AMBIGUOUS: Final[Any] = object()


class _Frame:
    """One open container plus the slot it occupies in its parent."""

    __slots__ = ("container", "is_object", "key", "slot")

    def __init__(self, container: Any, *, is_object: bool, slot: Any) -> None:
        self.container = container
        self.is_object = is_object
        self.key: str | None = None
        self.slot = slot


class _IncrementalPartialView:
    """Live mirror of :meth:`PartialJSONParser.parse` over a growing buffer.

    Consumes the same characters the accumulator sees and keeps the value tree
    built so far, so producing a partial snapshot costs the open spine instead
    of re-parsing and re-repairing the whole buffer on every chunk. That is the
    difference between a stream whose total cost is linear in its length and
    one whose cost is cubic.

    The mirror covers the JSON grammar only. Anything outside it — an unknown
    escape, a raw control character inside a string, a token neither JSON nor a
    prefix of one, a structural character where the grammar allows none — sets
    :attr:`supported` to ``False``, and the caller falls back to the text
    parser, whose leniency (Python-literal rescue, tolerated oddities) is what
    decides those cases and cannot be mirrored by a grammar alone.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop all state, as after a complete value was emitted."""
        self.supported = True
        # ``tree_dirty`` covers changes already folded into the value tree;
        # ``pending`` vs ``emitted_pending`` covers the scalar still being
        # typed, whose text can grow without its value changing (``1000.``
        # and ``1000.0`` are the same number).
        self.tree_dirty = False
        self.pending: Any = _UNSET
        self.emitted_pending: Any = _UNSET
        self._state = _ST_BEFORE
        self._root: Any = None
        self._stack: list[_Frame] = []
        self._token: list[str] = []
        self._string: list[str] = []
        self._string_is_key = False
        self._escape = _ESC_NONE
        self._hex = ""
        self._high_surrogate: int | None = None
        self._trailing_space = 0
        self._tail = ""

    # -- consumption --------------------------------------------------------

    def consume(self, chunk: str) -> None:
        """Feed the same characters the accumulator just took."""
        if not self.supported:
            return
        # ``strip_thinking`` runs ahead of the text parser and would excise a
        # thinking span wherever it appears, including inside a string. The
        # mirror does not model that, so its presence hands the buffer back.
        if "<think" in (self._tail + chunk).lower():
            self.supported = False
            return
        self._tail = (self._tail + chunk)[-_THINK_TAG_LOOKBEHIND:]
        for char in chunk:
            self._feed(char)
            if not self.supported:
                return

    def _unsupported(self) -> None:
        self.supported = False

    def _feed(self, char: str) -> None:
        state = self._state
        if state == _ST_STRING:
            self._feed_string(char)
            return
        if state == _ST_TOKEN:
            if char in _TOKEN_CHARS:
                self._token.append(char)
                return
            if not self._close_token():
                return
            state = self._state
        if char in _JSON_WHITESPACE:
            return
        if state == _ST_BEFORE:
            if char == "{":
                self._open(is_object=True)
            elif char == "[":
                self._open(is_object=False)
            return
        if state == _ST_VALUE:
            self._feed_value(char)
            return
        if state == _ST_OBJ_KEY:
            if char == '"':
                self._open_string(is_key=True)
            elif char == "}":
                self._close_container()
            else:
                self._unsupported()
            return
        if state == _ST_OBJ_COLON:
            if char == ":":
                self._state = _ST_VALUE
            else:
                self._unsupported()
            return
        if state == _ST_AFTER:
            self._feed_after(char)
            return
        self._unsupported()

    def _feed_value(self, char: str) -> None:
        if char == '"':
            self._open_string(is_key=False)
            return
        if char == "{":
            self._open(is_object=True)
            return
        if char == "[":
            self._open(is_object=False)
            return
        if char in "}]":
            # A closer where a value was promised: the repair path drops the
            # value-less key or the trailing comma and closes the container.
            frame = self._stack[-1] if self._stack else None
            if frame is None or frame.is_object != (char == "}"):
                self._unsupported()
                return
            frame.key = None
            self._close_container()
            return
        if char in _TOKEN_CHARS:
            self._token = [char]
            self._state = _ST_TOKEN
            return
        self._unsupported()

    def _feed_after(self, char: str) -> None:
        frame = self._stack[-1] if self._stack else None
        if frame is None:
            self._unsupported()
            return
        if char == ",":
            self._state = _ST_OBJ_KEY if frame.is_object else _ST_VALUE
            return
        if char in "}]" and frame.is_object == (char == "}"):
            self._close_container()
            return
        self._unsupported()

    # -- structure ----------------------------------------------------------

    def _attach(self, value: Any) -> Any:
        """Place a finished value in the open container; return its slot."""
        if not self._stack:
            self._root = value
            return None
        frame = self._stack[-1]
        if frame.is_object:
            key = frame.key
            frame.key = None
            if key is None:
                self._unsupported()
                return None
            frame.container[key] = value
            return key
        frame.container.append(value)
        return len(frame.container) - 1

    def _open(self, *, is_object: bool) -> None:
        container: Any = {} if is_object else []
        slot = self._attach(container)
        if not self.supported:
            return
        self._stack.append(_Frame(container, is_object=is_object, slot=slot))
        self._state = _ST_OBJ_KEY if is_object else _ST_VALUE
        self.tree_dirty = True

    def _close_container(self) -> None:
        self._stack.pop()
        self._state = _ST_AFTER if self._stack else _ST_DONE

    def _close_token(self) -> bool:
        token = "".join(self._token)
        try:
            value = json.loads(token)
        except ValueError:
            self._unsupported()
            return False
        self._attach(value)
        if not self.supported:
            return False
        self._settle(value)
        self._state = _ST_AFTER
        return True

    # -- strings ------------------------------------------------------------

    def _open_string(self, *, is_key: bool) -> None:
        self._state = _ST_STRING
        self._string = []
        self._string_is_key = is_key
        self._escape = _ESC_NONE
        self._hex = ""
        self._high_surrogate = None
        self._trailing_space = 0

    def _emit_char(self, char: str, *, from_escape: bool) -> None:
        self._flush_high_surrogate()
        self._string.append(char)
        if not from_escape and char.isspace():
            # ``repair`` strips the buffer before closing an open string, so
            # literal trailing whitespace is not part of the partial value yet.
            self._trailing_space += 1
            return
        self._trailing_space = 0

    def _flush_high_surrogate(self) -> None:
        if self._high_surrogate is not None:
            self._string.append(chr(self._high_surrogate))
            self._high_surrogate = None
            self._trailing_space = 0

    def _feed_string(self, char: str) -> None:
        if self._escape == _ESC_BACKSLASH:
            if char == "u":
                self._escape = _ESC_HEX
                self._hex = ""
                return
            mapped = _SIMPLE_ESCAPES.get(char)
            if mapped is None:
                self._unsupported()
                return
            self._escape = _ESC_NONE
            self._emit_char(mapped, from_escape=True)
            return
        if self._escape == _ESC_HEX:
            if char not in _HEX_DIGITS:
                self._unsupported()
                return
            self._hex += char
            if len(self._hex) < _UNICODE_ESCAPE_DIGITS:
                return
            self._escape = _ESC_NONE
            code_point = int(self._hex, 16)
            self._hex = ""
            self._emit_code_point(code_point)
            return
        if char == '"':
            self._close_string()
            return
        if char == "\\":
            self._escape = _ESC_BACKSLASH
            return
        if char < " ":
            # Raw control characters are rejected by the JSON scanner but
            # tolerated by the literal-eval rescue behind the text parser.
            self._unsupported()
            return
        self._emit_char(char, from_escape=False)

    def _emit_code_point(self, code_point: int) -> None:
        if self._high_surrogate is not None and (
            _SURROGATE_LOW_FIRST <= code_point <= _SURROGATE_LOW_LAST
        ):
            high = self._high_surrogate - _SURROGATE_HIGH_FIRST
            low = code_point - _SURROGATE_LOW_FIRST
            self._high_surrogate = None
            self._string.append(
                chr(_SURROGATE_BASE + ((high << _SURROGATE_SHIFT) | low)),
            )
            self._trailing_space = 0
            return
        self._flush_high_surrogate()
        if _SURROGATE_HIGH_FIRST <= code_point <= _SURROGATE_HIGH_LAST:
            self._high_surrogate = code_point
            return
        self._emit_char(chr(code_point), from_escape=True)

    def _close_string(self) -> None:
        self._flush_high_surrogate()
        text = "".join(self._string)
        self._string = []
        if self._string_is_key:
            if not self._stack or not self._stack[-1].is_object:
                self._unsupported()
                return
            self._stack[-1].key = text
            self._state = _ST_OBJ_COLON
            return
        self._attach(text)
        if not self.supported:
            return
        self._settle(text)
        self._state = _ST_AFTER
        self._trailing_space = 0

    def _settle(self, value: Any) -> None:
        """Fold a finished scalar into the tree, keeping the dirty flag honest.

        The scalar was already visible in the snapshot as ``pending``; moving
        it into the tree changes what a caller sees only if its value differs
        from the one last handed out.
        """
        if value != self.emitted_pending:
            self.tree_dirty = True
        self.emitted_pending = _UNSET

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> Any:
        """Return the partial value, ``None``, or ``_AMBIGUOUS``.

        ``_AMBIGUOUS`` means the repair path's answer depends on leniency this
        mirror does not model, and the caller must ask the text parser.
        """
        state = self._state
        self.pending = _UNSET
        if state == _ST_BEFORE or self._root is None:
            return None
        pending: Any = _UNSET
        if state == _ST_OBJ_COLON:
            # ``{"a"`` — a key whose colon has not arrived. The repair path
            # drops the key, so the mirror shows the object without it; the
            # key becomes visible when its value does.
            pass
        elif state == _ST_STRING:
            if self._escape == _ESC_BACKSLASH:
                # The repair path closes an open string by appending a quote,
                # and a lone backslash swallows it: the string stays open, the
                # closers land inside it and nothing parses. Only the text
                # parser knows what it makes of that.
                return None
            if self._escape != _ESC_NONE and not self._string_is_key:
                return None
            if not self._string_is_key:
                # A key still being typed is dropped by the repair path for
                # the same reason, and nothing about it is visible yet.
                pending = "".join(self._string)
                if self._high_surrogate is not None:
                    pending += chr(self._high_surrogate)
                elif self._trailing_space:
                    pending = pending[: len(pending) - self._trailing_space]
                self.pending = pending
        elif state == _ST_TOKEN:
            try:
                pending = json.loads("".join(self._token))
            except ValueError:
                return _AMBIGUOUS
            self.pending = pending
        if not self._stack:
            return self._root
        copies: list[Any] = [
            dict(frame.container) if frame.is_object else list(frame.container)
            for frame in self._stack
        ]
        if pending is not _UNSET:
            frame = self._stack[-1]
            if frame.is_object:
                if frame.key is None:
                    return _AMBIGUOUS
                copies[-1][frame.key] = pending
            else:
                copies[-1].append(pending)
        for index in range(len(copies) - 1, 0, -1):
            copies[index - 1][self._stack[index].slot] = copies[index]
        return copies[0]


class RobustStreamingJSONParser:
    """Streaming parser with partial-repair fallback during generation.

    Wraps :class:`StreamingJSONParser` to emit partial-repair results for
    live UI updates: as the model streams tokens, callers see progressively
    more-complete JSON, then a final fully-validated payload on close.

    Deduplicates emitted partials by canonical JSON fingerprint so callers
    do not see redundant updates between meaningful state changes.
    """

    def __init__(self, *, max_depth: int = MAX_DATA_NESTING_DEPTH) -> None:
        self.streaming = StreamingJSONParser(max_depth=max_depth)
        self.partial = PartialJSONParser(max_depth=max_depth)
        self._view = _IncrementalPartialView()
        self._last_partial: Any = _UNSET
        self._recheck = False
        self._last_complete: Any | None = None

    def consume(self, chunk: str, *, emit_partial: bool = True) -> Any | None:
        """Consume a chunk; return a complete or partial JSON value, or ``None``."""
        try:
            complete = self.streaming.consume(chunk)
        except JSONNestingDepthExceeded:
            self._view.reset()
            raise
        if complete is not None:
            self._last_complete = complete
            self._last_partial = _UNSET
            self._recheck = False
            self._view.reset()
            return complete

        self._view.consume(chunk)
        if not emit_partial:
            return None
        if self._view.supported:
            snapshot = self._view.snapshot()
            if snapshot is not _AMBIGUOUS:
                if snapshot is None:
                    return None
                view = self._view
                if not view.tree_dirty and view.pending == view.emitted_pending:
                    return None
                if self._recheck and snapshot == self._last_partial:
                    self._settle_emission(snapshot)
                    return None
                self._settle_emission(snapshot)
                return snapshot
        # The mirror declined this buffer: the repair path decides it instead,
        # and its answer may not be one the mirror can reproduce, so the next
        # emission is compared against it in full.
        partial = self.partial.parse(self.streaming.buffer_text)
        duplicate = partial is None or partial == self._last_partial
        self._settle_emission(partial)
        self._recheck = True
        return None if duplicate else partial

    def _settle_emission(self, value: Any) -> None:
        """Record that ``value`` is what a caller has now been shown."""
        self._view.tree_dirty = False
        self._view.emitted_pending = self._view.pending
        self._recheck = False
        if value is not None:
            self._last_partial = value

    def finalize(self, raw_fallback: str = "") -> Any:
        """Return the last complete value, or attempt one final partial repair."""
        if self._last_complete is not None:
            return self._last_complete
        candidate = self.streaming.buffer_text or raw_fallback
        parsed = self.partial.parse(candidate)
        if parsed is None:
            raise OutputParserException("stream_parse_failed")
        return parsed


# JSON Pointer / RFC6902 diff (used by JsonOutputParser.parse_stream) -------


def _escape_json_pointer_token(token: str) -> str:
    """Escape a JSON Pointer path segment per RFC 6901.

    ``~`` → ``~0``, ``/`` → ``~1`` (order matters: tilde first).
    """
    return token.replace("~", "~0").replace("/", "~1")


def _json_diff(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    """Compute lightweight RFC6902-style diff between JSON values."""
    patches: list[dict[str, Any]] = []

    if type(old) is not type(new):
        return [{"op": "replace", "path": path or "/", "value": new}]

    if isinstance(old, dict):
        old_keys = set(old.keys())
        new_keys = set(new.keys())
        for key in old_keys - new_keys:
            escaped = _escape_json_pointer_token(key)
            patches.append({"op": "remove", "path": f"{path}/{escaped}"})
        for key in new_keys - old_keys:
            escaped = _escape_json_pointer_token(key)
            patches.append({"op": "add", "path": f"{path}/{escaped}", "value": new[key]})
        for key in old_keys & new_keys:
            escaped = _escape_json_pointer_token(key)
            patches.extend(_json_diff(old[key], new[key], f"{path}/{escaped}"))
        return patches

    if isinstance(old, list):
        if old != new:
            return [{"op": "replace", "path": path or "/", "value": new}]
        return patches

    if old != new:
        return [{"op": "replace", "path": path or "/", "value": new}]
    return patches


# Pydantic-aware extractor ---------------------------------------------------


class JsonOutputParser[TModel: BaseModel]:
    """Lightweight Pydantic-aware JSON output parser.

    Mirrors the LangChain ``JsonOutputParser`` shape so adapters can plug
    in without a LangChain dependency. ``pydantic_object=None`` means "no
    schema validation — return raw parsed JSON".

    Methods:
        - :meth:`get_format_instructions`: returns a JSON Schema prompt
          suffix for the model.
        - :meth:`parse`: one-shot parse of complete output.
        - :meth:`parse_result`: parse a list of generation-shaped objects.
        - :meth:`parse_stream`: iterate over chunks; yield validated values
          (or RFC6902 diffs if ``yield_diffs=True``).
        - :meth:`parse_stream_final`: consume all chunks; return the
          final validated value.
    """

    def __init__(
        self,
        pydantic_object: type[TModel] | None = None,
        *,
        max_depth: int = MAX_DATA_NESTING_DEPTH,
    ) -> None:
        self.pydantic_object = pydantic_object
        self.max_depth = max_depth
        self.partial_parser = PartialJSONParser(max_depth=max_depth)

    def get_format_instructions(self) -> str:
        """Return JSON Schema-formatted prompt instruction suffix."""
        if self.pydantic_object is None:
            return "Return a valid JSON object and nothing else."
        schema = self.pydantic_object.model_json_schema()
        return (
            "Return a JSON object that strictly matches this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=True, indent=2)}"
        )

    def parse(self, text: str) -> Any:
        """Parse ``text`` to a Pydantic instance (or raw value if no schema)."""
        parsed = self.partial_parser.parse(text)
        if parsed is None:
            raise OutputParserException("invalid_json_output")
        return self._validate(parsed)

    def parse_result(self, generations: list[Any], *, partial: bool = False) -> Any:
        """Parse a list of generation-shaped objects (str, ``.text``, or dict)."""
        if not generations:
            raise OutputParserException("empty_generation_result")
        text = _coerce_generation_text(generations[0])
        if partial:
            parsed = self.partial_parser.parse(text)
            return None if parsed is None else self._validate(parsed)
        return self.parse(text)

    def parse_stream(
        self,
        chunks: Iterable[str],
        *,
        yield_diffs: bool = False,
        include_partial: bool = True,
    ) -> Iterator[Any]:
        """Stream-parse ``chunks``; yield validated values or RFC6902 diffs."""
        parser = RobustStreamingJSONParser(max_depth=self.max_depth)
        previous: Any | None = None
        for chunk in chunks:
            parsed = parser.consume(chunk, emit_partial=include_partial)
            if parsed is None:
                continue
            validated = self._validate(parsed)
            if yield_diffs and previous is not None:
                yield _json_diff(previous, validated)
            else:
                yield validated
            previous = validated

    def parse_stream_final(self, chunks: Iterable[str]) -> Any:
        """Consume all ``chunks``; return the final validated value."""
        parser = RobustStreamingJSONParser(max_depth=self.max_depth)
        final: Any | None = None
        raw_fragments: list[str] = []
        for chunk in chunks:
            raw_fragments.append(chunk)
            parsed = parser.consume(chunk, emit_partial=True)
            if parsed is not None:
                final = parsed
        if final is None:
            final = parser.finalize("".join(raw_fragments))
        return self._validate(final)

    def _validate(self, data: Any) -> Any:
        if self.pydantic_object is None:
            return data
        try:
            return self.pydantic_object.model_validate(data)
        except ValidationError as exc:
            raise OutputParserException(f"pydantic_validation_failed:{exc}") from exc


__all__ = [
    "JSONNestingDepthExceeded",
    "JsonOutputParser",
    "OutputParserException",
    "PartialJSONParser",
    "RobustStreamingJSONParser",
    "StreamingJSONParser",
    "is_strict_json_text",
    "parse_complete_json",
    "parse_complete_json_any",
    "strip_thinking",
    "strip_thinking_tokens",
    "structured_json_candidates",
    "structured_json_strings",
]
