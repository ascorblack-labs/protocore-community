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
        self._last_partial_fingerprint: str | None = None
        self._last_complete: Any | None = None

    def consume(self, chunk: str, *, emit_partial: bool = True) -> Any | None:
        """Consume a chunk; return a complete or partial JSON value, or ``None``."""
        complete = self.streaming.consume(chunk)
        if complete is not None:
            self._last_complete = complete
            self._last_partial_fingerprint = None
            return complete

        if not emit_partial:
            return None
        partial = self.partial.parse(self.streaming.buffer_text)
        if partial is None:
            return None
        try:
            fingerprint = json.dumps(
                partial,
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            # Partial parse may contain non-serializable values; skip dedup.
            return partial
        if fingerprint == self._last_partial_fingerprint:
            return None
        self._last_partial_fingerprint = fingerprint
        return partial

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
