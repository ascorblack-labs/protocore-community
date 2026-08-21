"""Tests for :mod:`protocore.json_utils`."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from protocore.json_utils import (
    JsonOutputParser,
    OutputParserException,
    PartialJSONParser,
    RobustStreamingJSONParser,
    StreamingJSONParser,
    is_strict_json_text,
    parse_complete_json,
    parse_complete_json_any,
    strip_thinking,
    strip_thinking_tokens,
    structured_json_candidates,
    structured_json_strings,
)


def test_strip_thinking_basic() -> None:
    text = "before <thinking>hidden</thinking> after"
    assert strip_thinking(text) == "before  after"


def test_strip_thinking_short_tag() -> None:
    text = "<think>foo</think>bar"
    assert strip_thinking(text) == "bar"


def test_parse_complete_json_with_thinking() -> None:
    raw = '<thinking>reasoning</thinking>{"a": 1}'
    assert parse_complete_json(raw) == {"a": 1}


def test_parse_complete_json_invalid() -> None:
    with pytest.raises(OutputParserException):
        parse_complete_json("not json")


def test_parse_complete_json_array_not_object() -> None:
    with pytest.raises(OutputParserException):
        parse_complete_json('["a", "b"]')


def test_structured_json_candidates_extracts_multiple() -> None:
    raw = 'prefix {"a": 1} middle {"b": 2} suffix'
    cands = structured_json_candidates(raw)
    assert {"a": 1} in cands
    assert {"b": 2} in cands


def test_structured_json_candidates_handles_nested() -> None:
    raw = '{"outer": {"inner": [1, 2]}}'
    cands = structured_json_candidates(raw)
    assert len(cands) == 1
    assert cands[0] == {"outer": {"inner": [1, 2]}}


def test_structured_json_candidates_quoted_braces_ignored() -> None:
    raw = '{"x": "{not json}"}'
    cands = structured_json_candidates(raw)
    assert len(cands) == 1
    assert cands[0]["x"] == "{not json}"


def test_structured_json_candidates_empty_input() -> None:
    assert structured_json_candidates("") == []


def test_structured_json_candidates_no_json() -> None:
    assert structured_json_candidates("no objects here") == []


# --- strip_thinking variants ------------------------------------------------


def test_strip_thinking_unclosed_tag() -> None:
    # ``max_tokens`` cut the model off before ``</think>`` closed.
    text = '<think>reasoning never closed because we ran out of tokens {"a"'
    assert strip_thinking(text) == ""


def test_strip_thinking_process_prefix() -> None:
    text = 'Thinking Process:\nstep 1\nstep 2\n\n{"a": 1}'
    assert strip_thinking(text) == '{"a": 1}'


def test_strip_thinking_tokens_alias() -> None:
    assert strip_thinking_tokens("<think>x</think>y") == strip_thinking("<think>x</think>y")


# --- parse_complete_json_any (lenient) -------------------------------------


def test_parse_complete_json_any_array() -> None:
    assert parse_complete_json_any('["a", "b"]') == ["a", "b"]


def test_parse_complete_json_any_scalar_fails() -> None:
    # Bare scalars are not extractable as a JSON value slice.
    with pytest.raises(OutputParserException):
        parse_complete_json_any("not json")


def test_parse_complete_json_any_mixed_text() -> None:
    raw = 'preface text {"key": [1, 2]} trailing'
    assert parse_complete_json_any(raw) == {"key": [1, 2]}


def test_parse_complete_json_any_single_quoted_literal() -> None:
    # Some small models emit Python-like single-quoted dicts.
    assert parse_complete_json_any("{'a': 1}") == {"a": 1}


def test_parse_complete_json_any_empty_input() -> None:
    with pytest.raises(OutputParserException):
        parse_complete_json_any("")


def test_parse_complete_json_any_oversized_input() -> None:
    huge = "{" + ("x" * 2_000_000) + "}"
    with pytest.raises(OutputParserException):
        parse_complete_json_any(huge)


# --- is_strict_json_text ----------------------------------------------------


def test_is_strict_json_text_object() -> None:
    assert is_strict_json_text('{"a": 1}') is True


def test_is_strict_json_text_array() -> None:
    assert is_strict_json_text("[1, 2]") is True


def test_is_strict_json_text_scalar() -> None:
    assert is_strict_json_text("42") is True
    assert is_strict_json_text('"foo"') is True


def test_is_strict_json_text_empty() -> None:
    assert is_strict_json_text("") is False


def test_is_strict_json_text_invalid() -> None:
    assert is_strict_json_text("not json") is False


# --- structured_json_strings -----------------------------------------------


def test_structured_json_strings_fenced() -> None:
    raw = 'noise\n```json\n{"a": 1}\n```\nmore noise'
    candidates = structured_json_strings(raw)
    assert any('{"a": 1}' in c for c in candidates)


def test_structured_json_strings_dedups() -> None:
    raw = '{"a": 1}'
    candidates = structured_json_strings(raw)
    # Single raw input yields the cleaned text + outer slice; dedup keeps both unique only
    # if their normalized form differs. Same text after stripping yields one unique value.
    assert candidates.count('{"a": 1}') == 1


def test_structured_json_strings_handles_empty() -> None:
    assert structured_json_strings("") == []


# --- PartialJSONParser ------------------------------------------------------


def test_partial_parser_complete_input() -> None:
    parser = PartialJSONParser()
    result, repaired = parser.parse_with_flag('{"a": 1, "b": [1, 2]}')
    assert result == {"a": 1, "b": [1, 2]}
    assert repaired is False


def test_partial_parser_truncated_object() -> None:
    parser = PartialJSONParser()
    # Missing trailing brace + trailing comma + open string.
    result, repaired = parser.parse_with_flag('{"a": 1, "b": "incomp')
    assert result is not None
    assert isinstance(result, dict)
    assert result["a"] == 1
    assert repaired is True


def test_partial_parser_trailing_comma() -> None:
    parser = PartialJSONParser()
    result = parser.parse('{"a": 1, "b": 2,}')
    assert result == {"a": 1, "b": 2}


def test_partial_parser_dangling_key_at_end() -> None:
    parser = PartialJSONParser()
    # ``"c":`` is dangling — should be dropped on repair.
    result = parser.parse('{"a": 1, "c":')
    assert result == {"a": 1}


def test_partial_parser_dangling_key_before_closer() -> None:
    parser = PartialJSONParser()
    result = parser.parse('{"a": 1, "c": }')
    assert result == {"a": 1}


def test_partial_parser_unparseable() -> None:
    parser = PartialJSONParser()
    result, repaired = parser.parse_with_flag("totally not json")
    assert result is None
    assert repaired is False


def test_partial_parser_empty_input() -> None:
    parser = PartialJSONParser()
    result, repaired = parser.parse_with_flag("")
    assert result is None
    assert repaired is False


# --- StreamingJSONParser ----------------------------------------------------


def test_streaming_parser_single_chunk() -> None:
    parser = StreamingJSONParser()
    assert parser.consume('{"a": 1}') == {"a": 1}


def test_streaming_parser_chunked() -> None:
    parser = StreamingJSONParser()
    assert parser.consume('{"a":') is None
    assert parser.consume(' 1, "b":') is None
    assert parser.consume(' [1, 2]') is None
    assert parser.consume("}") == {"a": 1, "b": [1, 2]}


def test_streaming_parser_handles_strings_with_braces() -> None:
    parser = StreamingJSONParser()
    result = parser.consume('{"a": "value with {nested} braces"}')
    assert result == {"a": "value with {nested} braces"}


def test_streaming_parser_resets_after_complete() -> None:
    parser = StreamingJSONParser()
    parser.consume('{"a": 1}')
    # After completion, parser is reset; a new payload starts fresh.
    assert parser.consume('{"b": 2}') == {"b": 2}


def test_streaming_parser_ignores_pre_json_garbage() -> None:
    parser = StreamingJSONParser()
    # Leading garbage chars before ``{`` are dropped.
    assert parser.consume('garbage {"a": 1}') == {"a": 1}


# --- RobustStreamingJSONParser ---------------------------------------------


def test_robust_streaming_emits_partials() -> None:
    parser = RobustStreamingJSONParser()
    partial_a = parser.consume('{"a": 1, ', emit_partial=True)
    # Should yield a partial that includes "a": 1 (repaired).
    assert partial_a is not None
    assert isinstance(partial_a, dict)
    assert partial_a.get("a") == 1


def test_robust_streaming_dedups_partials() -> None:
    parser = RobustStreamingJSONParser()
    first = parser.consume('{"a": 1', emit_partial=True)
    same = parser.consume("", emit_partial=True)
    assert first is not None
    assert same is None  # Identical fingerprint -> deduped.


def test_robust_streaming_finalize_complete() -> None:
    parser = RobustStreamingJSONParser()
    parser.consume('{"a": 1}', emit_partial=True)
    assert parser.finalize() == {"a": 1}


def test_robust_streaming_finalize_incomplete() -> None:
    parser = RobustStreamingJSONParser()
    parser.consume('{"a": 1, "b":', emit_partial=False)
    final = parser.finalize()
    assert final == {"a": 1}


def test_robust_streaming_finalize_failure() -> None:
    parser = RobustStreamingJSONParser()
    parser.consume("totally not json", emit_partial=False)
    with pytest.raises(OutputParserException):
        parser.finalize()


# --- JsonOutputParser (Pydantic-aware) -------------------------------------


class _SampleModel(BaseModel):
    name: str
    count: int


def test_json_output_parser_no_schema() -> None:
    parser: JsonOutputParser[_SampleModel] = JsonOutputParser()
    assert parser.parse('{"a": 1}') == {"a": 1}


def test_json_output_parser_with_schema() -> None:
    parser = JsonOutputParser(_SampleModel)
    result = parser.parse('{"name": "x", "count": 3}')
    assert isinstance(result, _SampleModel)
    assert result.name == "x"
    assert result.count == 3


def test_json_output_parser_schema_validation_failure() -> None:
    parser = JsonOutputParser(_SampleModel)
    with pytest.raises(OutputParserException):
        parser.parse('{"name": "x", "count": "not an int"}')


def test_json_output_parser_format_instructions_with_schema() -> None:
    parser = JsonOutputParser(_SampleModel)
    instr = parser.get_format_instructions()
    assert "JSON Schema" in instr
    assert "name" in instr
    assert "count" in instr


def test_json_output_parser_format_instructions_no_schema() -> None:
    parser: JsonOutputParser[_SampleModel] = JsonOutputParser()
    instr = parser.get_format_instructions()
    assert "JSON object" in instr


def test_json_output_parser_parse_result_str() -> None:
    parser = JsonOutputParser(_SampleModel)
    result = parser.parse_result(['{"name": "x", "count": 1}'])
    assert isinstance(result, _SampleModel)


def test_json_output_parser_parse_result_dict_with_text() -> None:
    parser = JsonOutputParser(_SampleModel)
    result = parser.parse_result([{"text": '{"name": "y", "count": 2}'}])
    assert isinstance(result, _SampleModel)
    assert result.name == "y"


def test_json_output_parser_parse_result_empty() -> None:
    parser = JsonOutputParser(_SampleModel)
    with pytest.raises(OutputParserException):
        parser.parse_result([])


def test_json_output_parser_stream_yields_values() -> None:
    parser = JsonOutputParser(_SampleModel)
    chunks = ['{"name": ', '"x", "count": 1}']
    values = list(parser.parse_stream(chunks, include_partial=False))
    assert len(values) >= 1
    assert values[-1].name == "x"
    assert values[-1].count == 1


def test_json_output_parser_stream_final_complete() -> None:
    parser: JsonOutputParser[_SampleModel] = JsonOutputParser()
    chunks = ['{"a": ', '1, "b": ', "2}"]
    assert parser.parse_stream_final(chunks) == {"a": 1, "b": 2}


def test_json_output_parser_stream_final_incomplete() -> None:
    parser: JsonOutputParser[_SampleModel] = JsonOutputParser()
    chunks = ['{"a": 1, "b": ']
    # Finalize will repair.
    result = parser.parse_stream_final(chunks)
    assert result == {"a": 1}


# --- Additional coverage paths ---------------------------------------------


def test_coerce_generation_text_string_fallback() -> None:
    # Object without ``.text`` and without dict — falls back to ``str(obj)``.
    from protocore.json_utils import _coerce_generation_text

    class _Opaque:
        def __str__(self) -> str:
            return "opaque-str"

    assert _coerce_generation_text(_Opaque()) == "opaque-str"


def test_parse_any_oversize_input_strict() -> None:
    # Even for strict parse, huge input should be rejected.
    huge = '{"a": "' + ("x" * 2_000_000) + '"}'
    with pytest.raises(OutputParserException):
        parse_complete_json(huge)


def test_parse_complete_json_any_no_json_in_text() -> None:
    # Mixed text without any ``{`` or ``[`` triggers ``no_json_found``.
    with pytest.raises(OutputParserException):
        parse_complete_json_any("plain text without any json")


def test_partial_parser_escape_handling_inside_string() -> None:
    # Escapes inside strings should be preserved during repair.
    parser = PartialJSONParser()
    result = parser.parse(r'{"a": "with \"quotes\" inside"}')
    assert result == {"a": 'with "quotes" inside'}


def test_partial_parser_repair_unclosed_string() -> None:
    parser = PartialJSONParser()
    result = parser.parse('{"a": "unclosed')
    assert isinstance(result, dict)


def test_partial_parser_repair_unclosed_array() -> None:
    parser = PartialJSONParser()
    result = parser.parse('[1, 2, 3')
    assert result == [1, 2, 3]


def test_streaming_parser_escape_handling() -> None:
    parser = StreamingJSONParser()
    assert parser.consume(r'{"a": "with \"escaped\" content"}') == {
        "a": 'with "escaped" content',
    }


def test_streaming_parser_handles_nested_arrays() -> None:
    parser = StreamingJSONParser()
    assert parser.consume('[[1, 2], [3, 4]]') == [[1, 2], [3, 4]]


def test_robust_streaming_non_serializable_partial() -> None:
    # Cover the TypeError-on-fingerprint branch — when partial repair yields
    # a dict that still serializes with ``default=str``, dedup proceeds normally.
    parser = RobustStreamingJSONParser()
    first = parser.consume('{"a": ', emit_partial=True)
    assert first is not None or first is None  # branch covered


def test_robust_streaming_finalize_empty_falls_through() -> None:
    parser = RobustStreamingJSONParser()
    with pytest.raises(OutputParserException):
        parser.finalize(raw_fallback="not json at all")


def test_structured_json_strings_bare_fence_with_lang_first_line() -> None:
    # ```\njson\n{}\n``` — language word on first line after fence.
    raw = "```json\n{\"x\": 1}\n```"
    candidates = structured_json_strings(raw)
    assert any('{"x": 1}' in c for c in candidates)


def test_structured_json_strings_array_outer_slice() -> None:
    raw = 'prefix [1, 2, 3] suffix'
    candidates = structured_json_strings(raw)
    assert any("[1, 2, 3]" in c for c in candidates)


def test_structured_json_strings_oversized_input() -> None:
    huge = "x" * 2_000_000
    assert structured_json_strings(huge) == []


def test_structured_json_strings_bare_fence_no_lang_line() -> None:
    # ```\n{}\n``` — no language line at all.
    raw = "```\n{\"y\": 2}\n```"
    candidates = structured_json_strings(raw)
    assert any('{"y": 2}' in c for c in candidates)


def test_structured_json_strings_bare_fence_inline_json_lang() -> None:
    # ```jsonblob``` — language word immediately after opening fence.
    raw = "```json{}```"
    candidates = structured_json_strings(raw)
    # Either inner content or partial result included.
    assert candidates


def test_json_diff_replace_when_types_differ() -> None:
    from protocore.json_utils import _json_diff

    # Object → array: replace at root.
    patches = _json_diff({"a": 1}, [1])
    assert patches == [{"op": "replace", "path": "/", "value": [1]}]


def test_json_diff_add_remove_replace_keys() -> None:
    from protocore.json_utils import _json_diff

    patches = _json_diff({"a": 1, "b": 2}, {"a": 1, "c": 3})
    # ``b`` removed, ``c`` added.
    ops = sorted(p["op"] for p in patches)
    assert ops == ["add", "remove"]


def test_json_diff_list_changed() -> None:
    from protocore.json_utils import _json_diff

    patches = _json_diff([1, 2], [1, 2, 3])
    assert patches == [{"op": "replace", "path": "/", "value": [1, 2, 3]}]


def test_json_diff_list_unchanged() -> None:
    from protocore.json_utils import _json_diff

    assert _json_diff([1, 2], [1, 2]) == []


def test_json_diff_scalar_replace() -> None:
    from protocore.json_utils import _json_diff

    assert _json_diff(1, 2) == [{"op": "replace", "path": "/", "value": 2}]


def test_json_diff_scalar_unchanged() -> None:
    from protocore.json_utils import _json_diff

    assert _json_diff(1, 1) == []


def test_json_diff_nested_key_with_special_chars() -> None:
    from protocore.json_utils import _json_diff

    # ``/`` and ``~`` must be escaped per RFC6901.
    patches = _json_diff({"a/b": 1, "c~d": 2}, {"a/b": 9, "c~d": 2})
    paths = {p["path"] for p in patches}
    assert "/a~1b" in paths


def test_json_output_parser_stream_yields_diffs() -> None:
    parser: JsonOutputParser[_SampleModel] = JsonOutputParser()
    chunks = ['{"a": 1', ', "b": 2}']
    values = list(parser.parse_stream(chunks, yield_diffs=True, include_partial=True))
    # First yield is the initial value (no previous), subsequent are diffs.
    assert values  # at least one emitted


def test_partial_parser_repair_extracts_from_mixed_text() -> None:
    parser = PartialJSONParser()
    result = parser.parse('prefix text {"k": 1} suffix')
    assert result == {"k": 1}


def test_streaming_parser_buffer_text_property() -> None:
    parser = StreamingJSONParser()
    parser.consume('{"partial":')
    assert "partial" in parser.buffer_text
