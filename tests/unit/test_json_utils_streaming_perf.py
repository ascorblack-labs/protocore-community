"""Cost and fidelity of the streaming JSON parser under chunked input.

The streaming parser used to re-parse and re-repair the whole accumulated
buffer on every chunk, which made a stream cost roughly the cube of its length:
28 KiB arriving in 64-byte chunks took 4.74 s, and 88 KiB never finished. These
tests pin the shape of the cost, not only its size — a budget alone would still
pass on a fast machine running a quadratic algorithm — and check that the fast
path answers exactly what the repair path answers.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any

import pytest

from protocore.json_utils import (
    PartialJSONParser,
    RobustStreamingJSONParser,
    StreamingJSONParser,
)
from tests._fixtures.legacy_json_repair import (
    PartialJSONParser as LegacyPartialJSONParser,
)

CHUNK_SIZE = 64
LARGE_INPUT_BYTES = 28_672
SMALL_INPUT_BYTES = 7_168
LARGE_INPUT_BUDGET_SECONDS = 1.0
# Four times the input for at most six times the cost: linear, with room for a
# loaded machine. The pre-fix implementation grew by a factor of about 15 here.
MAX_GROWTH_RATIO = 6.0
TIMING_REPEATS = 5
PROPERTY_DOCUMENTS = 500
PROPERTY_SEED = 20260906


def _fixture(size_bytes: int) -> str:
    """A truncated document of nested records with escapes and non-ASCII text.

    Escapes matter: a quoted run containing backslashes is what made the old
    repair pass quadratic, and non-ASCII text keeps the byte and character
    counts from coinciding.
    """
    document = (
        '{"files": ['
        + ",".join(
            f'{{"path": "/a/b_{index}.py", "text": "line \\"q\\" \\\\ {index} абв"}}'
            for index in range(1000)
        )
        + "]}"
    )
    assert len(document) >= size_bytes
    return document[:size_bytes]


def _stream_once(text: str) -> float:
    """Seconds to feed ``text`` to the parser in fixed-size chunks."""
    parser = RobustStreamingJSONParser()
    started = time.perf_counter()
    for offset in range(0, len(text), CHUNK_SIZE):
        parser.consume(text[offset : offset + CHUNK_SIZE])
    return time.perf_counter() - started


def _stream_seconds(text: str) -> float:
    """Best of a few runs feeding ``text`` to the parser in fixed chunks."""
    return min(_stream_once(text) for _ in range(TIMING_REPEATS))


@pytest.mark.perf
def test_streaming_28_kib_stays_within_budget() -> None:
    elapsed = _stream_seconds(_fixture(LARGE_INPUT_BYTES))
    assert elapsed < LARGE_INPUT_BUDGET_SECONDS, (
        f"28 KiB in {CHUNK_SIZE}-byte chunks took {elapsed:.3f} s, "
        f"budget is {LARGE_INPUT_BUDGET_SECONDS} s"
    )


@pytest.mark.perf
def test_streaming_cost_grows_no_faster_than_input() -> None:
    """Four times the input for no more than six times the time.

    The two sizes are timed next to each other and the best ratio of several
    rounds is taken, rather than the ratio of two separately-taken bests: this
    suite runs on parallel workers, and a stall that lands on only one of the
    two measurements would otherwise read as superlinear growth.
    """
    small_text = _fixture(SMALL_INPUT_BYTES)
    large_text = _fixture(LARGE_INPUT_BYTES)
    rounds = [
        (_stream_once(small_text), _stream_once(large_text))
        for _ in range(TIMING_REPEATS)
    ]
    small, large = min(rounds, key=lambda pair: pair[1] / pair[0])
    ratio = large / small
    assert ratio <= MAX_GROWTH_RATIO, (
        f"four times the input cost {ratio:.1f} times the time "
        f"({small:.4f} s -> {large:.4f} s)"
    )


# --- fidelity against the repair path ---------------------------------------


def _reference_emissions(text: str, chunk_size: int) -> list[Any]:
    """What the parser would emit if it re-repaired the buffer every chunk.

    This is the behaviour the incremental path replaces, spelled out directly:
    accumulate, ask :class:`PartialJSONParser` about the whole buffer, and drop
    an answer identical to the previous one.
    """
    accumulator = StreamingJSONParser()
    repairer = PartialJSONParser()
    previous: Any = object()
    emissions: list[Any] = []
    for offset in range(0, len(text), chunk_size):
        complete = accumulator.consume(text[offset : offset + chunk_size])
        if complete is not None:
            emissions.append(complete)
            previous = object()
            continue
        partial = repairer.parse(accumulator.buffer_text)
        if partial is None or partial == previous:
            continue
        previous = partial
        emissions.append(partial)
    return emissions


def _actual_emissions(text: str, chunk_size: int) -> list[Any]:
    parser = RobustStreamingJSONParser()
    emissions: list[Any] = []
    for offset in range(0, len(text), chunk_size):
        value = parser.consume(text[offset : offset + chunk_size])
        if value is not None:
            emissions.append(value)
    return emissions


_SCALARS: tuple[Any, ...] = (
    0,
    1,
    -2,
    3.5,
    1e3,
    True,
    False,
    None,
    "",
    "ascii",
    "кириллица",
    'q"uo\\te',
    "tab\tnl\n",
    " leading",
    "trailing  ",
    "  pad  ",
    "emoji \U0001f600",
    "ӿ",
    "日本語",
)
_MAX_GENERATED_DEPTH = 3
_SCALAR_SHARE = 0.25
_OBJECT_SHARE = 0.6
_MAX_MEMBERS = 4


def _random_value(rnd: random.Random, depth: int = 0) -> Any:
    roll = rnd.random()
    if depth > _MAX_GENERATED_DEPTH or roll < _SCALAR_SHARE:
        return rnd.choice(_SCALARS)
    if roll < _OBJECT_SHARE:
        return {
            (f"k{index}" if rnd.random() < 0.8 else f'к\\"{index}'): _random_value(
                rnd,
                depth + 1,
            )
            for index in range(rnd.randint(0, _MAX_MEMBERS))
        }
    return [_random_value(rnd, depth + 1) for _ in range(rnd.randint(0, _MAX_MEMBERS))]


def test_incremental_emissions_match_the_repair_path() -> None:
    """Every emission, on 500 truncated documents, is what repair would give.

    Seeded rather than randomised per run: a fidelity test that fails only
    sometimes tells the next reader nothing about which change broke it.
    """
    rnd = random.Random(PROPERTY_SEED)
    for _ in range(PROPERTY_DOCUMENTS):
        document = json.dumps(
            _random_value(rnd),
            ensure_ascii=rnd.random() < 0.5,
        )
        truncated = document[: rnd.randint(1, len(document))]
        chunk_size = rnd.choice([1, 2, 3, 7, CHUNK_SIZE])
        assert _actual_emissions(truncated, chunk_size) == _reference_emissions(
            truncated,
            chunk_size,
        ), f"diverged on {truncated!r} at chunk size {chunk_size}"


def test_repair_of_an_escaped_string_is_linear() -> None:
    """A truncated string full of escapes no longer costs quadratic time.

    The three cleanup patterns this parser used to run in a loop each retried a
    quoted-run match at every quote in the buffer, and an unterminated string
    made every one of those attempts walk to the end of the input. 64 KiB took
    18.6 s.
    """
    repairer = PartialJSONParser()
    timings: list[float] = []
    for size in (16_384, 65_536):
        payload = ('{"a": "' + 'x\\"y\\\\z' * size)[:size]
        started = time.perf_counter()
        repairer.parse(payload)
        timings.append(time.perf_counter() - started)
    assert timings[1] / timings[0] <= MAX_GROWTH_RATIO, (
        f"four times the input cost {timings[1] / timings[0]:.1f} times the time"
    )


# --- fidelity against the implementation this one replaced ------------------


def _sets_become_empty_objects(value: Any) -> Any:
    """The one deliberate difference, applied to a reference answer.

    The previous repair left a truncated object key standing, so a body cut
    off mid-key came back carrying a Python ``set`` where a JSON document can
    only hold an object. The current repair drops the value-less key instead,
    which turns that leaf into an empty object. Every other leaf must match.
    """
    if isinstance(value, set):
        return {}
    if isinstance(value, dict):
        return {key: _sets_become_empty_objects(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sets_become_empty_objects(item) for item in value]
    return value


def test_repair_recovers_everything_the_previous_implementation_recovered() -> None:
    """New repair vs. the frozen previous one, on the same 500 documents.

    The fast-path test above compares the incremental parser with the repair
    path, and both are current code — agreement there says nothing about
    whether the rewrite lost ground. This one holds the rewrite against the
    implementation it replaced: a document the old parser could recover must
    still be recoverable, and the only value that may differ is the set leaf
    the old parser produced for a truncated key.
    """
    rnd = random.Random(PROPERTY_SEED)
    current = PartialJSONParser()
    previous = LegacyPartialJSONParser()
    for _ in range(PROPERTY_DOCUMENTS):
        document = json.dumps(_random_value(rnd), ensure_ascii=rnd.random() < 0.5)
        truncated = document[: rnd.randint(1, len(document))]
        before, before_repaired = previous.parse_with_flag(truncated)
        after, after_repaired = current.parse_with_flag(truncated)
        if before is None:
            continue
        assert after is not None, f"lost a recoverable document: {truncated!r}"
        assert after_repaired == before_repaired, truncated
        assert after == _sets_become_empty_objects(before), truncated


def test_a_body_truncated_inside_a_key_keeps_the_records_before_it() -> None:
    """The realistic truncation: an output limit cuts a list of records.

    Everything already complete has to survive. Refusing the whole document
    because its last record is half-written throws away the records that are
    whole, which for a caller extracting facts from a model's answer is the
    difference between a partial answer and none at all.
    """
    raw = (
        '{"facts": [{"text": "prefers dark mode", "confidence": 0.9}, '
        '{"text": "works in UTC+3", "confidence": 0.8}, {"te'
    )
    value, repaired = PartialJSONParser().parse_with_flag(raw)
    assert repaired is True
    assert value is not None
    assert value["facts"][:2] == [
        {"text": "prefers dark mode", "confidence": 0.9},
        {"text": "works in UTC+3", "confidence": 0.8},
    ]
