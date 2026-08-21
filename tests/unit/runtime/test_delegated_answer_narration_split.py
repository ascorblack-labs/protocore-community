# ruff: noqa: RUF001 — Cyrillic is the subject matter: these are
# verbatim answer openings, quoted so the detector is tested on what it will meet.
"""The leading-narration split on a delegating leader's answer.

Two layers, because the change has two halves that can fail independently:

* the detector, tested in BOTH directions — it must fire on the openers that
  were measured on real runs, and must leave alone the openings of real answers
  that merely resemble them;
* the loop, driven end to end through ``engine.run`` — the split has to reach
  the live stream and durable history as the SAME two blocks, only when the run
  actually delegated, and it must never take text away from either.

The openers below are quoted from delivered answers. The negatives matter more
than the positives: a detector validated only on the text it was written for is
not validated.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    BlockVisibility,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResult,
)
from protocore.runtime.answer_narration import leading_narration_span
from protocore.runtime.events import EventType, TurnEvent

from ._tool_fixtures import MockTool

SCAN = 600
FLOOR = 200

# A body long enough to clear the surviving-answer floor, so each case tests the
# detector rather than the floor.
BODY = (
    "\n\nМетод конечных элементов — численный метод решения краевых задач для "
    "дифференциальных уравнений в частных производных. Область разбивается на "
    "конечные элементы, на каждом из которых искомая функция приближается "
    "полиномом невысокой степени, а условия сопряжения в узлах дают систему "
    "алгебраических уравнений относительно узловых значений."
)

# ---------------------------------------------------------------------------
# Measured narration openers. Every one is first-person about the run's own
# state or possession, followed by an announcement of intent.
# ---------------------------------------------------------------------------
NARRATION_OPENERS = [
    "Теперь у меня есть все материалы. Составлю итоговый обзор.",
    "Теперь у меня есть все пять подборок. Составлю обзор.",
    "Все пять файлов с источниками прочитаны. Теперь составлю итоговый обзор.",
    "Отлично, у меня достаточно источников по всем пяти аспектам. Теперь составлю обзор.",
    "Хорошо, у меня достаточно источников. Теперь напишу итоговый обзор.",
    "Теперь у меня достаточно материалов. Собставлю полный обзор.",
    "Теперь у меня достаточно данных для формирования итогового ответа.",
    "Файлы прочитаны. Теперь у меня есть вся информация для итогового ответа.",
    "Теперь у меня есть полная картина по всем пяти аспектам. Составлю обзор.",
    "I now have a good collection of sources. Let me compile the article.",
    "Now I have enough source material. Let me compose the article.",
    "I have a good set of sources now. Let me compile the article from these references.",
]

# ---------------------------------------------------------------------------
# Openings of real answers that must survive untouched. A scope-framing opener
# is about the SUBJECT, in the third person, and several of these name the same
# work material the narration does — which is exactly why they are here.
# ---------------------------------------------------------------------------
ANSWER_OPENERS = [
    "# Метод конечных элементов в задачах теории упругости",
    "**Метод конечных элементов (МКЭ)** — это численный метод решения "
    "дифференциальных уравнений с частными производными.",
    "Метод конечных элементов (МКЭ) — один из фундаментальных численных методов "
    "механики деформируемого твёрдого тела.",
    "Обзор охватывает пять аспектов темы и опирается только на найденные публикации.",
    "Ниже — полный связный обзор, собранный из всех источников, найденных за эту беседу.",
    "Вот обзор литературы по теме, подготовленный по каталогу.",
    "Вот полная статья:",
    "Написана научная статья объёмом около 1000 слов.",
    "Статья объёмом около 1000 слов сохранена в файле `article.md`.",
    "Текст статьи написан полностью. Все утверждения снабжены идентификаторами.",
    "Теория упругости является фундаментом расчётов в машиностроении.",
    "По запросу в каталоге не нашлось работ, напрямую посвящённых этой теме.",
    "Based on the sources found, here is the answer:",
    "Два плюс два будет 4.",
]


def _span(text: str, *, scan: int = SCAN):
    return leading_narration_span(text, scan_chars=scan, complete=True)


@pytest.mark.parametrize("opener", NARRATION_OPENERS)
def test_measured_narration_openers_are_found(opener: str) -> None:
    span = _span(opener + BODY)
    assert span.found, f"narration not detected: {opener!r}"
    assert span.settled
    # The cut lands on the narration and nowhere inside the answer.
    assert (opener + BODY)[: span.length].strip() == opener.strip()


@pytest.mark.parametrize("opener", ANSWER_OPENERS)
def test_real_answer_openings_are_left_whole(opener: str) -> None:
    assert not _span(opener + BODY).found, f"answer opening was cut: {opener!r}"


def test_first_person_answer_without_an_announcement_is_left_whole() -> None:
    """"Do you have everything?" — "I have all the data you asked for: …".

    The possession claim on its own is ordinary prose. Requiring the
    announcement of intent alongside it is what separates the two.
    """

    opener = "Да, у меня есть все данные, которые вы просили." + BODY
    assert not _span(opener).found


def test_announcement_without_a_state_claim_is_left_whole() -> None:
    assert not _span("Составлю обзор." + BODY).found


def test_presentative_sentence_ends_the_scan() -> None:
    """A sentence handing the answer over belongs to the answer.

    «Вот полная научная статья:» reads like an announcement and is not one — it
    addresses the reader, so it stays visible while the narration before it
    does not.
    """

    text = "Теперь у меня достаточно источников для написания статьи. Вот полная научная статья:" + BODY
    span = _span(text)
    assert span.found
    assert text[: span.length] == "Теперь у меня достаточно источников для написания статьи."


def test_scan_stops_at_the_first_sentence_that_is_not_narration() -> None:
    text = (
        "Теперь у меня есть все материалы. Составлю обзор. "
        "Первый аспект раскрыт в трёх публикациях." + BODY
    )
    span = _span(text)
    assert text[: span.length] == "Теперь у меня есть все материалы. Составлю обзор."


def test_scan_never_leaves_the_first_paragraph() -> None:
    text = "Теперь у меня есть все материалы.\n\nСоставлю обзор." + BODY
    # The announcement is in the next paragraph, so the shape is incomplete and
    # nothing is cut.
    assert not _span(text).found


def test_scan_ceiling_bounds_the_cut() -> None:
    opener = "Теперь у меня есть все материалы. Составлю итоговый обзор."
    assert _span(opener + BODY, scan=len(opener)).length == len(opener)
    assert not _span(opener + BODY, scan=10).found


def test_a_span_is_unsettled_while_the_narration_run_can_still_grow() -> None:
    """The streaming contract: never act on a verdict that can still change."""

    text = "Теперь у меня есть все материалы. Составлю итоговый обзор.\n\nПервый аспект…"
    partial = leading_narration_span(text[:20], scan_chars=SCAN, complete=False)
    assert not partial.settled and not partial.found

    # Both narration sentences have arrived but a third could still follow.
    open_run = leading_narration_span(
        "Теперь у меня есть все материалы. Составлю итоговый обзор.",
        scan_chars=SCAN,
        complete=False,
    )
    assert not open_run.settled

    settled = leading_narration_span(text, scan_chars=SCAN, complete=False)
    assert settled.settled
    assert text[: settled.length] == "Теперь у меня есть все материалы. Составлю итоговый обзор."


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class _DelegationTool(MockTool):
    """Stands in for the host subagent-dispatch tool."""

    is_concurrent_safe = False
    is_parallel_delegation = True

    async def invoke(self, context, arguments: dict[str, Any]) -> ToolResult:  # type: ignore[no-untyped-def]
        self.calls.append(dict(arguments))
        return ToolResult(tool_call_id="", content="subtask done", is_error=False)


ANSWER_TURN_TEXT = NARRATION_OPENERS[0] + BODY


def _rc(**overrides: Any) -> RuntimeConstants:
    return RuntimeConstants(
        model_context_window=4_096,
        finalize_prose_gate_enabled=False,
        terminal_tool_nudge_enabled=False,
        **overrides,
    )


def _delegation_turn(tool_name: str) -> list[LLMStreamEvent]:
    args = {"prompt": "gather sources"}
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(
            name="tool_use_start",
            payload={"tool_call_id": "sub-1", "tool_name": tool_name},
        ),
        LLMStreamEvent(
            name="tool_use_input_delta",
            payload={"tool_call_id": "sub-1", "partial_input_json": json.dumps(args)},
        ),
        LLMStreamEvent(
            name="tool_use_stop",
            payload={"tool_call_id": "sub-1", "final_input": args},
        ),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
        ),
    ]


def _answer_turn(text: str, *, chunk: int = 17) -> list[LLMStreamEvent]:
    """The answer, streamed in small deltas the way a provider sends it."""

    events: list[LLMStreamEvent] = [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
    ]
    for start in range(0, len(text), chunk):
        events.append(
            LLMStreamEvent(
                name="content_block_delta",
                payload={"text": text[start : start + chunk], "kind": "text"},
            )
        )
    events.extend(
        [
            LLMStreamEvent(name="content_block_stop", payload={}),
            LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
            ),
        ]
    )
    return events


def _text_blocks_on_the_wire(events: list[TurnEvent]) -> list[dict[str, Any]]:
    """Reassemble each streamed text block as ``{visibility, text}``.

    Keyed by ``(turn_id, block_idx)``: block indices restart per turn, so a
    run-wide index alone merges the first turn's block with the second's and
    the last ``content_block_stop`` silently wins.
    """

    blocks: dict[tuple[Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any]] = []
    for evt in events:
        idx = evt.payload.get("block_idx")
        if not isinstance(idx, int):
            continue
        key = (evt.payload.get("turn_id"), idx)
        if evt.type is EventType.CONTENT_BLOCK_START and evt.payload.get("kind") == "text":
            blocks[key] = {"visibility": evt.payload.get("visibility"), "text": ""}
            order.append(key)
        elif evt.type is EventType.CONTENT_BLOCK_DELTA and key in blocks:
            delta = evt.payload.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                blocks[key]["text"] += delta.get("text") or ""
        elif evt.type is EventType.CONTENT_BLOCK_STOP and key in blocks:
            blocks[key]["visibility"] = evt.payload.get("visibility")
    return [blocks[key] for key in order]


def _final_assistant_text_blocks(engine) -> list[TextBlock]:  # type: ignore[no-untyped-def]
    for message in reversed(engine.history):
        if message.role is not MessageRole.assistant:
            continue
        blocks = [b for b in message.content_blocks if isinstance(b, TextBlock)]
        if blocks:
            return blocks
    return []


async def _run_scripted(
    engine_factory,  # type: ignore[no-untyped-def]
    in_memory_runtime: dict[str, object],
    *,
    rc: RuntimeConstants,
    delegate: bool,
) -> tuple[Any, list[TurnEvent]]:
    engine = engine_factory(rc=rc)
    if delegate:
        in_memory_runtime["tools"].register(_DelegationTool(tool_name="Agent"))
        in_memory_runtime["llm"]._scripted_streams.append(_delegation_turn("Agent"))
    in_memory_runtime["llm"]._scripted_streams.append(_answer_turn(ANSWER_TURN_TEXT))
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="обзор?")])
    events = [evt async for evt in engine.run(user)]
    return engine, events


@pytest.mark.asyncio
async def test_delegating_run_splits_its_answer_live_and_in_history(
    engine_factory, in_memory_runtime
) -> None:
    engine, events = await _run_scripted(
        engine_factory, in_memory_runtime, rc=_rc(), delegate=True
    )
    assert engine._run_delegated

    wire = _text_blocks_on_the_wire(events)
    assert len(wire) == 2, wire
    assert wire[0]["visibility"] == BlockVisibility.COLLAPSED.value
    assert wire[0]["text"] == NARRATION_OPENERS[0]
    assert wire[1]["visibility"] == BlockVisibility.PUBLIC.value
    # Nothing is dropped: the two blocks are the model's text, exactly.
    assert wire[0]["text"] + wire[1]["text"] == ANSWER_TURN_TEXT

    blocks = _final_assistant_text_blocks(engine)
    assert len(blocks) == 2
    assert blocks[0].visibility is BlockVisibility.COLLAPSED
    assert blocks[0].text == NARRATION_OPENERS[0]
    assert blocks[1].visibility is BlockVisibility.PUBLIC
    assert blocks[0].text + blocks[1].text == ANSWER_TURN_TEXT


@pytest.mark.asyncio
async def test_run_that_never_delegated_keeps_one_public_block(
    engine_factory, in_memory_runtime
) -> None:
    """The gate. Same answer, no subtask dispatched, nothing touched."""

    engine, events = await _run_scripted(
        engine_factory, in_memory_runtime, rc=_rc(), delegate=False
    )
    assert not engine._run_delegated

    wire = _text_blocks_on_the_wire(events)
    assert len(wire) == 1
    assert wire[0]["visibility"] == BlockVisibility.PUBLIC.value
    assert wire[0]["text"] == ANSWER_TURN_TEXT

    blocks = _final_assistant_text_blocks(engine)
    assert len(blocks) == 1
    assert blocks[0].visibility is BlockVisibility.PUBLIC


@pytest.mark.asyncio
async def test_switch_off_leaves_a_delegating_run_whole(
    engine_factory, in_memory_runtime
) -> None:
    engine, events = await _run_scripted(
        engine_factory,
        in_memory_runtime,
        rc=_rc(delegated_answer_narration_split_enabled=False),
        delegate=True,
    )
    assert engine._run_delegated
    wire = _text_blocks_on_the_wire(events)
    assert len(wire) == 1
    assert wire[0]["text"] == ANSWER_TURN_TEXT
    assert len(_final_assistant_text_blocks(engine)) == 1


@pytest.mark.asyncio
async def test_answer_too_short_behind_the_narration_is_left_visible(
    engine_factory, in_memory_runtime
) -> None:
    """A reply that is narration and little else keeps its narration.

    Collapsing it would leave an empty bubble, so the floor refuses the split
    and the reader still sees every character the model wrote.
    """

    engine = engine_factory(rc=_rc())
    in_memory_runtime["tools"].register(_DelegationTool(tool_name="Agent"))
    in_memory_runtime["llm"]._scripted_streams.append(_delegation_turn("Agent"))
    short = NARRATION_OPENERS[0] + "\n\nОбзор во вложении."
    in_memory_runtime["llm"]._scripted_streams.append(_answer_turn(short))
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="обзор?")])
    events = [evt async for evt in engine.run(user)]

    wire = _text_blocks_on_the_wire(events)
    assert len(wire) == 1
    assert wire[0]["visibility"] == BlockVisibility.PUBLIC.value
    assert wire[0]["text"] == short


@pytest.mark.asyncio
async def test_between_tool_narration_stays_one_collapsed_block(
    engine_factory, in_memory_runtime
) -> None:
    """Text that shares a turn with a tool call is already narration.

    The split may still fire on it; what must not happen is a public block
    appearing in a turn the structural rule collapses whole.
    """

    engine = engine_factory(rc=_rc())
    in_memory_runtime["tools"].register(_DelegationTool(tool_name="Agent"))
    narrating_dispatch = _answer_turn(ANSWER_TURN_TEXT)[:-1] + _delegation_turn("Agent")[1:]
    in_memory_runtime["llm"]._scripted_streams.append(narrating_dispatch)
    in_memory_runtime["llm"]._scripted_streams.append(_answer_turn("Готово." + BODY))
    user = Message(role=MessageRole.user, content_blocks=[TextBlock(text="обзор?")])
    events = [evt async for evt in engine.run(user)]

    first_turn_blocks = _text_blocks_on_the_wire(events)[:1]
    assert first_turn_blocks
    assert first_turn_blocks[0]["visibility"] == BlockVisibility.COLLAPSED.value


@pytest.mark.asyncio
async def test_delegation_fact_survives_a_snapshot_resume(engine_factory) -> None:
    """A run re-driven on another pod must render its answer the same way."""

    engine = engine_factory(rc=_rc())
    engine._run_delegated = True
    snapshot = engine.snapshot()
    assert snapshot["run_delegated"] is True

    resumed = engine_factory(rc=_rc())
    await resumed.resume_from_snapshot(snapshot)
    assert resumed._run_delegated is True


@pytest.mark.asyncio
async def test_a_snapshot_without_the_field_resumes_as_not_delegated(
    engine_factory,
) -> None:
    engine = engine_factory(rc=_rc())
    snapshot = engine.snapshot()
    snapshot.pop("run_delegated")
    await engine.resume_from_snapshot(snapshot)
    assert engine._run_delegated is False
