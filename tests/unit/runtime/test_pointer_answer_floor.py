# ruff: noqa: RUF001 — the measured shapes are Russian; the Cyrillic is the point.
"""A reply that only says where the file is, to a user who cannot open it.

The substantive-answer floor is a length test, and the failure measured on the
live stand walks straight past it. The user asks for an article; the agent
writes 13 KB of one into a workspace file and reports back "Готово. Статья …
сохранена в ``workspace/article_fem_elasticity.md`` (13 031 байт, ~960 слов).
Структура статьи: 1. Введение …". That notice is 1 200-1 900 characters, so it
clears any floor an operator would set, and over three hours of production runs
the floor fired exactly once — the failure was never presented to the mechanism
that exists to correct it. It happens in roughly two runs in three.

What separates a filing notice from an answer is not its length but its length
RELATIVE to what the run produced. These tests pin that comparison: the run's
own history holds both sides of it (the write call's arguments hold the content,
the assistant text is the answer), and where the user has a file browser the
whole thing stays inert, because there a path really is an answer.

They also pin what the refusal is allowed to SPEND. Measured on the stand it
recognises the failure perfectly and repairs none of it: told in so many words
to write the answer out, the model filed a second notice, both times. So the
refusal carries an attempt budget rather than the single shot the length floor
gets, and the tests below fix both ends of it — every attempt is spent when the
model will not be corrected, the run then finishes on whatever it has and says
so out loud, and a model that answers on the first ask never reaches the second.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import (
    _finalize_prose_gate_applies,
    _plain_stop_answer_floor_applies,
)
from protocore.tests_support.adapters import InMemoryLLMProvider, InMemoryToolRegistry

from ._tool_fixtures import MockTool

_LOGGER = "protocore.runtime.query"

#: The deliverable's path, as the model typed it into the write call.
ARTICLE_PATH = "workspace/article_fem_elasticity.md"

#: The deliverable. Only its size carries meaning here — 13 031 characters is
#: what the stand measured, and it is comfortably above the written-content
#: floor that separates a document from a scratch file.
ARTICLE = (
    "Метод конечных элементов в задачах теории упругости. " * 300
)[:13_031]

#: The reply the user actually received. Long enough to clear any sane length
#: floor, and it contains nothing of the article.
FILING_NOTICE = (
    "Готово. Статья «Метод конечных элементов в задачах теории упругости» "
    "сохранена в `workspace/article_fem_elasticity.md` (13 031 байт, ~960 "
    "слов). Структура статьи: 1. Введение. 2. Вариационная постановка. "
    "3. Дискретизация области. 4. Численные примеры. 5. Заключение."
)

#: The same notice, written by a model whose deployment sets a high length
#: floor. Long enough that the SHORT-ANSWER test is satisfied by it and only the
#: pointer test can object — which is the shape that used to be unreachable,
#: because the floor firing first spent the shot both tests drew on.
LONG_FILING_NOTICE = (
    "Готово. Полный текст статьи «Метод конечных элементов в задачах теории "
    "упругости» сохранён в `workspace/article_fem_elasticity.md`. "
    + (
        "В файле пять разделов, все формулы набраны в LaTeX, а список "
        "литературы оформлен по ГОСТ и вынесен в конец документа. "
    )
    * 5
).strip()

#: A reply that carries the work back AND says where the file is — the shape
#: the mechanism must never touch.
SUBSTANTIVE_ANSWER = (
    "Кратко по статье (полный текст — в `workspace/article_fem_elasticity.md`).\n\n"
    + (
        "Вариационная постановка задачи теории упругости сводится к минимизации "
        "функционала энергии на пространстве допустимых перемещений; "
        "дискретизация треугольными элементами первого порядка даёт систему "
        "линейных уравнений с разреженной матрицей жёсткости, а сходимость по "
        "энергетической норме имеет первый порядок по шагу сетки. "
    )
    * 12
).strip()

#: One chunk of a file written in several calls. Below the deliverable floor on
#: its own; two of them are above it.
ARTICLE_CHUNK = ARTICLE[:3_000]

#: A run legitimately writes small files in passing. This is one.
SCRATCH_PATH = "workspace/notes.md"
SCRATCH_NOTE = "Черновые заметки по ходу работы. " * 12

#: A length floor high enough that a filing notice satisfies it, as a
#: deployment that has tuned the short-answer test upward would have.
HIGH_FLOOR = 400


def _prose_stream(text: str) -> list[LLMStreamEvent]:
    """A plain-stop turn: some visible text, no tool call, ``end_turn``."""
    stream = [LLMStreamEvent(name="message_start", payload={})]
    if text:
        stream.extend(
            [
                LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
                LLMStreamEvent(
                    name="content_block_delta",
                    payload={"text": text, "kind": "text"},
                ),
                LLMStreamEvent(name="content_block_stop", payload={}),
            ]
        )
    stream.append(
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )
    )
    return stream


def _build(engine_factory, in_memory_runtime, **rc_kwargs: Any):
    """An agent with a write tool and no terminal-tool contract.

    The context window is deliberately large: these runs carry a 13 KB write in
    history and the point is that it sits there untouched, so compaction must
    not be part of what is being measured.
    """
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(
        MockTool(
            tool_name="Write",
            description="Write a file",
            parameters_schema={
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        )
    )
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=1_048_576, **rc_kwargs)
    )
    return engine, in_memory_runtime["llm"]


def _write(
    llm: InMemoryLLMProvider,
    *,
    path: str = ARTICLE_PATH,
    content: str = ARTICLE,
    call_id: str = "toolu_write_1",
) -> None:
    """Script the turn that produces the deliverable."""
    llm.queue_tool_call_response(
        tool_call_id=call_id,
        tool_name="Write",
        tool_input={"path": path, "content": content},
    )


def _repair_turns(engine) -> list[Message]:
    return [
        m
        for m in engine.history
        if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)
        == SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR
    ]


def _assistant_texts(engine) -> list[str]:
    return [
        b.text.strip()
        for m in engine.history
        if m.role is MessageRole.assistant
        for b in m.content_blocks
        if isinstance(b, TextBlock) and b.text.strip()
    ]


async def _drive(engine, prompt: str = "напиши статью про МКЭ в теории упругости") -> None:
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=prompt)])
    ):
        pass


def _diag(caplog, event: str) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith(f"DIAG query.finalize_prose_gate.{event} ")
    ]


def test_the_measured_shape_is_what_these_tests_claim_it_is() -> None:
    """The fixtures, checked against the defaults they are meant to straddle.

    Every assertion below rests on these relationships, and a default change
    that quietly moved one of them would leave the tests passing for the wrong
    reason.
    """
    rc = RuntimeConstants()
    assert len(ARTICLE) == 13_031
    assert len(ARTICLE) >= rc.finalize_prose_gate_pointer_min_written_chars
    # The notice clears the length floor and is still a fraction of the article.
    assert len(FILING_NOTICE) > rc.finalize_prose_gate_min_chars
    assert len(FILING_NOTICE) < (
        rc.finalize_prose_gate_pointer_max_answer_fraction * len(ARTICLE)
    )
    # The long notice clears a floor set high enough to make the short-answer
    # test inapplicable, and is STILL a fraction of the article. Both halves
    # matter: the first is what makes it reach the pointer test at all, the
    # second is what the pointer test then objects to.
    assert len(LONG_FILING_NOTICE) > HIGH_FLOOR
    assert len(LONG_FILING_NOTICE) < (
        rc.finalize_prose_gate_pointer_max_answer_fraction * len(ARTICLE)
    )
    # Bounded and non-zero. This assertion used to demand MORE than one
    # attempt, on the reasoning that a single repair turn had been seen to
    # change nothing. Measured on the stand with only this value moved, the
    # article scenario was acceptable in 2 of 3 runs at one attempt and 0 of 3
    # at three: extra asking spends the run's turns and grows the context the
    # answer is written from, and a request repeated does not become a
    # compulsion. What the tests below rest on is that the budget exists and
    # ends — not how large it is.
    assert rc.finalize_prose_gate_pointer_max_repair_attempts >= 1
    # A real summary of the same article is not.
    assert len(SUBSTANTIVE_ANSWER) >= (
        rc.finalize_prose_gate_pointer_max_answer_fraction * len(ARTICLE)
    )
    # The scratch note is below the deliverable floor.
    assert len(SCRATCH_NOTE) < rc.finalize_prose_gate_pointer_min_written_chars


# ---------------------------------------------------------------------------
# The loop property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_filing_notice_after_a_large_write_buys_one_more_turn(
    engine_factory, in_memory_runtime
) -> None:
    """The headline case. The article is written, the reply is a notice about
    the file, and on a surface where the user cannot open that file the run does
    NOT complete on it: it is asked once more, and the second reply is the one
    the user keeps."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))

    await _drive(engine)

    # Three round-trips: the write, the filing notice, the repaired answer.
    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    # One attempt of the pointer refusal's OWN budget, and the short-answer
    # floor's shot untouched — a model corrected on the first ask has cost the
    # run nothing else, and the floor is still armed for the failure it owns.
    assert engine._pointer_answer_repair_attempts == 1
    assert engine._finalize_prose_gate_used is False
    assert len(_repair_turns(engine)) == 1
    assert (
        _repair_turns(engine)[0].content_blocks[0].text
        == engine.config.rc.finalize_prose_gate_repair_text
    )
    # The answer is in history, and the notice was not retracted — the floor
    # asks for more, it never erases what the model already said.
    assert SUBSTANTIVE_ANSWER in _assistant_texts(engine)
    assert FILING_NOTICE in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_a_model_that_files_the_same_notice_every_time_spends_the_budget(
    engine_factory, in_memory_runtime
) -> None:
    """The bound, from both sides. A model that answers every correction with
    the same notice is asked exactly as many times as the budget allows and then
    left alone: the run completes on the last notice rather than trading it
    forever, and the turn after the budget is never taken."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )
    budget = engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts

    _write(llm)
    # One notice for the first stop, one for each repair turn the budget buys.
    for _ in range(budget + 1):
        llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    # And one more that must never be reached.
    llm._scripted_streams.append(_prose_stream("этот ход никто не запрашивал"))

    await _drive(engine)

    # The write, the first notice, and one round-trip per attempt.
    assert len(llm.calls) == 2 + budget
    assert engine.state is LoopState.COMPLETED
    assert len(_repair_turns(engine)) == budget
    assert engine._pointer_answer_repair_attempts == budget
    assert "этот ход никто не запрашивал" not in _assistant_texts(engine)
    # The short-answer floor never entered into it and is still armed.
    assert engine._finalize_prose_gate_used is False


@pytest.mark.asyncio
async def test_giving_up_is_said_out_loud(
    engine_factory, in_memory_runtime, caplog
) -> None:
    """A run that spends every attempt and still ships a filing notice is the
    outcome an operator has to be able to find, and it is otherwise invisible —
    from the outside a run that gave up looks exactly like one that succeeded on
    its last try. One line, once, carrying what was refused and how often."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )
    budget = engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts

    _write(llm)
    for _ in range(budget + 1):
        llm._scripted_streams.append(_prose_stream(FILING_NOTICE))

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _drive(engine)

    assert len(_diag(caplog, "pointer_answer_repair")) == budget
    released = _diag(caplog, "pointer_answer_budget_spent")
    assert len(released) == 1
    assert f"attempts={budget}/{budget}" in released[0]
    assert f"path={ARTICLE_PATH}" in released[0]
    assert f"written_chars={len(ARTICLE)}" in released[0]


@pytest.mark.asyncio
async def test_a_run_that_was_repaired_announces_nothing(
    engine_factory, in_memory_runtime, caplog
) -> None:
    """The counterpart, and the reason the release is reported where the run
    ends rather than where the last attempt is charged: a budget spent on a
    repair that WORKED is not a failure and says nothing. Charging time could
    not draw this distinction — the answer it would be reporting on has not been
    written yet."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_pointer_max_repair_attempts=1,
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _drive(engine)

    # The budget IS spent — and the run still has nothing to confess.
    assert engine._pointer_answer_repair_attempts == 1
    assert _diag(caplog, "pointer_answer_budget_spent") == []


@pytest.mark.asyncio
async def test_the_length_floor_no_longer_spends_the_pointer_refusal(
    engine_factory, in_memory_runtime
) -> None:
    """The two tests, disentangled. This run trips the SHORT-answer floor first
    — it did the work and said seven characters about it — and answers the
    correction with a notice long enough to satisfy that floor and nothing else.

    While both drew on one shot, the second reply was unreachable: the floor had
    spent it, and the run shipped the notice. Now the floor's shot pays for the
    floor's repair, the pointer refusal's budget pays for its own, and the run
    is asked the second question it was always owed."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_min_chars=HIGH_FLOOR,
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream("Готово."))
    llm._scripted_streams.append(_prose_stream(LONG_FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))

    await _drive(engine)

    assert len(llm.calls) == 4
    assert engine.state is LoopState.COMPLETED
    # Two repairs from two different mechanisms, one attempt each.
    assert len(_repair_turns(engine)) == 2
    assert engine._finalize_prose_gate_used is True
    assert engine._pointer_answer_repair_attempts == 1
    assert SUBSTANTIVE_ANSWER in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_the_short_answer_floor_is_still_a_single_shot(
    engine_factory, in_memory_runtime
) -> None:
    """What was NOT widened. The floor's own failure — a run that did real work
    and reports it in a sentence — still buys exactly one correction, and a
    model that answers it with another thin sentence completes on that. Nothing
    about the pointer refusal's budget reaches this test."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_min_chars=HIGH_FLOOR,
    )

    _write(llm, path=SCRATCH_PATH, content=SCRATCH_NOTE)
    llm._scripted_streams.append(_prose_stream("Готово."))
    llm._scripted_streams.append(_prose_stream("Готово, всё сделано."))
    llm._scripted_streams.append(_prose_stream("этот ход никто не запрашивал"))

    await _drive(engine)

    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    assert len(_repair_turns(engine)) == 1
    assert engine._finalize_prose_gate_used is True
    assert engine._pointer_answer_repair_attempts == 0
    assert "этот ход никто не запрашивал" not in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_a_budget_of_one_is_the_single_shot_it_replaced(
    engine_factory, in_memory_runtime
) -> None:
    """The floor of the new knob is the behaviour it grew out of: at 1 the
    refusal asks once, is ignored once, and the run completes on the notice —
    exactly what the shared latch did, which is what the stand measured and
    found insufficient."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_pointer_max_repair_attempts=1,
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream("этот ход никто не запрашивал"))

    await _drive(engine)

    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    assert len(_repair_turns(engine)) == 1
    assert engine._pointer_answer_repair_attempts == 1
    assert "этот ход никто не запрашивал" not in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_a_budget_of_zero_switches_the_refusal_off(
    engine_factory, in_memory_runtime, caplog
) -> None:
    """The third way to disable the pointer test, for an operator who keeps the
    hidden workspace and the length floor but wants the notice to stand: no
    repair turn, nothing charged, and no release either — a mechanism that never
    engaged has no outcome to report."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_pointer_max_repair_attempts=0,
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream("never reached"))

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert _repair_turns(engine) == []
    assert engine._pointer_answer_repair_attempts == 0
    assert engine._finalize_prose_gate_used is False
    assert _diag(caplog, "pointer_answer_budget_spent") == []
    assert _assistant_texts(engine) == [FILING_NOTICE]


@pytest.mark.asyncio
async def test_a_visible_workspace_is_left_exactly_as_it_was(
    engine_factory, in_memory_runtime
) -> None:
    """The default, stated as the product fact it is. Where the user can open
    the workspace, "it is in this file" IS an answer, and the identical history
    completes on the notice with nothing spent — no extra round-trip, no latch,
    no scaffolding."""
    engine, llm = _build(engine_factory, in_memory_runtime)
    assert engine.config.rc.workspace_visible_to_user is True

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert engine._pointer_answer_repair_attempts == 0
    assert _repair_turns(engine) == []
    assert _assistant_texts(engine) == [FILING_NOTICE]


@pytest.mark.asyncio
async def test_a_summary_that_also_gives_the_path_is_an_answer(
    engine_factory, in_memory_runtime
) -> None:
    """The boundary that matters most. Naming the file is not the offence —
    naming it INSTEAD of answering is. A reply that carries the substance and
    also says where the full text lives passes untouched."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )
    assert ARTICLE_PATH in SUBSTANTIVE_ANSWER

    _write(llm)
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert _repair_turns(engine) == []


@pytest.mark.asyncio
async def test_a_small_write_is_not_a_deliverable(
    engine_factory, in_memory_runtime
) -> None:
    """A run that jots a note down and mentions it has not withheld an article.
    Below the written-content floor the comparison is not made at all, however
    terse the reply."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )

    _write(llm, path=SCRATCH_PATH, content=SCRATCH_NOTE)
    llm._scripted_streams.append(
        _prose_stream(f"Готово, заметки сохранены в `{SCRATCH_PATH}`.")
    )
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert _repair_turns(engine) == []


@pytest.mark.asyncio
async def test_the_kill_switch_leaves_the_filing_notice_alone(
    engine_factory, in_memory_runtime
) -> None:
    """``finalize_prose_gate_enabled=False`` is the one switch for the whole
    substantive-answer machinery, this test included: with the workspace hidden
    and the article written, the notice still completes the run."""
    engine, llm = _build(
        engine_factory,
        in_memory_runtime,
        workspace_visible_to_user=False,
        finalize_prose_gate_enabled=False,
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert _repair_turns(engine) == []
    assert _assistant_texts(engine) == [FILING_NOTICE]


@pytest.mark.asyncio
async def test_the_log_says_which_file_and_how_much_was_withheld(
    engine_factory, in_memory_runtime, caplog
) -> None:
    """Production keeps WARNING and nothing below it, and the two ways into the
    repair turn need opposite reading. The pointer line carries what makes this
    one what it is: both sizes and the file they were measured against."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, workspace_visible_to_user=False
    )

    _write(llm)
    llm._scripted_streams.append(_prose_stream(FILING_NOTICE))
    llm._scripted_streams.append(_prose_stream(SUBSTANTIVE_ANSWER))

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _drive(engine)

    lines = _diag(caplog, "pointer_answer_repair")
    assert len(lines) == 1
    assert f"run={engine.config.run_id}" in lines[0]
    # Which attempt this was, out of how many — the same shape the read-back
    # driver's forced turns carry, and the thing that makes a repeated refusal
    # legible as one sequence instead of several unrelated events.
    budget = engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    assert f"attempt=1/{budget}" in lines[0]
    assert f"answer_chars={len(FILING_NOTICE)}" in lines[0]
    assert f"written_chars={len(ARTICLE)}" in lines[0]
    assert f"path={ARTICLE_PATH}" in lines[0]
    # The plain length-floor line describes a different failure and must not
    # also be emitted for this one.
    assert _diag(caplog, "plain_stop_repair") == []


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


def _seed_write_run(
    engine,
    *,
    answer: str,
    path: str = ARTICLE_PATH,
    content: str = ARTICLE,
    is_error: bool = False,
    call_id: str = "t-write",
) -> None:
    """History for a run that wrote a file and then said something about it.

    Everything the floor tests except the pointer comparison is arranged to say
    "do not fire": there is a visible answer, it comes after the work, and it
    clears the default length floor. Whatever the predicate then returns is the
    pointer test's doing and nothing else.
    """
    engine.history.extend(
        [
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="напиши статью")],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id=call_id,
                        name="Write",
                        arguments_json=_write_args(path, content),
                    )
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=call_id,
                        content="written" if not is_error else "disk full",
                        is_error=is_error,
                    )
                ],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text=answer)],
            ),
        ]
    )


def _write_args(path: str, content: str) -> str:
    """A write call's arguments as the loop records them in history."""
    return json.dumps({"path": path, "content": content}, ensure_ascii=False)


def _hidden(engine_factory, **rc_kwargs: Any):
    return engine_factory(
        rc=RuntimeConstants(
            model_context_window=1_048_576,
            workspace_visible_to_user=False,
            **rc_kwargs,
        )
    )


def test_the_notice_is_refused_and_the_summary_is_not(engine_factory) -> None:
    """The whole mechanism in two lines: same run, same file, two replies."""
    pointer_run = _hidden(engine_factory)
    _seed_write_run(pointer_run, answer=FILING_NOTICE)
    assert _plain_stop_answer_floor_applies(pointer_run) is True

    answered_run = _hidden(engine_factory)
    _seed_write_run(answered_run, answer=SUBSTANTIVE_ANSWER)
    assert _plain_stop_answer_floor_applies(answered_run) is False


def test_a_write_that_failed_produced_nothing_to_point_at(engine_factory) -> None:
    """A rejected write left no file, so the reply cannot be withholding one.
    Counting the attempt would have this run demand a longer answer about
    content that does not exist anywhere."""
    engine = _hidden(engine_factory)
    _seed_write_run(engine, answer=FILING_NOTICE, is_error=True)

    assert _plain_stop_answer_floor_applies(engine) is False


def test_an_answer_about_something_else_is_not_a_pointer(engine_factory) -> None:
    """The reference is what ties the reply to the file. A short answer that
    never mentions the article is a different question — possibly a bad answer,
    but not one this test can call a filing notice for THIS file."""
    engine = _hidden(engine_factory)
    _seed_write_run(engine, answer="Да, метод сходится по энергетической норме.")

    assert _plain_stop_answer_floor_applies(engine) is False


def test_the_bare_filename_counts_as_naming_the_file(engine_factory) -> None:
    """The model writes through ``workspace/…`` and reports back the filename
    alone about as often as the full path; both are the same reference."""
    engine = _hidden(engine_factory)
    _seed_write_run(
        engine,
        answer=(
            "Готово — всё сложено в article_fem_elasticity.md, там пять "
            "разделов и список литературы."
        ),
    )

    assert _plain_stop_answer_floor_applies(engine) is True


def test_chunked_writes_to_one_path_are_one_deliverable(engine_factory) -> None:
    """A long file arrives as ``Write`` plus a run of ``AppendFile`` calls, and
    what the run produced is their sum — neither chunk clears the floor alone."""
    engine = _hidden(engine_factory)
    floor = engine.config.rc.finalize_prose_gate_pointer_min_written_chars
    assert len(ARTICLE_CHUNK) < floor <= 2 * len(ARTICLE_CHUNK)
    _seed_write_run(
        engine, answer=FILING_NOTICE, content=ARTICLE_CHUNK, call_id="t-write-1"
    )
    # The second chunk lands on the same path, ahead of the answer.
    engine.history.insert(
        3,
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="t-write-2",
                    name="AppendFile",
                    arguments_json=_write_args(ARTICLE_PATH, ARTICLE_CHUNK),
                )
            ],
        ),
    )
    engine.history.insert(
        4,
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="t-write-2", content="appended", is_error=False
                )
            ],
        ),
    )

    assert _plain_stop_answer_floor_applies(engine) is True


def test_two_small_files_do_not_add_up_to_a_deliverable(engine_factory) -> None:
    """The counterpart: the floor is per FILE. A run that wrote two notes has
    produced two notes, not one document, and the reply is asked about each of
    them separately."""
    engine = _hidden(engine_factory)
    _seed_write_run(
        engine, answer=FILING_NOTICE, content=ARTICLE_CHUNK, call_id="t-write-1"
    )
    engine.history.insert(
        3,
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="t-write-2",
                    name="Write",
                    arguments_json=_write_args(SCRATCH_PATH, ARTICLE_CHUNK),
                )
            ],
        ),
    )
    engine.history.insert(
        4,
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="t-write-2", content="written", is_error=False
                )
            ],
        ),
    )

    assert _plain_stop_answer_floor_applies(engine) is False


def test_either_knob_at_zero_disables_the_comparison(engine_factory) -> None:
    """Two ways to switch this off without touching the length floor, for an
    operator who wants one and not the other."""
    no_fraction = _hidden(
        engine_factory, finalize_prose_gate_pointer_max_answer_fraction=0.0
    )
    _seed_write_run(no_fraction, answer=FILING_NOTICE)
    assert _plain_stop_answer_floor_applies(no_fraction) is False

    no_floor = _hidden(engine_factory, finalize_prose_gate_pointer_min_written_chars=0)
    _seed_write_run(no_floor, answer=FILING_NOTICE)
    assert _plain_stop_answer_floor_applies(no_floor) is False


def test_the_length_floor_still_does_its_own_job(engine_factory) -> None:
    """Nothing here narrows what was already caught: a run that did work and
    then said almost nothing still trips the floor, with no file in sight."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            workspace_visible_to_user=False,
            finalize_prose_gate_min_chars=400,
        )
    )
    engine.history.extend(
        [
            Message(
                role=MessageRole.user, content_blocks=[TextBlock(text="проверь сервис")]
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="t-work", name="Agent", arguments_json="{}"
                    )
                ],
            ),
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id="t-work", content="done", is_error=False
                    )
                ],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="Готово.")],
            ),
        ]
    )

    assert _plain_stop_answer_floor_applies(engine) is True


def test_the_dispatch_seam_refuses_the_same_reply(
    engine_factory, in_memory_runtime
) -> None:
    """The terminal-tool path is the same gate at the other seam, so it makes
    the same judgement: a background terminal about to seal a run whose only
    prose is a filing notice is vetoed exactly as a prose-less one is."""
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(
        MockTool(
            tool_name="Finalize",
            description="End the run",
            parameters_schema={"declared_deliverables": {"type": "array"}},
        )
    )
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=1_048_576, workspace_visible_to_user=False
        ),
        expected_terminal_tool="Finalize",
    )
    call = ToolCall(id="t-final", name="Finalize", arguments={})

    _seed_write_run(engine, answer=SUBSTANTIVE_ANSWER)
    assert _finalize_prose_gate_applies(engine, call) is False

    pointer_run = engine_factory(
        rc=RuntimeConstants(
            model_context_window=1_048_576, workspace_visible_to_user=False
        ),
        expected_terminal_tool="Finalize",
    )
    _seed_write_run(pointer_run, answer=FILING_NOTICE)
    assert _finalize_prose_gate_applies(pointer_run, call) is True

    # And it answers to the same budget: once that is spent the terminal tool
    # goes through, so a run with a terminal contract cannot be held at the
    # dispatch seam any longer than one without a terminal tool is held at its
    # completion.
    pointer_run._pointer_answer_repair_attempts = (
        pointer_run.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    )
    assert _finalize_prose_gate_applies(pointer_run, call) is False


def test_a_spent_budget_stops_the_refusal_at_both_seams(engine_factory) -> None:
    """The bound is on the RUN, not on a seam. Whichever completion path a
    deployment ends its runs on, an exhausted budget means the same thing there:
    stop asking."""
    engine = _hidden(engine_factory)
    _seed_write_run(engine, answer=FILING_NOTICE)
    assert _plain_stop_answer_floor_applies(engine) is True

    engine._pointer_answer_repair_attempts = (
        engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    )
    assert _plain_stop_answer_floor_applies(engine) is False


def test_a_spent_short_answer_latch_does_not_bind_the_pointer_test(
    engine_factory,
) -> None:
    """The entanglement, stated as the predicate-level fact it was. A run that
    already spent the floor's single shot is still asked about a filing notice;
    before the split, the latch alone answered for both and this returned
    False."""
    engine = _hidden(engine_factory)
    _seed_write_run(engine, answer=FILING_NOTICE)
    engine._finalize_prose_gate_used = True

    assert _plain_stop_answer_floor_applies(engine) is True


def test_a_spent_pointer_budget_does_not_bind_the_short_answer_floor(
    engine_factory,
) -> None:
    """And the other direction, which matters just as much: a run that has
    burnt every pointer attempt on one file may still be corrected for a reply
    that is simply too short to be an answer at all."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            workspace_visible_to_user=False,
            finalize_prose_gate_min_chars=HIGH_FLOOR,
        )
    )
    _seed_write_run(engine, answer="Готово.")
    engine._pointer_answer_repair_attempts = (
        engine.config.rc.finalize_prose_gate_pointer_max_repair_attempts
    )

    assert _plain_stop_answer_floor_applies(engine) is True
