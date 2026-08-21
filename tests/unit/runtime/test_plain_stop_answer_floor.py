"""A run that stops without answering does not get to call that a success.

The prose gate proper sits at the terminal-tool dispatch seam, so it only ever
sees runs that call a terminal tool. A run where the model simply stops —
``finish_reason='stop'``, no tool call — walks straight past it to the
completion, and on a deployment that declares no terminal tool that is how
nearly every run ends. The measured failure: a leader delegated correctly, its
subagents wrote five result files, and the reply the user received was under a
hundred characters. The gate was enabled throughout and never participated.

These tests drive the real loop to that completion and assert on what the run
does there — how many times the model is asked again, and what ends up in
history — rather than on a predicate in isolation. The predicate tests at the
bottom pin the one place the terminal path's answer-field exemption still means
something once no terminal tool has been called.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_PROSE_GATE_REPAIR,
    TERMINAL_TOOL_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _plain_stop_answer_floor_applies
from protocore.tests_support.adapters import InMemoryLLMProvider, InMemoryToolRegistry

from ._tool_fixtures import MockTool

#: A pointer, not an answer — the shape measured in production. Its exact
#: length does not matter, only that it sits under the scope's floor.
THIN_ANSWER = "Done. The results are saved in reports/ — see the files there."

#: A reply that actually carries the work back to the user.
FULL_ANSWER = (
    "Across the five reports the reviewers agree on three findings: the "
    "release job never verifies the image it just pushed, the retry budget is "
    "shared between two unrelated callers, and the queue lane leaks one slot "
    "per cancelled run. Details and line references are in reports/."
)

#: Above THIN_ANSWER, below FULL_ANSWER — the per-scope tuning the deployment
#: applies. The default floor of 1 is deliberately left alone.
FLOOR = 200


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
    """A leader with one work tool and no terminal-tool contract.

    ``expected_terminal_tool`` stays None on purpose: that is the deployment on
    which the dispatch-seam gate can never fire, so it is the deployment the
    floor has to hold up.
    """
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(MockTool(tool_name="Agent", description="Delegate work"))
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            finalize_prose_gate_min_chars=FLOOR,
            **rc_kwargs,
        )
    )
    return engine, in_memory_runtime["llm"]


def _delegate(llm: InMemoryLLMProvider, call_id: str = "toolu_agent_1") -> None:
    """Script a turn that does real work and produces a tool result."""
    llm.queue_tool_call_response(
        tool_call_id=call_id,
        tool_name="Agent",
        tool_input={"subagent_type": "reviewer"},
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


async def _drive(engine, prompt: str = "review the service and report") -> None:
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=prompt)])
    ):
        pass


# ---------------------------------------------------------------------------
# The loop property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_pointer_instead_of_an_answer_buys_one_more_turn(
    engine_factory, in_memory_runtime
) -> None:
    """The headline case. The model delegates, gets its result, and reports the
    work in a sentence too short to be the answer. The run does NOT complete on
    that: it is asked once more, and the second reply is the one the user
    keeps."""
    engine, llm = _build(engine_factory, in_memory_runtime)
    assert len(THIN_ANSWER) < FLOOR <= len(FULL_ANSWER)

    _delegate(llm)
    llm._scripted_streams.append(_prose_stream(THIN_ANSWER))
    llm._scripted_streams.append(_prose_stream(FULL_ANSWER))

    await _drive(engine)

    # Three round-trips: the delegation, the thin stop, the repaired answer.
    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is True
    # Exactly one repair turn, carrying the gate's own text.
    assert len(_repair_turns(engine)) == 1
    assert (
        _repair_turns(engine)[0].content_blocks[0].text
        == engine.config.rc.finalize_prose_gate_repair_text
    )
    # The substantive answer is in history, and the thin one was not erased —
    # the floor asks for more, it does not retract what the model already said.
    assert FULL_ANSWER in _assistant_texts(engine)
    assert THIN_ANSWER in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_a_model_that_ignores_the_repair_still_finishes(
    engine_factory, in_memory_runtime
) -> None:
    """One shot, and only one. A model that answers the repair turn with the
    same thin reply completes on it — the run must never be able to loop
    here."""
    engine, llm = _build(engine_factory, in_memory_runtime)

    _delegate(llm)
    llm._scripted_streams.append(_prose_stream(THIN_ANSWER))
    llm._scripted_streams.append(_prose_stream(THIN_ANSWER))
    # A fourth stream that must never be reached.
    llm._scripted_streams.append(_prose_stream("this turn was never asked for"))

    await _drive(engine)

    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    assert len(_repair_turns(engine)) == 1
    assert "this turn was never asked for" not in _assistant_texts(engine)


@pytest.mark.asyncio
async def test_a_substantive_answer_is_never_touched(
    engine_factory, in_memory_runtime
) -> None:
    """The healthy shape costs nothing: one work turn, one real answer, done.
    No extra round-trip, no latch spent, no scaffolding in history."""
    engine, llm = _build(engine_factory, in_memory_runtime)

    _delegate(llm)
    llm._scripted_streams.append(_prose_stream(FULL_ANSWER))
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert _repair_turns(engine) == []
    assert _assistant_texts(engine) == [FULL_ANSWER]


@pytest.mark.asyncio
async def test_the_kill_switch_leaves_the_thin_answer_alone(
    engine_factory, in_memory_runtime
) -> None:
    """``finalize_prose_gate_enabled=False`` restores the prior behaviour on
    THIS path too: the thin reply completes the run exactly as it used to."""
    engine, llm = _build(
        engine_factory, in_memory_runtime, finalize_prose_gate_enabled=False
    )

    _delegate(llm)
    llm._scripted_streams.append(_prose_stream(THIN_ANSWER))
    llm._scripted_streams.append(_prose_stream("never reached"))

    await _drive(engine)

    assert len(llm.calls) == 2
    assert engine.state is LoopState.COMPLETED
    assert engine._finalize_prose_gate_used is False
    assert _repair_turns(engine) == []
    assert _assistant_texts(engine) == [THIN_ANSWER]


@pytest.mark.asyncio
async def test_narration_before_the_work_is_not_the_answer(
    engine_factory, in_memory_runtime
) -> None:
    """Under the DEFAULT floor of 1. The model announces what it is about to do
    at length, does it, and then stops with nothing — so every visible word it
    produced predates the result it was supposed to report on. That is progress
    narration, not an answer, and the floor still fires.

    This is also the shape where the tail of history is a tool result rather
    than an assistant turn, so it pins that the repair turn is appended
    validly there.
    """
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(MockTool(tool_name="Agent", description="Delegate work"))
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    assert engine.config.rc.finalize_prose_gate_min_chars == 1
    llm: InMemoryLLMProvider = in_memory_runtime["llm"]

    llm.queue_tool_call_response(
        tool_call_id="toolu_agent_1",
        tool_name="Agent",
        tool_input={"subagent_type": "reviewer"},
        text_prefix="I will delegate the review and then summarise the findings.",
    )
    # A stop carrying nothing at all: the narration above is the only prose,
    # and it came BEFORE the work.
    llm._scripted_streams.append(_prose_stream(""))
    llm._scripted_streams.append(_prose_stream(FULL_ANSWER))

    await _drive(engine)

    assert len(llm.calls) == 3
    assert engine.state is LoopState.COMPLETED
    assert len(_repair_turns(engine)) == 1
    assert FULL_ANSWER in _assistant_texts(engine)


# ---------------------------------------------------------------------------
# The exemption: what the answer-field question means once nothing was called
# ---------------------------------------------------------------------------


def _seed_unanswered_run(engine) -> None:
    """History for a run that narrated, then worked, and never came back.

    Everything the floor tests EXCEPT the exemption is arranged to say "fire":
    there is visible assistant prose (so this is not the empty-completion
    guard's case), and it all predates the last real work (so nothing in it can
    be the answer to that work). Whatever the predicate then returns is the
    exemption's doing and nothing else.
    """
    engine.history.extend(
        [
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="review the service")],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="I will delegate the review now.")],
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
                        tool_call_id="t-work", content="wrote 5 files", is_error=False
                    )
                ],
            ),
        ]
    )


def _terminal_result(engine, *, tool: str, call_id: str = "t-final") -> None:
    """Put a satisfied terminal-tool submission into history."""
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id=call_id, name=tool, arguments_json="{}")
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=call_id,
                    content="submitted",
                    is_error=False,
                    metadata={TERMINAL_TOOL_METADATA_KEY: True},
                )
            ],
        )
    )


def test_an_answer_already_submitted_in_tool_args_exempts_the_run(
    engine_factory, in_memory_runtime
) -> None:
    """The one reading of the answer-field exemption that survives onto this
    path: the answer already reached the user, through a terminal tool that
    carries it in its own args. Demanding prose on top would ask the run to say
    everything twice."""
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(
        MockTool(
            tool_name="Answer",
            description="Submit the answer",
            parameters_schema={"message": {"type": "string"}},
        )
    )
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096), expected_terminal_tool="Answer"
    )
    _seed_unanswered_run(engine)
    _terminal_result(engine, tool="Answer")

    assert _plain_stop_answer_floor_applies(engine) is False


def test_a_background_terminal_does_not_exempt_the_run(
    engine_factory, in_memory_runtime
) -> None:
    """A terminal tool with no answer-carrying field submits nothing the user
    reads — it only ends the run. Its result in history is therefore no
    evidence that an answer exists, and the floor still applies."""
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(
        MockTool(
            tool_name="Finalize",
            description="End the run",
            parameters_schema={"declared_deliverables": {"type": "array"}},
        )
    )
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096),
        expected_terminal_tool="Finalize",
    )
    _seed_unanswered_run(engine)
    _terminal_result(engine, tool="Finalize")

    assert _plain_stop_answer_floor_applies(engine) is True


def test_an_uncalled_message_carrying_terminal_exempts_nothing(
    engine_factory, in_memory_runtime
) -> None:
    """The reason the terminal path's condition is not copied verbatim. Read
    off the schema alone, a message-carrying terminal tool would exempt this
    run — but it was never called, so nothing was submitted through it and the
    user has nothing at all. Keyed on the submission instead of the schema, the
    floor fires."""
    registry: InMemoryToolRegistry = in_memory_runtime["tools"]
    registry.register(
        MockTool(
            tool_name="Answer",
            description="Submit the answer",
            parameters_schema={"message": {"type": "string"}},
        )
    )
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096), expected_terminal_tool="Answer"
    )
    _seed_unanswered_run(engine)

    assert _plain_stop_answer_floor_applies(engine) is True


def test_the_latch_is_shared_with_the_dispatch_seam(
    engine_factory, in_memory_runtime
) -> None:
    """One shot for the whole mechanism, not one per path: a run whose terminal
    dispatch was already vetoed cannot also be repaired here."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    _seed_unanswered_run(engine)
    assert _plain_stop_answer_floor_applies(engine) is True

    engine._finalize_prose_gate_used = True
    assert _plain_stop_answer_floor_applies(engine) is False


def test_a_short_answer_with_no_tool_work_behind_it_is_left_alone(
    engine_factory,
) -> None:
    """A run that answered without touching a tool has nothing to under-report.

    The floor is a length test, and a length test cannot on its own tell a
    reply that collapsed from one that is correctly brief. What separates them
    is whether there was anything to report. Here a greeting is answered in one
    turn, far below the floor, and must pass untouched — otherwise the repair
    turn's only achievable effect is to pad a correct answer up to the
    threshold.
    """
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, finalize_prose_gate_min_chars=400
        )
    )
    engine.history.extend(
        [
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text="Привет! Как дела?")],
            ),
            Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="Привет! Всё хорошо, спасибо.")],
            ),
        ]
    )

    assert _plain_stop_answer_floor_applies(engine) is False


def test_the_same_short_answer_after_real_work_does_trip_the_floor(
    engine_factory,
) -> None:
    """The companion to the test above: the discriminator is the work, not the length.

    The same short reply and the same floor; the only difference is that this
    run called a tool first. That is exactly the shape the floor exists for —
    a run that went and did something and then said almost nothing about it.
    """
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096, finalize_prose_gate_min_chars=400
        )
    )
    _seed_unanswered_run(engine)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="Привет! Всё хорошо, спасибо.")],
        )
    )

    assert _plain_stop_answer_floor_applies(engine) is True
