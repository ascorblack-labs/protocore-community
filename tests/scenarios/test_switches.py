"""Every switch, on and off, at the level of a whole run.

A boolean an installation can flip has two behaviours, and a suite that
only ever exercises the default has verified one of them. Each pair here
feeds the SAME script to the loop twice and differs only in the constant,
so the difference in the events and the requests is the switch and nothing
else.
"""
from __future__ import annotations

import builtins
from collections.abc import Sequence

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.types import StopReason
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool, default_rc

PHRASE = "the same sentence forever"
THOUGHT = "the same thought forever"


def _repeating_text_stream(times: int = 8) -> Sequence[LLMStreamEvent]:
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(
            name="content_block_delta",
            payload={"text": " ".join([PHRASE] * times), "kind": "text"},
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


def _repeating_thinking_stream(times: int = 8) -> Sequence[LLMStreamEvent]:
    """A model that loops in its reasoning and never says anything out loud."""
    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "thinking"}),
        LLMStreamEvent(
            name="content_block_delta",
            payload={"text": " ".join([THOUGHT] * times), "kind": "thinking"},
        ),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


def _guard_rc(**overrides: object) -> object:
    values: dict[str, object] = {
        "loop_guard_enabled": True,
        "loop_guard_repeat_min_chars": 16,
        "loop_guard_repeat_window_tokens": 4_096,
    }
    values.update(overrides)
    return default_rc(**values)


# ----------------------------------------------------------------------
# loop_guard_enabled — repeating prose
# ----------------------------------------------------------------------


async def test_a_looping_answer_is_cut_when_the_guard_is_on(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=_guard_rc())
    run.llm.queue_scripted_stream(_repeating_text_stream())

    produced = await run.run("go")

    assert EventType.LOOP_GUARD_FIRED in [evt.type for evt in produced]
    stored = "".join(run.history_texts())
    assert stored.count(PHRASE) < 8


async def test_the_same_looping_answer_runs_to_the_end_with_the_guard_off(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=default_rc(loop_guard_enabled=False))
    run.llm.queue_scripted_stream(_repeating_text_stream())

    produced = await run.run("go")

    assert EventType.LOOP_GUARD_FIRED not in [evt.type for evt in produced]
    assert "".join(run.history_texts()).count(PHRASE) == 8


# ----------------------------------------------------------------------
# loop_guard_enabled — repeating reasoning with nothing said out loud
# ----------------------------------------------------------------------


async def test_a_run_that_loops_only_in_its_reasoning_is_cut_too(
    scenario: ScenarioFactory,
) -> None:
    """The guard reads the thinking channel, not only what reached the reader."""
    run = scenario(rc=_guard_rc())
    run.llm.queue_scripted_stream(_repeating_thinking_stream())
    run.llm.queue_response(text="fine, here is the answer")

    produced = await run.run("go")

    assert EventType.LOOP_GUARD_FIRED in [evt.type for evt in produced]


async def test_the_reasoning_loop_is_left_alone_with_the_guard_off(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=default_rc(loop_guard_enabled=False))
    run.llm.queue_scripted_stream(_repeating_thinking_stream())
    run.llm.queue_response(text="fine, here is the answer")

    produced = await run.run("go")

    assert EventType.LOOP_GUARD_FIRED not in [evt.type for evt in produced]


# ----------------------------------------------------------------------
# loop_guard_identical_tool_limit — the same call over and over
# ----------------------------------------------------------------------


async def test_the_same_call_repeated_is_stopped_at_the_limit(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Read", content="same body")
    run = scenario(
        rc=_guard_rc(loop_guard_identical_tool_limit=2, max_turns_per_run=20),
        tools=[tool],
    )
    for index in range(6):
        run.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}",
            tool_name="Read",
            tool_input={"v": "same.txt"},
        )
    run.llm.queue_response(text="I will stop")

    produced = await run.run("read it")

    guards = [evt for evt in produced if evt.type is EventType.LOOP_GUARD_FIRED]
    assert any(evt.payload.get("kind") == "identical_tool" for evt in guards)
    assert len(tool.invocations) < 6


async def test_the_same_call_repeated_is_allowed_when_the_limit_is_off(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Read", content="same body")
    run = scenario(
        rc=default_rc(loop_guard_enabled=False, max_turns_per_run=20), tools=[tool]
    )
    for index in range(4):
        run.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}",
            tool_name="Read",
            tool_input={"v": "same.txt"},
        )
    run.llm.queue_response(text="done at last")

    produced = await run.run("read it")

    assert EventType.LOOP_GUARD_FIRED not in [evt.type for evt in produced]
    assert len(tool.invocations) == 4


# ----------------------------------------------------------------------
# run_settled_enabled
# ----------------------------------------------------------------------


async def test_the_settle_is_announced_when_the_installation_wants_it(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=default_rc(run_settled_enabled=True))
    run.llm.queue_response(text="answer")

    produced = await run.run("go")

    assert EventType.RUN_SETTLED in [evt.type for evt in produced]


async def test_the_settle_is_silent_when_it_is_switched_off(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(rc=default_rc(run_settled_enabled=False))
    run.llm.queue_response(text="answer")

    produced = await run.run("go")

    assert EventType.RUN_SETTLED not in [evt.type for evt in produced]


# ----------------------------------------------------------------------
# background_tasks_enabled
# ----------------------------------------------------------------------


class _WakingPool:
    """A pool holding one finished command the run has not been told about."""

    def __init__(self) -> None:
        self.drained = False

    def mark_session_attached(self, session_id: str) -> None:
        return None

    async def ensure_session_attached(self, session_id: str) -> bool:
        return True

    def list(self, session_id: str) -> builtins.list[object]:
        return []

    def get(self, task_id: str) -> object | None:
        return None

    async def refresh(self, task_id: str) -> None:
        return None

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        if self.drained:
            return []
        self.drained = True
        return ["bg-1"]


async def test_a_finished_background_command_wakes_the_run_when_enabled(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        rc=default_rc(background_tasks_enabled=True),
        background_pool=_WakingPool(),
    )
    run.llm.queue_response(text="thanks for telling me")

    produced = await run.run("go")

    assert EventType.BACKGROUND_WAKE in [evt.type for evt in produced]


async def test_the_work_pool_is_reachable_without_being_switched_on(
    scenario: ScenarioFactory,
) -> None:
    """The session work pool is the default, not an opt-in.

    It was an opt-in while it only offered Bash somewhere to put a long
    command. Delegation then started drawing on the same switch, so leaving it
    off made a background delegation a refusal — and a caller that reads a
    refusal as advice re-issues the batch inline and blocks for the child's
    whole run in the tool it asked not to block in. The default is stated as a
    behaviour rather than as a field value: an unconfigured run reaches the
    pool.
    """
    run = scenario(background_pool=_WakingPool())
    run.llm.queue_response(text="thanks for telling me")

    produced = await run.run("go")

    assert EventType.BACKGROUND_WAKE in [evt.type for evt in produced]


async def test_the_same_pool_is_ignored_when_background_work_is_switched_off(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        rc=default_rc(background_tasks_enabled=False),
        background_pool=_WakingPool(),
    )
    run.llm.queue_response(text="nothing to report")

    produced = await run.run("go")

    assert EventType.BACKGROUND_WAKE not in [evt.type for evt in produced]


# ----------------------------------------------------------------------
# soft_stop_enabled
# ----------------------------------------------------------------------


async def test_the_wind_down_is_offered_when_it_is_enabled(
    scenario: ScenarioFactory,
) -> None:
    """A run at its turn ceiling gets one narrowed turn to answer with."""
    tool = ScriptedTool(tool_name="Read")
    run = scenario(
        rc=default_rc(soft_stop_enabled=True, max_turns_per_run=2),
        tools=[tool],
    )
    for index in range(6):
        run.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}", tool_name="Read", tool_input={"v": str(index)}
        )
    run.llm.queue_response(text="the wind-down answer")

    produced = await run.run("go")

    assert "soft_stop_notified" in [
        str(evt.payload.get("reason", ""))
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
    ]


async def test_the_run_ends_without_a_wind_down_when_it_is_disabled(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Read")
    run = scenario(
        rc=default_rc(soft_stop_enabled=False, max_turns_per_run=2),
        tools=[tool],
    )
    for index in range(6):
        run.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}", tool_name="Read", tool_input={"v": str(index)}
        )
    run.llm.queue_response(text="the wind-down answer")

    produced = await run.run("go")

    assert "soft_stop_notified" not in [
        str(evt.payload.get("reason", ""))
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
    ]


# ----------------------------------------------------------------------
# finalize_prose_gate_enabled
# ----------------------------------------------------------------------


async def test_a_run_that_under_reports_its_work_is_sent_back_for_an_answer(
    scenario: ScenarioFactory,
) -> None:
    """Work was done and four characters were said about it: not an answer.

    The floor sends the turn round again with a correction in history, and the
    repair round is what the caller finally reads. What is asserted is what a
    caller sees: a further request reached the model, and the run said which
    test fired.
    """
    tool = ScriptedTool(tool_name="Search", content="five results")
    run = scenario(
        rc=default_rc(
            finalize_prose_gate_enabled=True, finalize_prose_gate_min_chars=200
        ),
        tools=[tool],
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Search")
    run.llm.queue_response(text="done")
    run.llm.queue_response(text="x" * 400)

    await run.run("search and report")

    assert "finalize_prose_gate_plain_stop_repair" in run.state_reasons()
    assert len(run.requests) == 3
    assert "x" * 400 in "".join(run.history_texts())


async def test_the_same_thin_answer_is_accepted_with_the_floor_off(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Search", content="five results")
    run = scenario(
        rc=default_rc(
            finalize_prose_gate_enabled=False, finalize_prose_gate_min_chars=200
        ),
        tools=[tool],
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Search")
    run.llm.queue_response(text="done")
    run.llm.queue_response(text="x" * 400)

    await run.run("search and report")

    assert "finalize_prose_gate_plain_stop_repair" not in run.state_reasons()
    assert len(run.requests) == 2


# ----------------------------------------------------------------------
# empty_completion_guard_enabled
# ----------------------------------------------------------------------


async def test_a_finish_that_delivered_nothing_is_driven_again(
    scenario: ScenarioFactory,
) -> None:
    """An empty finish is not an answer, so the run gets another attempt.

    Sealing it would report a finished run whose history holds nothing — the
    turn is lost silently, and a reload shows a success with no answer in it.
    """
    run = scenario(
        rc=default_rc(
            empty_completion_guard_enabled=True, empty_completion_guard_max_redrives=1
        ),
    )
    run.llm.queue_response(text="")
    run.llm.queue_response(text="the answer the re-drive produced")

    await run.run("go")

    assert "empty_completion_redrive" in run.state_reasons()
    assert len(run.requests) == 2
    assert "the answer the re-drive produced" in "".join(run.history_texts())


async def test_the_same_empty_finish_is_sealed_with_the_guard_off(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        rc=default_rc(
            empty_completion_guard_enabled=False, empty_completion_guard_max_redrives=1
        ),
    )
    run.llm.queue_response(text="")
    run.llm.queue_response(text="the answer nobody asked for")

    await run.run("go")

    assert "empty_completion_redrive" not in run.state_reasons()
    assert len(run.requests) == 1
