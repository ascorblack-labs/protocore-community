"""What a run made of itself, kept where compaction cannot reach it.

Two facts about a finished run that nothing downstream could previously read.

**Which tools it called.** History does not keep them. Compaction replaces a
turn with prose about that turn and preserves no tool names, so a run long
enough to be compacted arrives at the user's screen as whichever handful of
calls happened to escape the sweep — ninety-eight calls rendered as six. The
ledger is written AT the dispatch for that reason, and reading it back out of
history would defeat the whole point of having it.

**Whether the user was answered.** ``status`` says the loop finished cleanly and
``stop_reason`` says why it stopped; neither says whether a single readable
sentence was produced. A run can end ``completed`` / ``end_turn`` having said
nothing, and one that ended in error may have delivered its answer first.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    SESSION_HISTORY_SEED_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState

from ._tool_fixtures import MockTool


def _user(text: str = "go") -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


# ----------------------------------------------------------------------
# The tool-call ledger
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_dispatched_call_lands_in_the_ledger_in_order(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["tools"].register(MockTool(tool_name="Grep"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c2", tool_name="Grep", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(_user()):
        pass

    assert engine.tool_call_ledger == [
        {"seq": 1, "name": "Read", "ok": True},
        {"seq": 2, "name": "Grep", "ok": True},
    ]
    assert engine.tool_call_ledger_truncated is False


@pytest.mark.asyncio
async def test_a_failed_call_is_recorded_as_a_call_that_failed(
    engine_factory, in_memory_runtime
) -> None:
    """``ok`` is the whole of the outcome — no message, no arguments, no result."""
    engine = engine_factory()
    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", response_is_error=True)
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(_user()):
        pass

    assert engine.tool_call_ledger == [{"seq": 1, "name": "Read", "ok": False}]


@pytest.mark.asyncio
async def test_the_ledger_carries_nothing_but_the_ordinal_name_and_outcome(
    engine_factory, in_memory_runtime
) -> None:
    """A record that grew arguments or results would be a second transcript."""
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={"path": "/etc/secret"}
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(_user()):
        pass

    assert set(engine.tool_call_ledger[0]) == {"seq", "name", "ok"}


@pytest.mark.asyncio
async def test_the_ledger_outlives_the_history_it_was_never_read_from(
    engine_factory, in_memory_runtime
) -> None:
    """The property the whole mechanism exists for.

    Compaction rewrites the turn a tool call lived in and keeps no tool names.
    Wiping history outright is the same loss taken to its limit: if the ledger
    were derived from the transcript it would empty with it.
    """
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(_user()):
        pass

    engine.history.clear()

    assert engine.tool_call_ledger == [{"seq": 1, "name": "Read", "ok": True}]


def test_the_ledger_stops_growing_at_the_cap_and_says_so(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(run_tool_call_ledger_max_entries=3))

    for index in range(10):
        engine.record_tool_call(f"Tool{index}", ok=True)

    assert [entry["seq"] for entry in engine.tool_call_ledger] == [1, 2, 3]
    assert engine.tool_call_ledger_truncated is True


def test_the_ordinal_keeps_counting_past_the_cap(engine_factory) -> None:
    """``seq`` is the call's position in the RUN, not its index in this list.

    A ledger whose ordinals restarted, or stopped, at the cap would make a
    truncated record silently disagree with the run it describes.
    """
    engine = engine_factory(rc=RuntimeConstants(run_tool_call_ledger_max_entries=2))

    for index in range(5):
        engine.record_tool_call(f"Tool{index}", ok=True)
    engine.record_tool_call("Last", ok=True)

    assert engine._tool_call_ledger_seq == 6


def test_a_zero_cap_keeps_no_ledger_at_all(engine_factory) -> None:
    engine = engine_factory(rc=RuntimeConstants(run_tool_call_ledger_max_entries=0))

    engine.record_tool_call("Read", ok=True)

    assert engine.tool_call_ledger == []
    assert engine.tool_call_ledger_truncated is True


def test_the_ledger_a_caller_reads_is_a_copy(engine_factory) -> None:
    engine = engine_factory()
    engine.record_tool_call("Read", ok=True)

    engine.tool_call_ledger[0]["name"] = "tampered"

    assert engine.tool_call_ledger[0]["name"] == "Read"


@pytest.mark.asyncio
async def test_the_ledger_survives_a_cross_pod_resume(engine_factory) -> None:
    """Losing it on resume loses the only copy of a compacted turn's calls."""
    engine = engine_factory(rc=RuntimeConstants(run_tool_call_ledger_max_entries=2))
    engine.record_tool_call("Read", ok=True)
    engine.record_tool_call("Grep", ok=False)
    engine.record_tool_call("Bash", ok=True)  # dropped, latches truncation
    snapshot = engine.snapshot()

    resumed = engine_factory(rc=RuntimeConstants(run_tool_call_ledger_max_entries=2))
    await resumed.resume_from_snapshot(snapshot)

    assert resumed.tool_call_ledger == [
        {"seq": 1, "name": "Read", "ok": True},
        {"seq": 2, "name": "Grep", "ok": False},
    ]
    assert resumed.tool_call_ledger_truncated is True
    resumed.record_tool_call("Next", ok=True)
    assert resumed._tool_call_ledger_seq == 4


def test_the_snapshot_carries_the_ledger_and_the_outcome(engine_factory) -> None:
    engine = engine_factory()
    engine.record_tool_call("Read", ok=True)

    snapshot = engine.snapshot()

    assert snapshot["tool_call_ledger"] == [{"seq": 1, "name": "Read", "ok": True}]
    assert snapshot["tool_call_ledger_truncated"] is False
    assert snapshot["has_final_answer"] is False


# ----------------------------------------------------------------------
# has_final_answer
# ----------------------------------------------------------------------


def test_a_run_that_has_only_called_tools_has_not_answered(engine_factory) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="c1", name="Read", arguments_json="{}")],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="c1", content="body")],
        )
    )

    assert engine.has_final_answer is False


def test_a_run_that_wrote_prose_has_answered(engine_factory) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="Here is what I found.")],
        )
    )

    assert engine.has_final_answer is True


def test_an_earlier_run_of_the_session_does_not_answer_for_this_one(
    engine_factory,
) -> None:
    """The exact way an unanswered run reads as answered.

    Cross-run seeding prepends the previous turns verbatim, and the newest
    assistant prose in the transcript is then a fluent, complete reply to a
    question this run was never asked.
    """
    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="A full answer, from the run before.")],
            metadata={SESSION_HISTORY_SEED_METADATA_KEY: True},
        )
    )

    assert engine.has_final_answer is False


def test_the_runtimes_own_recovery_scaffolding_does_not_count_as_an_answer(
    engine_factory,
) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="(empty)")],
            metadata={SYNTHETIC_RECOVERY_METADATA_KEY: "post_tool_empty_nudge"},
        )
    )

    assert engine.has_final_answer is False


def test_whitespace_is_not_an_answer(engine_factory) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text="   \n\t ")],
        )
    )

    assert engine.has_final_answer is False


@pytest.mark.asyncio
async def test_the_terminal_message_stop_reports_whether_the_user_was_answered(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="the answer")

    events = [evt async for evt in engine.run(_user())]

    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["has_final_answer"] is True
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_a_run_that_ends_without_prose_says_so_on_the_terminal_stop(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="")

    events = [evt async for evt in engine.run(_user())]

    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    terminal = [s for s in stops if s.payload.get("stop_reason") != "tool_use"]
    assert terminal
    assert terminal[-1].payload["has_final_answer"] is False


@pytest.mark.asyncio
async def test_a_mid_run_round_boundary_is_not_stamped_with_a_verdict(
    engine_factory, in_memory_runtime
) -> None:
    """``stop_reason="tool_use"`` closes a round, not a run.

    A consumer that latched "no answer yet" from a round boundary would report
    a run unanswered on the strength of its first tool call.
    """
    engine = engine_factory()
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="done")

    events = [evt async for evt in engine.run(_user())]

    rounds = [
        e
        for e in events
        if e.type is EventType.MESSAGE_STOP
        and e.payload.get("stop_reason") == "tool_use"
    ]
    assert rounds
    assert all("has_final_answer" not in e.payload for e in rounds)


@pytest.mark.asyncio
async def test_a_crashed_run_reports_no_answer_rather_than_no_information(
    engine_factory, in_memory_runtime
) -> None:
    """The incident shape: twenty-seven minutes of work, error, nothing readable."""
    from collections.abc import AsyncIterator

    from protocore.contracts.llm import LLMRequest, LLMStreamEvent

    class _Exploding:
        @property
        def calls(self):  # type: ignore[no-untyped-def]
            return []

        async def stream_with_tools(  # type: ignore[no-untyped-def]
            self, request: LLMRequest
        ) -> AsyncIterator[LLMStreamEvent]:
            if False:  # pragma: no cover
                yield LLMStreamEvent(name="never", payload={})
            raise RecursionError("maximum recursion depth exceeded")

        async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
            raise RuntimeError("unused")

        def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
            return max(1, len(text) // 4)

    engine = engine_factory()
    engine.llm = _Exploding()  # type: ignore[assignment]

    events = [evt async for evt in engine.run(_user())]

    stops = [e for e in events if e.type is EventType.MESSAGE_STOP]
    assert stops[-1].payload["has_final_answer"] is False


@pytest.mark.asyncio
async def test_the_settle_event_carries_the_whole_outcome_once_per_run(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=RuntimeConstants(run_settled_enabled=True))
    in_memory_runtime["tools"].register(MockTool(tool_name="Read"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="c1", tool_name="Read", tool_input={}
    )
    in_memory_runtime["llm"].queue_response(text="the answer")

    events = [evt async for evt in engine.run(_user())]

    settled = [e for e in events if e.type is EventType.RUN_SETTLED]
    assert len(settled) == 1
    assert settled[0].payload["has_final_answer"] is True
    assert settled[0].payload["tool_calls"] == [
        {"seq": 1, "name": "Read", "ok": True}
    ]
    assert settled[0].payload["tool_calls_truncated"] is False
