"""What a run can still say about a tool call after it stops mid-call.

A pod dies between the moment a tool is invoked and the moment its result
reaches history. What survives is a ``tool_use`` block with nothing after it,
and from that alone four very different situations look identical: the call was
parked at a gate and never ran, it ran and nobody recorded the outcome, it is
waiting for the user to answer a question, or it finished. These tests pin the
record that tells them apart and what each one makes a resumed run do.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    HookEvent,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType
from protocore.runtime.intent import (
    DISPATCHED,
    PAUSED_ASK_USER,
    PENDING_APPROVAL,
    SETTLED,
    IntentPauseMismatch,
    IntentRecord,
    assert_pause_matches,
    commit_intent,
    find_intent,
    mark_dispatched,
    mark_paused_ask_user,
    mark_pending_approval,
    orphaned_intents,
    repeat_is_safe_for,
    settle_intent,
    settle_unknown,
    should_dispatch,
    unknown_outcome_text,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _query as query
from protocore.runtime.query import resume_approved_tool
from protocore.tests_support.adapters import HookActionKind, HookResult

from ._tool_fixtures import MockTool


def _rc(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "approval_gate_web_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _tool_use(call_id: str, name: str, arguments_json: str) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(tool_call_id=call_id, name=name, arguments_json=arguments_json)
        ],
    )


def _results(engine) -> list[ToolResultBlock]:
    return [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


def test_a_fresh_record_stands_for_a_call_about_to_be_made() -> None:
    record = commit_intent(
        tool_name="Write",
        tool_call_id="c1",
        rc=_rc(),
        arguments={"path": "f.txt"},
    )
    assert record.state == DISPATCHED
    assert record.outcome == "pending"
    assert record.replay == "never"
    assert record.repeat_is_safe is False
    assert record.idempotency_key.startswith("c1:")
    assert should_dispatch(record) is True
    assert should_dispatch(None) is True


def test_the_key_separates_two_calls_and_survives_a_round_trip() -> None:
    one = commit_intent(tool_name="Write", tool_call_id="c1", rc=_rc(), arguments={"a": 1})
    two = commit_intent(tool_name="Write", tool_call_id="c1", rc=_rc(), arguments={"a": 2})
    assert one.idempotency_key != two.idempotency_key
    # Argument order is not part of the identity of a call.
    reordered = commit_intent(
        tool_name="Write", tool_call_id="c1", rc=_rc(), arguments={"a": 1}
    )
    assert reordered.idempotency_key == one.idempotency_key
    restored = IntentRecord.from_dict(one.to_dict())
    assert restored.idempotency_key == one.idempotency_key
    assert restored.state == one.state


def test_an_unserialisable_argument_still_yields_a_key() -> None:
    record = commit_intent(
        tool_name="Write", tool_call_id="c1", rc=_rc(), arguments={"h": object()}
    )
    assert record.arguments_fingerprint


def test_a_record_from_junk_falls_back_to_the_in_flight_reading() -> None:
    restored = IntentRecord.from_dict(
        {"tool_call_id": "c1", "state": "nonsense", "outcome": "nonsense", "result": 7}
    )
    assert restored.state == DISPATCHED
    assert restored.outcome == "pending"
    assert restored.result is None
    assert restored.pause_fingerprint is None
    assert restored.replay == "safe"


def test_each_transition_moves_the_record_and_settling_ends_it() -> None:
    record = commit_intent(tool_name="Write", tool_call_id="c1", rc=_rc())
    assert mark_pending_approval(record).state == PENDING_APPROVAL
    assert mark_dispatched(record).state == DISPATCHED
    assert mark_paused_ask_user(record, pause_payload={"q": 1}).state == PAUSED_ASK_USER
    assert record.pause_fingerprint
    settle_intent(record, result="done")
    assert record.state == SETTLED
    assert record.outcome == "known"
    assert should_dispatch(record) is False
    # A settled call does not move again, whatever arrives late.
    mark_pending_approval(record)
    mark_dispatched(record)
    mark_paused_ask_user(record)
    assert record.state == SETTLED


def test_a_read_and_a_write_are_told_apart_by_what_a_repeat_costs() -> None:
    rc = _rc()
    write = commit_intent(tool_name="Write", tool_call_id="c1", rc=rc)
    read = commit_intent(tool_name="Read", tool_call_id="c2", rc=rc)
    assert read.repeat_is_safe is True
    assert "safe" in unknown_outcome_text(read, rc)
    assert "never recorded" in unknown_outcome_text(write, rc)
    assert unknown_outcome_text(write, rc) != unknown_outcome_text(read, rc)


def test_only_an_in_flight_call_with_no_result_counts_as_orphaned() -> None:
    rc = _rc()
    flight = commit_intent(tool_name="Write", tool_call_id="c1", rc=rc)
    answered = commit_intent(tool_name="Write", tool_call_id="c2", rc=rc)
    parked = mark_pending_approval(commit_intent(tool_name="Write", tool_call_id="c3", rc=rc))
    asking = mark_paused_ask_user(commit_intent(tool_name="AskUser", tool_call_id="c4", rc=rc))
    done = settle_intent(commit_intent(tool_name="Write", tool_call_id="c5", rc=rc), result="x")
    records = [flight, answered, parked, asking, done]
    assert orphaned_intents(records, resolved_tool_call_ids={"c2"}) == [flight]
    assert find_intent(records, "c3") is parked
    assert find_intent(records, "nope") is None
    assert settle_unknown(flight).outcome_is_unknown is True


# ---------------------------------------------------------------------------
# (a) A call parked for approval is not executed by a resume
# ---------------------------------------------------------------------------


async def test_a_call_parked_for_approval_is_not_executed_on_resume(
    engine_factory, in_memory_runtime
) -> None:
    """The record says parked, so the resumed run waits instead of running it.

    This is the pairing that has to hold: the record is written before the
    dispatch, and the dispatch is what a gate stops. A resumed run that read
    the record as "in flight" would tell the model the outcome of a call the
    operator never approved is unknown — and a resumed run that executed it
    would run it without the approval.
    """
    engine = engine_factory(rc=_rc())
    tool = MockTool(tool_name="Write", response_content="written")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "f.txt"}
    )

    async for _ in engine.run(_user("write it")):
        pass

    assert tool.calls == []
    assert engine.state is LoopState.AWAITING
    record = find_intent(engine.open_intents, "w1")
    assert record is not None
    assert record.state == PENDING_APPROVAL

    # Cross-pod resume: the record travels in the snapshot and still says
    # parked, so nothing about this call is executed or declared unknown.
    snapshot = engine.snapshot()
    resumed = engine_factory(rc=_rc())
    await resumed.resume_from_snapshot(snapshot)
    revived = find_intent(resumed.open_intents, "w1")
    assert revived is not None
    assert revived.state == PENDING_APPROVAL

    resumed.state = LoopState.PENDING
    in_memory_runtime["llm"].queue_response(text="waiting")
    async for _ in query(resumed):
        pass

    assert tool.calls == []
    unknown = [
        block for block in _results(resumed) if block.tool_call_id == "w1"
    ]
    assert unknown == []


# ---------------------------------------------------------------------------
# (b) An approved call runs exactly once
# ---------------------------------------------------------------------------


async def test_an_approved_call_with_a_record_runs_exactly_once(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    tool = MockTool(tool_name="Write", response_content="written once")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "f.txt"}
    )
    async for _ in engine.run(_user("write it")):
        pass
    assert find_intent(engine.open_intents, "w1") is not None

    approved = ToolCall(id="w1", name="Write", arguments={"path": "f.txt"})
    async for _ in resume_approved_tool(engine, approved):
        pass
    assert tool.calls == [{"path": "f.txt"}]
    assert len([b for b in _results(engine) if b.tool_call_id == "w1"]) == 1
    # The record is gone: history holds the result now.
    assert find_intent(engine.open_intents, "w1") is None

    # The same approval arriving twice does not write the file twice.
    async for _ in resume_approved_tool(engine, approved):
        pass
    assert tool.calls == [{"path": "f.txt"}]
    assert len([b for b in _results(engine) if b.tool_call_id == "w1"]) == 1


async def test_a_settled_record_answers_instead_of_calling_the_tool_again(
    engine_factory, in_memory_runtime
) -> None:
    """A record left settled short-circuits a repeat of the same call id."""
    engine = engine_factory(rc=_rc())
    tool = MockTool(tool_name="Write", response_content="written")
    in_memory_runtime["tools"].register(tool)
    record = commit_intent(
        tool_name="Write", tool_call_id="w1", rc=engine.config.rc, arguments={"path": "f"}
    )
    settle_unknown(record)
    engine.open_intents.append(record)
    engine.history.append(_user("go"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f"}'))

    from protocore.runtime.query import _dispatch_tool

    events = [
        evt
        async for evt in _dispatch_tool(
            engine, ToolCall(id="w1", name="Write", arguments={"path": "f"})
        )
    ]
    assert tool.calls == []
    results = [evt for evt in events if evt.type is EventType.TOOL_RESULT]
    assert len(results) == 1
    assert results[0].payload["is_error"] is False
    assert "never recorded" in results[0].payload["content"]

    settle_intent(record, result="written")
    replayed = [
        evt
        async for evt in _dispatch_tool(
            engine, ToolCall(id="w1", name="Write", arguments={"path": "f"})
        )
    ]
    assert tool.calls == []
    assert replayed[0].payload["content"] == "written"


# ---------------------------------------------------------------------------
# (c) A pause that disagrees with the record is refused outright
# ---------------------------------------------------------------------------


def test_a_pause_that_names_another_call_is_refused() -> None:
    record = commit_intent(
        tool_name="Write", tool_call_id="w1", rc=_rc(), arguments={"path": "f.txt"}
    )
    assert_pause_matches(
        record, tool_call_id="w1", tool_name="Write", arguments={"path": "f.txt"}
    )
    with pytest.raises(IntentPauseMismatch, match="not the recorded intent"):
        assert_pause_matches(record, tool_call_id="w2", tool_name="Write")
    with pytest.raises(IntentPauseMismatch, match="names tool"):
        assert_pause_matches(record, tool_call_id="w1", tool_name="Bash")
    with pytest.raises(IntentPauseMismatch, match="arguments that differ"):
        assert_pause_matches(
            record,
            tool_call_id="w1",
            tool_name="Write",
            arguments={"path": "other.txt"},
        )
    # A record written without arguments makes no claim about them.
    bare = commit_intent(tool_name="Write", tool_call_id="w1", rc=_rc())
    bare.arguments_fingerprint = ""
    assert_pause_matches(bare, tool_call_id="w1", tool_name="Write", arguments={"x": 1})


async def test_an_approval_carrying_other_arguments_stops_the_run(
    engine_factory, in_memory_runtime
) -> None:
    """The record and the approval disagree, so neither is acted on.

    An approval that arrives describing different arguments from the ones the
    run recorded is not a stale copy to be tolerated: one of the two describes
    the call an operator saw and agreed to, and executing the other is the
    thing approval exists to prevent.
    """
    engine = engine_factory(rc=_rc())
    tool = MockTool(tool_name="Write", response_content="written")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "f.txt"}
    )
    async for _ in engine.run(_user("write it")):
        pass

    # The pending tool_use in history still says ``f.txt``; only the record is
    # consulted first, and it refuses before the dispatcher is reached.
    from protocore.runtime.query import _dispatch_tool

    with pytest.raises(IntentPauseMismatch, match="arguments that differ"):
        async for _ in _dispatch_tool(
            engine,
            ToolCall(id="w1", name="Write", arguments={"path": "elsewhere.txt"}),
            preapproved=True,
        ):
            pass
    assert tool.calls == []


# ---------------------------------------------------------------------------
# The orphan a crash leaves behind
# ---------------------------------------------------------------------------


async def test_an_orphaned_call_is_reported_as_unrecorded_not_as_a_failure(
    engine_factory, in_memory_runtime
) -> None:
    """The model is told the outcome is unknown, never that the call failed.

    A synthetic ``is_error`` result says the tool did not do its work, and the
    ordinary response to that is to call it again — which, for a tool that
    already wrote the file, writes it twice.
    """
    engine = engine_factory(rc=_rc())
    tool = MockTool(tool_name="Write", response_content="written")
    in_memory_runtime["tools"].register(tool)
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    engine.open_intents.append(
        commit_intent(
            tool_name="Write",
            tool_call_id="w1",
            rc=engine.config.rc,
            arguments={"path": "f.txt"},
        )
    )

    in_memory_runtime["llm"].queue_response(text="I will check the file first")
    engine.state = LoopState.PENDING
    events = [evt async for evt in query(engine)]

    reported = [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]
    assert len(reported) == 1
    assert reported[0].payload["is_error"] is False
    assert reported[0].payload["idempotency_key"].startswith("w1:")

    blocks = [block for block in _results(engine) if block.tool_call_id == "w1"]
    assert len(blocks) == 1
    assert blocks[0].is_error is False
    assert "never recorded" in blocks[0].content
    assert "Check the current state" in blocks[0].content
    # The tool is not called again on the way to saying so.
    assert tool.calls == []
    record = find_intent(engine.open_intents, "w1")
    assert record is not None and record.outcome_is_unknown


async def test_an_orphaned_read_is_reported_as_safe_to_repeat(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    in_memory_runtime["tools"].register(MockTool(tool_name="Read", response_content="hi"))
    engine.history.append(_user("read it"))
    engine.history.append(_tool_use("r1", "Read", '{"path": "f.txt"}'))
    engine.open_intents.append(
        commit_intent(tool_name="Read", tool_call_id="r1", rc=engine.config.rc)
    )
    in_memory_runtime["llm"].queue_response(text="reading again")
    engine.state = LoopState.PENDING
    async for _ in query(engine):
        pass
    blocks = [block for block in _results(engine) if block.tool_call_id == "r1"]
    assert len(blocks) == 1
    assert blocks[0].is_error is False
    assert "safe" in blocks[0].content


async def test_a_record_whose_result_already_landed_is_simply_dropped(
    engine_factory, in_memory_runtime
) -> None:
    """A call answered before the stop needs no announcement at all."""
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="w1", content="written", is_error=False)
            ],
        )
    )
    engine.open_intents.append(
        commit_intent(tool_name="Write", tool_call_id="w1", rc=engine.config.rc)
    )
    in_memory_runtime["llm"].queue_response(text="done")
    engine.state = LoopState.PENDING
    events = [evt async for evt in query(engine)]
    assert engine.open_intents == []
    assert [b.content for b in _results(engine)] == ["written"]
    assert not [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]


async def test_the_honest_result_lands_next_to_the_call_it_answers(
    engine_factory, in_memory_runtime
) -> None:
    """Position matters: a provider rejects a result that drifted out of place."""
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    engine.history.append(_user("are you there?"))
    engine.open_intents.append(
        commit_intent(tool_name="Write", tool_call_id="w1", rc=engine.config.rc)
    )
    in_memory_runtime["llm"].queue_response(text="checking")
    engine.state = LoopState.PENDING
    async for _ in query(engine):
        pass
    roles = [message.role for message in engine.history[:4]]
    assert roles[:3] == [MessageRole.user, MessageRole.assistant, MessageRole.tool]


async def test_a_record_whose_call_left_no_trace_reports_nothing(
    engine_factory, in_memory_runtime
) -> None:
    """No ``tool_use`` to attach to means no result is invented for one."""
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("go"))
    engine.open_intents.append(
        commit_intent(tool_name="Write", tool_call_id="ghost", rc=engine.config.rc)
    )
    in_memory_runtime["llm"].queue_response(text="ok")
    engine.state = LoopState.PENDING
    async for _ in query(engine):
        pass
    assert _results(engine) == []


# ---------------------------------------------------------------------------
# The record is durable before the tool is touched
# ---------------------------------------------------------------------------


async def test_the_record_is_persisted_before_the_tool_is_invoked(
    engine_factory, in_memory_runtime
) -> None:
    """What the snapshot says at the moment the tool starts work.

    The whole scheme rests on this ordering. If the record reached durable
    storage only after the call returned, the crash it exists to survive would
    take it with the call.
    """
    engine = engine_factory(rc=_rc())
    seen: list[list[str]] = []

    async def on_invoke(_args: dict[str, object]) -> None:
        snapshot = engine.snapshot()
        seen.append(
            [
                f"{item['tool_call_id']}:{item['state']}"
                for item in snapshot["open_intents"]
            ]
        )

    in_memory_runtime["tools"].register(
        MockTool(tool_name="Write", response_content="ok", on_invoke=on_invoke)
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "f.txt"}
    )
    in_memory_runtime["llm"].queue_response(text="done")
    async for _ in engine.run(_user("write it")):
        pass
    assert seen == [["w1:DISPATCHED"]]
    # …and it is gone once history carries the result.
    assert engine.open_intents == []


async def test_a_read_is_not_charged_a_snapshot_it_does_not_need(
    engine_factory, in_memory_runtime
) -> None:
    """Durability buys nothing for a call whose repeat costs nothing."""
    engine = engine_factory(rc=_rc())
    writes: list[int] = []
    original = engine._persist_snapshot

    async def counting() -> None:
        writes.append(1)
        await original()

    engine._persist_snapshot = counting  # type: ignore[method-assign]
    in_memory_runtime["tools"].register(MockTool(tool_name="Read", response_content="hi"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="r1", tool_name="Read", tool_input={"path": "f.txt"}
    )
    in_memory_runtime["llm"].queue_response(text="done")
    reads = len(writes)
    async for _ in engine.run(_user("read it")):
        pass
    read_writes = len(writes) - reads

    engine2 = engine_factory(rc=_rc())
    hits: list[int] = []
    original2 = engine2._persist_snapshot

    async def counting2() -> None:
        hits.append(1)
        await original2()

    engine2._persist_snapshot = counting2  # type: ignore[method-assign]
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", response_content="ok"))
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="w1", tool_name="Write", tool_input={"path": "f.txt"}
    )
    in_memory_runtime["llm"].queue_response(text="done")
    async for _ in engine2.run(_user("write it")):
        pass
    assert len(hits) == read_writes + 1


# ---------------------------------------------------------------------------
# A run killed mid-call can be driven again
# ---------------------------------------------------------------------------


async def test_a_history_ending_in_an_unanswered_call_can_be_driven(
    engine_factory, in_memory_runtime
) -> None:
    """The most common way a run dies is also the one it must recover from."""
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    engine.open_intents.append(
        commit_intent(tool_name="Write", tool_call_id="w1", rc=engine.config.rc)
    )
    in_memory_runtime["llm"].queue_response(text="I checked; the file is there")
    events = [evt async for evt in engine.run()]
    assert events
    assert engine.state is LoopState.COMPLETED


async def test_a_history_ending_in_a_tool_result_can_be_driven(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="w1", content="written", is_error=False)
            ],
        )
    )
    in_memory_runtime["llm"].queue_response(text="the file is written")
    async for _ in engine.run():
        pass
    assert engine.state is LoopState.COMPLETED


async def test_a_plain_assistant_answer_is_still_not_something_to_answer(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("hello"))
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi")])
    )
    with pytest.raises(ValueError, match="must end with"):
        async for _ in engine.run():
            pass
    assert len(in_memory_runtime["llm"].calls) == 0


async def test_an_answered_call_at_the_tail_is_not_something_to_answer(
    engine_factory, in_memory_runtime
) -> None:
    """A call whose result is already in history leaves nothing outstanding."""
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("write it"))
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="w1", content="written", is_error=False)
            ],
        )
    )
    engine.history.append(_tool_use("w1", "Write", '{"path": "f.txt"}'))
    with pytest.raises(ValueError, match="must end with"):
        async for _ in engine.run():
            pass


# ---------------------------------------------------------------------------
# A call waiting on the user
# ---------------------------------------------------------------------------


async def test_a_call_waiting_for_an_answer_is_neither_run_nor_declared_unknown(
    engine_factory, in_memory_runtime
) -> None:
    """The answer is the result, so nothing else may stand in for it."""
    from protocore.tools.ask_user import AskUserInput, AskUserPauseRequested

    engine = engine_factory(rc=_rc())

    async def ask(_args: dict[str, object]) -> None:
        raise AskUserPauseRequested(
            AskUserInput.model_validate(
                {
                    "questions": [
                        {
                            "question": "which file?",
                            "options": [{"label": "a"}, {"label": "b"}],
                        }
                    ]
                }
            )
        )

    in_memory_runtime["tools"].register(
        MockTool(tool_name="AskUser", description="ask", on_invoke=ask)
    )
    engine.history.append(_user("ask me"))
    engine.history.append(_tool_use("q1", "AskUser", "{}"))
    engine.open_intents.append(
        commit_intent(
            tool_name="AskUser", tool_call_id="q1", rc=engine.config.rc, arguments={}
        )
    )

    from protocore.runtime.query import _dispatch_tool

    events = [
        evt
        async for evt in _dispatch_tool(engine, ToolCall(id="q1", name="AskUser", arguments={}))
    ]
    assert any(evt.type is EventType.TOOL_CALL_PENDING for evt in events)
    record = find_intent(engine.open_intents, "q1")
    assert record is not None
    assert record.state == PAUSED_ASK_USER
    # No result was invented for a question that has not been answered.
    assert _results(engine) == []

    # …and the next turn does not turn the wait into an unknown outcome.
    in_memory_runtime["llm"].queue_response(text="still waiting")
    engine.state = LoopState.PENDING
    later = [evt async for evt in query(engine)]
    assert not [
        evt
        for evt in later
        if evt.type is EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]


# ---------------------------------------------------------------------------
# (f) A paused question stops being carried once its answer is in history
# ---------------------------------------------------------------------------


async def test_an_answered_question_stops_riding_along_in_every_snapshot(
    engine_factory, in_memory_runtime
) -> None:
    """The record for a paused call is dropped once history holds its answer.

    The answer to a question a tool asked arrives as a tool result appended by
    whatever collected it, not by a second pass through the dispatch, so the
    record is never settled where every other record is. Left alone it would
    stay open for the life of the run and be written into every snapshot after
    it, describing a wait that ended turns ago.
    """
    engine = engine_factory(rc=_rc())
    engine.history.append(_user("ask me"))
    engine.history.append(_tool_use("q1", "AskUser", "{}"))
    record = commit_intent(
        tool_name="AskUser", tool_call_id="q1", rc=engine.config.rc, arguments={}
    )
    mark_paused_ask_user(record, pause_payload={})
    engine.open_intents.append(record)

    # The answer lands the way a host delivers it: a tool result in history.
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="q1", content="blue")],
        )
    )

    in_memory_runtime["llm"].queue_response(text="thanks")
    engine.state = LoopState.PENDING
    events = [evt async for evt in query(engine)]

    assert find_intent(engine.open_intents, "q1") is None
    # The answer already in history is the result; none was invented for it.
    assert [block.content for block in _results(engine)] == ["blue"]
    assert not [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]


# ---------------------------------------------------------------------------
# (g) A parallel batch is recorded before the gate but not claimed to be running
# ---------------------------------------------------------------------------


async def test_a_batch_is_recorded_before_the_gate_without_claiming_it_ran(
    engine_factory, in_memory_runtime
) -> None:
    """The batch write is a reservation; the gate still stands between it and the tool.

    A batch has to be made durable in one snapshot, before the gather, because
    an await placed inside a member decides which sibling reaches its tool
    first. But between that snapshot and the tool there is still a permission
    gate, a hook and a precondition check, any of which refuses the call
    outright. If the snapshot already said "dispatched", a run that died in
    that window would come back and tell the model the outcome of a refused
    call is unknown and its effects may already be in place.
    """
    from protocore.runtime.intent import RESERVED
    from protocore.runtime.query import _record_batch_tool_intents

    engine = engine_factory(rc=_rc())
    engine.history.append(_user("do both"))
    engine.history.append(_tool_use("b1", "Write", "{}"))
    engine.history.append(_tool_use("b2", "Write", "{}"))

    await _record_batch_tool_intents(
        engine,
        [
            ToolCall(id="b1", name="Write", arguments={"path": "a"}),
            ToolCall(id="b2", name="Write", arguments={"path": "b"}),
        ],
    )
    assert [item.state for item in engine.open_intents] == [RESERVED, RESERVED]
    assert orphaned_intents(engine.open_intents, resolved_tool_call_ids=set()) == []

    # A run resumed from that snapshot says nothing at all about the two
    # calls: neither was tried, so neither has an outcome to report.
    resumed = engine_factory(rc=_rc())
    await resumed.resume_from_snapshot(engine.snapshot())
    resumed.state = LoopState.PENDING
    in_memory_runtime["llm"].queue_response(text="nothing to recover")
    events = [evt async for evt in query(resumed)]

    assert _results(resumed) == []
    assert not [
        evt
        for evt in events
        if evt.type is EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]


async def test_a_batch_member_marks_its_own_record_when_it_reaches_its_tool(
    engine_factory, in_memory_runtime
) -> None:
    """Each call moves its record past the gate itself, at the tool's door."""
    from protocore.runtime.intent import RESERVED
    from protocore.runtime.query import (
        _drain_dispatch_tool_deferred,
        _record_batch_tool_intents,
    )

    seen: list[str] = []
    tool = MockTool(
        tool_name="Write",
        response_content="written",
        on_invoke=lambda _args: seen.append(
            find_intent(engine.open_intents, "b1").state
        ),
    )
    engine = engine_factory(rc=_rc())
    in_memory_runtime["tools"].register(tool)
    engine.history.append(_user("write it"))
    engine.history.append(_tool_use("b1", "Write", "{}"))

    call = ToolCall(id="b1", name="Write", arguments={"path": "a"})
    await _record_batch_tool_intents(engine, [call])
    assert engine.open_intents[0].state == RESERVED

    await _drain_dispatch_tool_deferred(engine, call)

    assert seen == [DISPATCHED]
    # The result is in history's keeping now, so the record is gone.
    assert find_intent(engine.open_intents, "b1") is None


def test_which_repeats_are_safe_is_an_operator_setting() -> None:
    """The list lives with the other tool-name settings, not in the code.

    A tool whose repeat is harmless costs no durability write and is described
    to the model differently after a crash. Which tools those are depends on
    what the surrounding layer named them, so an installation that calls its
    reader something else can say so instead of being told what it has.
    """
    rc = _rc(intent_repeat_safe_tools="Read,Lookup")

    assert repeat_is_safe_for("Lookup", rc) is True
    assert repeat_is_safe_for("Grep", rc) is False

    record = commit_intent(
        tool_name="Lookup", tool_call_id="c9", rc=rc, arguments={"q": "x"}
    )
    assert record.repeat_is_safe is True
    assert unknown_outcome_text(record, rc) == (
        rc.tool_result_unknown_outcome_repeatable_placeholder
    )

    # And the answer travels with the record: a run picked up on another pod
    # describes the call the way the run that made it did, not the way a list
    # edited in the meantime would.
    revived = IntentRecord.from_dict(record.to_dict())
    assert revived.repeat_is_safe is True
    assert unknown_outcome_text(revived, _rc()) == (
        rc.tool_result_unknown_outcome_repeatable_placeholder
    )


def test_a_reading_tool_is_repeat_safe_from_its_role_alone() -> None:
    """The tools answer it themselves; the operator list only adds to them.

    A host that renamed its reading tools no longer has to keep a second list
    in step with its registry: a call that only looks at state costs a second
    look if it runs twice, whatever it is called.
    """
    from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
    from protocore.runtime.intent import repeat_is_safe_for

    rc = LoopConstants(intent_repeat_safe_tools="")
    roles = ToolRoleMap.declare(
        {
            "OpenDoc": [ToolRole.reads_path],
            "PutDoc": [ToolRole.writes_path],
            "HandOff": [ToolRole.delegates_work],
        }
    )

    assert repeat_is_safe_for("OpenDoc", rc, roles) is True
    # A write is not repeat-safe, and neither is a delegation: repeating it
    # starts a second subtree.
    assert repeat_is_safe_for("PutDoc", rc, roles) is False
    assert repeat_is_safe_for("HandOff", rc, roles) is False
    # An operator may still name a tool the roles cannot speak for.
    named = LoopConstants(intent_repeat_safe_tools="Lookup")
    assert repeat_is_safe_for("Lookup", named, roles) is True
