"""End-to-end query() + ToolDispatcher integration tests.

Verifies the wiring between :func:`protocore.runtime.query.query`,
:class:`ToolDispatcher`, :class:`ToolPermissionGate`, and the engine's
history/state mutations.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    TERMINAL_TOOL_METADATA_KEY,
    TERMINAL_TOOL_STATUS_COMPLETED,
    TERMINAL_TOOL_STATUS_METADATA_KEY,
    HookEvent,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.tool_dispatch import (
    TOOL_CALL_SOFT_CAP_METADATA_KEY,
    TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY,
    TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY,
    TOOL_CALL_SOFT_CAPS_HELPER_KEY,
)

from ._tool_fixtures import MockTool


class _ContextCapturingTool(MockTool):
    """Records the ToolContext each dispatch receives."""

    contexts: list[Any]

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        del arguments
        if not hasattr(self, "contexts"):
            self.contexts = []
        self.contexts.append(context)
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=False,
        )


class _TerminalTool(MockTool):
    """Mock tool that marks a successful result as terminal for the loop."""

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=False,
            metadata={
                TERMINAL_TOOL_METADATA_KEY: True,
                TERMINAL_TOOL_STATUS_METADATA_KEY: TERMINAL_TOOL_STATUS_COMPLETED,
            },
        )


class _SoftErrorMetadataTool(MockTool):
    """Mock tool that returns is_error=True with structured metadata."""

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=True,
            metadata={
                "error_kind": "synthetic_soft_error",
                TERMINAL_TOOL_METADATA_KEY: True,
            },
        )


# ----------------------------------------------------------------------
# Happy path — tool emits result, follow-up text turn completes
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_tool_dispatch_cycle(engine_factory, in_memory_runtime) -> None:
    """Tool call → execute → follow-up assistant turn → end_turn."""
    engine = engine_factory()
    tool = MockTool(
        tool_name="MyTool",
        description="Mock tool",
        response_content="tool-output-here",
    )
    in_memory_runtime["tools"].register(tool)
    # Per L-1: hook_fired(pre) only emits when stage == hook. Queue a
    # MODIFY-action so the gate exits at PermissionStage.hook and the
    # ``pre_tool_use`` event surfaces.
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={}),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_abc",
        tool_name="MyTool",
        tool_input={"v": "hello"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    types = [e.type for e in events]

    # tool_use_start (LLM stream) → tool_use_input_delta → tool_use_stop →
    # hook_fired (pre) → hook_fired (post) → tool_result
    assert EventType.TOOL_USE_START in types
    assert EventType.TOOL_USE_INPUT_DELTA in types
    assert EventType.TOOL_USE_STOP in types
    assert EventType.TOOL_RESULT in types

    # Pre & post hook events fire.
    hook_events = [e for e in events if e.type is EventType.HOOK_FIRED]
    hook_names = [e.payload["hook_event"] for e in hook_events]
    # At least 1 pre-tool-use AND 1 post-tool-use.
    assert HookEvent.pre_tool_use.value in hook_names
    assert HookEvent.post_tool_use.value in hook_names

    # Tool actually invoked with structured args.
    assert tool.calls == [{"v": "hello"}]

    # History carries the tool result.
    tool_results = [b for msg in engine.history for b in msg.content_blocks if isinstance(b, ToolResultBlock)]
    assert len(tool_results) == 1
    assert tool_results[0].content == "tool-output-here"
    assert tool_results[0].is_error is False

    # End state COMPLETED.
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_tool_call_soft_cap_warning_is_provider_visible(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Subagent soft-cap diagnostics annotate tool_result without blocking."""
    engine = engine_factory()
    engine._helpers = {  # type: ignore[attr-defined]
        TOOL_CALL_SOFT_CAPS_HELPER_KEY: {"MyTool": 1},
        TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY: {"counts": {}, "warnings": []},
    }
    tool = MockTool(
        tool_name="MyTool",
        description="Mock tool",
        response_content="tool-output-here",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_soft_cap",
        tool_name="MyTool",
        tool_input={"v": "hello"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events = [evt async for evt in engine.run(user_msg)]

    assert tool.calls == [{"v": "hello"}]
    result_evt = next(evt for evt in events if evt.type is EventType.TOOL_RESULT)
    result_text = result_evt.payload["content_blocks"][0]["text"]
    assert "tool-output-here" in result_text
    assert "[Tool call soft-cap warning]" in result_text
    warning = result_evt.payload["metadata"][TOOL_CALL_SOFT_CAP_METADATA_KEY]
    assert warning["tool_name"] == "MyTool"
    assert warning["limit"] == 1
    assert warning["count"] == 1
    assert warning["status"] == "reached"
    assert "soft cap 1 reached" in warning["message"]

    tool_result = next(
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    )
    assert "[Tool call soft-cap warning]" in tool_result.content
    assert tool_result.metadata[TOOL_CALL_SOFT_CAP_METADATA_KEY]["status"] == "reached"
    assert len(tool_result.metadata[TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY]) == 1
    assert (
        engine._helpers[TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY]["counts"]["MyTool"]  # type: ignore[attr-defined]
        == 1
    )


@pytest.mark.asyncio
async def test_tool_call_soft_cap_zero_is_unlimited(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A limit of 0 disables warnings for that tool."""
    engine = engine_factory()
    engine._helpers = {  # type: ignore[attr-defined]
        TOOL_CALL_SOFT_CAPS_HELPER_KEY: {"MyTool": 0},
        TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY: {"counts": {}, "warnings": []},
    }
    tool = MockTool(
        tool_name="MyTool",
        description="Mock tool",
        response_content="tool-output-here",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_soft_cap_zero",
        tool_name="MyTool",
        tool_input={"v": "hello"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events = [evt async for evt in engine.run(user_msg)]

    assert tool.calls == [{"v": "hello"}]
    result_evt = next(evt for evt in events if evt.type is EventType.TOOL_RESULT)
    result_text = result_evt.payload["content_blocks"][0]["text"]
    assert "tool-output-here" in result_text
    assert "[Tool call soft-cap warning]" not in result_text
    assert TOOL_CALL_SOFT_CAP_METADATA_KEY not in result_evt.payload.get("metadata", {})
    assert engine._helpers[TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY]["counts"] == {}  # type: ignore[attr-defined]
    assert engine._helpers[TOOL_CALL_SOFT_CAP_STATE_HELPER_KEY]["warnings"] == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_cumulative_tool_budget_stops_the_run_rather_than_advising_it(
    engine_factory,
    in_memory_runtime,
) -> None:
    """The budget used to append a paragraph to the tool result and change nothing.

    It said, in the same breath, both "begin wrapping up now" and "this is
    advisory — tools still run". In the run this was written for it fired once
    at call eighty and the agent made another eighteen calls. Reaching the
    budget now starts the wind-down, and the wind-down takes the tools away.
    """
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, leader_tool_call_soft_cap=1),
    )
    tool = MockTool(
        tool_name="MyTool",
        description="Mock tool",
        response_content="tool-output-here",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_total_cap",
        tool_name="MyTool",
        tool_input={"v": "hello"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events = [evt async for evt in engine.run(user_msg)]

    # The call that reached the budget still ran and its result is intact —
    # nothing is appended to what the model reads.
    assert tool.calls == [{"v": "hello"}]
    result_evt = next(evt for evt in events if evt.type is EventType.TOOL_RESULT)
    result_text = result_evt.payload["content_blocks"][0]["text"]
    assert result_text == "tool-output-here"
    assert "soft-cap warning" not in result_text

    reasons = [
        e.payload.get("reason") for e in events if e.type is EventType.STATE_CHANGED
    ]
    assert "soft_stop_notified" in reasons
    assert "soft_stop_tools_withdrawn" in reasons


@pytest.mark.asyncio
async def test_a_zero_cumulative_tool_budget_never_stops_the_run(
    engine_factory,
    in_memory_runtime,
) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, leader_tool_call_soft_cap=0),
    )
    tool = MockTool(
        tool_name="MyTool", description="Mock tool", response_content="ok"
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_total_cap_off",
        tool_name="MyTool",
        tool_input={"v": "x"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events = [evt async for evt in engine.run(user_msg)]

    reasons = [
        e.payload.get("reason") for e in events if e.type is EventType.STATE_CHANGED
    ]
    assert "soft_stop_notified" not in reasons


@pytest.mark.asyncio
async def test_terminal_tool_result_completes_without_followup_llm(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A successful terminal tool closes the loop immediately."""
    engine = engine_factory()
    tool = _TerminalTool(
        tool_name="pcm_answer",
        description="Submit the terminal answer",
        response_content='{"submitted": true}',
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_answer",
        tool_name="pcm_answer",
        tool_input={"message": "done", "outcome": "OUTCOME_OK"},
    )
    in_memory_runtime["llm"].queue_response(text="should not be called")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert len(in_memory_runtime["llm"].calls) == 1
    assert tool.calls == [{"message": "done", "outcome": "OUTCOME_OK"}]
    assert engine.state is LoopState.COMPLETED

    stop_events = [evt for evt in events if evt.type is EventType.MESSAGE_STOP]
    assert stop_events[-1].payload["stop_reason"] == "end_turn"

    result_evts = [evt for evt in events if evt.type is EventType.TOOL_RESULT]
    assert result_evts[-1].payload["metadata"][TERMINAL_TOOL_METADATA_KEY] is True

    tool_results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_results[-1].metadata[TERMINAL_TOOL_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_soft_error_tool_metadata_is_preserved_without_terminal_stop(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Soft-error metadata survives, but terminal metadata is ignored on errors."""
    engine = engine_factory()
    tool = _SoftErrorMetadataTool(
        tool_name="SoftErrorTool",
        description="Return a structured soft error",
        response_content="soft failure",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_soft_error",
        tool_name="SoftErrorTool",
        tool_input={"v": "bad"},
    )
    in_memory_runtime["llm"].queue_response(text="handled")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    assert len(in_memory_runtime["llm"].calls) == 2
    assert engine.state is LoopState.COMPLETED

    result_evt = next(evt for evt in events if evt.type is EventType.TOOL_RESULT)
    metadata = result_evt.payload["metadata"]
    assert metadata["error_kind"] == "synthetic_soft_error"
    assert metadata[TERMINAL_TOOL_METADATA_KEY] is True
    assert metadata["tool_dispatch.replay_error_kind"] == "execution"

    tool_result = next(
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
        and block.tool_call_id == "toolu_soft_error"
    )
    assert tool_result.is_error is True
    assert tool_result.metadata["error_kind"] == "synthetic_soft_error"
    assert tool_result.metadata[TERMINAL_TOOL_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_tool_context_metadata_includes_tool_call_id(
    engine_factory,
    in_memory_runtime,
) -> None:
    """ToolContext metadata carries the provider tool-call id into tools."""
    engine = engine_factory()
    tool = _ContextCapturingTool(
        tool_name="Agent",
        description="Dispatch a subagent",
        response_content="child done",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_parent_123",
        tool_name="Agent",
        tool_input={"prompt": "work"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _ in engine.run(user_msg):
        pass

    assert len(tool.contexts) == 1
    assert tool.contexts[0].metadata["tool_call_id"] == "toolu_parent_123"


@pytest.mark.asyncio
async def test_forged_run_metadata_cannot_shadow_tool_call_id(
    engine_factory,
    in_memory_runtime,
) -> None:
    """: a forged ``tool_call_id`` (or any ``protocore.*`` key) in the
    operator-supplied per-run metadata envelope must NOT shadow the
    runtime-internal authoritative tool-call id on a normal tool call.

    The public ``POST /v1/runs.metadata`` envelope flows into the helper
    bag's ``run_metadata`` and is merged onto ``ToolContext.metadata`` by
    the dispatcher path; a forged ``tool_call_id`` would otherwise poison
    every tool-result correlation / subagent-parent edge / answer RPC
    binding for the whole run."""
    engine = engine_factory()
    # Simulate the executor wiring the helper bag with a *forged* run_metadata
    # envelope. A legit (non-internal) key still flows through.
    engine._helpers = {  # type: ignore[attr-defined]
        "run_metadata": {
            "tool_call_id": "FORGED_BY_OPERATOR",
            "protocore.helpers": "forged-bag",
            "protocore.synthetic_recovery": "forged-recovery",
            "pac_trial_id": "trial-42",
        },
    }
    tool = _ContextCapturingTool(
        tool_name="MyTool",
        description="A tool",
        response_content="ok",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_authoritative_999",
        tool_name="MyTool",
        tool_input={"v": "x"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    async for _ in engine.run(user_msg):
        pass

    assert len(tool.contexts) == 1
    md = tool.contexts[0].metadata
    # The authoritative runtime id wins — the forged value is rejected.
    assert md["tool_call_id"] == "toolu_authoritative_999"
    # The helper bag namespace is never replaced by a forged value.
    assert md["protocore.helpers"] != "forged-bag"
    # No forged ``protocore.*`` runtime-internal key leaks through.
    assert md.get("protocore.synthetic_recovery") != "forged-recovery"
    # A genuinely operator-scoped (non-internal) key DOES still pass through.
    assert md["pac_trial_id"] == "trial-42"


# ----------------------------------------------------------------------
# Permission gate denies — error result appended, loop continues
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_denied_by_safety_policy(engine_factory, in_memory_runtime) -> None:
    """rm -rf / is denied; tool not invoked; error surfaces."""
    engine = engine_factory()
    bash = MockTool(tool_name="Bash", description="run shell commands")
    in_memory_runtime["tools"].register(bash)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_bash",
        tool_name="Bash",
        tool_input={"command": "rm -rf /"},
    )
    in_memory_runtime["llm"].queue_response(text="aborted")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="run")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # Tool NEVER invoked.
    assert bash.calls == []

    # TOOL_RESULT carries the permission error.
    result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert result_evts
    error_payload = result_evts[0].payload.get("error")
    assert error_payload is not None
    assert error_payload["kind"] == "permission"

    # Loop completes (the follow-up assistant message lands end_turn).
    assert engine.state is LoopState.COMPLETED


# ----------------------------------------------------------------------
# Tool not registered — error surfaces, loop continues
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_tool_surfaces_unknown_tool_error(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory()
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_x",
        tool_name="Ghost",  # never registered
        tool_input={},
    )
    in_memory_runtime["llm"].queue_response(text="hmm")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    result_evts = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert result_evts
    err = result_evts[0].payload.get("error")
    assert err is not None
    assert err["kind"] == "unknown_tool"

    # Tool result appended to history as is_error.
    tool_result_blocks = [b for msg in engine.history for b in msg.content_blocks if isinstance(b, ToolResultBlock)]
    assert tool_result_blocks
    assert tool_result_blocks[0].is_error is True


# ----------------------------------------------------------------------
# Sandbox cold-start event — adapter owns emission, NOT core
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_does_not_emit_sandbox_starting_for_bash(engine_factory, in_memory_runtime) -> None:
    """The sandbox adapter is the sole emitter of ``sandbox_starting``.
    The core dispatcher cannot tell hot vs cold pod, so it MUST NOT emit.
    The adapter's integration test
    ``protocore-the host/tests/integration/sandbox/test_dispatcher.py::
    test_first_dispatch_spawns_pod`` validates the adapter side.
    """
    engine = engine_factory()
    bash = MockTool(tool_name="Bash", description="run shell", response_content="ok")
    in_memory_runtime["tools"].register(bash)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_bash",
        tool_name="Bash",
        tool_input={"command": "echo hi"},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    sandbox = [e for e in events if e.type is EventType.SANDBOX_STARTING]
    assert sandbox == []


# ----------------------------------------------------------------------
# Approval flow puts engine into AWAITING (kill-switch ON = CLI mode)
# ----------------------------------------------------------------------
#
# -G (2026-05-20) — these tests assert the CLI-mode
# behaviour where ``RuntimeConstants.approval_gate_web_enabled=True``
# preserves the pause / TOOL_CALL_PENDING / AWAITING flow. The default
# (False) downgrades approvals; see ``test_approval_downgrade_*`` below.


@pytest.mark.asyncio
async def test_approval_required_transitions_to_awaiting(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, approval_gate_web_enabled=True),
    )
    tool = MockTool(tool_name="MyTool")
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
        tool_call_id="toolu_q",
        tool_name="MyTool",
        tool_input={},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # TOOL_CALL_PENDING event emitted.
    pending = [e for e in events if e.type is EventType.TOOL_CALL_PENDING]
    assert len(pending) == 1
    assert pending[0].payload["approval_token"] == "tok-1"

    # Tool was NOT invoked.
    assert tool.calls == []

    # Engine ends in AWAITING.
    assert engine.state is LoopState.AWAITING


@pytest.mark.asyncio
async def test_resume_approved_tool_executes_pending_call_once(
    engine_factory,
    in_memory_runtime,
) -> None:
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, approval_gate_web_enabled=True),
    )
    tool = MockTool(tool_name="MyTool", response_content="approved-output")
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
        tool_call_id="toolu_q",
        tool_name="MyTool",
        tool_input={"v": "approved"},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    initial_events = [evt async for evt in engine.run(user_msg)]

    assert [evt.type for evt in initial_events].count(EventType.TOOL_CALL_PENDING) == 1
    assert tool.calls == []
    assert engine.state is LoopState.AWAITING

    resumed_events = [
        evt
        async for evt in resume_approved_tool(
            engine,
            ToolCall(
                id="toolu_q",
                name="MyTool",
                arguments={"v": "approved"},
            ),
        )
    ]

    assert tool.calls == [{"v": "approved"}]
    assert [evt.type for evt in resumed_events].count(EventType.TOOL_RESULT) == 1
    assert EventType.TOOL_CALL_PENDING not in [evt.type for evt in resumed_events]
    assert engine.state is LoopState.COMPLETED

    tool_results = [
        block for msg in engine.history_snapshot() for block in msg.content_blocks if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "toolu_q"
    assert tool_results[0].content == "approved-output"
    assert tool_results[0].is_error is False


@pytest.mark.asyncio
async def test_resume_approved_tool_preserves_post_tool_hook(
    engine_factory,
    in_memory_runtime,
) -> None:
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory()
    tool = MockTool(tool_name="MyTool", response_content="raw-output")
    in_memory_runtime["tools"].register(tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_post",
                    name="MyTool",
                    arguments_json='{"v": "approved"}',
                )
            ],
        )
    )
    engine.mark_pending_approval("toolu_post")
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.AWAITING)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
        ),
    )
    in_memory_runtime["hooks"].queue_action(
        HookEvent.post_tool_use,
        HookResult(
            action=HookActionKind.MODIFY,
            modifications={"tool_output": "post-hook-output"},
        ),
    )

    events = [
        evt
        async for evt in resume_approved_tool(
            engine,
            ToolCall(
                id="toolu_post",
                name="MyTool",
                arguments={"v": "approved"},
            ),
        )
    ]

    assert tool.calls == [{"v": "approved"}]
    assert EventType.TOOL_CALL_PENDING not in [evt.type for evt in events]
    result = next(
        block for msg in engine.history_snapshot() for block in msg.content_blocks if isinstance(block, ToolResultBlock)
    )
    assert result.content == "post-hook-output"
    assert [evt.type for evt in events].count(EventType.TOOL_RESULT) == 1


@pytest.mark.asyncio
async def test_resume_approved_tool_is_idempotent_when_result_exists(
    engine_factory,
    in_memory_runtime,
) -> None:
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory()
    tool = MockTool(tool_name="MyTool", response_content="should-not-run")
    in_memory_runtime["tools"].register(tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_done",
                    name="MyTool",
                    arguments_json="{}",
                )
            ],
        )
    )
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id="toolu_done",
                    content="existing",
                    is_error=False,
                )
            ],
        )
    )

    events = [
        evt
        async for evt in resume_approved_tool(
            engine,
            ToolCall(id="toolu_done", name="MyTool", arguments={}),
        )
    ]

    assert events == []
    assert tool.calls == []
    tool_results = [
        block for msg in engine.history_snapshot() for block in msg.content_blocks if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1


@pytest.mark.asyncio
async def test_resume_approved_tool_rejects_unapproved_sibling_tool_call(
    engine_factory,
    in_memory_runtime,
) -> None:
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory()
    sibling = MockTool(tool_name="SiblingTool", response_content="sibling-output")
    pending = MockTool(tool_name="PendingTool", response_content="pending-output")
    in_memory_runtime["tools"].register(sibling)
    in_memory_runtime["tools"].register(pending)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_sibling",
                    name="SiblingTool",
                    arguments_json="{}",
                ),
                ToolUseBlock(
                    tool_call_id="toolu_pending",
                    name="PendingTool",
                    arguments_json="{}",
                ),
            ],
        )
    )
    engine.mark_pending_approval("toolu_pending")
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.AWAITING)

    with pytest.raises(ValueError, match="not the pending approval"):
        _ = [
            evt
            async for evt in resume_approved_tool(
                engine,
                ToolCall(id="toolu_sibling", name="SiblingTool", arguments={}),
            )
        ]

    assert sibling.calls == []
    assert pending.calls == []
    assert not [
        block for msg in engine.history_snapshot() for block in msg.content_blocks if isinstance(block, ToolResultBlock)
    ]


@pytest.mark.asyncio
async def test_resume_approved_tool_still_enforces_blocked_policy(
    engine_factory,
    in_memory_runtime,
) -> None:
    from protocore.runtime.query import resume_approved_tool

    engine = engine_factory()
    engine.config = replace(
        engine.config,
        tool_visibility_policy=ToolVisibilityPolicy(blocked={"BlockedTool"}),
    )
    tool = MockTool(tool_name="BlockedTool", response_content="should-not-run")
    in_memory_runtime["tools"].register(tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="toolu_blocked",
                    name="BlockedTool",
                    arguments_json="{}",
                )
            ],
        )
    )
    engine.mark_pending_approval("toolu_blocked")
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.AWAITING)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-repeat"},
        ),
    )

    events = [
        evt
        async for evt in resume_approved_tool(
            engine,
            ToolCall(id="toolu_blocked", name="BlockedTool", arguments={}),
        )
    ]

    assert tool.calls == []
    assert EventType.TOOL_CALL_PENDING not in [evt.type for evt in events]
    assert [evt.type for evt in events].count(EventType.TOOL_RESULT) == 1
    result = next(
        block for msg in engine.history_snapshot() for block in msg.content_blocks if isinstance(block, ToolResultBlock)
    )
    assert result.is_error is True
    assert "blocked list" in result.content


# ----------------------------------------------------------------------
# -G — approval-gate kill-switch (web mode default)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_downgrade_web_mode_default_runs_tool_and_no_pending_event(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Default RC (``approval_gate_web_enabled=False``) downgrades approvals.

 The PreToolUse hook returns ``requires_approval=True``; in web mode
 (the production default) the loop transparently re-dispatches with
 ``preapproved=True`` and the tool actually executes. No
 ``TOOL_CALL_PENDING`` envelope reaches the consumer and the engine
 completes the turn normally — sandbox isolation +
 ``dangerous_commands.py`` deny patterns are the safety boundary.

 Note: the hook fixture is stateless
 (returns ``requires_approval=True`` on EVERY invocation) so the
 re-dispatch path actually exercises
 :meth:`ToolPermissionGate.check`'s
 ``skip_pre_tool_approval and decision.requires_approval`` short-circuit
 (``tool_permission.py:463-468``). Without that short-circuit the second
 dispatch would also yield ``approval_required=True``, the engine would
 transition to ``AWAITING`` and ``tool.calls`` would stay empty — so
 the strict ``tool.calls == [{"v": "go"}]`` +
 ``engine.state is COMPLETED`` assertions below proxy as a positive
 test for the short-circuit. The second ``HOOK_FIRED(pre_tool_use)``
 event's ``outcome`` is also asserted to be ``allow`` (not
 ``require_approval``) which can only happen via the gate's
 short-circuit branch.
 """
    # ``RuntimeConstants()`` defaults to ``approval_gate_web_enabled=False``.
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    assert engine.config.rc.approval_gate_web_enabled is False  # sanity
    tool = MockTool(tool_name="MyTool", response_content="ran-anyway")
    in_memory_runtime["tools"].register(tool)
    # Stateless hook: returns ``requires_approval=True`` on every invocation
    # so the re-dispatch must rely on ``skip_pre_tool_approval`` rather than
    # an empty-queue ALLOW fallback. Replaces the default
    # ``InMemoryHookManager.invoke`` for this test only.
    hook_manager = in_memory_runtime["hooks"]

    async def _stateless_invoke(
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        hook_manager.invocations.append((event, payload, tenant_id))
        if event is HookEvent.pre_tool_use:
            return HookResult(
                action=HookActionKind.ALLOW,
                modifications={"requires_approval": True, "approval_token": "tok-1"},
                reason="operator hook says approve",
            )
        return HookResult(action=HookActionKind.ALLOW)

    hook_manager.invoke = _stateless_invoke  # type: ignore[method-assign]
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_q",
        tool_name="MyTool",
        tool_input={"v": "go"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="run")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    # No TOOL_CALL_PENDING envelope reaches the outer loop.
    assert EventType.TOOL_CALL_PENDING not in [e.type for e in events]
    # Tool actually executed EXACTLY ONCE. This is the load-bearing
    # assertion: with a stateless ``require_approval`` hook the only way
    # to reach this state is via the gate's ``skip_pre_tool_approval``
    # short-circuit on the re-dispatch (otherwise the second pass would
    # also yield ``approval_required`` and the engine would AWAIT
    # without executing the tool).
    assert tool.calls == [{"v": "go"}]
    # PreToolUse hook fires on BOTH passes (first dispatch + re-dispatch).
    pre_tool_use_invocations = [
        inv for inv in hook_manager.invocations if inv[0] is HookEvent.pre_tool_use
    ]
    assert len(pre_tool_use_invocations) == 2, (
        "stateless hook must be invoked on both the first dispatch and "
        "the re-dispatch; observed only "
        f"{len(pre_tool_use_invocations)} pre_tool_use invocation(s)"
    )
    # Both HOOK_FIRED(pre_tool_use) events surface; the second one carries
    # ``outcome=allow`` which is only produced by the gate's
    # ``skip_pre_tool_approval`` short-circuit at
    # ``tool_permission.py:463-468`` — the hook itself still returns
    # ``require_approval``. Stage stays ``hook`` because the gate's hook
    # branch is what emits the ALLOW.
    hook_fired = [
        e
        for e in events
        if e.type is EventType.HOOK_FIRED
        and e.payload.get("hook_event") == HookEvent.pre_tool_use.value
    ]
    assert len(hook_fired) == 2, (
        f"expected two HOOK_FIRED(pre_tool_use) events; got {len(hook_fired)}"
    )
    assert hook_fired[0].payload["outcome"] == "require_approval"
    assert hook_fired[1].payload["outcome"] == "allow", (
        "second pass must short-circuit to ``allow`` via "
        "``skip_pre_tool_approval``; observed "
        f"{hook_fired[1].payload['outcome']!r}"
    )
    # History carries the tool result.
    tool_results = [
        b for msg in engine.history for b in msg.content_blocks if isinstance(b, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].content == "ran-anyway"
    assert tool_results[0].is_error is False
    # Engine terminates COMPLETED (not AWAITING).
    assert engine.state is LoopState.COMPLETED


@pytest.mark.asyncio
async def test_approval_kill_switch_on_preserves_awaiting_flow(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``approval_gate_web_enabled=True`` keeps the pause / AWAITING flow.

    Symmetric inverse of the web-mode downgrade — when the operator
    flips the kill-switch ON (future CLI mode) the dispatcher's
    ``TOOL_CALL_PENDING`` envelope reaches the outer loop, the engine
    transitions to ``AWAITING`` and the tool is NOT invoked. The same
    hook payload as the downgrade test is used so the only differing
    variable is the RC flag.
    """
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, approval_gate_web_enabled=True),
    )
    tool = MockTool(tool_name="MyTool", response_content="should-not-run")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-cli"},
            reason="cli mode",
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_cli",
        tool_name="MyTool",
        tool_input={"v": "wait"},
    )

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="run")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    pending = [e for e in events if e.type is EventType.TOOL_CALL_PENDING]
    assert len(pending) == 1
    assert pending[0].payload["approval_token"] == "tok-cli"
    assert tool.calls == []
    assert engine.state is LoopState.AWAITING


@pytest.mark.asyncio
async def test_approval_downgrade_emits_warning_log(
    engine_factory,
    in_memory_runtime,
    caplog,
) -> None:
    """The downgrade path emits a ``WARNING`` audit log so the event is grep-able.

 Per the rule for production-visible
 diagnostics is ``logger.warning``; the downgrade is intentionally
 not ``DIAG``-prefixed because it is a structural policy decision
 rather than per-iteration diagnostic.
 """
    import logging

    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    tool = MockTool(tool_name="MyTool", response_content="ran")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-warn"},
            reason="log me",
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_warn",
        tool_name="MyTool",
        tool_input={},
    )
    in_memory_runtime["llm"].queue_response(text="done")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="x")])
    with caplog.at_level(logging.WARNING, logger="protocore.runtime.query"):
        async for _ in engine.run(user_msg):
            pass

    downgrade_records = [
        r for r in caplog.records if "approval.downgrade" in r.getMessage()
    ]
    assert len(downgrade_records) == 1
    assert downgrade_records[0].levelno == logging.WARNING
    # Message carries the run id + tool name so a kubectl logs grep can
    # tie the downgrade to a specific (run, tool).
    msg = downgrade_records[0].getMessage()
    assert "tool=MyTool" in msg
    assert "web_mode_default_off" in msg


# ----------------------------------------------------------------------
# PostToolUse hook modifies content end-to-end
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_hook_modifies_tool_output_in_history(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory()
    tool = MockTool(tool_name="MyTool", response_content="raw secret")
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.post_tool_use,
        HookResult(
            action=HookActionKind.MODIFY,
            modifications={"tool_output": "[redacted]"},
        ),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_secret",
        tool_name="MyTool",
        tool_input={},
    )
    in_memory_runtime["llm"].queue_response(text="ok")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    tool_results = [b for msg in engine.history for b in msg.content_blocks if isinstance(b, ToolResultBlock)]
    assert tool_results[0].content == "[redacted]"

    # Final TOOL_RESULT envelope also shows the redacted text.
    final_result = [e for e in events if e.type is EventType.TOOL_RESULT][-1]
    assert final_result.payload["content_blocks"][0]["text"] == "[redacted]"


# ----------------------------------------------------------------------
# Tool execution exception classified as execution error
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_exception_classified_execution(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory()
    tool = MockTool(
        tool_name="Boom",
        raise_exception=RuntimeError("boom-boom"),
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_boom",
        tool_name="Boom",
        tool_input={},
    )
    in_memory_runtime["llm"].queue_response(text="acknowledged")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    result_evt = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result_evt.payload["error"]["kind"] == "execution"
    assert "boom-boom" in result_evt.payload["error"]["message"]

    # Loop completed (with acknowledgement turn).
    assert engine.state is LoopState.COMPLETED


# ----------------------------------------------------------------------
# Event ordering invariant — full sequence
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_ordering_invariant(engine_factory, in_memory_runtime) -> None:
    """tool_use_start < hook_fired(pre) < tool_result < hook_fired(post)?
    The PRE-hook fires AFTER the LLM stream ends but BEFORE execute,
    and the POST-hook fires AFTER execute but BEFORE the tool_result
    is emitted. Order within the dispatcher:

        tool_use_stop (LLM stream)  ──┐
        hook_fired(pre)             ──┤
        (tool.invoke)               ──┤
        hook_fired(post)            ──┤
        tool_result                 ──┘

    ``sandbox_starting`` is emitted by the sandbox adapter on cold
    start only.
    """
    engine = engine_factory()
    tool = MockTool(tool_name="MyTool", response_content="x")
    in_memory_runtime["tools"].register(tool)
    # Per L-1: ``hook_fired(pre)`` only emits at PermissionStage.hook.
    # Queue a MODIFY action so the gate reaches that stage.
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={}),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_x",
        tool_name="MyTool",
        tool_input={"v": "x"},
    )
    in_memory_runtime["llm"].queue_response(text="ok")

    user_msg = Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    types = [e.type for e in events]
    indices = {t: types.index(t) for t in set(types) if t in types}

    assert EventType.TOOL_USE_START in indices
    assert EventType.TOOL_USE_STOP in indices
    assert EventType.TOOL_RESULT in indices

    assert indices[EventType.TOOL_USE_START] < indices[EventType.TOOL_USE_STOP]
    assert indices[EventType.TOOL_USE_STOP] < indices[EventType.TOOL_RESULT]

    # Pre hook fires before tool_result; post hook fires before tool_result.
    hook_events = [(i, e) for i, e in enumerate(events) if e.type is EventType.HOOK_FIRED]
    pre_idx = next(i for i, e in hook_events if e.payload["hook_event"] == "pre_tool_use")
    post_idx = next(i for i, e in hook_events if e.payload["hook_event"] == "post_tool_use")
    assert pre_idx < indices[EventType.TOOL_RESULT]
    assert post_idx < indices[EventType.TOOL_RESULT]
    # Pre fires BEFORE post.
    assert pre_idx < post_idx
