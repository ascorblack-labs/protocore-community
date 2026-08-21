"""Typed evidence ingress from dispatch, independent of model-visible output."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
)
from protocore.contracts.verification import (
    EvidenceLedger,
    EvidenceProducerBinding,
    EvidenceRecord,
    RunTreeOrigin,
)
from protocore.runtime.events import EventType

from ._tool_fixtures import MockTool


def _record(context: ToolContext, record_id: str) -> EvidenceRecord:
    origin = context.evidence_origin or RunTreeOrigin(
        run_id=context.run_id, root_run_id=context.run_id, depth=0
    )
    return EvidenceRecord(
        record_id=record_id,
        origin=origin,
        producer_id="trusted-tool",
        producer_revision="revision-1",
        subject_id=f"subject-{record_id}",
        subject_reference=f"reference-{record_id}",
        digest=f"digest-{record_id}",
    )


class _EvidenceTool(MockTool):
    is_concurrent_safe = True
    is_destructive = False

    @property
    def evidence_producer(self) -> EvidenceProducerBinding:
        return EvidenceProducerBinding(producer_id="trusted-tool", producer_revision="revision-1")

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(float(arguments.get("delay", 0)))
        record_id = str(arguments["record_id"])
        return ToolResult(
            tool_call_id="",
            content=f"result-{record_id}",
            evidence_records=(_record(context, record_id),),
        )


class _ForgedProducerEvidenceTool(_EvidenceTool):
    """A tool may lie in its result; the dispatcher must overwrite it."""

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        record = _record(context, str(arguments["record_id"])).model_copy(
            update={"producer_id": "forged-producer", "producer_revision": "forged-revision"}
        )
        return ToolResult(tool_call_id="", content="forged", evidence_records=(record,))


class _ErrorSideChannelAttemptTool(MockTool):
    """Regression: context offers no callable path into the evidence ledger."""

    is_concurrent_safe = False
    is_destructive = False

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        del arguments
        assert not hasattr(context, "evidence_admitter")
        assert not any(
            callable(value)
            for name, value in context.__dict__.items()
            if "evidence" in name
        )
        return ToolResult(tool_call_id="", content="failed after attempted side channel", is_error=True)


class _PrerequisiteEvidenceTool(_EvidenceTool):
    """Successful-looking prerequisite whose duplicate evidence is rejected."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(properties={"record_id": {"type": "string"}}),
        )


class _DependentTool(MockTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(properties={"record_id": {"type": "string"}}),
            preconditions=["Produce"],
        )


def _queue_tools(
    llm: Any, calls: list[tuple[str, dict[str, Any]]], *, tool_name: str = "Read"
) -> None:
    stream: list[LLMStreamEvent] = [LLMStreamEvent(name="message_start", payload={})]
    for call_id, arguments in calls:
        stream.extend(
            (
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={"tool_call_id": call_id, "partial_input_json": json.dumps(arguments)},
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={"tool_call_id": call_id, "final_input": arguments},
                ),
            )
        )
    stream.append(LLMStreamEvent(name="message_stop", payload={"stop_reason": StopReason.tool_use.value}))
    llm._scripted_streams.append(stream)


def _queue_named_tools(llm: Any, calls: list[tuple[str, str, dict[str, Any]]]) -> None:
    stream: list[LLMStreamEvent] = [LLMStreamEvent(name="message_start", payload={})]
    for call_id, tool_name, arguments in calls:
        stream.extend(
            (
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={"tool_call_id": call_id, "partial_input_json": json.dumps(arguments)},
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={"tool_call_id": call_id, "final_input": arguments},
                ),
            )
        )
    stream.append(LLMStreamEvent(name="message_stop", payload={"stop_reason": StopReason.tool_use.value}))
    llm._scripted_streams.append(stream)


@pytest.mark.asyncio
async def test_serial_dispatch_persists_typed_evidence_without_history_or_sse_leak(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="run-evidence")
    engine.begin_evidence_collection(ledger_id="ledger-evidence")
    in_memory_runtime["tools"].register(_EvidenceTool(tool_name="Read", description="read"))
    _queue_tools(in_memory_runtime["llm"], [("call-1", {"record_id": "record-1"})])
    in_memory_runtime["llm"].queue_response(text="done")

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
        )
    ]

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["record-1"]
    assert engine.snapshot()["verification"]["ledger"]["records"][0]["record_id"] == "record-1"
    visible = [
        block
        for message in engine.history
        for block in message.content_blocks
        if getattr(block, "tool_call_id", None) == "call-1"
    ]
    assert len(visible) == 2  # tool_use plus the model-visible tool_result only
    result_event = next(event for event in events if event.type is EventType.TOOL_RESULT)
    assert "evidence_records" not in result_event.payload
    assert "evidence_records" not in result_event.payload.get("metadata", {})


def test_error_tool_result_cannot_carry_typed_evidence() -> None:
    context = ToolContext(run_id="run", tenant_id="tenant", session_id="session")
    with pytest.raises(ValueError, match="error ToolResult"):
        ToolResult(
            tool_call_id="call",
            content="failed",
            is_error=True,
            evidence_records=(_record(context, "record"),),
        )


@pytest.mark.asyncio
async def test_invalid_evidence_rejects_dispatch_without_partial_ledger_mutation(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="duplicate-run")
    engine.begin_evidence_collection(ledger_id="duplicate-ledger")
    existing_context = ToolContext(
        run_id="duplicate-run", tenant_id="tenant-test", session_id="sess-test"
    )
    engine.append_tool_evidence((_record(existing_context, "duplicate"),))
    in_memory_runtime["tools"].register(_EvidenceTool(tool_name="Read", description="read"))
    _queue_tools(in_memory_runtime["llm"], [("call-duplicate", {"record_id": "duplicate"})])
    in_memory_runtime["llm"].queue_response(text="done")

    events = [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
        )
    ]

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["duplicate"]
    result = next(
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "call-duplicate"
    )
    assert result.is_error is True
    assert result.content.startswith("tool evidence rejected:")
    result_event = next(event for event in events if event.type is EventType.TOOL_RESULT)
    assert result_event.payload["success"] is False
    assert result_event.payload["error"]["kind"] == "execution"


@pytest.mark.asyncio
async def test_rejected_evidence_never_satisfies_a_serial_dependency_or_resets_failure_state(
    engine_factory, in_memory_runtime
) -> None:
    """Evidence rejection is a dispatch failure before all success side effects."""

    engine = engine_factory(
        run_id="serial-dependency-run",
        rc=RuntimeConstants(model_context_window=4_096, tool_preconditions_enabled=True),
    )
    engine.begin_evidence_collection(ledger_id="serial-dependency-ledger")
    engine._helpers = {"rc": engine.config.rc}
    existing_context = ToolContext(
        run_id=engine.config.run_id,
        tenant_id=engine.config.tenant_id,
        session_id=engine.config.session_id,
    )
    engine.append_tool_evidence((_record(existing_context, "duplicate"),))
    producer = _PrerequisiteEvidenceTool(tool_name="Produce", description="produce")
    dependent = _DependentTool(tool_name="Consume", description="consume")
    in_memory_runtime["tools"].register(producer)
    in_memory_runtime["tools"].register(dependent)
    _queue_named_tools(
        in_memory_runtime["llm"],
        [
            ("produce", "Produce", {"record_id": "duplicate"}),
            ("consume", "Consume", {"record_id": "unused"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert results[0].is_error is True
    assert results[0].content.startswith("tool evidence rejected:")
    assert results[1].is_error is True
    assert "PRECONDITION NOT MET" in results[1].content
    assert "'Produce'" in results[1].content
    assert dependent.calls == []
    # A failed evidence admission must advance, not clear, the normal dispatch
    # failure streak.  This is the state the following call observes.
    assert engine._helpers["tool_dispatch.consecutive_error_state"]["tool_name"] == "Consume"


@pytest.mark.asyncio
async def test_child_tool_receives_typed_run_tree_origin_and_appends_to_child_ledger(
    engine_factory, in_memory_runtime
) -> None:
    """A descendant tool uses injected provenance; it does not infer ancestry."""

    child = engine_factory(
        run_id="child-run",
        root_run_id="root-run",
        parent_run_id="parent-run",
        subagent_id="researcher",
    )
    child.begin_evidence_collection(ledger_id="child-ledger")
    in_memory_runtime["tools"].register(_EvidenceTool(tool_name="Read", description="read"))
    _queue_tools(in_memory_runtime["llm"], [("child-call", {"record_id": "child-record"})])
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in child.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = child.verification_lifecycle.ledger
    assert ledger is not None
    assert ledger.records[0].origin == RunTreeOrigin(
        run_id="child-run",
        root_run_id="root-run",
        depth=1,
        parent_run_id="parent-run",
        subagent_id="researcher",
    )


@pytest.mark.asyncio
async def test_serial_dispatch_stamps_registered_producer_binding_over_tool_claim(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="producer-stamp-serial")
    engine.begin_evidence_collection(ledger_id="producer-stamp-serial-ledger")
    in_memory_runtime["tools"].register(
        _ForgedProducerEvidenceTool(tool_name="Read", description="read")
    )
    _queue_tools(in_memory_runtime["llm"], [("call-1", {"record_id": "record-1"})])
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert ledger.records[0].producer_id == "trusted-tool"
    assert ledger.records[0].producer_revision == "revision-1"
    assert ledger.records[0].origin == engine._engine_evidence_origin()


@pytest.mark.asyncio
async def test_parallel_replay_stamps_registered_producer_binding_in_llm_order(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="producer-stamp-parallel")
    engine.begin_evidence_collection(ledger_id="producer-stamp-parallel-ledger")
    in_memory_runtime["tools"].register(
        _ForgedProducerEvidenceTool(tool_name="Read", description="read")
    )
    _queue_tools(
        in_memory_runtime["llm"],
        [("first", {"record_id": "first", "delay": 0.02}), ("second", {"record_id": "second"})],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["first", "second"]
    assert {(record.producer_id, record.producer_revision) for record in ledger.records} == {
        ("trusted-tool", "revision-1")
    }


@pytest.mark.asyncio
async def test_serial_error_tool_cannot_mutate_evidence_from_context(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="no-side-channel")
    engine.begin_evidence_collection(ledger_id="no-side-channel-ledger")
    in_memory_runtime["tools"].register(
        _ErrorSideChannelAttemptTool(tool_name="Read", description="read")
    )
    _queue_tools(in_memory_runtime["llm"], [("serial", {})])
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert ledger.records == ()


@pytest.mark.asyncio
async def test_parallel_error_tools_cannot_mutate_evidence_from_context(
    engine_factory, in_memory_runtime
) -> None:
    class _ParallelErrorSideChannelAttemptTool(_ErrorSideChannelAttemptTool):
        is_concurrent_safe = True

    engine = engine_factory(run_id="no-side-channel-parallel")
    engine.begin_evidence_collection(ledger_id="no-side-channel-parallel-ledger")
    in_memory_runtime["tools"].register(
        _ParallelErrorSideChannelAttemptTool(tool_name="Read", description="read")
    )
    _queue_tools(in_memory_runtime["llm"], [("first", {}), ("second", {})])
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert ledger.records == ()


async def _assert_rejected_evidence_locks_out_same_batch_dependency(
    engine_factory,
    in_memory_runtime,
    *,
    producer: _PrerequisiteEvidenceTool,
    dependent: _DependentTool,
) -> None:
    engine = engine_factory(
        run_id=f"{producer.tool_name}-parallel-run",
        rc=RuntimeConstants(model_context_window=4_096, tool_preconditions_enabled=True),
    )
    engine.begin_evidence_collection(ledger_id=f"{producer.tool_name}-parallel-ledger")
    engine._helpers = {"rc": engine.config.rc}
    existing_context = ToolContext(
        run_id=engine.config.run_id,
        tenant_id=engine.config.tenant_id,
        session_id=engine.config.session_id,
    )
    engine.append_tool_evidence((_record(existing_context, "duplicate"),))
    in_memory_runtime["tools"].register(producer)
    in_memory_runtime["tools"].register(dependent)
    _queue_named_tools(
        in_memory_runtime["llm"],
        [
            ("produce", producer.tool_name, {"record_id": "duplicate", "delay": 0.02}),
            ("consume", dependent.tool_name, {"record_id": "unused"}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    results = [
        block
        for message in engine.history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert results[0].is_error is True
    assert results[0].content.startswith("tool evidence rejected:")
    assert results[1].is_error is True
    assert "PRECONDITION NOT MET" in results[1].content
    assert dependent.calls == []
    assert engine._helpers["tool_dispatch.consecutive_error_state"]["tool_name"] == dependent.tool_name


@pytest.mark.asyncio
async def test_parallel_safe_replay_treats_rejected_evidence_as_failure_before_dependencies(
    engine_factory, in_memory_runtime
) -> None:
    class _ParallelProducer(_PrerequisiteEvidenceTool):
        is_concurrent_safe = True
        is_destructive = False

    class _ParallelDependent(_DependentTool):
        is_concurrent_safe = True
        is_destructive = False

    await _assert_rejected_evidence_locks_out_same_batch_dependency(
        engine_factory,
        in_memory_runtime,
        producer=_ParallelProducer(tool_name="Produce", description="produce"),
        dependent=_ParallelDependent(tool_name="Consume", description="consume"),
    )


@pytest.mark.asyncio
async def test_delegation_fanout_replay_treats_rejected_evidence_as_failure_before_dependencies(
    engine_factory, in_memory_runtime
) -> None:
    class _DelegationProducer(_PrerequisiteEvidenceTool):
        is_concurrent_safe = False
        is_parallel_delegation = True

    class _DelegationDependent(_DependentTool):
        is_concurrent_safe = False
        is_parallel_delegation = True

    await _assert_rejected_evidence_locks_out_same_batch_dependency(
        engine_factory,
        in_memory_runtime,
        producer=_DelegationProducer(tool_name="Produce", description="produce"),
        dependent=_DelegationDependent(tool_name="Consume", description="consume"),
    )


@pytest.mark.asyncio
async def test_parallel_dispatch_replays_evidence_in_llm_order_after_inverse_completion(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(run_id="parallel-run")
    engine.begin_evidence_collection(ledger_id="parallel-ledger")
    in_memory_runtime["tools"].register(_EvidenceTool(tool_name="Read", description="read"))
    _queue_tools(
        in_memory_runtime["llm"],
        [
            ("call-slow", {"record_id": "first", "delay": 0.05}),
            ("call-fast", {"record_id": "second", "delay": 0.0}),
        ],
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["first", "second"]
    snapshot_ledger = engine.snapshot()["verification"]["ledger"]
    assert EvidenceLedger.model_validate(snapshot_ledger).digest == ledger.digest


@pytest.mark.asyncio
async def test_parallel_delegation_replays_evidence_in_llm_order_after_inverse_completion(
    engine_factory, in_memory_runtime
) -> None:
    """The separate delegation fan-out has the same deterministic ledger order."""

    class _EvidenceDelegationTool(_EvidenceTool):
        is_concurrent_safe = False
        is_parallel_delegation = True

    engine = engine_factory(run_id="delegation-run")
    engine.begin_evidence_collection(ledger_id="delegation-ledger")
    in_memory_runtime["tools"].register(
        _EvidenceDelegationTool(tool_name="Agent", description="delegate")
    )
    _queue_tools(
        in_memory_runtime["llm"],
        [
            ("call-slow", {"record_id": "first", "delay": 0.05}),
            ("call-fast", {"record_id": "second", "delay": 0.0}),
        ],
        tool_name="Agent",
    )
    in_memory_runtime["llm"].queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    ledger = engine.verification_lifecycle.ledger
    assert ledger is not None
    assert [record.record_id for record in ledger.records] == ["first", "second"]
