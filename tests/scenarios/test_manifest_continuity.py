"""A run picked up elsewhere asks for the same thing it was going to ask for.

This is the property the request manifest exists to make checkable, and it is
checked here through the entry points a host actually drives — the constructor,
``run()``, ``snapshot()``, ``resume_from_snapshot()`` and ``rearm()`` — with no
private symbol of the engine touched anywhere in the file. A test that reached
inside would be asserting about an implementation rather than about the
guarantee a host is given.

The guarantee: whatever boundary a run is cut off at, the process that picks it
up builds the SAME next request. Three ways of arriving at that next request
are covered, because they are three different code paths and only one of them
was ever exercised:

* **fresh** — a run built and driven from nothing;
* **warm** — the same live engine re-armed for the next question, which is what
  a host does when a session continues in the same process;
* **cold** — a new engine in a new process, rehydrated from the snapshot, which
  is what a host does when the pod it was running on died.

The recorded run is the oracle: its requests are replayed by
``ReplayLLMProvider``, which refuses anything it does not recognise, so a drift
in assembly fails here rather than being papered over by a fresh provider that
answers whatever it is asked.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from protocore.contracts.llm import LLMRequest
from protocore.contracts.observability import request_digest
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext, ToolResult
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryRequestManifestSink,
    InMemorySkillStore,
    InMemoryToolRegistry,
    ReplayLLMProvider,
)

MODEL = "test-model-continuity"
RUN_ID = "run-continuity"
TURNS = ("first question", "second question", "third question")


class _EchoTool(Tool):
    """A tool with no side effect, so a re-driven turn may call it again."""

    @property
    def name(self) -> str:
        return "Echo"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Echo",
            description="Echo the text back.",
            parameters=ToolParameterSchema(
                properties={"text": {"type": "string"}}, required=["text"]
            ),
        )

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        del context
        return ToolResult(content=str(arguments.get("text", "")))


def _build_engine(
    *, llm: Any, sink: InMemoryRequestManifestSink | None
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    registry.register(_EchoTool())
    return QueryEngine(
        config=QueryEngineConfig(
            run_id=RUN_ID,
            tenant_id="tenant-continuity",
            session_id="sess-continuity",
            model_name=MODEL,
            rc=LoopConstants(model_context_window=8_192),
            request_manifest_sink=sink,
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


def _scripted_provider(answers: Sequence[str]) -> InMemoryLLMProvider:
    llm = InMemoryLLMProvider()
    for answer in answers:
        llm.queue_response(text=answer, stop_reason=StopReason.end_turn)
    return llm


async def _ask(engine: QueryEngine, text: str) -> None:
    [
        event
        async for event in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
        )
    ]


def _recording(llm: InMemoryLLMProvider, answers: Sequence[str]) -> ReplayLLMProvider:
    """The requests a run made, paired with the answers it got."""
    return ReplayLLMProvider.from_recording(
        [
            (request_digest(request), _stream_for(answer))
            for request, answer in zip(llm.calls, answers, strict=True)
        ]
    )


def _stream_for(answer: str) -> list[Any]:
    from protocore.contracts.llm import LLMStreamEvent

    return [
        LLMStreamEvent(name="message_start", payload={}),
        LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
        LLMStreamEvent(name="content_block_delta", payload={"text": answer}),
        LLMStreamEvent(name="content_block_stop", payload={}),
        LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        ),
    ]


async def _drive_recorded_run() -> tuple[
    InMemoryLLMProvider,
    InMemoryRequestManifestSink,
    list[dict[str, Any]],
]:
    """One uninterrupted conversation: its requests, manifests and snapshots.

    A snapshot is taken at every turn boundary — the boundaries a host's store
    actually holds — so each one is a point the run could have been killed at.
    """
    answers = [f"answer to {text}" for text in TURNS]
    llm = _scripted_provider(answers)
    sink = InMemoryRequestManifestSink()
    engine = _build_engine(llm=llm, sink=sink)
    snapshots: list[dict[str, Any]] = []
    for index, text in enumerate(TURNS):
        if index:
            engine.rearm()
        await _ask(engine, text)
        snapshots.append(engine.snapshot())
    return llm, sink, snapshots


# ── fresh ───────────────────────────────────────────────────────────────────


async def test_a_fresh_run_asks_what_the_recording_holds() -> None:
    llm, sink, _ = await _drive_recorded_run()
    answers = [f"answer to {text}" for text in TURNS]

    replay = _recording(llm, answers)
    replay_sink = InMemoryRequestManifestSink()
    engine = _build_engine(llm=replay, sink=replay_sink)
    for index, text in enumerate(TURNS):
        if index:
            engine.rearm()
        await _ask(engine, text)

    assert replay.exhausted
    assert [item.manifest_id for item in replay_sink.manifests] == [
        item.manifest_id for item in sink.manifests
    ]


# ── warm ────────────────────────────────────────────────────────────────────


async def test_a_re_armed_engine_asks_what_the_recording_holds() -> None:
    """The same live engine, next question. Nothing is rebuilt, so a value that
    should have been reset and was not shows up as a changed request."""
    llm, _sink, _ = await _drive_recorded_run()
    answers = [f"answer to {text}" for text in TURNS]

    replay = _recording(llm, answers)
    engine = _build_engine(llm=replay, sink=None)
    await _ask(engine, TURNS[0])
    engine.rearm()
    await _ask(engine, TURNS[1])

    assert [request_digest(item) for item in replay.calls] == [
        request_digest(item) for item in llm.calls[:2]
    ]


# ── cold, at every boundary the run has ─────────────────────────────────────


async def test_a_cold_resume_at_every_boundary_asks_the_same_next_thing() -> None:
    """Kill the run after each turn, pick it up in a new engine, and check that
    the request it builds next is the one the uninterrupted run built.

    The manifest id is what is compared, and it is the stronger comparison: it
    covers the ordered messages, the full tool definitions, the budgets, the
    extras, the constants digest and the attempt id, so a resume that rebuilt
    the history one message short, or under a different constants snapshot,
    fails here even though the run would have carried on looking healthy.
    """
    llm, sink, snapshots = await _drive_recorded_run()
    answers = [f"answer to {text}" for text in TURNS]

    for boundary in range(len(TURNS) - 1):
        replay = _recording(llm, answers)
        # Position the recording at the call the resumed run will make.
        for made in llm.calls[: boundary + 1]:
            [event async for event in replay.stream_with_tools(made)]

        resumed_sink = InMemoryRequestManifestSink()
        resumed = _build_engine(llm=replay, sink=resumed_sink)
        await resumed.resume_from_snapshot(snapshots[boundary])
        resumed.rearm()
        await _ask(resumed, TURNS[boundary + 1])

        assert len(resumed_sink.manifests) == 1
        expected = sink.manifests[boundary + 1]
        assert resumed_sink.manifests[0].manifest_id == expected.manifest_id
        assert resumed_sink.manifests[0].attempt_id == expected.attempt_id


async def test_a_cold_resume_carries_the_last_manifest_reference() -> None:
    """The resumed run knows which call it was on: the id travels in the
    snapshot, the manifest itself stays wherever the host put it."""
    llm, sink, snapshots = await _drive_recorded_run()

    replay = _recording(llm, [f"answer to {text}" for text in TURNS])
    resumed = _build_engine(llm=replay, sink=None)
    await resumed.resume_from_snapshot(snapshots[0])

    reference = resumed.last_request_manifest
    assert reference is not None
    assert reference["manifest_id"] == sink.manifests[0].manifest_id
    # Addressed, not copied: the reference is three small fields, not a
    # transcript.
    assert set(reference) == {
        "manifest_id",
        "manifest_schema_version",
        "attempt_id",
    }


async def test_the_replay_provider_needs_no_endpoint_and_no_tokens() -> None:
    """The cheap half of the same mechanism: a recorded run re-drives with no
    provider at all, which is what makes an incident into a regression test."""
    llm, _, _ = await _drive_recorded_run()
    answers = [f"answer to {text}" for text in TURNS]
    replay = _recording(llm, answers)

    engine = _build_engine(llm=replay, sink=None)
    for index, text in enumerate(TURNS):
        if index:
            engine.rearm()
        await _ask(engine, text)

    assert replay.exhausted
    assert len(replay.calls) == len(TURNS)
    assert isinstance(replay.calls[0], LLMRequest)
