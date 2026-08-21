"""An agent handed files it has not read cannot answer until it reads them.

The loop-level property the read-back gate exists for. A leader delegates, the
subagent writes its findings to two files and returns the paths, and the leader
answers from the one-line pointer — measured on a live stand, with zero
citations where the files held dozens. So while a declared path is unread the
stream builder names the read tool in
``LLMRequest.extra['forced_tool_choice']``: the provider must call it, and a
final answer is not reachable. Reading the last path releases the gate in the
same turn, with no operator action.

These tests assert on the forced slot as it lands on the outbound request, and
on the release — not merely that a driver function was called.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    PENDING_READS_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolPrecondition,
)
from protocore.runtime import longfile_convergence as _longfile
from protocore.runtime import pending_reads as _pending_reads
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.manager import ContextBundle
from protocore.runtime.events import EventType
from protocore.runtime.query import _drive_one_stream, _StreamAttemptResult

DELEGATED_FILES = ["reports/findings-a.md", "reports/findings-b.md"]


def _tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"fake {name}",
        parameters=ToolParameterSchema(properties={}, required=[]),
    )


#: The realistic surface a leader is given: it can delegate, read and answer.
FULL_SURFACE = (
    _tool_def("Agent"),
    _tool_def("Read"),
    _tool_def("Grep"),
    _tool_def("SubmitAnswer"),
)


def _make_context(engine, tools: tuple[ToolDefinition, ...]) -> ContextBundle:
    return ContextBundle(
        system_prompt_sections=(),
        tools=tools,
        messages=tuple(engine.history),
        active_language="en",
        budgets=derive_budgets(engine.config.rc),
    )


class _StreamRecorder:
    """Captures every outbound :class:`LLMRequest` and replies with a stop."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def __call__(self, request: object) -> AsyncIterator[ProviderDelta]:
        self.requests.append(request)

        async def _gen() -> AsyncIterator[ProviderDelta]:
            yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")

        return _gen()

    def forced_choice(self, index: int = -1) -> object:
        extra = getattr(self.requests[index], "extra", {})
        return extra.get("forced_tool_choice")


async def _run_one_stream(engine, tools=FULL_SURFACE) -> list[object]:
    """Drive one assistant stream; return the events it emitted."""
    events: list[object] = []
    result = _StreamAttemptResult()
    async for evt in _drive_one_stream(engine, _make_context(engine, tools), result):
        events.append(evt)
    return events


def _observe_delegation(engine, paths: list[str] | None = None) -> None:
    """Fold in a delegation result that declares files the caller must open."""
    _pending_reads.observe_tool_result(
        engine,
        ToolCall(id="agent-1", name="Agent", arguments={"subagent_type": "worker"}),
        {PENDING_READS_METADATA_KEY: list(DELEGATED_FILES if paths is None else paths)},
        is_error=False,
    )


def _observe_read(engine, path: str, *, call_id: str = "read-1") -> None:
    _pending_reads.observe_tool_result(
        engine,
        ToolCall(id=call_id, name="Read", arguments={"file_path": path}),
        {},
        is_error=False,
    )


def _released_reasons(events: list[object]) -> list[str]:
    reasons = []
    for evt in events:
        if getattr(evt, "type", None) is EventType.STATE_CHANGED:
            reason = evt.payload.get("reason")
            if isinstance(reason, str):
                reasons.append(reason)
    return reasons


@pytest.fixture
def leader(engine_factory):
    """An engine mid-run, with a recorder wired in place of the provider."""
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True))
    engine.history.append(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="review the service and report")],
        )
    )
    recorder = _StreamRecorder()
    engine.llm.stream_with_tools = recorder  # type: ignore[method-assign]
    return engine, recorder


@pytest.mark.asyncio
async def test_no_answer_until_both_declared_files_are_read(leader) -> None:
    """The headline property: two declared files, two forced reads, and only
    then a stream the model is free to answer on."""
    engine, recorder = leader
    _observe_delegation(engine)

    # Stream 1 — one file has been read: none. The read tool is forced, so the
    # provider cannot return a final answer on this turn.
    await _run_one_stream(engine)
    assert recorder.forced_choice() == "Read"

    # The model reads the first file. One still outstanding ⇒ still forced.
    _observe_read(engine, DELEGATED_FILES[0])
    await _run_one_stream(engine)
    assert recorder.forced_choice() == "Read"

    # It reads the second. The debt is paid — the gate releases ITSELF, with no
    # operator action and no timeout, and the whole surface is the agent's.
    _observe_read(engine, DELEGATED_FILES[1], call_id="read-2")
    await _run_one_stream(engine)
    assert recorder.forced_choice() is None
    assert _pending_reads.pending_paths(engine) == ()

    # Three streams, and the gate spoke on exactly the two that owed a read.
    assert [recorder.forced_choice(i) for i in range(3)] == ["Read", "Read", None]


@pytest.mark.asyncio
async def test_a_run_that_declares_nothing_is_untouched(leader) -> None:
    """Inert for every run in which no tool declares a file — which is nearly
    all of them."""
    engine, recorder = leader

    await _run_one_stream(engine)

    assert recorder.forced_choice() is None


@pytest.mark.asyncio
async def test_kill_switch_off_is_byte_identical(engine_factory) -> None:
    """An operator can disable the whole mechanism; a disabled build forces
    nothing and tracks nothing."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=False)
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    recorder = _StreamRecorder()
    engine.llm.stream_with_tools = recorder  # type: ignore[method-assign]
    _observe_delegation(engine)

    await _run_one_stream(engine)

    assert recorder.forced_choice() is None
    assert _pending_reads.pending_paths(engine) == ()


@pytest.mark.asyncio
async def test_a_precondition_wins_the_slot_and_the_debt_survives(
    engine_factory,
) -> None:
    """A run-level precondition is a promise with a deadline; the read-back
    debt is not, so it waits its turn — untouched, uncharged, still owed."""
    engine = engine_factory(
        rc=RuntimeConstants(model_context_window=4_096, pending_reads_enabled=True),
        tool_preconditions=(ToolPrecondition(tool="Grep", calls=1),),
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    recorder = _StreamRecorder()
    engine.llm.stream_with_tools = recorder  # type: ignore[method-assign]
    _observe_delegation(engine)

    await _run_one_stream(engine)

    assert recorder.forced_choice() == "Grep"
    assert _pending_reads.pending_paths(engine) == tuple(DELEGATED_FILES)
    # No attempt was charged for a stream the gate never owned.
    assert engine._pending_reads_forced_attempts == 0


@pytest.mark.asyncio
async def test_the_convergence_hint_wins_the_slot_and_the_debt_survives(
    leader,
) -> None:
    """A half-written file gets the transient slot it needs; the durable
    pending set loses nothing by waiting a turn."""
    engine, recorder = leader
    _observe_delegation(engine)
    _longfile.set_force_next_tool(engine, "AppendFile")

    await _run_one_stream(engine, tools=(*FULL_SURFACE, _tool_def("AppendFile")))

    assert recorder.forced_choice() == "AppendFile"
    assert _pending_reads.pending_paths(engine) == tuple(DELEGATED_FILES)
    assert engine._pending_reads_forced_attempts == 0


@pytest.mark.asyncio
async def test_read_missing_from_the_surface_consumes_nothing(leader) -> None:
    """A BM25-clipped surface can drop the read tool. Forcing an unadvertised
    name would reject the whole request, so the gate stays silent — and does
    not spend an attempt on a stream the model was never offered it on."""
    engine, recorder = leader
    _observe_delegation(engine)

    clipped = (_tool_def("Agent"), _tool_def("Grep"), _tool_def("SubmitAnswer"))
    await _run_one_stream(engine, tools=clipped)

    assert recorder.forced_choice() is None
    assert _pending_reads.pending_paths(engine) == tuple(DELEGATED_FILES)
    assert engine._pending_reads_forced_attempts == 0

    # A later stream whose surface DOES advertise the tool still forces it.
    await _run_one_stream(engine)
    assert recorder.forced_choice() == "Read"


@pytest.mark.asyncio
async def test_the_attempt_bound_releases_the_run_rather_than_wedging_it(
    engine_factory,
) -> None:
    """A declared file that cannot be read — it was never written, or the model
    reads something else every time — costs a bounded number of turns and then
    releases, saying so on the run's event stream."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            pending_reads_enabled=True,
            pending_reads_max_forced_attempts=2,
        )
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    recorder = _StreamRecorder()
    engine.llm.stream_with_tools = recorder  # type: ignore[method-assign]
    _observe_delegation(engine, ["reports/never-written.md"])

    # Two forced streams, and the model reads nothing that clears the debt.
    assert _released_reasons(await _run_one_stream(engine)) == []
    assert _released_reasons(await _run_one_stream(engine)) == []
    assert [recorder.forced_choice(i) for i in range(2)] == ["Read", "Read"]

    events = await _run_one_stream(engine)

    assert recorder.forced_choice() is None
    assert "pending_reads_released" in _released_reasons(events)
    assert _pending_reads.pending_paths(engine) == ()

    # And it stays released: the abandoned path is never forced again.
    await _run_one_stream(engine)
    assert recorder.forced_choice() is None


# ── What the gate leaves behind in a production log ────────────────────────
#
# A mechanism that narrows the model's surface and records nothing can be
# neither confirmed nor refuted after the fact. These pin that each of the
# three states the gate can be in is distinguishable from the others in a
# WARNING-level log, which is the only level production keeps.

_LOGGER = "protocore.runtime.pending_reads"


def _diag(caplog, event: str) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith(f"DIAG pending_reads.{event} ")
    ]


@pytest.mark.asyncio
async def test_forcing_a_read_says_so(leader, caplog) -> None:
    """The engaged state: which run, which attempt out of how many, and how
    much is still owed."""
    engine, _ = leader
    _observe_delegation(engine)

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _run_one_stream(engine)

    lines = _diag(caplog, "forced")
    assert len(lines) == 1
    assert f"run={engine.config.run_id}" in lines[0]
    assert "attempt=1/" in lines[0]
    assert "pending=2" in lines[0]
    assert DELEGATED_FILES[0] in lines[0]


@pytest.mark.asyncio
async def test_a_read_that_could_not_be_forced_is_not_silent(leader, caplog) -> None:
    """The state that mattered most and had no trace at all. A clipped surface
    drops the read tool, so the gate stays quiet and — correctly — charges
    nothing; on the wire that is indistinguishable from a run where no tool
    ever declared a read-back. The two need opposite fixes, so they must not
    look the same in the log."""
    engine, recorder = leader
    _observe_delegation(engine)

    clipped = (_tool_def("Agent"), _tool_def("Grep"), _tool_def("SubmitAnswer"))
    with caplog.at_level("WARNING", logger=_LOGGER):
        await _run_one_stream(engine, tools=clipped)

    assert recorder.forced_choice() is None
    lines = _diag(caplog, "not_forced")
    assert len(lines) == 1
    assert "reason=tool_not_on_surface" in lines[0]
    assert f"run={engine.config.run_id}" in lines[0]
    assert "pending=2" in lines[0]
    # The turn was NOT charged, and the line says so rather than implying an
    # attempt was spent.
    assert "attempt=0/" in lines[0]
    assert _diag(caplog, "forced") == []


@pytest.mark.asyncio
async def test_a_run_that_owes_nothing_logs_nothing(leader, caplog) -> None:
    """Inert runs stay inert: no declaration, no lines."""
    engine, _ = leader

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _run_one_stream(engine)

    assert caplog.records == []


@pytest.mark.asyncio
async def test_giving_up_names_the_files_it_abandoned(engine_factory, caplog) -> None:
    """The released state, with the paths that were never opened — the run
    continues without them, and the log says which ones."""
    engine = engine_factory(
        rc=RuntimeConstants(
            model_context_window=4_096,
            pending_reads_enabled=True,
            pending_reads_max_forced_attempts=1,
        )
    )
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    engine.llm.stream_with_tools = _StreamRecorder()  # type: ignore[method-assign]
    _observe_delegation(engine, ["reports/never-written.md"])

    with caplog.at_level("WARNING", logger=_LOGGER):
        await _run_one_stream(engine)
        await _run_one_stream(engine)

    lines = _diag(caplog, "release_exhausted")
    assert len(lines) == 1
    assert f"run={engine.config.run_id}" in lines[0]
    assert "reports/never-written.md" in lines[0]
