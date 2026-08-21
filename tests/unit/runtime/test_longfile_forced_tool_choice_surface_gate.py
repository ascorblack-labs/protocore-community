"""Forced ``tool_choice`` is NOT consumed when the forced tool is not on the
per-turn surface.

The convergence driver decides to force ``AppendFile`` or ``FinalizeFile`` on the
next stream and charges the per-kind budget; ``_drive_one_stream`` then
threads the name onto ``LLMRequest.extra["forced_tool_choice"]`` so
the host adapter translates it into a native ``tool_choice``. The
unconditional ``take_force_next_tool`` at request assembly would POP the
hint even when the tool is not on the surface (a BM25 clip, a compacted
surface, or a tenant that does not include ``AppendFile`` /
``FinalizeFile`` in its ``tool_surface_forced_pins``) — silently dropping
the force while the continue message + ``commit_forced_*`` charge are
already settled. The fix is a non-destructive ``peek_force_next_tool``
that the stream builder uses to gate the pop on the surface including
the tool, so a future stream that does include the tool picks the hint
back up.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
)
from protocore.runtime import longfile_convergence as lfc
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.context.manager import ContextBundle
from protocore.runtime.query import (
    _drive_one_stream,
    _StreamAttemptResult,
)


def _user_message(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _finish_deltas() -> list[ProviderDelta]:
    return [ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")]


def _make_context(
    engine, tools: tuple[ToolDefinition, ...]
) -> ContextBundle:
    """Build a :class:`ContextBundle` whose ``budgets`` is a real derived
    instance (the stream builder reads ``context.budgets.max_context`` to
    size the output cap)."""
    return ContextBundle(
        system_prompt_sections=(),
        tools=tools,
        messages=tuple(engine.history),
        active_language="en",
        budgets=derive_budgets(engine.config.rc),
    )


def _stub_stream(deltas: list[ProviderDelta]):
    """Return a ``stream_with_tools`` stub yielding ``deltas`` verbatim."""

    def _stream_with_tools(request: object) -> AsyncIterator[ProviderDelta]:
        del request

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in deltas:
                yield delta

        return _gen()

    return _stream_with_tools


def _tool_def(name: str) -> ToolDefinition:
    """Build a minimal :class:`ToolDefinition` for a surface-only test."""
    return ToolDefinition(
        name=name,
        description=f"fake {name}",
        parameters=ToolParameterSchema(properties={}, required=[]),
    )


@pytest.mark.asyncio
async def test_force_hint_not_consumed_when_forced_tool_not_on_surface(
    engine_factory,
) -> None:
    """The stream builder must NOT pop the
    ``force_next_tool`` hint when the forced tool is not on the per-turn
    surface (``AppendFile`` is in the hint but not in ``context.tools``).
    Subsequent ``take_force_next_tool`` calls return the same value, so a
    future stream whose surface DOES include ``AppendFile`` still forces it.

    FAILS on pre-fix code: the unconditional ``take_force_next_tool`` pops
    the hint, so a downstream consumer sees ``None`` and the force is
    silently dropped.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("do work"))
    engine.llm.stream_with_tools = _stub_stream(_finish_deltas())  # type: ignore[method-assign]

    # Set the forced-tool hint (simulating ``maybe_force_next_tool`` having
    # decided to drive an append on the next stream).
    lfc.set_force_next_tool(engine, "AppendFile")
    assert lfc.peek_force_next_tool(engine) == "AppendFile"

    # The surface is ``[Read, Write]`` — ``AppendFile`` is NOT in it (the
    # realistic BM25-clip / non-pinned shape the bug fires on).
    context = _make_context(
        engine, (_tool_def("Read"), _tool_def("Write"))
    )
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    # The hint was NOT consumed (a downstream stream can still force it).
    assert lfc.peek_force_next_tool(engine) == "AppendFile", (
        "force_next_tool hint was popped even though the forced tool was not on "
        "the per-turn surface — a future stream whose surface includes the tool "
        "can no longer be forced"
    )


@pytest.mark.asyncio
async def test_force_hint_consumed_when_forced_tool_on_surface(
    engine_factory,
) -> None:
    """When the surface DOES include the forced
    tool, the pop happens exactly once and ``forced_tool_choice`` lands on
    the outbound ``LLMRequest.extra``. The hint is consumed so the next
    stream is not double-forced.
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("do work"))

    captured: list[object] = []

    def _capture(request: object) -> AsyncIterator[ProviderDelta]:
        captured.append(request)

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in _finish_deltas():
                yield delta

        return _gen()

    engine.llm.stream_with_tools = _capture  # type: ignore[method-assign]

    lfc.set_force_next_tool(engine, "AppendFile")
    context = _make_context(
        engine,
        (_tool_def("Read"), _tool_def("Write"), _tool_def("AppendFile")),
    )
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    # The hint was consumed (the surface includes the tool).
    assert lfc.peek_force_next_tool(engine) is None
    # The forced choice was threaded onto the request.
    request = captured[0]
    extra = getattr(request, "extra", {})
    assert extra.get("forced_tool_choice") == "AppendFile"


@pytest.mark.asyncio
async def test_force_hint_no_op_when_unset(engine_factory) -> None:
    """When no forced-tool hint is pending, the
    stream builder must not set ``forced_tool_choice`` on the request and
    must not throw (idempotent on the no-hint path).
    """
    engine = engine_factory(rc=RuntimeConstants(model_context_window=4_096))
    engine.history.append(_user_message("hello"))
    captured: list[object] = []

    def _capture(request: object) -> AsyncIterator[ProviderDelta]:
        captured.append(request)

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in _finish_deltas():
                yield delta

        return _gen()

    engine.llm.stream_with_tools = _capture  # type: ignore[method-assign]
    # No set_force_next_tool — the hint is unset.
    context = _make_context(
        engine, (_tool_def("Read"), _tool_def("AppendFile"))
    )
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, context, result):
        pass

    request = captured[0]
    extra = getattr(request, "extra", {})
    assert "forced_tool_choice" not in extra
