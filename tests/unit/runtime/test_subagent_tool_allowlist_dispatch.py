"""The declared-toolset allow-list, enforced at dispatch.

``ToolPermissionGate`` has always been able to refuse a tool outside an agent's
declared set, but no production dispatch passed the argument, so the stage never
ran. These tests drive :func:`protocore.runtime.query._dispatch_tool` end-to-end
against a real :class:`ToolDispatcher` and assert the refusal happens when the
tool is CALLED — an assertion about the advertised surface would pass just as
well against a tool that is merely unlisted and still callable.

The invariant that protects existing deployments is the last two tests: an agent
that declares nothing is unrestricted, exactly as before.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import ToolCall
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query import _dispatch_tool
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

from ._tool_fixtures import MockTool

RESTRICTED = "ConsoleConstantSearch"
ORDINARY = "Grep"


def _engine(
    *,
    declared: frozenset[str] = frozenset(),
    policy: ToolVisibilityPolicy | None = None,
) -> QueryEngine:
    registry = InMemoryToolRegistry()
    for name in (RESTRICTED, ORDINARY):
        registry.register(MockTool(tool_name=name))
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-allowlist",
            tenant_id="tenant-allowlist",
            session_id="sess-allowlist",
            model_name="qwen3.6-35b-a3b",
            rc=RuntimeConstants(
                model_context_window=4_096,
                # An empty floor keeps these tests about the declaration alone.
                # The floor's own interaction with it is asserted separately.
                tool_surface_forced_pins=(),
            ),
            tool_visibility_policy=policy or ToolVisibilityPolicy(),
            subagent_tool_allowlist=declared,
        ),
        llm_provider=object(),  # type: ignore[arg-type]  # never streamed here
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    engine._helpers = {}  # type: ignore[attr-defined]
    return engine


async def _dispatch(engine: QueryEngine, name: str) -> list[TurnEvent]:
    call = ToolCall(id=f"call-{name}", name=name, arguments={})
    return [evt async for evt in _dispatch_tool(engine, call)]


def _last_result(events: list[TurnEvent]) -> TurnEvent:
    results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert results, "dispatch produced no tool_result"
    return results[-1]


def _result_text(events: list[TurnEvent]) -> str:
    blocks = _last_result(events).payload.get("content_blocks") or []
    return " ".join(
        str(block.get("text", "")) for block in blocks if isinstance(block, dict)
    )


def _refused(events: list[TurnEvent]) -> bool:
    return not bool(_last_result(events).payload.get("success"))


@pytest.mark.asyncio
async def test_declared_tool_is_callable_by_the_agent_that_declared_it() -> None:
    engine = _engine(declared=frozenset({RESTRICTED}))
    events = await _dispatch(engine, RESTRICTED)
    assert not _refused(events)


@pytest.mark.asyncio
async def test_undeclared_tool_is_refused_at_dispatch() -> None:
    """A different agent's declaration does not reach this one's tool.

    The tool is REGISTERED and not blocked by any tenant policy — the only
    thing standing between the call and the tool is the declaration.
    """
    engine = _engine(declared=frozenset({ORDINARY}))
    events = await _dispatch(engine, RESTRICTED)
    assert _refused(events)
    assert "not in subagent whitelist" in _result_text(events)


@pytest.mark.asyncio
async def test_agent_declaring_nothing_is_unrestricted() -> None:
    """The compatibility invariant: no declaration ⇒ the stage never runs.

    Every subagent that exists today declares nothing, and this is what keeps
    their surface exactly as it was.
    """
    engine = _engine(declared=frozenset())
    assert engine.effective_subagent_tool_allowlist is None
    for name in (RESTRICTED, ORDINARY):
        events = await _dispatch(engine, name)
        assert not _refused(events), f"{name} was refused for an undeclared agent"


@pytest.mark.asyncio
async def test_declaration_narrows_only_beyond_the_forced_pin_floor() -> None:
    """A floor tool stays callable even when the declaration omits it.

    ``forced_pinned`` is advertised to the model unconditionally, so refusing
    one would hand out a callable schema that always fails — the exact
    advertise/dispatch mismatch the gate's own comments warn about.
    """
    engine = _engine(declared=frozenset({RESTRICTED}))
    object.__setattr__(
        engine.config,
        "rc",
        RuntimeConstants(
            model_context_window=4_096,
            tool_surface_forced_pins=(ORDINARY,),
        ),
    )
    allowlist = engine.effective_subagent_tool_allowlist
    assert allowlist == frozenset({RESTRICTED, ORDINARY})
    assert not _refused(await _dispatch(engine, ORDINARY))


@pytest.mark.asyncio
async def test_tenant_block_still_wins_over_a_declaration() -> None:
    """Declaring a tool does not grant it past the tenant policy."""
    engine = _engine(
        declared=frozenset({RESTRICTED}),
        policy=ToolVisibilityPolicy(blocked={RESTRICTED}),
    )
    events = await _dispatch(engine, RESTRICTED)
    assert _refused(events)
    assert "blocked list" in _result_text(events)
