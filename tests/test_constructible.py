"""Smoke test: in-memory runtime wires up end-to-end on mocks only.

Zero host dependencies in core tests.
"""
from __future__ import annotations

from protocore.contracts.agent_dispatch import IAgentDispatch
from protocore.contracts.blob import IBlobStore
from protocore.contracts.events import IEventStream
from protocore.contracts.hooks import IHookManager
from protocore.contracts.llm import ILLMProvider
from protocore.contracts.run import IRunStore
from protocore.contracts.search import ISearchIndex
from protocore.contracts.session import ISessionStore
from protocore.contracts.skills import ISkillStore
from protocore.contracts.todo import ITodoStorage
from protocore.contracts.tool_registry import IToolRegistry
from protocore.contracts.types import StopReason
from protocore.testing import build_in_memory_runtime


def test_build_in_memory_runtime_smoke() -> None:
    runtime = build_in_memory_runtime()
    assert runtime is not None
    assert runtime.tenant_id == "test-tenant"
    assert runtime.rc.model_context_window > 0


def test_build_in_memory_runtime_implements_all_interfaces() -> None:
    runtime = build_in_memory_runtime()
    assert isinstance(runtime.llm, ILLMProvider)
    assert isinstance(runtime.blob, IBlobStore)
    assert isinstance(runtime.search, ISearchIndex)
    assert isinstance(runtime.skills, ISkillStore)
    assert isinstance(runtime.agents, IAgentDispatch)
    assert isinstance(runtime.sessions, ISessionStore)
    assert isinstance(runtime.runs, IRunStore)
    assert isinstance(runtime.events, IEventStream)
    assert isinstance(runtime.hooks, IHookManager)
    assert isinstance(runtime.todos, ITodoStorage)
    assert isinstance(runtime.tools, IToolRegistry)


def test_rc_overrides_applied() -> None:
    runtime = build_in_memory_runtime(rc_overrides={"max_iterations": 100})
    assert runtime.rc.max_iterations == 100


async def test_llm_provider_scripts_response() -> None:
    runtime = build_in_memory_runtime()
    runtime.llm.queue_response(text="hello world", stop_reason=StopReason.end_turn)

    from protocore.contracts.llm import LLMRequest
    from protocore.contracts.types import Message, MessageRole

    request = LLMRequest(
        model="qwen3.6-35b-a3b",
        messages=[Message(role=MessageRole.user, content_blocks=[])],
    )
    events = []
    async for event in runtime.llm.stream_with_tools(request):
        events.append(event)

    names = [e.name for e in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert any(e.name == "content_block_delta" for e in events)
