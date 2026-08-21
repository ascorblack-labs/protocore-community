"""``build_in_memory_runtime`` — wire all 12 mocks into a Runtime dataclass.

Every loop test starts with::

 runtime = build_in_memory_runtime
 runtime.llm.queue_response(text="hi")
 ...

Zero host dependencies in core tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.events import EventBus
from protocore.runtime.runtime_constants import (
    StaticRuntimeConstantsProvider,
    default_runtime_constants,
)
from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryRunStore,
    InMemorySearchIndex,
    InMemorySessionStore,
    InMemorySkillStore,
    InMemoryTodoStorage,
    InMemoryToolRegistry,
)


@dataclass(slots=True)
class InMemoryRuntime:
    """Fully-wired in-memory runtime bundle for tests.

 All adapters are programmable mocks; the RC snapshot is fixed at
 construction time. 's QueryEngine will consume this same bundle.
 """

    tenant_id: str
    rc: RuntimeConstants
    rc_provider: StaticRuntimeConstantsProvider
    event_bus: EventBus
    llm: InMemoryLLMProvider
    blob: InMemoryBlobStore
    search: InMemorySearchIndex
    skills: InMemorySkillStore
    agents: InMemoryAgentDispatch
    sessions: InMemorySessionStore
    runs: InMemoryRunStore
    events: InMemoryEventStream
    hooks: InMemoryHookManager
    todos: InMemoryTodoStorage
    tools: InMemoryToolRegistry


def build_in_memory_runtime(
    *,
    tenant_id: str = "test-tenant",
    rc_overrides: dict[str, Any] | None = None,
) -> InMemoryRuntime:
    """Wire up the full in-memory runtime."""
    rc = default_runtime_constants(**(rc_overrides or {}))
    return InMemoryRuntime(
        tenant_id=tenant_id,
        rc=rc,
        rc_provider=StaticRuntimeConstantsProvider(rc),
        event_bus=EventBus(),
        llm=InMemoryLLMProvider(),
        blob=InMemoryBlobStore(),
        search=InMemorySearchIndex(),
        skills=InMemorySkillStore(),
        agents=InMemoryAgentDispatch(),
        sessions=InMemorySessionStore(),
        runs=InMemoryRunStore(),
        events=InMemoryEventStream(),
        hooks=InMemoryHookManager(),
        todos=InMemoryTodoStorage(),
        tools=InMemoryToolRegistry(),
    )


__all__ = ["InMemoryRuntime", "build_in_memory_runtime"]
