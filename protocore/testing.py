"""Convenience re-export from :mod:`protocore.tests_support`.

Real implementations live in :mod:`protocore.tests_support`. This module
gives consumers a shorter import path
(``from protocore.testing import build_in_memory_runtime``).
"""
from __future__ import annotations

from protocore.tests_support import (
    InMemoryAgentDispatch,
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryRunStore,
    InMemoryRuntime,
    InMemorySearchIndex,
    InMemorySessionStore,
    InMemorySkillStore,
    InMemoryTodoStorage,
    InMemoryToolRegistry,
    build_in_memory_runtime,
)

__all__ = [
    "InMemoryAgentDispatch",
    "InMemoryBlobStore",
    "InMemoryEventStream",
    "InMemoryHookManager",
    "InMemoryLLMProvider",
    "InMemoryRunStore",
    "InMemoryRuntime",
    "InMemorySearchIndex",
    "InMemorySessionStore",
    "InMemorySkillStore",
    "InMemoryTodoStorage",
    "InMemoryToolRegistry",
    "build_in_memory_runtime",
]
