"""In-memory fixtures for testing pure-core wire-up.

This module ships with the core package (NOT under ``tests/``) so that
downstream phases can ``from protocore.tests_support import build_in_memory_runtime``
without depending on test-collection layout.

"""
from __future__ import annotations

from protocore.tests_support.adapters import (
    InMemoryAgentDispatch,
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemoryMemory,
    InMemoryRunStore,
    InMemorySearchIndex,
    InMemorySessionStore,
    InMemorySkillStore,
    InMemoryTodoStorage,
    InMemoryToolRegistry,
)
from protocore.tests_support.runtime import InMemoryRuntime, build_in_memory_runtime

__all__ = [
    "InMemoryAgentDispatch",
    "InMemoryBlobStore",
    "InMemoryEventStream",
    "InMemoryHookManager",
    "InMemoryLLMProvider",
    "InMemoryMemory",
    "InMemoryRunStore",
    "InMemoryRuntime",
    "InMemorySearchIndex",
    "InMemorySessionStore",
    "InMemorySkillStore",
    "InMemoryTodoStorage",
    "InMemoryToolRegistry",
    "build_in_memory_runtime",
]
