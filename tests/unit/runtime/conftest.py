"""Fixtures for runtime tests.

Provides ``build_in_memory_engine`` helper that wires :class:`QueryEngine`
against the in-memory adapter doubles.
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.types import ToolPrecondition
from protocore.contracts.verification import VerificationDelivery
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
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


@pytest.fixture
def in_memory_runtime() -> dict[str, object]:
    """Return a dict of fresh in-memory mocks for the 12 interfaces."""
    return {
        "llm": InMemoryLLMProvider(),
        "tools": InMemoryToolRegistry(),
        "events": InMemoryEventStream(),
        "hooks": InMemoryHookManager(),
        "skills": InMemorySkillStore(),
        "blobs": InMemoryBlobStore(),
        "agents": InMemoryAgentDispatch(),
        "sessions": InMemorySessionStore(),
        "runs": InMemoryRunStore(),
        "search": InMemorySearchIndex(),
        "todos": InMemoryTodoStorage(),
    }


@pytest.fixture
def engine_factory(in_memory_runtime: dict[str, object]):
    """Return a ``build_engine(**overrides)`` callable for tests."""

    def build_engine(
        *,
        run_id: str = "run-test",
        tenant_id: str = "tenant-test",
        session_id: str = "sess-test",
        model_name: str = "qwen3.6-35b-a3b",
        account_id: str | None = None,
        rc: RuntimeConstants | None = None,
        expected_terminal_tool: str | None = None,
        root_run_id: str = "",
        parent_run_id: str | None = None,
        subagent_id: str | None = None,
        run_depth: int | None = None,
        pinned_skill_names: frozenset[str] = frozenset(),
        tool_preconditions: tuple[ToolPrecondition, ...] = (),
        verification_delivery: VerificationDelivery | None = None,
    ) -> QueryEngine:
        # A root run sits at depth 0 and a child one hop below it, which is
        # what a caller naming a parent almost always wants; tests that build a
        # deeper tree, or that exercise the depth binding itself, say so.
        if run_depth is None:
            run_depth = 0 if parent_run_id is None else 1
        return QueryEngine(
            config=QueryEngineConfig(
                run_id=run_id,
                tenant_id=tenant_id,
                # Default the account id to the scope id so callers that only
                # care about the scope==account case stay terse; the
                # scope≠account skill-resolution path passes account_id
                # explicitly.
                account_id=tenant_id if account_id is None else account_id,
                session_id=session_id,
                model_name=model_name,
                rc=rc or RuntimeConstants(model_context_window=4_096),
                expected_terminal_tool=expected_terminal_tool,
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
                subagent_id=subagent_id,
                run_depth=run_depth,
                pinned_skill_names=pinned_skill_names,
                tool_preconditions=tool_preconditions,
                verification_delivery=verification_delivery,
            ),
            llm_provider=in_memory_runtime["llm"],
            tool_registry=in_memory_runtime["tools"],
            event_stream=in_memory_runtime["events"],
            hook_manager=in_memory_runtime["hooks"],
            skill_store=in_memory_runtime["skills"],
            blob_store=in_memory_runtime["blobs"],
        )

    return build_engine
