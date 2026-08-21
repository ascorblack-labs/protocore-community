"""Shared Tool fixtures for tool-subsystem tests.

A ``MockTool`` family that exercises the :class:`Tool` ABC without
pulling in any the host plumbing. Each variant is scriptable from a
test (controlled exception / delay / output content).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from protocore.contracts.tools import Tool, ToolContext, ToolPolicyDenied
from protocore.contracts.types import ToolDefinition, ToolParameterSchema, ToolResult


@dataclass
class MockTool(Tool):
    """Programmable :class:`Tool` for unit tests.

 Parameters
 ----------
 tool_name:
 Stable name surfaced in registry lookups + LLM context.
 description:
 Human-readable description (also feeds BM25 search corpus).
 response_content:
 Body of the :class:`ToolResult.content` returned on success.
 response_is_error:
 Forces ``is_error=True`` on the response.
 raise_exception:
 Replaces the response with raising ``raise_exception``.
 sleep_seconds:
 Delay before returning (used by timeout tests).
 on_invoke:
 Optional callback invoked with the args dict — lets a test
 inspect / mutate state on each call.
 side_effect_class:
 Surfaces a ClassVar-style attribute that the permission gate
 reads via ``getattr(tool, "side_effect_class", None)``.
 parameters_schema:
 Optional JSON-Schema dict to inject into the
 :class:`ToolDefinition.parameters`.
 response_metadata:
 Optional dict stamped onto :class:`ToolResult.metadata`. Lets a test
 simulate a tool that classifies a soft ``is_error`` result via the
 ``count_as_tool_error`` / ``consecutive_error_cap_eligible`` flags
 without standing up the host ``TypedTool``.
 """

    tool_name: str = "Mock"
    description: str = "Mock tool"
    always_load: bool = False
    search_hint: str = ""
    response_content: str = "ok"
    response_is_error: bool = False
    raise_exception: BaseException | None = None
    sleep_seconds: float = 0.0
    on_invoke: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None
    side_effect_class: str | None = None
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(
                properties=self.parameters_schema or {"v": {"type": "string"}},
            ),
        )

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        if self.on_invoke is not None:
            result = self.on_invoke(dict(arguments))
            if asyncio.iscoroutine(result):
                await result
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_exception is not None:
            raise self.raise_exception
        return ToolResult(
            tool_call_id="",
            content=self.response_content,
            is_error=self.response_is_error,
            metadata=dict(self.response_metadata),
        )


@dataclass
class PolicyDeniedTool(MockTool):
    """Variant that always raises :class:`ToolPolicyDenied`."""

    raise_exception: BaseException | None = field(
        default_factory=lambda: ToolPolicyDenied("policy denied by adapter")
    )


def make_default_ctx(
    *,
    run_id: str = "run-1",
    tenant_id: str = "tenant-1",
    session_id: str = "sess-1",
) -> ToolContext:
    """Convenience: build a minimal :class:`ToolContext` for tests."""
    return ToolContext(
        run_id=run_id,
        tenant_id=tenant_id,
        session_id=session_id,
    )


__all__ = ["MockTool", "PolicyDeniedTool", "make_default_ctx"]
