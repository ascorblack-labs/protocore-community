"""Regression test for :func:`protocore.tools.tool` decorator.

An async tool function with a non-``context`` parameter that lacks an
annotation must be rejected at decoration time. Before the fix,
``_build_schema`` silently skipped the unannotated parameter, so it was
absent from the JSON schema + ``required`` list while remaining a required
positional argument of the callable. A caller validating ``arguments``
against the (incomplete) schema then produced a dict missing the key, and
``invoke`` raised ``TypeError: ... missing 1 required positional argument``
only on first invocation -- violating the ``schema matches the actual
callable`` invariant.
"""
from __future__ import annotations

import pytest

from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolResult
from protocore.tools import tool


def test_unannotated_param_raises_at_decoration_time() -> None:
    with pytest.raises(TypeError, match="must be annotated"):

        @tool(name="noanno", description="x")
        async def _f(  # type: ignore[no-untyped-def]
            context: ToolContext, text: str, extra
        ) -> ToolResult:
            return ToolResult(tool_call_id="t", content="")


def test_fully_annotated_tool_still_decorates() -> None:
    @tool(name="ok", description="x")
    async def _f(context: ToolContext, text: str) -> ToolResult:
        return ToolResult(tool_call_id="t", content=text)

    instance = _f()
    assert instance.name == "ok"
    assert "text" in instance.definition.parameters.required
