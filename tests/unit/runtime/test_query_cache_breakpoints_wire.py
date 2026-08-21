"""``query`` populates ``LLMRequest.extra["cache_breakpoints"]``.

The breakpoints come from :func:`protocore.runtime.prompt_caching.apply_system_and_3`
(pure function, tested separately in ``test_prompt_caching.py``). This
suite pins the *integration* — every assistant-turn stream the engine
opens carries a non-empty cache_breakpoint hint on
:attr:`LLMRequest.extra` so downstream Anthropic adapters can place
``cache_control`` markers.

Adapters that don't recognise the key (vLLM, OpenAI) ignore it; the
hint is forward-compatible.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import CacheBreakpoint
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
)
from protocore.runtime.events import TurnEvent


@pytest.mark.asyncio
async def test_cache_breakpoints_populated_on_first_turn(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """First assistant turn → ``LLMRequest.extra["cache_breakpoints"]`` non-empty."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="hello back")
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="hello")]
    )

    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    calls = in_memory_runtime["llm"].calls
    assert len(calls) >= 1
    request = calls[0]
    assert "cache_breakpoints" in request.extra
    breakpoints = request.extra["cache_breakpoints"]
    assert isinstance(breakpoints, list)
    # apply_system_and_3 caps at MAX_BREAKPOINTS=4.
    assert 0 < len(breakpoints) <= 4
    # Every entry is a CacheBreakpoint dataclass instance.
    for bp in breakpoints:
        assert isinstance(bp, CacheBreakpoint)
        assert bp.message_index >= 0


@pytest.mark.asyncio
async def test_cache_breakpoints_indices_in_range(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """Every breakpoint index addresses a message in ``LLMRequest.messages``."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="ok")
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="hi")]
    )
    async for _ in engine.run(user_msg):
        pass

    request = in_memory_runtime["llm"].calls[0]
    breakpoints = request.extra["cache_breakpoints"]
    n_messages = len(request.messages)
    for bp in breakpoints:
        assert 0 <= bp.message_index < n_messages, (
            f"breakpoint at {bp.message_index} is out of bounds "
            f"(messages count = {n_messages})"
        )


@pytest.mark.asyncio
async def test_cache_breakpoints_target_trailing_messages(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """``system_and_3`` strategy → at most 1 system + last 3 non-system."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(text="ok")
    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="hi")]
    )
    async for _ in engine.run(user_msg):
        pass

    request = in_memory_runtime["llm"].calls[0]
    breakpoints = request.extra["cache_breakpoints"]
    rationales = [bp.rationale for bp in breakpoints]
    # The first message in the request is the system-prompt prefix
    # (built by _prepend_system_sections); breakpoint 0 is the system
    # prefix, the rest are trailing messages.
    if request.messages and request.messages[0].role is MessageRole.system:
        assert "system_prefix" in rationales
    assert all(
        r in ("system_prefix", "trailing_message") for r in rationales
    )


@pytest.mark.asyncio
async def test_cache_breakpoints_recomputed_per_iteration(
    engine_factory: Any,
    in_memory_runtime: dict[str, Any],
) -> None:
    """Each provider stream gets its own freshly-computed breakpoint list.

    The hint is pure-function output of the current message history; as
    history grows (tool calls + results), the breakpoint indices shift
    to track the new last-3 messages. We assert that breakpoint indices
    on iteration 2 differ from iteration 1.
    """
    from protocore.contracts.hooks import HookActionKind, HookResult
    from protocore.contracts.types import HookEvent

    from ._tool_fixtures import MockTool

    engine = engine_factory()
    tool = MockTool(
        tool_name="EchoTool",
        description="echo",
        response_content="result-text",
    )
    in_memory_runtime["tools"].register(tool)
    in_memory_runtime["hooks"].queue_action(
        HookEvent.pre_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={}),
    )
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_x",
        tool_name="EchoTool",
        tool_input={"v": "in"},
    )
    in_memory_runtime["llm"].queue_response(text="all done")

    user_msg = Message(
        role=MessageRole.user, content_blocks=[TextBlock(text="go")]
    )
    async for _ in engine.run(user_msg):
        pass

    calls = in_memory_runtime["llm"].calls
    assert len(calls) == 2
    bp1 = calls[0].extra["cache_breakpoints"]
    bp2 = calls[1].extra["cache_breakpoints"]
    # Both lists are non-empty.
    assert bp1 and bp2
    # The second iteration has more messages (tool_use + tool_result),
    # so the trailing-3 window shifted; at least one index changed.
    indices_1 = [bp.message_index for bp in bp1]
    indices_2 = [bp.message_index for bp in bp2]
    assert indices_1 != indices_2, (
        "trailing-3 window must shift between iterations as history grows"
    )
