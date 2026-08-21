"""Unit tests for :mod:`protocore.runtime.prompt_caching`.

Covers the breakpoint-placement pass at the algorithm level — invariants:

* Output is a fresh list; the input is never mutated.
* At most :data:`MAX_BREAKPOINTS` returned.
* System-at-0 + 3 trailing non-system messages = canonical placement.
* No system prompt → 4 trailing breakpoints (entire window).
* History shorter than 4 → one breakpoint per message.
* Deterministic: same input → same output (idempotent function).

"""
from __future__ import annotations

import copy

import pytest

from protocore.contracts.llm import CacheBreakpoint
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.prompt_caching import MAX_BREAKPOINTS, apply_system_and_3

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _system(text: str = "You are a helpful agent.") -> Message:
    return Message(role=MessageRole.system, content_blocks=[TextBlock(text=text)])


def _user(text: str = "hi") -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant_text(text: str = "ack") -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _assistant_tool_use(call_id: str = "call-1", name: str = "Bash") -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json='{"command": "ls"}',
            )
        ],
    )


def _tool(call_id: str = "call-1", content: str = "result") -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(tool_call_id=call_id, content=content, is_error=False)
        ],
    )


# ----------------------------------------------------------------------
# Happy path — canonical system_and_3 placement
# ----------------------------------------------------------------------


def test_system_and_3_canonical_placement() -> None:
    """system + 3 user/assistant turns → all 4 breakpoints placed.

    Indices: 0 (system), 1 (user), 2 (assistant), 3 (user). The
    canonical placement: ``[system, last-3-non-system]``.
    """
    messages = [
        _system(),
        _user("first"),
        _assistant_text("first reply"),
        _user("second"),
    ]
    breakpoints = apply_system_and_3(messages)

    assert len(breakpoints) == MAX_BREAKPOINTS
    assert [b.message_index for b in breakpoints] == [0, 1, 2, 3]
    assert breakpoints[0].rationale == "system_prefix"
    assert all(b.rationale == "trailing_message" for b in breakpoints[1:])


def test_default_cache_ttl_is_5m() -> None:
    """``cache_ttl`` defaults to ``"5m"`` (Anthropic explicit short TTL)."""
    breakpoints = apply_system_and_3([_system(), _user()])
    assert all(b.cache_control_type == "5m" for b in breakpoints)


@pytest.mark.parametrize("ttl", ["ephemeral", "5m", "1h"])
def test_cache_ttl_propagates_to_all_breakpoints(ttl: str) -> None:
    """Each breakpoint carries the requested ``cache_ttl``."""
    messages = [_system(), _user(), _assistant_text(), _user()]
    breakpoints = apply_system_and_3(messages, cache_ttl=ttl)  # type: ignore[arg-type]
    for bp in breakpoints:
        assert bp.cache_control_type == ttl


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_empty_messages_returns_empty_list() -> None:
    """Empty input → empty output, no errors."""
    assert apply_system_and_3([]) == []


def test_only_system_message_returns_single_breakpoint() -> None:
    """System-only history → one breakpoint at index 0."""
    breakpoints = apply_system_and_3([_system()])
    assert len(breakpoints) == 1
    assert breakpoints[0].message_index == 0
    assert breakpoints[0].rationale == "system_prefix"


def test_only_user_messages_no_system_uses_all_4_trailing() -> None:
    """No system prompt → all 4 breakpoints go to trailing non-system messages."""
    messages = [_user("a"), _user("b"), _user("c"), _user("d"), _user("e")]
    breakpoints = apply_system_and_3(messages)
    assert len(breakpoints) == MAX_BREAKPOINTS
    # Last 4 of 5 → indices 1, 2, 3, 4.
    assert [b.message_index for b in breakpoints] == [1, 2, 3, 4]
    assert all(b.rationale == "trailing_message" for b in breakpoints)


def test_short_history_less_than_4_messages() -> None:
    """History shorter than MAX_BREAKPOINTS → one breakpoint per message.

    Edge case: a 2-message history (system + first user turn) should still
    cache both, not fail with "not enough trailing".
    """
    messages = [_system(), _user("hello")]
    breakpoints = apply_system_and_3(messages)
    assert len(breakpoints) == 2
    assert [b.message_index for b in breakpoints] == [0, 1]


def test_single_user_message_no_system() -> None:
    """Single non-system message → one trailing breakpoint."""
    breakpoints = apply_system_and_3([_user("hi")])
    assert len(breakpoints) == 1
    assert breakpoints[0].message_index == 0
    assert breakpoints[0].rationale == "trailing_message"


# ----------------------------------------------------------------------
# Edge case — tool-heavy session
# ----------------------------------------------------------------------


def test_tool_heavy_session_trailing_3() -> None:
    """Tool-use + tool-result sequences are treated as regular messages.

    A turn with [system, user, assistant(tool_use), tool, assistant(text)]
    yields breakpoints at [0, 2, 3, 4]. The algorithm doesn't
    special-case tool messages — they participate in the trailing-3
    window like any other non-system message.
    """
    messages = [
        _system(),
        _user("run ls"),
        _assistant_tool_use("call-1"),
        _tool("call-1", "file1\nfile2"),
        _assistant_text("Done."),
    ]
    breakpoints = apply_system_and_3(messages)
    assert len(breakpoints) == MAX_BREAKPOINTS
    # System at 0; trailing 3 non-system = indices 2, 3, 4.
    assert [b.message_index for b in breakpoints] == [0, 2, 3, 4]


def test_long_history_keeps_only_last_3_plus_system() -> None:
    """100-message history → breakpoints at [0, 97, 98, 99]."""
    messages = [_system()]
    for i in range(99):
        messages.append(_user(f"turn-{i}") if i % 2 == 0 else _assistant_text(f"r-{i}"))
    breakpoints = apply_system_and_3(messages)
    assert len(breakpoints) == MAX_BREAKPOINTS
    assert [b.message_index for b in breakpoints] == [0, 97, 98, 99]


# ----------------------------------------------------------------------
# Edge case — skills-only-no-tools
# ----------------------------------------------------------------------


def test_skills_only_no_tool_use_blocks() -> None:
    """Pure text turns with no tool-use blocks → standard system_and_3.

    Skills are injected as user/system text (per AGENTS.md "Skills inject
    as user-message"). The breakpoint placement is unaffected by content
    block kind — only role + position matter.
    """
    messages = [
        _system("System: with skill index block"),
        _user("Use the foo skill."),
        _assistant_text("Calling foo..."),
        _user("Now do bar."),
        _assistant_text("Done."),
    ]
    breakpoints = apply_system_and_3(messages)
    assert len(breakpoints) == MAX_BREAKPOINTS
    # System at 0; last 3 non-system = 2, 3, 4.
    assert [b.message_index for b in breakpoints] == [0, 2, 3, 4]


# ----------------------------------------------------------------------
# Immutability invariant
# ----------------------------------------------------------------------


def test_input_messages_are_not_mutated() -> None:
    """The function MUST NOT mutate the input list or its elements.

    An implementation that rewrites `cache_control` in place needs a
    defensive `copy.deepcopy`. This one returns indices only — no mutation
    is possible — but the invariant is asserted anyway.
    """
    messages = [
        _system(),
        _user("first"),
        _assistant_text("ack"),
        _user("second"),
    ]
    snapshot = copy.deepcopy(messages)
    _ = apply_system_and_3(messages)
    assert messages == snapshot, "input mutated"


def test_returns_new_list_each_call() -> None:
    """Each call returns a fresh list — caller is free to mutate it."""
    messages = [_system(), _user()]
    a = apply_system_and_3(messages)
    b = apply_system_and_3(messages)
    assert a == b
    assert a is not b  # different list objects


# ----------------------------------------------------------------------
# Determinism — same input → same output
# ----------------------------------------------------------------------


def test_deterministic_same_input_same_output() -> None:
    """Idempotent function — repeat calls with identical input return identical output."""
    messages = [
        _system("Stable system prompt."),
        _user("first"),
        _assistant_text("reply-1"),
        _user("second"),
        _assistant_text("reply-2"),
        _user("third"),
    ]
    out_a = apply_system_and_3(messages)
    out_b = apply_system_and_3(messages)
    out_c = apply_system_and_3(messages)
    assert out_a == out_b == out_c


def test_breakpoints_are_sorted_by_message_index() -> None:
    """Output ordering invariant — message_index ascending, no duplicates."""
    messages = [
        _system(),
        _user("a"),
        _assistant_text("b"),
        _user("c"),
        _assistant_text("d"),
    ]
    breakpoints = apply_system_and_3(messages)
    indices = [b.message_index for b in breakpoints]
    assert indices == sorted(indices), "breakpoints must be sorted by message_index"
    assert len(indices) == len(set(indices)), "no duplicate message_index"


# ----------------------------------------------------------------------
# Type / schema invariants
# ----------------------------------------------------------------------


def test_returns_frozen_dataclass_instances() -> None:
    """Each breakpoint is a :class:`CacheBreakpoint` (frozen dataclass)."""
    breakpoints = apply_system_and_3([_system(), _user()])
    for bp in breakpoints:
        assert isinstance(bp, CacheBreakpoint)
        # frozen=True — attempting to mutate raises.
        with pytest.raises((AttributeError, Exception)):
            bp.message_index = 999  # type: ignore[misc]


def test_max_breakpoints_constant_is_4() -> None:
    """Anthropic hard-cap: 4 explicit cache_control blocks per request."""
    assert MAX_BREAKPOINTS == 4
