"""Tool decorator helpers + core-defined tools.

Concrete adapter tools (Read/Write/Edit/Bash/...) live in the host; the
core package owns only the :class:`Tool` ABC (in
:mod:`protocore.contracts.tools`), the :func:`tool` decorator helper, and
the small set of tools whose entire surface is the protocol contract
itself (``AskUser`` — pauses the loop, no infra handles needed).
"""
from __future__ import annotations

from protocore.tools.ask_user import (
    ASK_USER_TOOL_NAME,
    AskUserInput,
    AskUserOutput,
    AskUserPauseRequested,
    AskUserTool,
)
from protocore.tools.decorator import tool
from protocore.tools.memory import (
    FORGET_TOOL_NAME,
    MEMORY_TOOL_NAMES,
    RECALL_TOOL_NAME,
    REMEMBER_TOOL_NAME,
    ForgetInput,
    ForgetTool,
    RecallInput,
    RecallTool,
    RememberInput,
    RememberTool,
    build_memory_tools,
)

__all__ = [
    "ASK_USER_TOOL_NAME",
    "FORGET_TOOL_NAME",
    "MEMORY_TOOL_NAMES",
    "RECALL_TOOL_NAME",
    "REMEMBER_TOOL_NAME",
    "AskUserInput",
    "AskUserOutput",
    "AskUserPauseRequested",
    "AskUserTool",
    "ForgetInput",
    "ForgetTool",
    "RecallInput",
    "RecallTool",
    "RememberInput",
    "RememberTool",
    "build_memory_tools",
    "tool",
]
