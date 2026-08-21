"""Pluggy hook specifications — 8 events.

Implementations register via :func:`hookimpl`; the manager invokes them
via :class:`HookManager.invoke`.
"""
from __future__ import annotations

from typing import Any

import pluggy

PROJECT_NAME = "protocore"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class AgentHookSpecs:
    """The 8 typed hook events.

    All hook implementations are coroutines returning ``dict[str, Any]``
    (or ``None`` if no opinion). The manager aggregates per event.
    """

    @hookspec
    async def pre_tool_use(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired before a tool is invoked. May veto (return ``{'action': 'deny'}``)."""

    @hookspec
    async def post_tool_use(
        self,
        tool_name: str,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired after a tool returns."""

    @hookspec
    async def user_prompt_submit(
        self,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired when a user message is appended to a session."""

    @hookspec
    async def session_start(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired at session creation."""

    @hookspec
    async def session_end(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired at session termination."""

    @hookspec
    async def pre_compact(
        self,
        plan: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired before a compaction pass executes. May veto."""

    @hookspec
    async def post_compact(
        self,
        summary: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired after a compaction pass completes."""

    @hookspec
    async def file_changed(
        self,
        path: str,
        kind: str,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fired when a workspace file is created / updated / deleted."""


__all__ = ["PROJECT_NAME", "AgentHookSpecs", "hookimpl", "hookspec"]
