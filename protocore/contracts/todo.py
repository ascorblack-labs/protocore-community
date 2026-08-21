"""ITodoStorage Protocol — TodoWrite tool persistence.

Reference shape: a relational store keyed by session
(Redis hot for active session + PG durable on session close).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from protocore.contracts.types import Todo


class TodoStorageError(Exception):
    """Base for todo-storage domain errors."""


@runtime_checkable
class ITodoStorage(Protocol):
    """Adapter Protocol over the per-session todo list."""

    async def read(self, session_id: str, tenant_id: str) -> Sequence[Todo]:
        """Read the current todo list for a session (chronological)."""
        ...

    async def write(
        self,
        session_id: str,
        tenant_id: str,
        todos: Sequence[Todo],
    ) -> None:
        """Overwrite the session's todo list (TodoWrite semantics)."""
        ...


__all__ = ["ITodoStorage", "TodoStorageError"]
