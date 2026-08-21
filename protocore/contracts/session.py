"""ISessionStore Protocol — multi-turn conversation persistence.

Reference shape: a relational store, one row per message.

Sessions are NEVER deleted. Messages are cursored — ``list_messages`` returns
items strictly newer than ``since`` (excludes the cursor message itself).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from protocore.contracts.types import Message, Session


class SessionStoreError(Exception):
    """Base for session-store domain errors."""


class SessionNotFoundError(SessionStoreError):
    """Requested session_id does not exist."""


@runtime_checkable
class ISessionStore(Protocol):
    """Adapter Protocol over PG sessions + messages tables."""

    async def create(self, session: Session) -> None:
        """Create a new session. Idempotent on ``session.id``."""
        ...

    async def get(self, session_id: str, tenant_id: str) -> Session:
        """Fetch session metadata. Raise :class:`SessionNotFoundError`."""
        ...

    async def append_message(self, session_id: str, tenant_id: str, message: Message) -> None:
        """Append a single message to the session log."""
        ...

    async def list_messages(
        self,
        session_id: str,
        tenant_id: str,
        *,
        since: datetime | None = None,
        limit: int = 200,
    ) -> Sequence[Message]:
        """List messages in chronological order. ``since`` is exclusive."""
        ...


__all__ = ["ISessionStore", "SessionNotFoundError", "SessionStoreError"]
