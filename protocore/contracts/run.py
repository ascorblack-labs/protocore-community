"""IRunStore Protocol — durable per-run state.

Reference shape: a relational summary row paired with an object-store
detail blob.

``flush_terminal`` is called once a run reaches a terminal status
(completed/error/cancelled) to write the S3 blob and update the PG row.

``IRunToolErrorCounter`` is the narrow protocol the core
:class:`~protocore.runtime.tool_dispatch.ToolDispatcher` uses to record
per-run tool dispatch errors. Reference shape: one atomic
``UPDATE runs SET tool_errors_count = tool_errors_count + N``. Keeping
the protocol in core lets the dispatcher remain host-agnostic while
still surfacing the per-run counter for the terminal classifier.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from protocore.contracts.types import Run, RunStatus


class RunStoreError(Exception):
    """Base for run-store domain errors."""


class RunNotFoundError(RunStoreError):
    """Requested run_id does not exist."""


@runtime_checkable
class IRunStore(Protocol):
    """Adapter Protocol over the PG ``runs`` table."""

    async def create(self, run: Run) -> None:
        """Create a new run row."""
        ...

    async def get(self, run_id: str, tenant_id: str) -> Run:
        """Fetch run row. Raise :class:`RunNotFoundError`."""
        ...

    async def update_status(self, run_id: str, tenant_id: str, status: RunStatus) -> None:
        """Update run status. Validates legal transitions."""
        ...

    async def list(
        self,
        tenant_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> Sequence[Run]:
        """List runs for a tenant (filtered, paginated)."""
        ...

    async def flush_terminal(
        self,
        run_id: str,
        tenant_id: str,
        detail_blob_ref: str,
    ) -> None:
        """Mark run as flushed to durable storage; bind detail blob ref."""
        ...


@runtime_checkable
class IRunToolErrorCounter(Protocol):
    """Adapter Protocol for atomic increments of ``runs.tool_errors_count``.

    Implementations MUST be atomic (single SQL ``UPDATE ... SET col = col + N``
    or equivalent), idempotent under concurrent calls, and non-raising — the
    dispatcher swallows counter failures so a flaky telemetry plane never
    corrupts the agent loop.
    """

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        """Increment ``runs.tool_errors_count`` for ``run_id`` by ``by``.

        ``by`` must be a positive integer; ``ValueError`` if zero or negative.
        Missing ``run_id`` is a no-op (the row may not exist yet during early
        bootstrap or in unit tests with mocked rows).
        """
        ...


__all__ = [
    "IRunStore",
    "IRunToolErrorCounter",
    "RunNotFoundError",
    "RunStoreError",
]
