"""ISearchIndex Protocol — pluggable search backend.

Reference shape: Postgres ``tsvector`` + ``pg_trgm`` + ``pgvector`` for a
single-database deployment, or a dedicated search cluster where one is
already available.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SearchError(Exception):
    """Base for search-index domain errors."""


class IndexDoc(BaseModel):
    """Document indexed for retrieval."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    tenant_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    embeddings: list[float] | None = None


class Hit(BaseModel):
    """Single result hit."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    score: float
    fields: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ISearchIndex(Protocol):
    """Pluggable search index (tenant-scoped)."""

    async def index(self, doc: IndexDoc) -> None:
        """Upsert a single document. Idempotent on ``doc_id``."""
        ...

    async def search(
        self,
        query: str,
        tenant_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> Sequence[Hit]:
        """Hybrid (lexical + vector) search; multi-tenant filter required."""
        ...

    async def delete(self, doc_id: str, tenant_id: str) -> bool:
        """Remove document; return ``True`` if deleted, ``False`` if absent."""
        ...


__all__ = ["Hit", "ISearchIndex", "IndexDoc", "SearchError"]
