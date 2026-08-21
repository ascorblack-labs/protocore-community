"""IBlobStore ABC + error hierarchy.

Reference shape: any S3-compatible object store (S3, MinIO, Ceph).
Stateful contract — use ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from protocore.contracts.types import BlobMetadata


class BlobStoreError(Exception):
    """Base class for blob-store domain errors."""


class BlobNotFoundError(BlobStoreError):
    """Requested blob ref is unknown."""


class BlobStoreUnavailableError(BlobStoreError):
    """Backing store is temporarily unreachable."""


class IBlobStore(ABC):
    """Content-addressed blob store.

    Contract:
        - ``put`` is idempotent on (tenant_id, content) — same input yields
          same SHA-256 and the same ref.
        - ``get`` raises :class:`BlobNotFoundError` on missing ref.
        - ``delete`` returns ``True`` if removed, ``False`` if absent.
        - Implementations enforce tenant isolation: ``ref`` carries the
          tenant prefix and ``get`` MUST validate that the caller's
          ``tenant_id`` matches before returning bytes.
    """

    @abstractmethod
    async def put(
        self,
        tenant_id: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> BlobMetadata:
        """Store ``content`` and return :class:`BlobMetadata`."""

    @abstractmethod
    async def get(self, tenant_id: str, ref: str) -> bytes:
        """Retrieve full blob bytes by ``ref``."""

    @abstractmethod
    def get_stream(self, tenant_id: str, ref: str) -> AsyncIterator[bytes]:
        """Stream blob bytes in chunks.

        Implementations are async generators; declared here without
        ``async def`` so the Protocol typechecks against async iterators.
        """

    @abstractmethod
    async def head(self, tenant_id: str, ref: str) -> BlobMetadata:
        """Return metadata without fetching content."""

    @abstractmethod
    async def exists(self, tenant_id: str, ref: str) -> bool:
        """Return ``True`` if blob exists."""

    @abstractmethod
    async def delete(self, tenant_id: str, ref: str) -> bool:
        """Delete blob. Return ``True`` if removed; ``False`` if absent."""

    @abstractmethod
    async def list_prefix(
        self,
        tenant_id: str,
        prefix: str,
        *,
        limit: int = 1000,
    ) -> list[BlobMetadata]:
        """List blobs by key prefix (paginated)."""


__all__ = [
    "BlobNotFoundError",
    "BlobStoreError",
    "BlobStoreUnavailableError",
    "IBlobStore",
]
