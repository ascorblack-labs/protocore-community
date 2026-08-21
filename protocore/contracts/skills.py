"""ISkillStore Protocol — skill catalog adapter.

Reference shape: relational rows with a vector index for retrieval and an
object-store blob for the body.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import SkillManifest

# Canonical bundle-relative path for the skill entry point — every
# multi-file skill bundle must contain exactly one row at this path.
# A legacy single-file skill projects as a single
# ``SkillFileRef(path=SKILL_ENTRY_PATH, ...)`` until the bundle is
# rewritten through the import surface.
SKILL_ENTRY_PATH: Final[str] = "SKILL.md"

# MIME type for the canonical entry-point row. The import surface will sniff
# real MIME types per file; for the lazy backward path the synthesis is always
# ``text/markdown`` because legacy ``skills.body_md`` is markdown by contract.
SKILL_ENTRY_MIME_TYPE: Final[str] = "text/markdown"


class SkillStoreError(Exception):
    """Base for skill-store domain errors."""


class SkillNotFoundError(SkillStoreError):
    """Raised when a skill cannot be resolved by id/name."""


class SkillConflictError(SkillStoreError):
    """Raised when a skill name collides with an existing account skill."""


class SkillIndexEntry(BaseModel):
    """Per-skill index entry (no body — fetch via :meth:`ISkillStore.load`)."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    enabled: bool = True


class SkillBundle(BaseModel):
    """Full skill payload (manifest + body markdown)."""

    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    body: str


class SkillUpsertInput(BaseModel):
    """CRUD payload for create / update via ``ISkillStore.upsert``."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    body_md: str
    enabled: bool = True


class SkillFileRef(BaseModel):
    """Index entry for one file in a multi-file skill bundle.

    A skill bundle is a directory tree containing :data:`SKILL_ENTRY_PATH` at
    the top level plus any number of helper files (helper scripts, sub-prompts,
    docs, examples). ``path`` is bundle-relative (no leading slash, no ``..``).
    The body bytes live in :class:`~protocore.contracts.blob.IBlobStore` and are
    fetched via :meth:`ISkillStore.load_file`.
    """

    model_config = ConfigDict(frozen=True)

    path: str = Field(
        description=(
            "Bundle-relative path (e.g. ``SKILL.md`` or ``helpers/util.py``)."
        ),
    )
    size_bytes: int = Field(ge=0)
    mime_type: str
    content_hash: str = Field(
        description="Lowercase hex SHA-256 of the file body bytes.",
    )


@runtime_checkable
class ISkillStore(Protocol):
    """Adapter Protocol over the skill catalog.

    Supports CRUD + body load + ``list_files`` / ``load_file``
    for multi-file skill bundle support.
    """

    async def list(self, tenant_id: str) -> Sequence[SkillIndexEntry]:
        """Return the account's skill index."""
        ...

    async def load(self, tenant_id: str, skill_id: str) -> SkillBundle:
        """Fetch full skill manifest + body markdown."""
        ...

    async def upsert(self, tenant_id: str, manifest: SkillManifest, body: str) -> None:
        """Create or update an account skill (body stored as blob)."""
        ...

    async def create(self, tenant_id: str, payload: SkillUpsertInput) -> SkillIndexEntry:
        """Insert a new account skill."""
        ...

    async def update(
        self,
        tenant_id: str,
        skill_id: str,
        payload: SkillUpsertInput,
    ) -> SkillIndexEntry:
        """Update an existing account skill."""
        ...

    async def delete(self, tenant_id: str, skill_id: str) -> None:
        """Delete an account skill by id."""
        ...

    async def set_enabled(self, tenant_id: str, skill_id: str, *, enabled: bool) -> None:
        """Soft toggle skill enabled flag."""
        ...

    async def list_subset(
        self,
        tenant_id: str,
        names: Sequence[str],
    ) -> Sequence[SkillIndexEntry]:
        """Return entries matching the given skill names.

        Used by subagent dispatch to enforce skill whitelist.

        NB: this whitelist-resolution path deliberately ignores the
        ``enabled`` flag (a whitelisted skill resolves regardless of toggle
        state). Prompt-surfacing callers (project pins) MUST use
        :meth:`list_enabled_subset` so a disabled skill is not resurfaced.
        """
        ...

    async def list_enabled_subset(
        self,
        tenant_id: str,
        names: Sequence[str],
    ) -> Sequence[SkillIndexEntry]:
        """Return ENABLED entries matching the given skill names.

        Identical resolution to :meth:`list_subset` (same account-scoped
        name matching) but additionally drops any entry whose ``enabled``
        flag is false. Used for prompt surfacing of project-pinned skills so
        an operator disabling a skill in the account-wide bank cannot be
        overridden by a stale project pin. ``list_subset`` keeps its
        whitelist-oriented "resolve regardless of toggle" semantics for the
        existing subagent-dispatch caller.
        """
        ...

    async def list_files(
        self,
        tenant_id: str,
        skill_id: str,
    ) -> Sequence[SkillFileRef]:
        """List every file in the skill bundle as ``SkillFileRef`` rows.

        Returns at minimum the canonical :data:`SKILL_ENTRY_PATH` entry.
        For legacy single-file skills the implementation MAY synthesise this
        entry lazily from the legacy ``body_md`` column. Implementations MUST
        NOT silently fall back across multiple sources; the lazy synthesis is
        the single documented bridge, not a chain.
        """
        ...

    async def load_file(
        self,
        tenant_id: str,
        skill_id: str,
        path: str,
    ) -> bytes | None:
        """Load the raw bytes for one file in the bundle, or ``None`` if absent.

        ``path`` is bundle-relative; the canonical entry point is
        :data:`SKILL_ENTRY_PATH`. ``None`` signals "no such file in this
        bundle" (callers can choose to 404 / fall back); the implementation
        must NOT raise :class:`SkillNotFoundError` when the skill exists but
        the path is unknown.
        """
        ...


__all__ = [
    "SKILL_ENTRY_MIME_TYPE",
    "SKILL_ENTRY_PATH",
    "ISkillStore",
    "SkillBundle",
    "SkillConflictError",
    "SkillFileRef",
    "SkillIndexEntry",
    "SkillNotFoundError",
    "SkillStoreError",
    "SkillUpsertInput",
]
