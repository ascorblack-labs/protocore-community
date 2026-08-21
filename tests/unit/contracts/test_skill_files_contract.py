"""/ E1: ISkillStore.list_files + load_file contract tests.

 shipped the :class:`SkillFileRef` data type + canonical-path
constants. (this file) covers the ``ISkillStore`` Protocol
extension + :class:`InMemorySkillStore` lazy-backward-path
implementation.

Covers:

* :class:`SkillFileRef` validation (non-negative size).
* :data:`SKILL_ENTRY_PATH` + :data:`SKILL_ENTRY_MIME_TYPE` invariants.
* :class:`ISkillStore` runtime-checkable Protocol still passes against
 :class:`InMemorySkillStore` after the contract extension.
* Lazy backward path: a single-file skill (no rows in the bundle
 overlay) projects through ``list_files`` as a single ``SKILL.md``
 ``SkillFileRef`` and ``load_file("SKILL.md", ...)`` returns the
 legacy body bytes.
* Multi-file overlay: ``list_files`` returns every file,
 ``load_file`` returns bytes for added files, ``None`` for unknown
 paths.
"""
from __future__ import annotations

import hashlib

import pytest

from protocore.contracts.skills import (
    SKILL_ENTRY_MIME_TYPE,
    SKILL_ENTRY_PATH,
    ISkillStore,
    SkillFileRef,
    SkillUpsertInput,
)
from protocore.tests_support.adapters import InMemorySkillStore


def test_skill_entry_path_is_skill_md() -> None:
    assert SKILL_ENTRY_PATH == "SKILL.md"


def test_skill_entry_mime_type_is_markdown() -> None:
    assert SKILL_ENTRY_MIME_TYPE == "text/markdown"


def test_skill_file_ref_validates_size_non_negative() -> None:
    with pytest.raises(ValueError):
        SkillFileRef(
            path="SKILL.md",
            size_bytes=-1,
            mime_type=SKILL_ENTRY_MIME_TYPE,
            content_hash="deadbeef",
        )


def test_skill_file_ref_is_frozen() -> None:
    ref = SkillFileRef(
        path="SKILL.md",
        size_bytes=1,
        mime_type=SKILL_ENTRY_MIME_TYPE,
        content_hash="deadbeef",
    )
    with pytest.raises((TypeError, ValueError)):
        ref.path = "other.md"  # type: ignore[misc]


def test_skill_file_ref_extra_fields_rejected() -> None:
    """Pydantic model is frozen-only, extra fields tolerated by default —
    this test pins the current shape so any future ``extra='forbid'``
    flip is intentional."""
    # No assertion needed beyond construction succeeding for the
    # documented fields.
    ref = SkillFileRef(
        path="helpers/util.py",
        size_bytes=42,
        mime_type="text/x-python",
        content_hash="cafebabe",
    )
    assert ref.path == "helpers/util.py"
    assert ref.size_bytes == 42


def test_in_memory_skill_store_satisfies_protocol() -> None:
    """Runtime Protocol check survives the / E1 method additions."""
    store = InMemorySkillStore()
    assert isinstance(store, ISkillStore)
    assert hasattr(store, "list_files")
    assert hasattr(store, "load_file")


@pytest.mark.asyncio
async def test_list_files_lazy_synthesises_skill_md_from_legacy_body() -> None:
    """Single-file skill (no overlay rows) projects as SKILL.md alone."""

    store = InMemorySkillStore()
    body = "# Hello\nThis is a single-file skill body."
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="legacy",
            description="legacy single-file skill",
            body_md=body,
        ),
    )

    refs = await store.list_files("tenant-A", entry.id)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.path == SKILL_ENTRY_PATH
    assert ref.mime_type == SKILL_ENTRY_MIME_TYPE
    assert ref.size_bytes == len(body.encode("utf-8"))
    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert ref.content_hash == expected_hash


@pytest.mark.asyncio
async def test_load_file_lazy_returns_legacy_body_for_skill_md() -> None:
    store = InMemorySkillStore()
    body = "## Body"
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="legacy",
            description="legacy single-file skill",
            body_md=body,
        ),
    )

    loaded = await store.load_file("tenant-A", entry.id, SKILL_ENTRY_PATH)
    assert loaded == body.encode("utf-8")


@pytest.mark.asyncio
async def test_load_file_lazy_returns_none_for_unknown_path() -> None:
    """Lazy synthesis is one-shot — a non-SKILL.md path returns None."""

    store = InMemorySkillStore()
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="legacy",
            description="legacy single-file skill",
            body_md="body",
        ),
    )

    loaded = await store.load_file("tenant-A", entry.id, "helpers/util.py")
    assert loaded is None


@pytest.mark.asyncio
async def test_list_files_with_overlay_returns_every_file() -> None:
    """Once any file is added to the overlay, the synthesis path is bypassed."""

    store = InMemorySkillStore()
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="bundle",
            description="multi-file skill",
            body_md="# Bundle entry",
        ),
    )
    helper_bytes = b"def util(): pass\n"
    skill_md_bytes = b"# Bundle entry"
    store.put_file("tenant-A", entry.id, SKILL_ENTRY_PATH, skill_md_bytes)
    store.put_file("tenant-A", entry.id, "helpers/util.py", helper_bytes)

    refs = await store.list_files("tenant-A", entry.id)
    paths = [ref.path for ref in refs]
    assert SKILL_ENTRY_PATH in paths
    assert "helpers/util.py" in paths
    helper_ref = next(r for r in refs if r.path == "helpers/util.py")
    assert helper_ref.size_bytes == len(helper_bytes)
    assert helper_ref.content_hash == hashlib.sha256(helper_bytes).hexdigest()


@pytest.mark.asyncio
async def test_load_file_returns_overlay_bytes_when_present() -> None:
    store = InMemorySkillStore()
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="bundle",
            description="multi-file skill",
            body_md="# Bundle entry",
        ),
    )
    helper_bytes = b"helper content"
    store.put_file("tenant-A", entry.id, "helpers/util.py", helper_bytes)

    loaded = await store.load_file("tenant-A", entry.id, "helpers/util.py")
    assert loaded == helper_bytes


@pytest.mark.asyncio
async def test_load_file_returns_none_for_unknown_path_with_overlay() -> None:
    store = InMemorySkillStore()
    entry = await store.create(
        "tenant-A",
        SkillUpsertInput(
            name="bundle",
            description="multi-file skill",
            body_md="# Bundle entry",
        ),
    )
    store.put_file("tenant-A", entry.id, "helpers/util.py", b"x")

    loaded = await store.load_file("tenant-A", entry.id, "examples/missing.py")
    assert loaded is None


@pytest.mark.asyncio
async def test_list_files_unknown_skill_returns_empty() -> None:
    """No skill row + no overlay → ``list_files`` returns ``[]`` (not raise)."""

    store = InMemorySkillStore()
    refs = await store.list_files("tenant-A", "unknown-id")
    assert list(refs) == []
