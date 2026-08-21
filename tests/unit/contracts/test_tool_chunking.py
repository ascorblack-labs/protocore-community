"""Unit tests for the shared chunkable-content-mutation predicate. This ONE
predicate is used by both core ``query.py`` and the host LLM client, so
the routing of a truncated tool call is identical in both layers.
"""
from __future__ import annotations

from protocore.contracts.tool_chunking import (
    CHUNKABLE_CONTENT_MUTATION_ALLOWLIST,
    is_chunkable_content_mutation,
)


def test_allowlisted_tools_with_required_content_qualify() -> None:
    for name in ("Write", "AppendFile"):
        assert is_chunkable_content_mutation(
            tool_name=name, required=["path", "content"], chunkable_flag=None
        ), name


def test_explicit_flag_qualifies_off_allowlist() -> None:
    """A per-tenant tool opts in via the flag without being on the allowlist."""
    assert is_chunkable_content_mutation(
        tool_name="TenantDoc", required=["target", "content"], chunkable_flag=True
    )


def test_dynamic_content_tool_without_flag_does_not_qualify() -> None:
    """The bug: a dynamic tool that merely REQUIRES ``content`` must NOT
    qualify unless it is allowlisted or explicitly flagged."""
    assert not is_chunkable_content_mutation(
        tool_name="PostComment", required=["target", "content"], chunkable_flag=None
    )
    assert not is_chunkable_content_mutation(
        tool_name="PostComment", required=["target", "content"], chunkable_flag=False
    )


def test_content_must_be_required_not_merely_present() -> None:
    """An OPTIONAL ``content`` (not in ``required``) is not the cut-body shape,
    even for an allowlisted name."""
    assert not is_chunkable_content_mutation(
        tool_name="Write", required=["path"], chunkable_flag=None
    )
    # …and the flag cannot rescue an optional content field either.
    assert not is_chunkable_content_mutation(
        tool_name="TenantDoc", required=["target"], chunkable_flag=True
    )


def test_unknown_or_empty_tool_name_does_not_qualify_via_allowlist() -> None:
    assert not is_chunkable_content_mutation(
        tool_name=None, required=["content"], chunkable_flag=None
    )
    assert not is_chunkable_content_mutation(
        tool_name="", required=["content"], chunkable_flag=None
    )
    # …but the flag still works regardless of name.
    assert is_chunkable_content_mutation(
        tool_name=None, required=["content"], chunkable_flag=True
    )


def test_edit_is_not_on_allowlist() -> None:
    """``Edit`` has no full-``content`` body, so it must not be allowlisted (and
    would never match the content-in-required test anyway)."""
    assert "Edit" not in CHUNKABLE_CONTENT_MUTATION_ALLOWLIST
