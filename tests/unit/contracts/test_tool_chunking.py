"""Unit tests for the shared chunkable-content-mutation predicate. This ONE
predicate is used by both core ``query.py`` and the host LLM client, so
the routing of a truncated tool call is identical in both layers.
"""
from __future__ import annotations

from protocore.contracts.tool_chunking import (
    chunkable_content_mutation_names,
    is_chunkable_content_mutation,
)
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap

#: One host's file tools, stated as roles the way a registration states them.
ROLES = ToolRoleMap.declare(
    {
        "Write": [ToolRole.writes_path],
        "AppendFile": [ToolRole.appends_path],
        "Edit": [ToolRole.edits_path],
    }
)


def test_byte_producing_tools_with_required_content_qualify() -> None:
    for name in ("Write", "AppendFile"):
        assert is_chunkable_content_mutation(
            tool_name=name, required=["path", "content"], chunkable_flag=None, roles=ROLES
        ), name


def test_explicit_flag_qualifies_without_a_role() -> None:
    """A per-tenant tool opts in via the flag without a declared role."""
    assert is_chunkable_content_mutation(
        tool_name="TenantDoc", required=["target", "content"], chunkable_flag=True, roles=ROLES
    )


def test_dynamic_content_tool_without_flag_does_not_qualify() -> None:
    """A dynamic tool that merely REQUIRES ``content`` must NOT qualify unless
    the host declared it byte-producing or it is explicitly flagged."""
    assert not is_chunkable_content_mutation(
        tool_name="PostComment", required=["target", "content"], chunkable_flag=None, roles=ROLES
    )
    assert not is_chunkable_content_mutation(
        tool_name="PostComment", required=["target", "content"], chunkable_flag=False, roles=ROLES
    )


def test_content_must_be_required_not_merely_present() -> None:
    """An OPTIONAL ``content`` (not in ``required``) is not the cut-body shape,
    even for a tool whose role says it produces bytes."""
    assert not is_chunkable_content_mutation(
        tool_name="Write", required=["path"], chunkable_flag=None, roles=ROLES
    )
    # …and the flag cannot rescue an optional content field either.
    assert not is_chunkable_content_mutation(
        tool_name="TenantDoc", required=["target"], chunkable_flag=True, roles=ROLES
    )


def test_unknown_or_empty_tool_name_does_not_qualify_by_role() -> None:
    assert not is_chunkable_content_mutation(
        tool_name=None, required=["content"], chunkable_flag=None, roles=ROLES
    )
    assert not is_chunkable_content_mutation(
        tool_name="", required=["content"], chunkable_flag=None, roles=ROLES
    )
    # …but the flag still works regardless of name.
    assert is_chunkable_content_mutation(
        tool_name=None, required=["content"], chunkable_flag=True, roles=ROLES
    )


def test_an_in_place_edit_tool_is_never_chunkable() -> None:
    """An edit tool has no full-``content`` body, so the chunk protocol has
    nothing to continue (and it would never match the content-in-required
    test anyway)."""
    assert "Edit" not in chunkable_content_mutation_names(ROLES)
