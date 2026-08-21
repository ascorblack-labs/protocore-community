"""Tests for the universal lean tool-surface contract.

Verifies the canonical lean pool shape, the read vs read_silent grounding
split, and the profile-selection helper — all pure-core, backend-agnostic.
"""
from __future__ import annotations

import pytest

from protocore.contracts import (
    GROUNDING_TRACKED_TOOLS,
    LEAN_TOOL_ANSWER,
    LEAN_TOOL_EXEC,
    LEAN_TOOL_NAMES,
    LEAN_TOOL_READ,
    LEAN_TOOL_READ_SILENT,
    TOOL_SURFACE_PROFILE_LEAN,
    TOOL_SURFACE_PROFILE_LEGACY,
    TOOL_SURFACE_PROFILES,
    is_lean_tool,
    lean_tool_definition,
    lean_tool_surface,
    select_tool_surface_profile,
)
from protocore.contracts.lean_tool_surface import (
    LEAN_TOOL_FIND,
    LEAN_TOOL_SEARCH,
    LEAN_TOOL_WRITE,
)
from protocore.contracts.types import ToolDefinition


def test_lean_pool_is_exactly_seven_canonical_verbs() -> None:
    assert LEAN_TOOL_NAMES == (
        LEAN_TOOL_EXEC,
        LEAN_TOOL_READ,
        LEAN_TOOL_READ_SILENT,
        LEAN_TOOL_WRITE,
        LEAN_TOOL_FIND,
        LEAN_TOOL_SEARCH,
        LEAN_TOOL_ANSWER,
    )
    # Lean = small. Guard against future default-tool bloat creeping in.
    assert len(LEAN_TOOL_NAMES) == 7
    assert len(set(LEAN_TOOL_NAMES)) == 7


def test_lean_tool_surface_returns_tool_definitions_in_stable_order() -> None:
    surface = lean_tool_surface()
    assert [d.name for d in surface] == list(LEAN_TOOL_NAMES)
    for d in surface:
        assert isinstance(d, ToolDefinition)
        assert d.description  # non-empty, capability-only
        assert d.parameters.type == "object"


def test_read_records_grounding_but_read_silent_does_not() -> None:
    # The +22.4pp mechanism: read is grounding-tracked,
    # read_silent is explicitly not.
    assert LEAN_TOOL_READ in GROUNDING_TRACKED_TOOLS
    assert LEAN_TOOL_READ_SILENT not in GROUNDING_TRACKED_TOOLS
    assert GROUNDING_TRACKED_TOOLS == frozenset({LEAN_TOOL_READ})


def test_read_and_read_silent_share_arg_shape() -> None:
    read = lean_tool_definition(LEAN_TOOL_READ)
    silent = lean_tool_definition(LEAN_TOOL_READ_SILENT)
    assert read.parameters.properties == silent.parameters.properties
    assert read.parameters.required == silent.parameters.required == ["path"]
    # Only the side-effect/description differs.
    assert read.description != silent.description


def test_exec_takes_path_args_stdin_not_a_shell_command() -> None:
    # exec is a registered-binary runner ({path, args, stdin}), NOT a shell.
    # The BitGN VM exec has no /bin/sh; a {command}->/bin/sh -c contract made
    # every exec NOT_FOUND. Path is the only required field.
    exec_def = lean_tool_definition(LEAN_TOOL_EXEC)
    props = exec_def.parameters.properties
    assert exec_def.parameters.required == ["path"]
    assert set(props) == {"path", "args", "stdin"}
    assert props["path"]["type"] == "string"
    assert props["args"]["type"] == "array"
    assert props["args"]["items"]["type"] == "string"
    assert props["stdin"]["type"] == "string"
    # The legacy shell-command contract is gone.
    assert "command" not in props
    # Description is capability-only AND states it is not a shell.
    low = exec_def.description.lower()
    assert "not a shell" in low
    assert "shell command" not in low


def test_answer_requires_message_and_accepts_outcome_and_refs() -> None:
    answer = lean_tool_definition(LEAN_TOOL_ANSWER)
    assert answer.parameters.required == ["message"]
    props = answer.parameters.properties
    assert "refs" in props
    # The answer contract surfaces an outcome disposition field so the model sets
    # it even before reading the environment's own docs. Core stays universal:
    # it owns only the field SHAPE, never a fixed value set, so the description
    # must NOT bake in any concrete enum literal.
    assert "outcome" in props
    assert props["outcome"]["type"] == "string"
    assert "OUTCOME_" not in props["outcome"]["description"]
    assert "refs" in props
    assert props["refs"]["type"] == "array"


def test_descriptions_carry_no_baked_in_step_hints() -> None:
    # Universal-core: descriptions are capability-only. No mandated bootstrap
    # sequence, exact vault paths, or "call X before answering" coaching.
    forbidden = (
        "/bin/date",
        "/bin/id",
        "/AGENTS.MD",
        "before anything else",
        "the scorer",
        "mandates",
        "policy-updates",
    )
    for d in lean_tool_surface():
        low = d.description.lower()
        for token in forbidden:
            assert token.lower() not in low, (token, d.name)


def test_is_lean_tool() -> None:
    assert is_lean_tool(LEAN_TOOL_EXEC)
    assert is_lean_tool(LEAN_TOOL_READ_SILENT)
    assert not is_lean_tool("remote_exec")
    assert not is_lean_tool("Bash")


def test_lean_tool_definition_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        lean_tool_definition("not_a_lean_verb")


def test_profiles_set() -> None:
    assert TOOL_SURFACE_PROFILES == (
        TOOL_SURFACE_PROFILE_LEGACY,
        TOOL_SURFACE_PROFILE_LEAN,
    )


def test_select_profile_falls_back_to_legacy_on_unknown() -> None:
    assert select_tool_surface_profile("lean") == TOOL_SURFACE_PROFILE_LEAN
    assert select_tool_surface_profile("legacy") == TOOL_SURFACE_PROFILE_LEGACY
    # Unknown / garbage falls back to the safe legacy default.
    assert select_tool_surface_profile("bogus") == TOOL_SURFACE_PROFILE_LEGACY
    assert select_tool_surface_profile("") == TOOL_SURFACE_PROFILE_LEGACY
