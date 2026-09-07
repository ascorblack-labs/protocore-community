"""The role map a test wires when it does not care what the tools are called.

A host declares what its tools do; the test tree plays that host. These are the
names the suite has always used for its file, shell and delegation tools, now
stated as roles the way a real registration states them. Any test that cares
about the naming builds its own map instead — several do, and that is the whole
demonstration: nothing in the runtime reads these names.
"""
from __future__ import annotations

from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRole, ToolRoleMap

CONVENTIONAL_TOOL_ROLES: ToolRoleMap = ToolRoleMap.declare(
    {
        "Read": [ToolRole.reads_path],
        "Write": [ToolRole.writes_path],
        "AppendFile": [ToolRole.appends_path],
        "FinalizeFile": [ToolRole.finalizes_path],
        "Edit": [ToolRole.edits_path],
        "Grep": [ToolRole.searches_workspace],
        "Glob": [ToolRole.searches_workspace],
        "List": [ToolRole.searches_workspace],
        "Bash": [ToolRole.runs_shell],
        "PythonExec": [ToolRole.runs_shell],
        "WebFetch": [ToolRole.fetches_url],
        "Agent": [ToolRole.delegates_work, ToolRole.never_delegated],
        "ToolSearch": [ToolRole.discovers_tools],
        "TodoWrite": [ToolRole.records_plan],
        "AskUser": [ToolRole.asks_user],
    },
    argument_aliases={
        ToolArgumentSlot.path: ["path", "file_path"],
        ToolArgumentSlot.content: ["content", "new_string", "text"],
        ToolArgumentSlot.shell_command: ["command", "cmd", "shell"],
        ToolArgumentSlot.url: ["url"],
        ToolArgumentSlot.answer: ["message", "answer", "text"],
    },
)

__all__ = ["CONVENTIONAL_TOOL_ROLES"]
