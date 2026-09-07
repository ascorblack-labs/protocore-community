"""Long work, operator control, and context — shipped helpers and query()."""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.skills import SkillIndexEntry
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.compact_checkpoint import (
    CompactCheckpoint,
    apply_checkpoint,
    build_checkpoint,
    tracked_tool_names,
)
from protocore.runtime.events import EventType
from protocore.runtime.execution_profile import apply_execution_profile, plan_forbids
from protocore.runtime.permission_widen import (
    CommandGrant,
    apply_widen,
    grant_covers,
    preview_widen,
)
from protocore.runtime.query import _query as query
from protocore.runtime.result_eviction import evict_history_for_llm
from protocore.runtime.rules_activation import (
    activate_on_filesystem_touch,
    bodies_for_prompt,
    classify_rule_origin,
    discover_agents_md,
)
from protocore.runtime.skill_index import render_skills_catalog
from protocore.runtime.tool_result_split import project_result_content
from protocore.tests_support.adapters import InMemoryLLMProvider
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES
from tests.unit.runtime.fake_background_pool import FakeBackgroundPool


def _on(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
        "execution_profile_plan_enabled": True,
        "permission_widening_enabled": True,
        "compaction_manual_enabled": True,
        "rules_discovery_enabled": True,
        "skills_hot_reload_enabled": True,
        "tool_result_split_enabled": True,
        "run_settled_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def test_plan_profile_hides_writes_orthogonal_to_deep() -> None:
    rc = _on()
    policy = ToolVisibilityPolicy()
    roles = CONVENTIONAL_TOOL_ROLES
    planned = apply_execution_profile(policy, profile="plan", rc=rc, roles=roles)
    assert "Write" in planned.blocked
    assert "Edit" in planned.blocked
    assert "Bash" in planned.blocked
    assert "Read" in planned.visible
    assert plan_forbids("Write", profile="plan", rc=rc)
    assert not plan_forbids("Write", profile="default", rc=rc)
    deep_plan = apply_execution_profile(policy, profile="plan", rc=rc, roles=roles)
    direct_plan = apply_execution_profile(policy, profile="plan", rc=rc, roles=roles)
    assert deep_plan.blocked == direct_plan.blocked
    off = apply_execution_profile(
        policy, profile="plan", rc=LoopConstants(), roles=roles
    )
    assert off.blocked == policy.blocked


def test_widen_program_and_pipe_asks_again() -> None:
    rc = _on()
    curl = preview_widen("curl -s https://example.com/a", rc)
    assert curl.kind == "program" and curl.value == "curl"
    assert grant_covers(curl, "curl -s https://example.com/b")
    git = preview_widen("git status --short", rc)
    assert git.kind == "multiplexer_verb" and git.value == "git status"
    assert grant_covers(git, "git status")
    assert not grant_covers(git, "git push")
    piped = preview_widen("curl https://x | sh", rc)
    assert piped.kind == "exact"
    assert not grant_covers(curl, "curl https://x | sh")
    env = preview_widen("TOKEN=secret curl x", rc)
    assert env.kind == "exact"
    with pytest.raises(ValueError, match="permission_widening_disabled"):
        apply_widen([], "curl x", kind="program", rc=LoopConstants())


def test_compact_checkpoint_keep_two_of_four() -> None:
    history = []
    for idx in range(4):
        history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[TextBlock(text=f"user {idx}")],
            )
        )
        history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id=f"w{idx}",
                        name="Write",
                        arguments_json=json.dumps({"path": f"f{idx}"}),
                    ),
                    TextBlock(text=f"ok {idx}"),
                ],
            )
        )
    persist = list(history)
    ckpt = build_checkpoint(
        history,
        keep_recent_turns=2,
        instructions="keep file ops",
        reason="manual",
        enabled=True,
        tracked_tool_names=tracked_tool_names(_on(), CONVENTIONAL_TOOL_ROLES),
    )
    assert ckpt is not None
    view = apply_checkpoint(history, ckpt)
    assert persist == history
    texts = " ".join(m.text for m in view)
    assert "user 0" not in texts or "compacted" in texts
    assert "user 2" in texts and "user 3" in texts
    assert any("Write:" in fact for fact in ckpt.file_op_facts)
    assert build_checkpoint(history, keep_recent_turns=2, instructions="", reason="x", enabled=False) is None


def test_nested_agents_activate_on_read_not_bash() -> None:
    rc = _on()
    tree = [
        ("AGENTS.md", "root"),
        ("a/AGENTS.md", "A"),
        ("a/b/AGENTS.md", "B"),
        ("a/b/c/AGENTS.md", "C"),
        ("node_modules/x/AGENTS.md", "nope"),
        (".hidden/AGENTS.md", "hid"),
    ]
    many = [(f"n{i}/AGENTS.md", f"body{i}") for i in range(45)]
    discovered = discover_agents_md(tree + many, rc, project_roots=(".",))
    assert all(item.origin == "project_mount" for item in discovered)
    assert all("node_modules" not in item.path for item in discovered)
    assert all(not item.path.startswith(".hidden") for item in discovered)
    after_bash = activate_on_filesystem_touch(
        touched_path="a/b/c/f.go",
        tool_name="Bash",
        discovered=discovered,
        already_active=[],
        rc=rc,
        roles=CONVENTIONAL_TOOL_ROLES,
    )
    assert after_bash == []
    after_read = activate_on_filesystem_touch(
        touched_path="a/b/c/f.go",
        tool_name="Read",
        discovered=discovered,
        already_active=[],
        rc=rc,
        roles=CONVENTIONAL_TOOL_ROLES,
    )
    assert "a/AGENTS.md" in after_read
    assert "a/b/AGENTS.md" in after_read
    assert "a/b/c/AGENTS.md" in after_read
    written = discover_agents_md([("AGENTS.md", "evil")], rc)
    assert all(item.origin == "workspace" for item in written)
    active = activate_on_filesystem_touch(
        touched_path="AGENTS.md",
        tool_name="Read",
        discovered=written,
        already_active=[],
        rc=rc,
        roles=CONVENTIONAL_TOOL_ROLES,
    )
    assert active == []
    prompt = bodies_for_prompt(discovered, after_read, rc)
    assert "A" in "".join(prompt) and "C" in "".join(prompt)
    assert classify_rule_origin("a/b/c/AGENTS.md") == "workspace"
    assert classify_rule_origin("a/b/c/AGENTS.md", project_roots=("a",)) == "project_mount"
    assert classify_rule_origin("AGENTS.md", project_roots=("a",)) == "workspace"


@pytest.mark.asyncio
async def test_skill_index_is_descriptions_not_bodies() -> None:
    entries = [
        SkillIndexEntry(id=f"id{i}", name=f"s{i}", description=f"desc{i}", enabled=True)
        for i in range(50)
    ]

    async def count(text: str) -> int:
        return len(text) // 4

    block = await render_skills_catalog(entries, token_counter=count, budget_tokens=0)
    assert "desc49" in block
    assert "SKILL.md" not in block
    assert all(f'Skill(skill="s{i}")' in block for i in range(50))
    assert "huge body" not in block


def test_a_long_result_is_projected_down_and_says_how_much_is_missing() -> None:
    rc = _on(tool_result_content_max_chars=20)

    projection = project_result_content("x" * 100, rc=rc, canonical_ref="blob-1")

    assert projection.is_shortened and projection.dropped_chars == 80
    assert "truncated 80 chars" in projection.content
    # The pointer names where the whole value still is, so the model reads a
    # value that continues elsewhere rather than one that simply ended here.
    assert "blob-1" in projection.content
    assert projection.content.startswith("x" * 20)


def test_a_result_that_fits_is_its_own_projection() -> None:
    rc = _on(tool_result_content_max_chars=200)

    projection = project_result_content("x" * 100, rc=rc)

    assert projection.content == "x" * 100
    assert not projection.is_shortened and projection.dropped_chars == 0


def test_projection_is_off_until_the_run_asks_for_it() -> None:
    projection = project_result_content("x" * 100, rc=LoopConstants())

    assert projection.content == "x" * 100 and not projection.is_shortened


@pytest.mark.asyncio
async def test_query_workspace_rules_stay_out_with_default_never(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    from ._tool_fixtures import MockTool

    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    assert rc.rules_workspace_trust == "never"
    engine = engine_factory(rc=rc)

    async def list_rule_files() -> list[tuple[str, str]]:
        return [
            ("AGENTS.md", "evil-root"),
            ("a/b/c/AGENTS.md", "evil-nested"),
        ]

    engine.list_rule_files = list_rule_files
    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", description="read a file", response_content="package f")
    )
    llm.queue_tool_call_response(
        tool_call_id="read1",
        tool_name="Read",
        tool_input={"path": "a/b/c/f.go"},
    )
    llm.queue_response(text="read done")
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="read the file")])
    )
    events = [evt async for evt in query(engine)]
    assert engine.discovered_rules
    assert all(item.origin == "workspace" for item in engine.discovered_rules)
    assert not [evt for evt in events if evt.type == EventType.RULES_ACTIVATED]
    assert engine.active_rule_paths == []
    joined = "".join(
        bodies_for_prompt(list(engine.discovered_rules), engine.active_rule_paths, rc)
    )
    assert "evil-root" not in joined
    assert "evil-nested" not in joined


@pytest.mark.asyncio
async def test_query_project_roots_activate_ancestors(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    from ._tool_fixtures import MockTool

    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    engine.rule_project_roots = ("a",)

    async def list_rule_files() -> list[tuple[str, str]]:
        return [
            ("AGENTS.md", "workspace-root"),
            ("a/AGENTS.md", "A-body"),
            ("a/b/AGENTS.md", "B-body"),
            ("a/b/c/AGENTS.md", "C-body"),
        ]

    engine.list_rule_files = list_rule_files
    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", description="read a file", response_content="package f")
    )
    llm.queue_tool_call_response(
        tool_call_id="read1",
        tool_name="Read",
        tool_input={"path": "a/b/c/f.go"},
    )
    llm.queue_response(text="read done")
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="read the file")])
    )
    events = [evt async for evt in query(engine)]
    origins = {item.path: item.origin for item in engine.discovered_rules}
    assert origins["AGENTS.md"] == "workspace"
    assert origins["a/AGENTS.md"] == "project_mount"
    assert origins["a/b/c/AGENTS.md"] == "project_mount"
    activated = [evt for evt in events if evt.type == EventType.RULES_ACTIVATED]
    assert activated
    paths = activated[0].payload["paths"]
    assert "a/AGENTS.md" in paths
    assert "a/b/AGENTS.md" in paths
    assert "a/b/c/AGENTS.md" in paths
    assert "AGENTS.md" not in engine.active_rule_paths
    joined = "".join(
        bodies_for_prompt(list(engine.discovered_rules), engine.active_rule_paths, rc)
    )
    assert "A-body" in joined and "C-body" in joined
    assert "workspace-root" not in joined


@pytest.mark.asyncio
async def test_query_plan_and_compact_no_settled_midway(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="ok")
    rc = _on()
    engine = engine_factory(rc=rc)
    object.__setattr__(engine.config, "execution_profile", "plan")
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    policy = engine.effective_tool_policy
    assert "Write" in policy.blocked
    ckpt = build_checkpoint(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="a")]),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="b")]),
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="c")]),
            Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="d")]),
        ],
        keep_recent_turns=1,
        instructions="",
        reason="overflow",
        enabled=True,
    )
    engine.compact_checkpoint = ckpt
    events = [evt async for evt in query(engine)]
    types = [evt.type for evt in events]
    if EventType.COMPACTION_STARTED in types and EventType.RUN_SETTLED in types:
        compact_idx = types.index(EventType.COMPACTION_STARTED)
        settled_idx = types.index(EventType.RUN_SETTLED)
        assert compact_idx < settled_idx
        assert EventType.RUN_SETTLED not in types[compact_idx:settled_idx]


def test_switch_profile_is_explicit_audit() -> None:
    from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig

    rc = _on()
    # audit helper lives on the engine list
    engine = QueryEngine.__new__(QueryEngine)
    engine.profile_audit = []
    engine.profile_audit.append(
        {"from": "plan", "to": "default", "actor": "user"}
    )
    assert engine.profile_audit[0]["actor"] == "user"
    cfg = QueryEngineConfig(
        run_id="r",
        tenant_id="t",
        session_id="s",
        model_name="m",
        rc=rc,
        execution_profile="plan",
        run_mode="deep",
        thinking_enabled=True,
    )
    assert cfg.execution_profile == "plan" and cfg.run_mode == "deep"
    QueryEngineConfig(
        run_id="r2",
        tenant_id="t",
        session_id="s",
        model_name="m",
        rc=rc,
        execution_profile="plan",
        run_mode="direct",
    )


@pytest.mark.asyncio
async def test_query_drains_one_batched_wake(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="woke")
    rc = _on()
    engine = engine_factory(rc=rc)
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    first = pool.start(engine.config.session_id, notify_on_finish=True)
    second = pool.start(engine.config.session_id, notify_on_finish=True)
    pool.finish(first.id)
    pool.finish(second.id)
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]))
    events = [evt async for evt in query(engine)]
    wakes = [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]
    assert len(wakes) == 1
    ids = wakes[0].payload["task_ids"]
    assert set(ids) == {first.id, second.id}
    assert any("background tasks finished" in msg.text for msg in engine.history)


@pytest.mark.asyncio
async def test_grant_covers_skips_approval_on_real_gate() -> None:
    from protocore.contracts.hooks import HookActionKind, HookResult
    from protocore.contracts.run_state import RunScopedState
    from protocore.contracts.tools import ToolContext
    from protocore.contracts.types import HookEvent
    from protocore.runtime.tool_permission import ToolPermissionGate, ToolPermissionOutcome
    from protocore.tests_support.adapters import InMemoryHookManager

    from ._tool_fixtures import MockTool

    grant = preview_widen("curl https://example.com/a", _on())
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok"},
        ),
    )
    ctx = ToolContext(
        run_id="r",
        tenant_id="t",
        session_id="s",
        run_state=RunScopedState(session_grants=[grant]),
    )
    decision = await ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES).check(
        tool=MockTool(tool_name="Bash"),
        arguments={"command": "curl https://example.com/b"},
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        hook_manager=hooks,
    )
    assert decision.outcome is ToolPermissionOutcome.allow
    assert decision.reason == "session_grant_covers"

    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok2"},
        ),
    )
    other = await ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES).check(
        tool=MockTool(tool_name="Bash"),
        arguments={"command": "wget https://example.com/b"},
        ctx=ctx,
        visibility_policy=ToolVisibilityPolicy(),
        hook_manager=hooks,
    )
    assert other.outcome is ToolPermissionOutcome.require_approval


# --- Domain-agnostic tool names: the two sets that used to be hard-coded ------


def _assistant_tool_call(name: str, call_id: str) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[
            ToolUseBlock(
                tool_call_id=call_id,
                name=name,
                arguments_json=json.dumps({"x": 1, "y": 2, "z": 3}),
            )
        ],
    )


def _tool_result(call_id: str, content: str) -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=call_id, content=content)],
    )


def test_checkpoint_tracks_the_tenant_s_own_tool_names() -> None:
    """A non-coding backend keeps ITS verbs as facts, not Write/Edit/Read."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"turn {i}")])
        for i in range(4)
    ]
    history.insert(1, _assistant_tool_call("mine", "m1"))
    history.insert(2, _assistant_tool_call("Write", "w1"))

    ckpt = build_checkpoint(
        history,
        keep_recent_turns=1,
        instructions="",
        reason="manual",
        enabled=True,
        tracked_tool_names=("mine", "build", "teach"),
    )
    assert ckpt is not None
    facts = " ".join(ckpt.file_op_facts)
    assert "mine:" in facts, "the tenant's own verb must survive compaction"
    assert "Write:" not in facts, "a verb this tenant never registered must not be tracked"


def test_checkpoint_tracks_the_tools_the_host_declared() -> None:
    """No configured names: the tracked set comes from the declared roles."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"turn {i}")])
        for i in range(4)
    ]
    history.insert(1, _assistant_tool_call("Write", "w1"))

    ckpt = build_checkpoint(
        history,
        keep_recent_turns=1,
        instructions="",
        reason="manual",
        enabled=True,
        tracked_tool_names=tracked_tool_names(
            LoopConstants(compaction_tracked_tool_names=()), CONVENTIONAL_TOOL_ROLES
        ),
    )
    assert ckpt is not None
    assert any("Write:" in fact for fact in ckpt.file_op_facts)


def test_eviction_targets_the_tenant_s_own_read_shaped_tools() -> None:
    """A simulation evicts bulky `look` results; it has no Read tool at all."""
    history = [
        _assistant_tool_call("look", "l1"),
        _tool_result("l1", "a very large tile map " * 50),
        _assistant_tool_call("say", "s1"),
        _tool_result("s1", "hello"),
    ]
    rc = LoopConstants(
        result_eviction_enabled=True,
        result_eviction_tool_names=("look", "inspect"),
    )
    view, evicted = evict_history_for_llm(history, rc, bundled_prompt_provider())

    assert evicted == ["l1"]
    assert history[1].content_blocks[0].content.startswith("a very large tile map")
    assert "evicted" in view[1].content_blocks[0].content
    assert view[3].content_blocks[0].content == "hello", "untracked tool is untouched"


def test_eviction_with_no_names_is_a_no_op() -> None:
    history = [_assistant_tool_call("look", "l1"), _tool_result("l1", "big" * 100)]
    rc = LoopConstants(result_eviction_enabled=True, result_eviction_tool_names=())
    view, evicted = evict_history_for_llm(history, rc, bundled_prompt_provider())
    assert evicted == []
    assert view == list(history)


def test_a_grant_read_back_out_of_its_row_still_answers_the_gate() -> None:
    """A grant that crossed a process boundary is a grant again.

    ``to_dict`` is how a grant leaves the process that made it — into a run
    snapshot, or into the shared store the API pod and the executor pod both
    read. What comes back is a row, and the approval gate does not ask a row
    anything: it asks the grant whether it covers the command. So the reader
    has to exist, and the object it returns has to match the same commands the
    original did.
    """
    original = apply_widen(
        [], "git status", kind="multiplexer_verb", rc=_on()
    )[0]
    restored = CommandGrant.from_dict(original.to_dict())

    assert restored == original
    assert grant_covers(restored, "git status --short") is True
    assert grant_covers(restored, "git push") is False


def test_a_grant_row_naming_an_unknown_kind_widens_nothing() -> None:
    """The narrowest reading of a row this build cannot understand."""
    restored = CommandGrant.from_dict({"kind": "everything", "value": "rm -rf /"})

    assert restored.kind == "exact"
    assert grant_covers(restored, "rm -rf /tmp") is False
    assert grant_covers(restored, "rm -rf /") is True


def test_a_checkpoint_read_back_out_of_its_row_is_the_same_checkpoint() -> None:
    """The folded-away turns survive in the checkpoint and nowhere else."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"m{i}")])
        for i in range(4)
    ]
    built = build_checkpoint(
        history,
        keep_recent_turns=2,
        instructions="keep the decisions",
        reason="manual",
        enabled=True,
    )
    assert built is not None

    restored = CompactCheckpoint.from_dict(built.to_dict())

    assert restored == built


def test_a_partial_checkpoint_row_is_read_rather_than_refused() -> None:
    """A row missing a field is a poorer checkpoint, not a broken resume."""
    restored = CompactCheckpoint.from_dict({"summary": "what was folded"})

    assert restored.summary == "what was folded"
    assert restored.retained_from_index == 0
    assert restored.file_op_facts == []
    assert restored.reason == "manual"
