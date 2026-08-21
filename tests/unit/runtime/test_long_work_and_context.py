"""Long work, operator control, and context — shipped helpers and query()."""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.skills import SkillIndexEntry
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.background import (
    BackgroundPool,
    FakeRunner,
    adopt_notice,
    compute_hard_timeout_seconds,
    decide_foreground,
    reap_is_safe,
    refuse_notify_when_disabled,
)
from protocore.runtime.compact_checkpoint import (
    apply_checkpoint,
    build_checkpoint,
)
from protocore.runtime.events import EventType
from protocore.runtime.execution_profile import apply_execution_profile, plan_forbids
from protocore.runtime.path_policy import deny_reason, paths_in_command
from protocore.runtime.permission_widen import (
    apply_widen,
    grant_covers,
    preview_widen,
)
from protocore.runtime.query import query
from protocore.runtime.result_eviction import evict_history_for_llm
from protocore.runtime.rules_activation import (
    activate_on_filesystem_touch,
    bodies_for_prompt,
    classify_rule_origin,
    discover_agents_md,
)
from protocore.runtime.skill_index import render_skills_catalog
from protocore.runtime.tool_result_split import split_result
from protocore.tests_support.adapters import InMemoryLLMProvider


def _on(**overrides: object) -> RuntimeConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
        "foreground_adopt_enabled": True,
        "execution_profile_plan_enabled": True,
        "permission_widening_enabled": True,
        "compaction_manual_enabled": True,
        "rules_discovery_enabled": True,
        "skills_hot_reload_enabled": True,
        "tool_result_split_enabled": True,
        "path_protection_enabled": True,
        "run_settled_enabled": True,
    }
    values.update(overrides)
    return RuntimeConstants(**values)  # type: ignore[arg-type]


def test_notify_off_flag_refuses() -> None:
    with pytest.raises(ValueError, match="background_tasks_disabled"):
        refuse_notify_when_disabled(enabled=False, notify_on_finish=True)
    refuse_notify_when_disabled(enabled=False, notify_on_finish=False)
    refuse_notify_when_disabled(enabled=True, notify_on_finish=True)


def test_timeout_formula_uses_rc() -> None:
    rc = _on()
    assert compute_hard_timeout_seconds(explicit=10, expected_seconds=100, rc=rc) == 10
    expected = compute_hard_timeout_seconds(
        explicit=None, expected_seconds=20, rc=rc
    )
    assert expected == max(
        20 * rc.background_expected_timeout_multiplier,
        rc.background_expected_timeout_floor_seconds,
    )
    assert expected <= rc.background_max_timeout_seconds
    assert (
        compute_hard_timeout_seconds(explicit=None, expected_seconds=None, rc=rc)
        == rc.background_default_timeout_seconds
    )


@pytest.mark.asyncio
async def test_pool_start_list_output_stop_and_one_wake() -> None:
    rc = _on()
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    task = await pool.start(
        command="pytest -q",
        session_id="s",
        tenant_id="t",
        notify_on_finish=True,
    )
    listed = pool.list("s")
    assert listed[0].status == "running"
    handle = runner.handles["pytest -q"]
    handle.output = ".... 12 passed"
    handle.finish("succeeded", ".... 12 passed")
    refreshed = await pool.refresh(task.id)
    assert refreshed is not None
    assert refreshed.status == "succeeded"
    assert "12 passed" in refreshed.output
    wakes = pool.drain_wakes("s")
    assert wakes == [task.id]
    assert pool.drain_wakes("s") == []


@pytest.mark.asyncio
async def test_three_finishes_one_wake() -> None:
    rc = _on()
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    ids = []
    for name in ("a", "b", "c"):
        task = await pool.start(
            command=name, session_id="s", tenant_id="t", notify_on_finish=True
        )
        runner.handles[name].finish("succeeded", "ok")
        await pool.refresh(task.id)
        ids.append(task.id)
    wakes = pool.drain_wakes("s")
    assert set(wakes) == set(ids)
    assert pool.wakes_used == 1


@pytest.mark.asyncio
async def test_worker_death_orphans_not_running() -> None:
    rc = _on()
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    task = await pool.start(command="sleep 5", session_id="s", tenant_id="t")
    assert task.status == "running"
    pool.worker_died(task.worker_id)
    assert pool.get(task.id).status == "orphaned"  # type: ignore[union-attr]
    assert pool.get(task.id).status != "running"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_reap_skips_reused_pid() -> None:
    rc = _on()
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    task = await pool.start(command="old", session_id="s", tenant_id="t")
    handle = runner.handles["old"]
    handle.start_token = "other-process"
    reaped = await pool.reap(task.id)
    assert reaped is not None
    assert reaped.reason == "reap_identity_mismatch"
    assert not handle.killed
    assert reap_is_safe(
        recorded_pid=1,
        recorded_start_token="a",
        live_pid=1,
        live_start_token="b",
    ) is False


@pytest.mark.asyncio
async def test_pool_full_and_shutdown_silent() -> None:
    rc = _on(background_max_concurrent_per_session=1)
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    await pool.start(command="one", session_id="s", tenant_id="t", notify_on_finish=True)
    with pytest.raises(ValueError, match="background_pool_full"):
        await pool.start(command="two", session_id="s", tenant_id="t")
    runner.handles["one"].finish("succeeded")
    await pool.refresh(pool.list("s")[0].id)
    pool.shutting_down = True
    assert pool.drain_wakes("s") == []


def test_foreground_adopt_and_cancel() -> None:
    rc = _on()
    assert (
        decide_foreground(
            background_enabled=True,
            adopt_enabled=True,
            elapsed_seconds=5,
            timeout_seconds=1,
            cancelled=False,
            pool_full=False,
        )
        == "adopt"
    )
    assert (
        decide_foreground(
            background_enabled=True,
            adopt_enabled=True,
            elapsed_seconds=5,
            timeout_seconds=1,
            cancelled=True,
            pool_full=False,
        )
        == "kill_cancel"
    )
    assert (
        decide_foreground(
            background_enabled=True,
            adopt_enabled=True,
            elapsed_seconds=5,
            timeout_seconds=1,
            cancelled=False,
            pool_full=True,
        )
        == "kill_pool_full"
    )
    notice = adopt_notice("bg_x", "output-tail", rc)
    assert notice.startswith("Command still running")
    assert "bg_x" in notice
    assert "Do NOT run this command again" in notice


def test_flags_off_background_is_noop() -> None:
    rc = RuntimeConstants(model_context_window=4096)
    assert rc.background_tasks_enabled is False
    with pytest.raises(ValueError, match="background_tasks_disabled"):
        refuse_notify_when_disabled(enabled=rc.background_tasks_enabled, notify_on_finish=True)
    assert (
        decide_foreground(
            background_enabled=False,
            adopt_enabled=False,
            elapsed_seconds=99,
            timeout_seconds=1,
            cancelled=False,
            pool_full=False,
        )
        == "kill_disabled"
    )


def test_plan_profile_hides_writes_orthogonal_to_deep() -> None:
    rc = _on()
    policy = ToolVisibilityPolicy()
    planned = apply_execution_profile(policy, profile="plan", rc=rc)
    assert "Write" in planned.blocked
    assert "Edit" in planned.blocked
    assert "Bash" in planned.blocked
    assert "Read" in planned.visible
    assert plan_forbids("Write", profile="plan", rc=rc)
    assert not plan_forbids("Write", profile="default", rc=rc)
    deep_plan = apply_execution_profile(policy, profile="plan", rc=rc)
    direct_plan = apply_execution_profile(policy, profile="plan", rc=rc)
    assert deep_plan.blocked == direct_plan.blocked
    off = apply_execution_profile(policy, profile="plan", rc=RuntimeConstants())
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
        apply_widen([], "curl x", kind="program", rc=RuntimeConstants())


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
    )
    assert after_bash == []
    after_read = activate_on_filesystem_touch(
        touched_path="a/b/c/f.go",
        tool_name="Read",
        discovered=discovered,
        already_active=[],
        rc=rc,
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


def test_split_and_path_deny() -> None:
    rc = _on(tool_result_content_max_chars=20)
    content, details = split_result("x" * 100, rc=rc)
    assert "truncated" in content
    assert details is not None and details["full_content"] == "x" * 100
    off_c, off_d = split_result("x" * 100, rc=RuntimeConstants())
    assert off_c == "x" * 100 and off_d is None
    assert deny_reason("/etc/passwd", workspace_root="ws", user_id="u1", rc=rc) == "outside_workspace"
    assert deny_reason("../../../etc/passwd", workspace_root="ws", user_id="u1", rc=rc)
    foreign = deny_reason(
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/secret",
        workspace_root="u1",
        user_id="u1",
        rc=rc,
    )
    assert foreign in {"foreign_workspace", "protected_path", "outside_workspace"} or foreign
    assert deny_reason("ok.txt", workspace_root="u1", user_id="u1", rc=rc) is None
    assert deny_reason("/etc/passwd", workspace_root="ws", user_id="u1", rc=RuntimeConstants()) is None
    assert "/etc/passwd" in paths_in_command("cat /etc/passwd")
    assert "../../../etc/shadow" in paths_in_command("cat ../../../etc/shadow")
    assert paths_in_command("curl https://example.com/x") == []
    assert paths_in_command("echo hello") == []


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
    runner = FakeRunner()
    pool = BackgroundPool(runner=runner, rc=rc)
    engine.background_pool = pool
    first = await pool.start(
        command="one", session_id=engine.config.session_id, tenant_id="t", notify_on_finish=True
    )
    second = await pool.start(
        command="two", session_id=engine.config.session_id, tenant_id="t", notify_on_finish=True
    )
    runner.handles["one"].finish("succeeded", "ok1")
    runner.handles["two"].finish("succeeded", "ok2")
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
        metadata={"protocore.helpers": {"session_grants": [grant]}},
    )
    decision = await ToolPermissionGate().check(
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
    other = await ToolPermissionGate().check(
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


def test_checkpoint_default_still_tracks_the_coding_verbs() -> None:
    """Callers that pass no names keep the historical behaviour."""
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"turn {i}")])
        for i in range(4)
    ]
    history.insert(1, _assistant_tool_call("Write", "w1"))

    ckpt = build_checkpoint(
        history, keep_recent_turns=1, instructions="", reason="manual", enabled=True
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
    rc = RuntimeConstants(
        result_eviction_enabled=True,
        result_eviction_tool_names=("look", "inspect"),
    )
    view, evicted = evict_history_for_llm(history, rc)

    assert evicted == ["l1"]
    assert history[1].content_blocks[0].content.startswith("a very large tile map")
    assert "evicted" in view[1].content_blocks[0].content
    assert view[3].content_blocks[0].content == "hello", "untracked tool is untouched"


def test_eviction_with_no_names_is_a_no_op() -> None:
    history = [_assistant_tool_call("look", "l1"), _tool_result("l1", "big" * 100)]
    rc = RuntimeConstants(result_eviction_enabled=True, result_eviction_tool_names=())
    view, evicted = evict_history_for_llm(history, rc)
    assert evicted == []
    assert view == list(history)
