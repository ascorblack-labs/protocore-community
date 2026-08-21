"""Tests for the skill-catalog wiring into the QueryEngine.

The full enabled skill catalog (name + one-line description) renders once
per run into a ``<system-reminder>`` block placed in the static
system-prompt prefix; trigger-style ``<command-name>NAME</command-name>``
references load full skill bodies into the context.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.skills import SkillIndexEntry, SkillUpsertInput
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.tests_support.adapters import InMemorySkillStore


def _system_text(in_memory_runtime: dict[str, object]) -> str:
    """Concatenate the system-role text of the most recent LLM call."""

    calls = in_memory_runtime["llm"].calls  # type: ignore[attr-defined]
    assert calls, "expected at least one LLM call"
    return "\n".join(
        block.text
        for msg in calls[-1].messages
        if msg.role is MessageRole.system
        for block in msg.content_blocks
        if hasattr(block, "text")
    )


@pytest.mark.asyncio
async def test_full_catalog_rendered_into_system_prefix(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Every enabled skill renders into the ``<system-reminder>`` catalog
    block in the LLM request's system_prompt_sections, alphabetical."""
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="git-flow",
            description="Use to commit and push changes properly.",
            body_md="# git flow body",
        ),
    )
    await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="frontend-design",
            description="Use to build distinctive frontends.",
            body_md="# frontend body",
        ),
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="anything at all")],
    )
    async for _evt in engine.run(user_msg):
        pass

    system_text = _system_text(in_memory_runtime)
    assert "<system-reminder>" in system_text
    # The full catalog surfaces every enabled skill, not just a query match.
    assert "git-flow" in system_text
    assert "frontend-design" in system_text
    # Alphabetical by name.
    assert system_text.index("frontend-design") < system_text.index("git-flow")


@pytest.mark.asyncio
async def test_disabled_skill_absent_from_catalog(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A skill the operator disabled is dropped from the catalog block."""
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    entry = await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="toggled",
            description="Toggle me off.",
            body_md="# body",
        ),
    )
    await skill_store.set_enabled(engine.config.tenant_id, entry.id, enabled=False)

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="anything")],
    )
    async for _evt in engine.run(user_msg):
        pass

    assert "toggled" not in _system_text(in_memory_runtime)


@pytest.mark.asyncio
async def test_command_triggers_full_skill_body(
    engine_factory,
    in_memory_runtime,
) -> None:
    """``<command-name>name</command-name>`` references in the user input
    must load the full skill body into the context (Layer 3)."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(
        text="done", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="release-prep",
            description="Use to prep a release branch.",
            body_md="# release prep\n\nStep 1: cut a branch.",
        ),
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[
            TextBlock(text="run <command-name>release-prep</command-name>"),
        ],
    )
    async for _evt in engine.run(user_msg):
        pass

    system_text = _system_text(in_memory_runtime)
    assert "<loaded-skill" in system_text
    assert "release-prep" in system_text
    assert "Step 1: cut a branch." in system_text


@pytest.mark.asyncio
async def test_catalog_budget_degrades_to_names_only(
    engine_factory,
    in_memory_runtime,
) -> None:
    """When the catalog overflows the budget, it degrades to names only."""
    rc = RuntimeConstants(
        model_context_window=4_000,
        skill_index_budget_ratio=0.01,
    )
    engine = engine_factory(rc=rc)
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    long_desc = (
        "this is a deliberately long description used to force the catalog "
        "renderer to degrade to names only because the full descriptions "
        "blow past the configured token budget for the catalog block"
    )
    for i in range(20):
        await skill_store.create(
            engine.config.tenant_id,
            SkillUpsertInput(
                name=f"skill-{i:02d}",
                description=long_desc,
                body_md=f"# body {i}",
            ),
        )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="anything")],
    )
    async for _evt in engine.run(user_msg):
        pass

    system_text = _system_text(in_memory_runtime)
    assert "<system-reminder>" in system_text
    # All 20 names present (names-only degrade keeps every skill)...
    assert system_text.count('Skill(skill="skill-') == 20
    # ...but none carry the long description after the degrade.
    assert long_desc not in system_text


@pytest.mark.asyncio
async def test_catalog_empty_when_no_store_skills(
    engine_factory,
    in_memory_runtime,
) -> None:
    """No skills in the store → no catalog block emitted (zero-cost)."""
    engine = engine_factory()
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="just a plain message")],
    )
    events: list[TurnEvent] = []
    async for evt in engine.run(user_msg):
        events.append(evt)

    system_text = _system_text(in_memory_runtime)
    assert "The following skills are available" not in system_text
    assert "Skills are tools, not files" not in system_text
    assert any(e.type is EventType.MESSAGE_STOP for e in events)


@pytest.mark.asyncio
async def test_project_pinned_skill_surfaces_in_catalog(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A project-pinned (bare-name) enabled skill is present in the catalog."""
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine = engine_factory(
        rc=rc,
        pinned_skill_names=frozenset({"deploy-runbook"}),
    )
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="deploy-runbook",
            description="Use to roll out a production deploy.",
            body_md="# deploy body",
        ),
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="What is the capital of France?")],
    )
    async for _evt in engine.run(user_msg):
        pass

    system_text = _system_text(in_memory_runtime)
    assert "<system-reminder>" in system_text
    assert "deploy-runbook" in system_text


@pytest.mark.asyncio
async def test_disabled_pinned_skill_not_surfaced(
    engine_factory,
    in_memory_runtime,
) -> None:
    """A pin for a skill the operator DISABLED is not resurfaced — the
    catalog drops disabled skills and the MISSING-pin fetch uses
    ``list_enabled_subset`` so a disabled skill stays off the catalog even
    with a stale pin (disable gates beat pins)."""
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine = engine_factory(
        rc=rc,
        pinned_skill_names=frozenset({"disabled-runbook"}),
    )
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    entry = await skill_store.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="disabled-runbook",
            description="Use to roll out a production deploy.",
            body_md="# deploy body",
        ),
    )
    await skill_store.set_enabled(engine.config.tenant_id, entry.id, enabled=False)

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="What is the capital of France?")],
    )
    async for _evt in engine.run(user_msg):
        pass

    assert "disabled-runbook" not in _system_text(in_memory_runtime)


@pytest.mark.asyncio
async def test_catalog_keys_on_account_not_scope(
    engine_factory,
    in_memory_runtime,
) -> None:
    """The catalog resolves the ACCOUNT's skills, not the scope's.

    The skill bank is account-wide (keyed on ``skills.account_id``) while the
    run's ``tenant_id`` is the scope id — which may differ from the account id
    (the default seed: scope ``…001`` owned by account ``…010``). The catalog
    build + pin merge MUST key on ``config.account_id``: a skill OWNED by the
    account surfaces, and a row keyed on the scope id does NOT (proving the
    lookup is not silently keying on the wrong id).
    """
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    scope_id = "00000000-0000-0000-0000-000000000001"
    account_id = "00000000-0000-0000-0000-000000000010"
    engine = engine_factory(
        rc=rc,
        tenant_id=scope_id,
        account_id=account_id,
        pinned_skill_names=frozenset({"pinned-account-skill"}),
    )
    assert engine.config.tenant_id != engine.config.account_id
    in_memory_runtime["llm"].queue_response(
        text="ok", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    # Owned by the ACCOUNT — must surface.
    await skill_store.create(
        account_id,
        SkillUpsertInput(
            name="account-skill",
            description="Account-wide enabled skill.",
            body_md="# account body",
        ),
    )
    # A disabled-by-default account skill, force-surfaced via a project pin —
    # exercises the ``_merge_pinned_skills`` account-keyed fetch.
    await skill_store.create(
        account_id,
        SkillUpsertInput(
            name="pinned-account-skill",
            description="Account skill surfaced via project pin.",
            body_md="# pinned body",
        ),
    )
    # A decoy row keyed on the SCOPE id — must NOT surface (the bug was that
    # the runtime keyed every lookup on this id).
    await skill_store.create(
        scope_id,
        SkillUpsertInput(
            name="scope-decoy-skill",
            description="Keyed on the scope id; the catalog must ignore it.",
            body_md="# decoy body",
        ),
    )

    async for _evt in engine.run(
        Message(
            role=MessageRole.user,
            content_blocks=[TextBlock(text="anything at all")],
        )
    ):
        pass

    system_text = _system_text(in_memory_runtime)
    assert "<system-reminder>" in system_text
    assert "account-skill" in system_text
    assert "pinned-account-skill" in system_text
    assert "scope-decoy-skill" not in system_text


@pytest.mark.asyncio
async def test_command_trigger_keys_on_account_not_scope(
    engine_factory,
    in_memory_runtime,
) -> None:
    """The ``<command-name>`` trigger loader (skill chaining) keys on the
    ACCOUNT id, not the scope id.

    A triggered skill OWNED by the account loads its full body even when the
    run's scope id differs from the account; a row keyed on the scope id does
    NOT load (proving the trigger loader is not silently keying on the wrong
    id). This is the progressive-disclosure / skill-chaining path.
    """
    scope_id = "00000000-0000-0000-0000-000000000001"
    account_id = "00000000-0000-0000-0000-000000000010"
    engine = engine_factory(tenant_id=scope_id, account_id=account_id)
    assert engine.config.tenant_id != engine.config.account_id
    in_memory_runtime["llm"].queue_response(
        text="done", stop_reason=StopReason.end_turn
    )

    skill_store = in_memory_runtime["skills"]
    # Owned by the ACCOUNT — must load when triggered.
    await skill_store.create(
        account_id,
        SkillUpsertInput(
            name="frontend-design",
            description="Use to build distinctive frontends.",
            body_md="# frontend-design\n\nProduce distinctive UI.",
        ),
    )
    # A decoy row keyed on the SCOPE id, same name — must NOT load.
    await skill_store.create(
        scope_id,
        SkillUpsertInput(
            name="scope-decoy",
            description="Keyed on the scope id; the trigger must ignore it.",
            body_md="# scope decoy body",
        ),
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[
            TextBlock(
                text=(
                    "run <command-name>frontend-design</command-name> and "
                    "<command-name>scope-decoy</command-name>"
                ),
            ),
        ],
    )
    async for _evt in engine.run(user_msg):
        pass

    system_text = _system_text(in_memory_runtime)
    # Account-owned triggered skill loaded its full body...
    assert "<loaded-skill" in system_text
    assert "Produce distinctive UI." in system_text
    # ...but the scope-keyed decoy never loaded.
    assert "scope decoy body" not in system_text


class _CountingSkillStore(InMemorySkillStore):
    """``InMemorySkillStore`` that counts ``list`` calls (catalog source)."""

    def __init__(self) -> None:
        super().__init__()
        self.list_calls = 0

    async def list(self, tenant_id: str) -> Sequence[SkillIndexEntry]:
        self.list_calls += 1
        return await super().list(tenant_id)


@pytest.mark.asyncio
async def test_catalog_built_once_per_run_across_rounds(
    engine_factory,
    in_memory_runtime,
) -> None:
    """The catalog source ``store.list`` is queried at most ONCE per run even
    though the run spans multiple LLM rounds (a tool-call round forces an
    inner-loop context rebuild + a recovery-style rebuild path), while a
    ``<command-name>`` trigger in the user message still force-loads its
    skill body this run (loaded bundles stay per turn, not behind the
    catalog sentinel)."""
    from ._tool_fixtures import MockTool

    counting = _CountingSkillStore()
    in_memory_runtime["skills"] = counting
    in_memory_runtime["tools"].register(MockTool(tool_name="Probe"))

    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine = engine_factory(rc=rc)

    # Round 1 → a tool call (drives the inner loop to rebuild the context
    # bundle, re-entering the catalog path). Round 2 → final answer.
    in_memory_runtime["llm"].queue_tool_call_response(
        tool_call_id="toolu_probe",
        tool_name="Probe",
        tool_input={},
    )
    in_memory_runtime["llm"].queue_response(text="done", stop_reason=StopReason.end_turn)

    await counting.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="catalog-skill",
            description="Always-listed catalog skill.",
            body_md="# catalog body",
        ),
    )
    await counting.create(
        engine.config.tenant_id,
        SkillUpsertInput(
            name="trigger-skill",
            description="Loaded only when triggered by command-name.",
            body_md="# trigger body\n\nDo the triggered thing.",
        ),
    )

    async for _evt in engine.run(
        Message(
            role=MessageRole.user,
            content_blocks=[
                TextBlock(text="run <command-name>trigger-skill</command-name>"),
            ],
        )
    ):
        pass

    # Two LLM rounds happened (tool-call + final), but the catalog was sourced
    # from the store exactly once for the whole run.
    assert len(in_memory_runtime["llm"].calls) >= 2
    assert counting.list_calls == 1, "catalog re-queried the store mid-run"

    system_text = _system_text(in_memory_runtime)
    # Catalog block present (reused across both rounds) ...
    assert "<system-reminder>" in system_text
    assert "catalog-skill" in system_text
    # ... and the per-turn trigger loaded the full body this run.
    assert "<loaded-skill" in system_text
    assert "Do the triggered thing." in system_text
