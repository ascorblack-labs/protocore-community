"""Byte-stability test for the system-prompt assembler.

System prompt assembly self-check: hash audit + factory contract that
forbids per-turn-mutating fields below day resolution.

Failure mode this test guards against: a stray ``datetime.now`` /
``uuid.uuid4`` / random IDs get accidentally introduced into the
cacheable prefix, busting the vLLM prefix cache on every turn (the
"invisible mutator" anti-pattern — most expensive in production).

Strategy: build the same context bundle twice in quick succession and
assert the resulting system_prompt_sections are byte-identical. Any
new injection of timestamp/UUID/random data into the prefix fails this
test.

CI gate: the test must remain in the strict default protocore pytest
collection so a regression on the prompt assembler is caught
pre-merge.
"""
from __future__ import annotations

import hashlib

import pytest

from protocore.contracts.runtime_constants import RuntimeConstants
from protocore.contracts.skills import SkillUpsertInput
from protocore.contracts.types import (
    Message,
    MessageRole,
    StopReason,
    TextBlock,
)


def _sections_hash(sections: tuple[str, ...]) -> str:
    """SHA-256 of the assembled prefix — stable across runs if the
    assembly path is deterministic."""
    joined = "\n\n".join(sections)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_skill_index_byte_stable_across_calls(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Two consecutive turns with identical user input must produce the
    same skill-index ``<system-reminder>`` block byte-for-byte."""
    rc = RuntimeConstants(
        model_context_window=100_000,
        skill_index_budget_ratio=0.05,
    )
    engine_one = engine_factory(rc=rc, run_id="run-stability-a")
    engine_two = engine_factory(rc=rc, run_id="run-stability-b")

    # Same tenant so both engines see the same skill set.
    tenant_id = engine_one.config.tenant_id
    engine_two_config_dict = engine_two.config.__dict__
    engine_two_config_dict["tenant_id"] = tenant_id  # ensure shared scope

    for _engine in (engine_one, engine_two):
        in_memory_runtime["llm"].queue_response(
            text="ok", stop_reason=StopReason.end_turn
        )

    skill_store = in_memory_runtime["skills"]
    await skill_store.create(
        tenant_id,
        SkillUpsertInput(
            name="git-flow",
            description="Use to commit changes.",
            body_md="# body",
        ),
    )
    await skill_store.create(
        tenant_id,
        SkillUpsertInput(
            name="safety-check",
            description="An account safety net.",
            body_md="# safety body",
        ),
    )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="commit and push these changes")],
    )

    async for _ in engine_one.run(user_msg):
        pass
    async for _ in engine_two.run(user_msg):
        pass

    calls = in_memory_runtime["llm"].calls
    # Two engines, one assistant call each = at least two calls.
    assert len(calls) >= 2

    def _system_text(call) -> str:
        return "\n".join(
            block.text
            for msg in call.messages
            if msg.role is MessageRole.system
            for block in msg.content_blocks
            if hasattr(block, "text")
        )

    system_one = _system_text(calls[0])
    system_two = _system_text(calls[-1])
    assert hashlib.sha256(system_one.encode()).hexdigest() == hashlib.sha256(
        system_two.encode()
    ).hexdigest(), (
        "system prompt assembly produced different bytes for identical "
        "inputs — likely a per-turn mutator (datetime.now / uuid / "
        "random) leaked into the prefix"
    )


@pytest.mark.asyncio
async def test_no_skills_byte_stable(
    engine_factory,
    in_memory_runtime,
) -> None:
    """Skill-store empty path: assembly stays byte-stable even with no
    skills (the no-op path must still be deterministic)."""
    rc = RuntimeConstants(model_context_window=8_000)
    engine_one = engine_factory(rc=rc, run_id="run-empty-a")
    engine_two = engine_factory(rc=rc, run_id="run-empty-b")

    for _engine in (engine_one, engine_two):
        in_memory_runtime["llm"].queue_response(
            text="ok", stop_reason=StopReason.end_turn
        )

    user_msg = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="some user prompt")],
    )

    async for _ in engine_one.run(user_msg):
        pass
    async for _ in engine_two.run(user_msg):
        pass

    calls = in_memory_runtime["llm"].calls
    assert len(calls) >= 2

    def _system_text(call) -> str:
        return "\n".join(
            block.text
            for msg in call.messages
            if msg.role is MessageRole.system
            for block in msg.content_blocks
            if hasattr(block, "text")
        )

    assert _system_text(calls[0]) == _system_text(calls[-1])


@pytest.mark.asyncio
async def test_context_manager_sections_byte_stable_directly() -> None:
    """Unit test on the assembler — independent of the LLM round-trip.

    Builds the same ContextBundle twice via ContextManager and asserts
    byte equality of system_prompt_sections.
    """
    from protocore.runtime.context.manager import ContextManager
    from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider

    rc = RuntimeConstants(model_context_window=10_000)
    cm = ContextManager(
        rc=rc,
        blob_store=InMemoryBlobStore(),
        compaction_llm=InMemoryLLMProvider(),
    )

    user = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="deterministic input")],
    )
    sections = ("system block one", "<context>\n<cwd>/tmp</cwd>\n</context>")

    bundle_one = cm.build_context(
        history=[user],
        tools=[],
        system_prompt_sections=sections,
        skill_index_block="<system-reminder>\nskills X Y Z\n</system-reminder>",
    )
    bundle_two = cm.build_context(
        history=[user],
        tools=[],
        system_prompt_sections=sections,
        skill_index_block="<system-reminder>\nskills X Y Z\n</system-reminder>",
    )

    hash_one = _sections_hash(bundle_one.system_prompt_sections)
    hash_two = _sections_hash(bundle_two.system_prompt_sections)
    assert hash_one == hash_two, (
        "ContextManager.build_context produced non-deterministic output"
    )
