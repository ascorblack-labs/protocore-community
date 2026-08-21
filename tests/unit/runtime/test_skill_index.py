"""Tests for ``protocore.runtime.skill_index`` — catalog rendering,
alphabetical ordering, names-only degrade, derived budget calculation.

Pure-core renderer: no PG, no embeddings, no ranking.
"""
from __future__ import annotations

import pytest

from protocore.contracts.skills import SkillIndexEntry
from protocore.runtime.skill_index import (
    derive_skill_index_budget_tokens,
    render_skills_catalog,
)


def _entry(name: str, desc: str) -> SkillIndexEntry:
    return SkillIndexEntry(id=name, name=name, description=desc)


async def _whitespace_token_counter(text: str) -> int:
    # Approximate token count = whitespace-split word count.
    return len(text.split())


def test_derive_budget_from_ratio() -> None:
    assert derive_skill_index_budget_tokens(
        model_context_window=49_152,
        skill_index_budget_ratio=0.01,
    ) == 491
    assert derive_skill_index_budget_tokens(
        model_context_window=0,
        skill_index_budget_ratio=0.01,
    ) == 0
    assert derive_skill_index_budget_tokens(
        model_context_window=100,
        skill_index_budget_ratio=0.5,
    ) == 50


@pytest.mark.asyncio
async def test_render_empty_entries_returns_empty() -> None:
    out = await render_skills_catalog(
        [], token_counter=_whitespace_token_counter, budget_tokens=1000
    )
    assert out == ""


@pytest.mark.asyncio
async def test_render_system_reminder_format() -> None:
    out = await render_skills_catalog(
        [_entry("a", "alpha desc")],
        token_counter=_whitespace_token_counter,
        budget_tokens=10_000,
    )
    assert out.startswith("<system-reminder>")
    assert "Skills are tools, not files" in out
    assert 'Skill(skill="<name>")' in out
    assert "/workspace/.skills/" in out
    assert 'Skill(skill="a") — alpha desc' in out
    assert out.endswith("</system-reminder>")


@pytest.mark.asyncio
async def test_render_is_alphabetical_by_name() -> None:
    out = await render_skills_catalog(
        [
            _entry("zebra", "z desc"),
            _entry("alpha", "a desc"),
            _entry("mango", "m desc"),
        ],
        token_counter=_whitespace_token_counter,
        budget_tokens=10_000,
    )
    assert (
        out.index('Skill(skill="alpha")')
        < out.index('Skill(skill="mango")')
        < out.index('Skill(skill="zebra")')
    )


@pytest.mark.asyncio
async def test_render_skips_blank_description() -> None:
    out = await render_skills_catalog(
        [_entry("bare", "   ")],
        token_counter=_whitespace_token_counter,
        budget_tokens=10_000,
    )
    # Blank description renders as a bare call shape (no em-dash description).
    assert 'Skill(skill="bare")' in out
    assert 'Skill(skill="bare") —' not in out


@pytest.mark.asyncio
async def test_render_zero_budget_is_unbounded() -> None:
    # budget_tokens <= 0 means no degrade — the full block is returned.
    out = await render_skills_catalog(
        [_entry("a", "a " * 100)],
        token_counter=_whitespace_token_counter,
        budget_tokens=0,
    )
    assert 'Skill(skill="a") —' in out


@pytest.mark.asyncio
async def test_render_degrades_to_names_only_when_over_budget() -> None:
    entries = [_entry(f"s{i}", "word " * 40) for i in range(5)]
    out = await render_skills_catalog(
        entries,
        token_counter=_whitespace_token_counter,
        budget_tokens=30,
    )
    # Over budget → every line is a bare call shape (no descriptions).
    for i in range(5):
        assert f'Skill(skill="s{i}")' in out
        assert f'Skill(skill="s{i}") — word' not in out
    # Call-shape-only block is still alphabetical + wrapped.
    assert out.startswith("<system-reminder>")
    assert out.index('Skill(skill="s0")') < out.index('Skill(skill="s4")')


@pytest.mark.asyncio
async def test_render_keeps_descriptions_when_within_budget() -> None:
    entries = [_entry("a", "short"), _entry("b", "also short")]
    out = await render_skills_catalog(
        entries,
        token_counter=_whitespace_token_counter,
        budget_tokens=10_000,
    )
    assert 'Skill(skill="a") — short' in out
    assert 'Skill(skill="b") — also short' in out


@pytest.mark.asyncio
async def test_render_degrade_decision_is_deterministic() -> None:
    """Same entries + budget → byte-identical output across calls (the degrade
    decision is content-stable so the catalog can be built once per run)."""
    entries = [_entry(f"s{i}", "word " * 40) for i in range(6)]
    out_one = await render_skills_catalog(
        entries, token_counter=_whitespace_token_counter, budget_tokens=25
    )
    out_two = await render_skills_catalog(
        entries, token_counter=_whitespace_token_counter, budget_tokens=25
    )
    assert out_one == out_two
