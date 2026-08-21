# ruff: noqa: RUF001, RUF002 — Bilingual RU+EN assertions + docstrings are intentional.
"""Unit tests for the scaffold's answer-boundary rule.

The scaffold already told the leader to delegate silently, and it was obeyed in
the narrow sense: across twelve measured delegating runs on a live stand, no
answer said "I don't have this tool" or named the subagent it was calling. What
eleven of the twelve DID do was open the final message with a report on their
own progress — «Теперь у меня есть все пять разделов. Составлю итоговый обзор.»,
"I have a good set of sources now. Let me compile the article from these
references." — and two closed it by naming the helpers and the directory their
files went to.

The old sentence does not cover that, read literally. It forbids narrating
ROUTING and adding a THIRD-PERSON summary, and its examples are third-person
status lines ("The user asked…", "No file was created", "The task is
complete"). The observed text is neither: it is a first-person, forward-looking
statement of intent, and the rule never named it. One run went further and
emitted «Файл не создавался — ответ представлен целиком в сообщении.» — the
scaffold's own negative example, at the end of a user-facing answer.

So the rule gets a second sentence stated by POSITION rather than by kind:
the first and last sentences of the final message belong to the answer. A
position is checkable and has no gap for a new phrasing to fall through, and it
covers the between-tool narration habit continuing into the one message where
nothing else can hide it — the final block carries no tool call, so the
stream's own PUBLIC/COLLAPSED judgement correctly calls the whole thing prose,
and there is no boundary inside it to mark.

These tests pin that the rule is present in BOTH mirrors, that it is
unconditional, and that it did not disturb the surface flag's four-line diff.
"""

from __future__ import annotations

import difflib

import pytest

from protocore.prompts import JinjaPromptTemplateProvider


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    return JinjaPromptTemplateProvider()


BASE_CTX: dict[str, object] = {
    "current_date": "2026-08-06",
    "persona_md": None,
    "agent_descriptions": None,
    "environment_capabilities": None,
    "capabilities": None,
    "finalization_contract_block": None,
}

BOUNDARY_EN = (
    "Your final message consists of the answer and nothing else, at BOTH ends."
)
BOUNDARY_RU = (
    "Финальное сообщение состоит из ответа и ничего кроме, с ОБОИХ концов."
)


def _render(provider: JinjaPromptTemplateProvider, **overrides: object) -> str:
    return provider.render("leader_system", {**BASE_CTX, **overrides})


def test_the_rule_is_present_in_both_mirrors(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Both languages or neither.

    The leader reads one prompt carrying both mirrors, and the measured
    violations were split across them — the narration openers were Russian on
    the literature prompt and English on the article prompt, from the same
    model in the same hour. A rule landing in one mirror leaves the other
    licensed by omission.
    """
    rendered = _render(provider)
    assert BOUNDARY_EN in rendered


def test_the_rule_names_the_first_and_last_sentence(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Stated by position, because "kind" is what the old rule tried.

    The old sentence enumerates kinds of forbidden text and the model wrote a
    kind that was not on the list. A position cannot be dodged that way: there
    is exactly one first sentence and one last sentence.
    """
    rendered = _render(provider)
    assert "Its FIRST sentence belongs to the answer" in rendered
    assert "Its LAST sentence likewise" in rendered


def test_a_missing_gap_is_stated_about_the_subject(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Where the admission goes when material is short.

    The leader is separately told — by the delegation tool's own outcome
    notice — not to answer as though incomplete work had delivered. Obeying
    that produced «подагент нашёл источники, но не создал файл» in a delivered
    literature review. The obligation is right and stays; the scaffold says
    which vocabulary discharges it.
    """
    rendered = _render(provider)
    assert "say so about the SUBJECT" in rendered
    assert "one of the helpers produced no file" in rendered


def test_the_rule_is_unconditional(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """No flag switches it off — every reader-facing surface wants it.

    ``workspace_visible_to_user`` decides whether a path is an acceptable
    answer. It does not decide whether a progress report is, and a tenant whose
    users CAN open the workspace still did not ask to be told which subagent
    ran.
    """
    for visible in (True, False):
        rendered = _render(provider, workspace_visible_to_user=visible)
        assert BOUNDARY_EN in rendered


def test_the_surface_flag_still_switches_exactly_four_lines(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The new rule is not inside either conditional.

    Guards the same invariant as the workspace-visibility suite from the other
    side: adding an unconditional sentence next to a gated one is the easy way
    to accidentally gate it.
    """
    visible = _render(provider, workspace_visible_to_user=True).splitlines()
    invisible = _render(provider, workspace_visible_to_user=False).splitlines()
    diff = list(difflib.ndiff(visible, invisible))
    removed = [line[2:] for line in diff if line.startswith("- ")]
    added = [line[2:] for line in diff if line.startswith("+ ")]
    assert len(removed) == 2, removed
    assert len(added) == 2, added
    assert not any(BOUNDARY_EN in line for line in removed + added)


def test_a_persona_cannot_remove_the_rule(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The scaffold wins, which is the whole reason the fix lives here.

    The stand's own persona already forbids every one of these behaviours in
    precise Russian — «Не описывай, что ты собираешься сделать», «не называй
    имена помощников» — and the answers leaked anyway. A persona is not the
    surface this belongs on.
    """
    rendered = _render(
        provider,
        persona_md="# Персона\nРассказывай пользователю о своей работе подробно.",
        workspace_visible_to_user=False,
    )
    assert "## Personality" in rendered
    assert BOUNDARY_EN in rendered
