"""Unit tests for ``leader_system.j2`` — persona is ADDITIVE.

A ``persona_md`` must NOT replace the bundled tool-use scaffolding.
The catastrophic "no tools -> prose" failure was caused by the empty BM25
surface, NOT the generic scaffolding. The real fix is to ensure the
always-on safeguards render regardless of whether a persona is set:

* the bundled tool-use scaffolding ("You are a Protocore agent" + full-tool
  access framing),
* the always-on language-matching directive (reply in the SAME language as the
  user's most recent message),
* the anti-leak ordering rule (``<finalization_contract>`` MUST come AFTER all
  Write/Edit/Bash work),
* the large-output chunking rule (seed with ``Write`` then append with ``Edit``),

``persona_md`` is appended on top as a PERSONALITY layer. These tests pin that
contract so a future edit cannot silently regress to persona-replaces-scaffolding.

The companion suite ``tests/unit/runtime/test_leader_system_prompt.py`` pins the
exact bilingual safeguard strings on the no-persona path; this file pins the
*additive* guarantee on the with-persona path.
"""

from __future__ import annotations

import pytest

from protocore.prompts import JinjaPromptTemplateProvider


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    """Default provider pointing at the bundled templates."""
    return JinjaPromptTemplateProvider()


PERSONA = "# Aurora\nYou are warm."


def test_persona_layers_on_top_of_always_on_scaffolding(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """With a non-empty persona the bundled tool-use scaffolding + the
    anti-leak ordering rule are STILL present, and the persona text is too.

    This is the core contract: persona is additive (personality on top),
    not a wholesale replacement of the safeguards.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-06-05",
            "persona_md": PERSONA,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    # Bundled tool-use scaffolding survives the persona.
    assert "You are a Protocore agent." in rendered
    # Always-on language-matching directive survives the persona (EN).
    assert (
        "ALWAYS write your reply in the SAME language as the user's most "
        "recent message." in rendered
    )
    # Anti-leak ordering rule survives (EN) — the load-bearing safeguard.
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    # Large-output chunking rule survives (EN).
    assert "first chunk with `Write`" in rendered
    # Persona body is appended on top as a personality layer.
    assert "# Aurora" in rendered
    assert "You are warm." in rendered


def test_persona_does_not_restore_a_russian_operating_mirror(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Reply language is the user's message; the scaffold stays English."""
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-06-05",
            "persona_md": PERSONA,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    assert "ТОЛЬКО ПОСЛЕ всех Write/Edit/Bash" not in rendered


def test_persona_appears_after_scaffolding_under_a_personality_header(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Persona is appended AFTER the scaffolding (personality layer), under a
    ``## Personality`` header so the model reads safeguards first.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-06-05",
            "persona_md": PERSONA,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    assert "## Personality" in rendered
    scaffolding_idx = rendered.index("You are a Protocore agent.")
    personality_idx = rendered.index("## Personality")
    persona_idx = rendered.index("You are warm.")
    # Scaffolding renders before the personality layer.
    assert scaffolding_idx < personality_idx < persona_idx


def test_persona_with_finalization_block_keeps_scaffolding_and_contract(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Persona + finalization gate ON: scaffolding, persona, AND the
    independently-gated contract block all render together.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-06-05",
            "persona_md": PERSONA,
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": (
                "<finalization_contract>X</finalization_contract>"
            ),
        },
    )
    # Always-on scaffolding.
    assert "You are a Protocore agent." in rendered
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    # Persona personality layer.
    assert "You are warm." in rendered
    # RC-gated contract block.
    assert "<finalization_contract>X</finalization_contract>" in rendered


def test_empty_persona_output_unchanged_from_no_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """No-regression guard: with ``persona_md`` empty/None the rendered output
    is byte-identical to today's no-persona scaffolding (no ``## Personality``
    header, no persona body), so the additive change does not perturb the
    universal no-persona path that every default-tenant run takes.
    """
    ctx_none = {
        "current_date": "2026-06-05",
        "persona_md": None,
        "agent_descriptions": None,
        "environment_capabilities": None,
        "capabilities": None,
        "finalization_contract_block": None,
    }
    rendered_none = provider.render("leader_system", ctx_none)
    # Scaffolding present; no personality layer emitted.
    assert "You are a Protocore agent." in rendered_none
    # Always-on language-matching directive present on the no-persona path too.
    assert (
        "ALWAYS write your reply in the SAME language as the user's most "
        "recent message." in rendered_none
    )
    assert "## Personality" not in rendered_none
    assert "# Aurora" not in rendered_none
    # Empty-string persona must behave exactly like None (falsy) — no header.
    rendered_empty = provider.render(
        "leader_system", {**ctx_none, "persona_md": ""}
    )
    assert rendered_empty == rendered_none
