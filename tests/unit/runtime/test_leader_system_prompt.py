
"""Unit tests for ``leader_system.j2`` template — efficiency + error handling.

Forensic audit of template regression (76.8% -> 72.4%) identified
the "Error Handling" block's escape phrase ("finalize with declared
deliverables") as the regression cause: Qwen3.6 read it as "give up
early and emit empty contract". The template was revised to compress
ReadDedupCache + Error Handling into a single tight bilingual
"Efficiency & Error Handling" block removing the escape phrase.

These tests pin the bilingual guidance into the rendered leader system
prompt so future template edits cannot silently drop the safeguards:
- Read `unchanged: true` framed as "NOT an error" (re-use prior Read).
- Retry cap at TWICE (proven win for dbg-*/multi_tool_chain).
- Pivot to a DIFFERENT tool after 2 failed retries (no "finalize" escape).
- `<finalization_contract>` anchored AFTER Write/Edit/Bash work.
"""

from __future__ import annotations

import pytest

from protocore.prompts import JinjaPromptTemplateProvider


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    """Default provider pointing at the bundled templates."""
    return JinjaPromptTemplateProvider()


@pytest.fixture()
def minimal_ctx() -> dict[str, object]:
    """Minimal valid context — every optional var set to None."""
    return {
        "current_date": "2026-05-20",
        "persona_md": None,
        "agent_descriptions": None,
        "environment_capabilities": None,
        "capabilities": None,
        "finalization_contract_block": None,
    }


def test_leader_system_renders_without_jinja_syntax_errors(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """The template parses and renders cleanly.

    Catches accidental Jinja syntax errors introduced when editing the
    template (e.g. unbalanced ``{%- if %}``, missing ``{%- endif %}``).
    """
    rendered = provider.render("leader_system", minimal_ctx)
    # Sanity: rendered prompt is non-empty and contains the scaffolding line.
    assert rendered.strip(), "leader_system render produced empty output"
    assert "You are a Protocore agent." in rendered


def test_leader_system_requires_skill_before_office_and_search_before_mixed_file(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """The early scaffold, not the buried persona, must name Skill().

    A model follows the first tool doctrine it reads. A reminder placed at
    the END of the prompt lost to the earlier "prefer Glob/Read" line: runs
    that needed an office binary went hunting through /workspace/.skills/
    and never called Skill at all.
    """
    rendered = provider.render("leader_system", minimal_ctx)
    assert 'Skill(skill="docx")' in rendered
    assert "Read/Glob cannot open `/workspace/.skills/`" in rendered
    assert "Source-retrieval tools already on your list" in rendered
    assert "Do not ToolSearch for a named skill" in rendered
    skill_at = rendered.index('Skill(skill="docx")')
    persona_at = rendered.find("## Personality")
    if persona_at != -1:
        assert skill_at < persona_at


def test_leader_system_documents_language_matching_directive(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """An always-on, language-neutral directive instructs the
    agent to reply in the SAME language as the user's most recent message.

    Root cause of the language_mismatch failures: the bundled prompt carries
    a large always-on Russian error-handling block but had NO explicit
    top-level language-matching directive (the only "respond in the user's
    language" hint lived buried in the optional ``persona_md``). The local
    model then answered English prompts in Russian → the judge flagged
    language_mismatch and correctness/completeness collapsed. This directive
    is load-bearing scaffolding (NOT inside the persona conditional) so it
    reaches every run, persona or not.
    """
    rendered = provider.render("leader_system", minimal_ctx)
    # The core directive: reply in the user's most recent message language.
    assert (
        "ALWAYS write your reply in the SAME language as the user's most "
        "recent message." in rendered
    )
    # Code/identifiers/paths stay unchanged regardless of reply language.
    assert (
        "Keep code, identifiers, file paths, commands, and quoted output "
        "unchanged" in rendered
    )
    # Anchored near the TOP — before the tool/delegation scaffolding so the
    # model reads it first (it biases every subsequent token).
    language_idx = rendered.index("ALWAYS write your reply in the SAME language")
    delegate_idx = rendered.index("Delegate only when")
    assert language_idx < delegate_idx


def test_leader_system_language_directive_survives_full_context_with_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The language-matching directive is always on and renders
    even with a persona set and every optional context var populated.

    Guards against the directive accidentally being gated inside the persona
    conditional (which would reintroduce the buried-hint failure mode).
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-06-08",
            "persona_md": "# Custom\nYou are a benchmark-specific agent.",
            "agent_descriptions": {"coder": "Writes code"},
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "default",
                "network_allowed": True,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "capabilities": {
                "delegation": True,
                "delegation_max": 3,
                "planning": True,
            },
            "finalization_contract_block": (
                "<finalization_contract>example</finalization_contract>"
            ),
        },
    )
    # Persona renders (personality layer) AND the always-on directive too.
    assert "You are a benchmark-specific agent." in rendered
    assert (
        "ALWAYS write your reply in the SAME language as the user's most "
        "recent message." in rendered
    )


def test_leader_system_documents_read_dedup_cache_unchanged_semantics(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """``unchanged: true`` Read response is
    explicitly framed as NOT an error in the merged efficiency block (EN).

    Long-en-001 seed1 produced 137 identical Read calls because the model
    interpreted ``content=""`` + ``unchanged=True`` as "file vanished" and
    looped retrying. The compact merged block preserves the "NOT an error"
    framing + "re-use the prior Read" remedy.
    """
    rendered = provider.render("leader_system", minimal_ctx)
    # The key flag the model sees + "this is NOT an error" framing.
    assert "unchanged: true" in rendered
    assert "NOT an error" in rendered
    # Explicit re-use remedy (the model needs a clear next-action signal).
    assert "re-use the prior Read" in rendered


def test_leader_system_language_match_is_the_bilingual_switch(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """One language-match line, not a full Russian copy of the scaffold.

    A full RU mirror doubled the prompt and biased the model toward Russian.
    Reply language is the user's latest message; operating rules stay English.
    """
    rendered = provider.render("leader_system", minimal_ctx)
    assert "if in Russian, answer in Russian" in rendered
    assert "## Эффективность и обработка ошибок" not in rendered


def test_leader_system_dedup_guidance_survives_full_context_without_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The merged efficiency block renders even
    when every optional context var (except ``persona_md``) is populated
    (agents, env, capabilities, finalization contract). Guards against
    accidental conditional gating *inside* the generic-scaffolding branch.

    2026-05-27 prompt-contamination fix: when ``persona_md`` IS provided,
    the entire generic scaffolding (including this efficiency block) is
    suppressed by design. The companion test
    ``test_leader_system_persona_suppresses_dedup_guidance`` pins that
    behaviour. We assert the survive-with-rest-populated guarantee on
    the no-persona branch only.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-20",
            "persona_md": None,
            "agent_descriptions": {"coder": "Writes code"},
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "default",
                "network_allowed": True,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "capabilities": {
                "delegation": True,
                "delegation_max": 3,
                "planning": True,
            },
            "finalization_contract_block": (
                "<finalization_contract>example</finalization_contract>"
            ),
        },
    )
    assert "NOT an error" in rendered
    assert "re-use the prior Read" in rendered
    # Finalization contract still appended at the end.
    assert "<finalization_contract>" in rendered


def test_leader_system_persona_keeps_dedup_guidance(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Persona is an ADDITIVE personality layer — it must NOT drop the
    dedup/efficiency safeguards.

    A custom persona must not suppress the generic scaffolding. The
    catastrophic "no tools -> prose" failure was the empty BM25 surface
    (fixed by the forced tool-surface pins), NOT the generic scaffolding.
    Letting a persona drop the dedup/efficiency block (and the anti-leak
    ordering rule it carries) was itself a bug. The block is now always-on;
    the persona layers on top.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-20",
            "persona_md": "# Custom\nYou are a benchmark-specific agent.",
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    # Persona body is rendered (personality layer).
    assert "You are a benchmark-specific agent." in rendered
    # Generic scaffolding + dedup/efficiency safeguards SURVIVE the persona.
    assert "NOT an error" in rendered
    assert "re-use the prior Read" in rendered
    assert "You are a Protocore agent." in rendered


def test_leader_system_persona_with_finalization_off_keeps_scaffolding(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Persona + finalization gate OFF still emits the always-on tool-use
 scaffolding; only the contract block is gated off.

 The anti-leak ordering rule (the ``<finalization_contract>`` MUST come
 AFTER all Write/Edit/Bash work) is a load-bearing safeguard and renders
 even when the finalization-contract BLOCK itself is gated off — so the
 bare tag appears in the always-on Efficiency block, while the gated
 JSON-sentinel block does not.
 """

    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-27",
            "persona_md": "# Custom Persona\nYou drive final_answer.",
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": None,
        },
    )
    # Persona body renders verbatim (personality layer).
    assert "You drive final_answer." in rendered
    # Always-on tool-use scaffolding survives the persona.
    assert "You are a Protocore agent." in rendered
    # The anti-leak ordering rule (carries the bare contract tag) survives.
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    assert "<finalization_contract>" in rendered
    # The gated JSON-sentinel contract block is NOT emitted (gate off):
    # the block body opens with the tag immediately followed by a JSON brace.
    assert "<finalization_contract>\n{" not in rendered
    # Dedup/efficiency guidance survives.
    assert "NOT an error" in rendered


def test_leader_system_persona_with_finalization_on_emits_scaffolding_and_contract(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Companion to the persona+finalization-off test.

    When ``persona_md`` is set AND the finalization gate is enabled
    (``rc.finalization_gate_enabled=true`` → executor passes a non-None
    ``finalization_contract_block``), the rendered prompt contains the
    always-on tool-use scaffolding PLUS the persona body PLUS the contract
    block. INTENTIONALLY reverses the prior "contract-only" expectation:
    the persona is additive, so the generic "Protocore agent" scaffolding
    and its safeguards remain.
    """

    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-27",
            "persona_md": "# Persona\nNarrow tenant body.",
            "agent_descriptions": None,
            "environment_capabilities": None,
            "capabilities": None,
            "finalization_contract_block": (
                "<finalization_contract>X</finalization_contract>"
            ),
        },
    )
    # Persona body renders verbatim (personality layer).
    assert "Narrow tenant body." in rendered
    # Contract block renders verbatim (RC-gated by finalization_gate_enabled).
    assert "<finalization_contract>X</finalization_contract>" in rendered
    # Always-on tool-use scaffolding + dedup/efficiency safeguards survive.
    assert "You are a Protocore agent." in rendered
    assert "NOT an error" in rendered


def test_leader_system_documents_retry_budget_cap_en(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """Leader prompt caps retries at TWICE and
 redirects to a DIFFERENT tool — no "finalize" escape clause (EN).

 Eval data (49% Bash-error streaks ≥3) drove the retry cap. An escape
 phrase ("finalize with declared deliverables") was removed because the
 model misread it as "give up early"; the contract is now anchored AFTER
 all mutation work.
 """
    rendered = provider.render("leader_system", minimal_ctx)
    # Section header present (merged efficiency + error handling).
    assert "## Efficiency & Error Handling" in rendered
    # Retry budget directive — exact canonical phrasing.
    assert "Retry at most TWICE" in rendered
    # Pivot guidance: try a DIFFERENT tool, do not give up.
    assert "try a DIFFERENT tool" in rendered
    # Anchoring: contract comes AFTER all mutation work.
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    # Escape phrase MUST be gone — it caused premature finalization.
    assert "finalize with declared deliverables" not in rendered


def test_leader_system_has_no_russian_operating_mirror(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """Operating rules are English; reply language is the user's message."""
    rendered = provider.render("leader_system", minimal_ctx)
    assert "## Эффективность и обработка ошибок" not in rendered
    assert "не более ДВУХ раз" not in rendered
    assert "if in Russian, answer in Russian" in rendered


def test_leader_system_retry_cap_survives_full_context_without_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Retry-cap + pivot block renders with every
    optional context var (except ``persona_md``) populated. Guards
    against accidental conditional gating inside the generic branch.

    See ``test_leader_system_persona_suppresses_dedup_guidance`` for the
    persona-suppression behaviour (2026-05-27 prompt-contamination fix).
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-20",
            "persona_md": None,
            "agent_descriptions": {"coder": "Writes code"},
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "default",
                "network_allowed": True,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "capabilities": {
                "delegation": True,
                "delegation_max": 3,
                "planning": True,
            },
            "finalization_contract_block": (
                "<finalization_contract>example</finalization_contract>"
            ),
        },
    )
    assert "## Efficiency & Error Handling" in rendered
    assert "Retry at most TWICE" in rendered
    assert "try a DIFFERENT tool" in rendered


def test_leader_system_documents_large_output_chunking_en(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """Leader prompt instructs the model to chunk LARGE outputs across
 the Write -> AppendFile -> FinalizeFile protocol (EN).

 Models fail when attempting a single oversized ``Write`` call that
 exhausts the token budget mid-call, truncating the artifact. The
 prompt teaches the real atomic protocol: seed with ``Write``, append
 each chunk with ``AppendFile``, then ``FinalizeFile`` to seal."""
    rendered = provider.render("leader_system", minimal_ctx)
    # Threshold trigger so the model knows when chunking applies.
    assert "LARGE outputs (>2 kB or >50 lines)" in rendered
    # Seed-then-append-then-finalize pattern (the real protocol).
    assert "first chunk with `Write`" in rendered
    assert "`AppendFile`" in rendered
    assert "`FinalizeFile`" in rendered
    # The wrong Edit-append instruction must be gone.
    assert "`Edit` (append mode)" not in rendered
    # Anti-pattern call-out: do NOT attempt one big Write.
    assert "Do NOT" in rendered and "one `Write` call" in rendered
    assert "token budgets can truncate" in rendered


def test_leader_system_chunking_is_not_duplicated_in_russian(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    rendered = provider.render("leader_system", minimal_ctx)
    assert "LARGE outputs (>2 kB or >50 lines)" in rendered
    assert "БОЛЬШИХ выводов (>2 KB или >50 строк)" not in rendered


def test_leader_system_chunking_guidance_survives_full_context_without_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Chunking guidance renders with every
    optional context var (except ``persona_md``) populated. Guards
    against accidental conditional gating inside the generic branch.

    2026-05-27 prompt-contamination fix: with ``persona_md`` set, the
    chunking guidance is suppressed because Write/Edit references in
    the generic block primed runs that have a narrower tool surface
    (e.g. ``remote_write``) toward the wrong tool names.
    """
    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-20",
            "persona_md": None,
            "agent_descriptions": {"coder": "Writes code"},
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "default",
                "network_allowed": True,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "capabilities": {
                "delegation": True,
                "delegation_max": 3,
                "planning": True,
            },
            "finalization_contract_block": (
                "<finalization_contract>example</finalization_contract>"
            ),
        },
    )
    assert "LARGE outputs (>2 kB or >50 lines)" in rendered
    assert "first chunk with `Write`" in rendered


def test_leader_system_documents_source_discipline_en(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """The always-on SOURCE DISCIPLINE rule renders in EN: cite only
    tool-returned sources, never fabricate provenance, and (generically)
    delegate source-gathering to a configured subagent via the subagent
    catalog — NO hardcoded tool/subagent/client name. Guards the
    anti-fabrication fix against a silent prompt-cleanup deletion."""

    rendered = provider.render("leader_system", minimal_ctx)
    assert "SOURCE DISCIPLINE" in rendered
    assert "the ONLY trustworthy sources are hits a tool actually returned" in rendered
    assert "NEVER invent or write from memory any source" in rendered
    # Generic delegate-for-sources rule (no client/tool/subagent name).
    assert "a configured subagent may provide them" in rendered
    assert "consult your subagent catalog" in rendered
    # The honest-count stop rule (no padding).
    assert "say so plainly and stop — do not pad" in rendered


def test_leader_system_source_discipline_is_not_duplicated_in_russian(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    rendered = provider.render("leader_system", minimal_ctx)
    assert "SOURCE DISCIPLINE" in rendered
    assert "ДИСЦИПЛИНА ИСТОЧНИКОВ" not in rendered


def test_leader_system_source_discipline_coexists_with_silent_delegation(
    provider: JinjaPromptTemplateProvider,
    minimal_ctx: dict[str, object],
) -> None:
    """The SOURCE DISCIPLINE rule must COEXIST with the silent-delegation
    rule (both are leader-routing safeguards): source discipline says
    'delegate source-gathering to a configured subagent', the delegation rule
    says 'delegate SILENTLY'. Both render together, EN + RU, so a future edit
    dropping either is caught."""

    rendered = provider.render("leader_system", minimal_ctx)
    assert "Delegate SILENTLY" in rendered
    assert "SOURCE DISCIPLINE" in rendered


def test_leader_system_source_discipline_survives_full_context_with_persona(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """Source discipline is always-on scaffolding — it renders even with a
    persona + every optional context var populated (additive contract)."""

    rendered = provider.render(
        "leader_system",
        {
            "current_date": "2026-05-20",
            "persona_md": "# Aurora\nYou are warm.",
            "agent_descriptions": {"researcher": "Source-backed writer"},
            "environment_capabilities": {
                "file_read": True,
                "file_write": True,
                "shell_profile": "default",
                "network_allowed": True,
                "package_install": False,
                "server_hosting": False,
                "long_running_processes": False,
            },
            "capabilities": {
                "delegation": True,
                "delegation_max": 3,
                "planning": True,
            },
            "finalization_contract_block": (
                "<finalization_contract>example</finalization_contract>"
            ),
        },
    )
    # Persona renders AND the always-on source-discipline rule (EN + RU) too.
    assert "You are warm." in rendered
    assert "SOURCE DISCIPLINE" in rendered
