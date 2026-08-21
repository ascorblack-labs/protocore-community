# ruff: noqa: RUF001 — Bilingual RU+EN assertions intentionally use Cyrillic characters.
"""Unit tests for ``leader_system.j2`` — the surface decides two directives.

The scaffold is wrapped around every leader persona, so whatever it says is
said to every tenant. Two of its sentences assumed a reader who can open a
file browser:

* the file-receipt directive — "your chat reply MUST reference it … and MUST
  NOT reproduce its full content in chat", plus "Completeness applies to the
  written file, not to your chat message";
* the large-output directive — "For LARGE outputs (>2 kB or >50 lines), write
  the first chunk with ``Write`` …", which reads as *when the answer is long,
  put it in a file*.

For a chat-only tenant both are wrong, and they are obeyed: asked for a
~1000-word article the agent writes the article to a file and replies with the
path, the byte count and a table of contents — the scaffold's own instruction,
followed to the letter, and an empty answer to a reader who cannot open it. A
per-tenant persona demanding the opposite does not help, because the scaffold
is deliberately not replaceable by a persona.

``workspace_visible_to_user`` therefore selects the wording. These tests pin
three things a later edit could quietly break:

1. the default is the OLD wording, byte-for-byte, so no deployment changes
   behaviour by upgrading;
2. with the workspace invisible, neither directive's receipt-shaped text
   survives in EITHER language — production traffic is RU+EN, so a directive
   that flips in English only recreates the contradiction for half of it;
3. the chunked-write protocol survives regardless, because a user who asks for
   a file still needs the file to be written correctly. Only the framing that
   treated length as a reason to file the answer is gone.
"""

from __future__ import annotations

import difflib

import pytest

from protocore.prompts import JinjaPromptTemplateProvider


@pytest.fixture()
def provider() -> JinjaPromptTemplateProvider:
    """Default provider pointing at the bundled templates."""
    return JinjaPromptTemplateProvider()


BASE_CTX: dict[str, object] = {
    "current_date": "2026-08-06",
    "persona_md": None,
    "agent_descriptions": None,
    "environment_capabilities": None,
    "capabilities": None,
    "finalization_contract_block": None,
}

# The four sentences the flag switches, in the wording every tenant gets today.
# Pinned verbatim: these ARE the "unchanged" the default promises, so a test
# that paraphrased them would pass over an edit that reworded them.
VISIBLE_RECEIPT_EN = (
    "When a deliverable has been written to a file or artifact, your chat "
    "reply MUST reference it — give the exact path and a concise summary"
)
VISIBLE_RECEIPT_COMPLETENESS_EN = (
    "Completeness applies to the written file, not to your chat message."
)
VISIBLE_RECEIPT_RU = (
    "Когда результат записан в файл или артефакт, ваш ОТВЕТ В ЧАТЕ ДОЛЖЕН "
    "ссылаться на него — указать точный путь и краткое содержание"
)
VISIBLE_RECEIPT_COMPLETENESS_RU = (
    "Требование полноты относится к записанному файлу, а не к сообщению в чате."
)
VISIBLE_CHUNKING_EN = (
    "For LARGE outputs (>2 kB or >50 lines), write the first chunk with `Write`"
)
VISIBLE_CHUNKING_RU = (
    "Для БОЛЬШИХ выводов (>2 KB или >50 строк), пишите первую часть через `Write`"
)


def _render(
    provider: JinjaPromptTemplateProvider, **overrides: object
) -> str:
    return provider.render("leader_system", {**BASE_CTX, **overrides})


# ---------------------------------------------------------------------------
# The default is today's prompt
# ---------------------------------------------------------------------------


def test_absent_flag_renders_identically_to_visible(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """A caller that never heard of the flag gets the visible-workspace prompt.

    The template guards with ``| default(true)`` and the service loader sets the
    variable explicitly rather than through its default-to-``None`` loop, because
    ``None`` is falsy in Jinja: a variable that is merely *listed* and never
    *set* would hand the chat-only wording to every tenant. Both halves of that
    invariant matter, and this is the template half.
    """
    assert _render(provider) == _render(provider, workspace_visible_to_user=True)


def test_visible_workspace_keeps_both_directives_verbatim(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """With a file browser in the surface, a path IS an answer — nothing changes.

    Pinned in both languages: the receipt directive and the chunking directive
    keep the exact wording they had before the flag existed.
    """
    rendered = _render(provider, workspace_visible_to_user=True)
    assert VISIBLE_RECEIPT_EN in rendered
    assert VISIBLE_RECEIPT_COMPLETENESS_EN in rendered
    assert VISIBLE_CHUNKING_EN in rendered


def test_flag_changes_exactly_the_four_directive_lines(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The flag is a wording switch, not a restructuring of the scaffold.

    Exactly four lines differ — file receipt and large output, once per
    language. Anything else moving is a regression: the rest of the scaffold
    (tool preference, source discipline, delegation, the anti-leak ordering
    rule) is about how the agent works, not about who reads the reply.
    """
    visible = _render(provider, workspace_visible_to_user=True).splitlines()
    invisible = _render(provider, workspace_visible_to_user=False).splitlines()
    diff = list(difflib.ndiff(visible, invisible))
    removed = [line[2:] for line in diff if line.startswith("- ")]
    added = [line[2:] for line in diff if line.startswith("+ ")]
    assert len(removed) == 2, removed
    assert len(added) == 2, added
    assert VISIBLE_RECEIPT_EN in removed[0]
    assert VISIBLE_CHUNKING_EN in removed[1]


# ---------------------------------------------------------------------------
# Chat-only surface: the reply is the deliverable
# ---------------------------------------------------------------------------


def test_invisible_workspace_drops_the_file_receipt_directive(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """No fragment of the receipt instruction may survive, in either language.

    Each of these is independently sufficient to produce the failure: telling
    the model to answer with a path, telling it NOT to reproduce the content,
    telling it completeness belongs to the file, or telling it that reading its
    own artifact is verification context rather than material for the answer.
    """
    rendered = _render(provider, workspace_visible_to_user=False)
    # English.
    assert "MUST reference it" not in rendered
    assert "give the exact path and a concise summary" not in rendered
    assert "MUST NOT reproduce its full content in chat" not in rendered
    assert VISIBLE_RECEIPT_COMPLETENESS_EN not in rendered
    assert "verification context, not text to copy into your answer" not in rendered
    # Russian.
    assert "ссылаться на него" not in rendered
    assert "указать точный путь и краткое содержание" not in rendered
    assert "НЕ должен воспроизводить его полное содержимое в чате" not in rendered
    assert VISIBLE_RECEIPT_COMPLETENESS_RU not in rendered
    assert "контекстом для проверки, а не текстом" not in rendered


def test_invisible_workspace_puts_the_substance_in_the_reply(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The replacement states the reader's situation and inverts the demand.

    Deleting the directive outright would leave the model to guess; the
    replacement says the reply is the only thing the user sees, that
    completeness therefore applies to it, and — naming the measured failure
    exactly — that a path plus a byte count plus a section list is not an
    answer. Writing the file is still required when it was asked for.
    """
    rendered = _render(provider, workspace_visible_to_user=False)
    # English.
    assert "The user reads ONLY this message and cannot open anything you write" in rendered
    assert "the reply itself is the deliverable and completeness applies to it" in rendered
    assert "the file is never the answer" in rendered
    assert "a path with a byte count and a list of sections is an empty reply" in rendered
    assert "required when the task asked for one" in rendered


def test_invisible_workspace_keeps_the_chunked_write_protocol(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The chunking mechanics are a token-budget fact, not a filing policy.

    A single oversized ``Write`` truncates mid-call whoever is reading, so a
    user who explicitly asked for a large file still needs the seed-append-seal
    protocol. What the chat-only wording drops is the *reason* — "for LARGE
    outputs", which invited the model to treat its own answer's length as
    grounds to move it into a file.
    """
    rendered = _render(provider, workspace_visible_to_user=False)
    # The framing that made length a reason to file the answer is gone.
    assert "For LARGE outputs" not in rendered
    assert "Для БОЛЬШИХ выводов" not in rendered
    # The protocol itself survives, keyed to a file the task asked for (EN).
    assert "Length is never a reason to move an answer out of the reply" in rendered
    assert "When you are writing a FILE the task asked for" in rendered
    assert "the first chunk with `Write`" in rendered
    assert "`AppendFile` for each subsequent chunk" in rendered
    assert "`FinalizeFile` to seal the file" in rendered
    assert "Do NOT use `Edit` for appending" in rendered
    assert "token budgets can truncate mid-call" in rendered


def test_invisible_workspace_leaves_the_other_safeguards_alone(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The always-on scaffolding is untouched by the surface flag.

    These are the safeguards a persona is not allowed to drop; the flag is not
    a licence to drop them either. Listed explicitly because the alternative —
    trusting the four-line diff test alone — would not survive someone
    reorganising the template.
    """
    rendered = _render(provider, workspace_visible_to_user=False)
    assert "You are a Protocore agent." in rendered
    assert (
        "ALWAYS write your reply in the SAME language as the user's most "
        "recent message." in rendered
    )
    assert "SOURCE DISCIPLINE:" in rendered
    assert "MUST come AFTER all Write/Edit/Bash work" in rendered
    assert "When a task asks you to create a file, call `Write`/`AppendFile`" in rendered


def test_persona_cannot_reintroduce_the_receipt_directive(
    provider: JinjaPromptTemplateProvider,
) -> None:
    """The flag holds with a persona layered on top.

    A persona renders under ``## Personality`` after the scaffold, so it cannot
    restore a directive the scaffold did not emit. This is the mirror of the
    additive-persona contract: the scaffold wins on safeguards, and here it
    wins on staying silent.
    """
    rendered = _render(
        provider,
        persona_md="# Aurora\nYou are warm.",
        workspace_visible_to_user=False,
    )
    assert "## Personality" in rendered
    assert "You are warm." in rendered
    assert "MUST reference it" not in rendered
    assert "ссылаться на него" not in rendered
