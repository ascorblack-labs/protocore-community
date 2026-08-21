# ruff: noqa: RUF001, RUF002 — Bilingual RU+EN assertions + docstrings are intentional.
"""Unit tests for :mod:`protocore.runtime.finalization_contract`.

/ A4 prompt-block scaffolding for the finalization
gate. Covers the build + parse round-trip + malformed-input degradation
contract.
"""
from __future__ import annotations

import json

from protocore.contracts.attempt_ledger import DeliverableDeclaration
from protocore.runtime.finalization_contract import (
    CONTRACT_CLOSE_TAG,
    CONTRACT_OPEN_TAG,
    build_finalization_contract_block,
    parse_finalization_contract,
)


def _wrap(json_payload: object) -> str:
    return (
        f"prefix text\n{CONTRACT_OPEN_TAG}\n"
        + json.dumps(json_payload, ensure_ascii=False)
        + f"\n{CONTRACT_CLOSE_TAG}\nsuffix"
    )


def test_build_contract_block_contains_sentinels() -> None:
    text = build_finalization_contract_block()
    assert CONTRACT_OPEN_TAG in text
    assert CONTRACT_CLOSE_TAG in text


def test_build_contract_block_is_bilingual() -> None:
    text = build_finalization_contract_block()
    # RU surface
    assert "Контракт финализации" in text
    # EN surface
    assert "Finalization contract" in text


def test_build_contract_block_anchors_after_mutation_tools() -> None:
    """Contract block is the FINAL signal — must
    be emitted AFTER Write/Edit/Bash work, never as a plan or upfront.

    Runs that emitted an empty contract early as a "graceful give-up"
    instead of completing mutation work trigger false completions. The
    contract prompt now anchors this explicitly in both languages.
    """
    text = build_finalization_contract_block()
    # EN anchoring sentence.
    assert "FINAL signal" in text
    assert "AFTER all tool calls" in text
    # RU anchoring sentence.
    assert "ФИНАЛЬНЫЙ сигнал" in text
    assert "ПОСЛЕ всех" in text


def test_parse_finalization_contract_round_trip() -> None:
    payload = {
        "declared_deliverables": [
            {
                "path": "site.html",
                "kind": "file",
                "required": True,
                "summary": "Landing page",
            },
            {
                "path": "data/output.json",
                "kind": "file",
                "required": False,
            },
        ]
    }
    text = _wrap(payload)
    result = parse_finalization_contract(text)
    assert len(result) == 2
    assert result[0].path == "site.html"
    assert result[0].required is True
    assert result[0].summary == "Landing page"
    assert result[1].path == "data/output.json"
    assert result[1].required is False


def test_parse_finalization_contract_missing_block_returns_empty() -> None:
    assert parse_finalization_contract("nothing relevant here") == []


def test_parse_finalization_contract_empty_string_returns_empty() -> None:
    assert parse_finalization_contract("") == []


def test_parse_finalization_contract_invalid_json_returns_empty() -> None:
    text = f"{CONTRACT_OPEN_TAG}\n{{ not valid json }}\n{CONTRACT_CLOSE_TAG}"
    assert parse_finalization_contract(text) == []


def test_parse_finalization_contract_non_object_root_returns_empty() -> None:
    text = f"{CONTRACT_OPEN_TAG}\n[1, 2, 3]\n{CONTRACT_CLOSE_TAG}"
    assert parse_finalization_contract(text) == []


def test_parse_finalization_contract_missing_key_returns_empty() -> None:
    text = _wrap({"unrelated_key": [{"path": "x"}]})
    assert parse_finalization_contract(text) == []


def test_parse_finalization_contract_empty_list_returns_empty() -> None:
    text = _wrap({"declared_deliverables": []})
    assert parse_finalization_contract(text) == []


def test_parse_finalization_contract_skips_malformed_entries() -> None:
    payload = {
        "declared_deliverables": [
            {"path": "ok.html"},
            {"missing_path": True},  # malformed: no path
            "not a dict",
            {"path": "also-ok.json", "kind": "file"},
        ]
    }
    text = _wrap(payload)
    result = parse_finalization_contract(text)
    assert [d.path for d in result] == ["ok.html", "also-ok.json"]


def test_parse_finalization_contract_handles_nested_blocks() -> None:
    """Outer block wins on first match — model echo should not duplicate."""
    payload_a = {"declared_deliverables": [{"path": "first.html"}]}
    payload_b = {"declared_deliverables": [{"path": "second.html"}]}
    text = _wrap(payload_a) + _wrap(payload_b)
    result = parse_finalization_contract(text)
    # Only first block parsed.
    assert [d.path for d in result] == ["first.html"]


def test_parse_finalization_contract_returns_list_of_DeliverableDeclaration() -> None:
    text = _wrap({"declared_deliverables": [{"path": "x.html"}]})
    result = parse_finalization_contract(text)
    assert all(isinstance(item, DeliverableDeclaration) for item in result)


def test_contract_block_warns_against_empty_bypass_ru() -> None:
    """Leader prompt must
 spell out that empty contract + short response = FAIL — the bypass
 attractor caused status=completed for zero work. The floor is 100 chars."""
    text = build_finalization_contract_block()
    # Must mention the 100-char floor + warn against the shortcut.
    assert "100" in text
    assert "FAIL" in text or "fail" in text.lower()
    # Must keep the existing tool-output declaration directive present
    # — now scoped to MUTATION tools after .
    assert "Write/Edit/Bash" in text


def test_contract_block_warns_against_empty_bypass_en() -> None:
    """English leader prompt mirrors the Russian directive."""
    text = build_finalization_contract_block()
    assert "substantive analytic response" in text
    assert "FAIL" in text


def test_contract_block_disambiguates_xml_vs_tool_en() -> None:
    """English directive must
 call out that ``<finalization_contract>`` is the format of a JSON
 block placed at the END of the assistant message, NOT a tool name
 to invoke.

 ``docgen-en-004`` failed because the leader emitted
 ``finalization_contract`` as a tool call name, the dispatcher rejected
 it (unknown tool), and the model only recovered on the next turn.
 The original disambiguation used the phrase "emit INLINE in your
 assistant text", which Qwen3.5 over-indexed on and started emitting
 the contract block in place of actually doing Write/Edit/Bash work
 (BAD-EARLY pattern). The current phrasing reframes it in work-first
 terms: the block goes AT THE END, AFTER Write/Edit/Bash."""
    text = build_finalization_contract_block()
    # Key disambiguation phrases — substring checks to allow future
    # wording tweaks without breaking the regression intent.
    assert "is NOT a tool name" in text
    assert "AT THE END" in text
    # Pointer to the real tool surface so the model knows where work goes.
    assert "Write" in text
    assert "Edit" in text
    assert "Bash" in text


def test_contract_block_disambiguates_xml_vs_tool_ru() -> None:
    """Russian mirror of the XML-vs-tool disambiguation. Production traffic
    is RU+EN; the directive must reach Russian runs.

    The current phrasing uses the work-first "В КОНЦЕ ... ПОСЛЕ выполнения
    всех Write / Edit / Bash действий" framing."""
    text = build_finalization_contract_block()
    assert "НЕ имя инструмента" in text
    assert "В КОНЦЕ" in text


def test_contract_prompt_does_not_use_emit_inline_phrasing() -> None:
    """Root cause of a BAD-EARLY regression was Qwen3.5 over-indexing on the
    literal phrase "emit INLINE" from the XML-vs-tool disambiguation, treating
    it as license to skip Write / Edit / Bash work. The phrase has been dropped;
    this test pins that decision so it is not silently reintroduced."""
    block = build_finalization_contract_block()
    assert "emit INLINE" not in block
    assert "emit inline" not in block.lower()


def test_parse_finalization_contract_skips_placeholder_path() -> None:
    """an echoed unfilled template block must NOT create a phantom
    deliverable.

    The leader prompt renders a fill-in template whose body is
    ``{"path": "...", ...}``. ``parse_finalization_contract`` uses
    ``.search`` (first match), so if the model echoes the unfilled template
    at or before the real contract block the placeholder ``path == "..."``
    is parsed as a real required deliverable. That literal path never
    verifies on disk, so a run that actually completed its declared work is
    downgraded to failed/partial on a phantom ``...`` deliverable.

    The placeholder entry must be skipped (consistent with the existing
    skip-malformed-entries contract), not aborted.
    """
    payload = {
        "declared_deliverables": [
            {"path": "...", "kind": "file", "required": True, "summary": "..."},
        ]
    }
    text = _wrap(payload)
    assert parse_finalization_contract(text) == []


def test_parse_finalization_contract_skips_placeholder_keeps_real_entries() -> None:
    """a placeholder entry mixed with real entries drops only the
    placeholder; the genuine deliverables survive."""
    payload = {
        "declared_deliverables": [
            {"path": "...", "kind": "file", "required": True, "summary": "..."},
            {"path": "site.html", "kind": "file", "required": True},
            {"path": "data/out.json", "kind": "file", "required": False},
        ]
    }
    text = _wrap(payload)
    result = parse_finalization_contract(text)
    assert [d.path for d in result] == ["site.html", "data/out.json"]


def test_parse_finalization_contract_skips_rendered_template_block() -> None:
    """parsing the EXACT rendered leader template (the unfilled
    placeholder the model is shown) yields no phantom deliverables.

    Regression guard: the prompt template is the canonical echo source, so
    feeding ``build_finalization_contract_block()`` straight back through the
    parser must produce an empty declaration list rather than a bogus
    ``...`` deliverable.
    """
    block = build_finalization_contract_block()
    assert parse_finalization_contract(block) == []


def test_parse_finalization_contract_skips_unicode_ellipsis_placeholder() -> None:
    """a model that renders the placeholder with the single-char
    Unicode ellipsis (``…``) instead of three ASCII dots is still skipped."""
    payload = {"declared_deliverables": [{"path": "…", "kind": "file"}]}
    text = _wrap(payload)
    assert parse_finalization_contract(text) == []


def test_contract_block_still_parseable() -> None:
    """Round-trip safety: the new prompt directive must not break the
    JSON template's parse contract. We render the block, then verify
    that filling the template with a valid payload still parses."""
    block = build_finalization_contract_block()
    # The block has the empty template at the bottom — replace it with
    # a populated declared_deliverables list, then confirm parse_finalization_contract
    # can extract them. We do this by appending a second contract block
    # AFTER the placeholder, which the parser matches first.
    populated = (
        f"prologue\n{CONTRACT_OPEN_TAG}\n"
        '{ "declared_deliverables": [{"path": "x.py", "required": true}] }\n'
        f"{CONTRACT_CLOSE_TAG}\nepilogue"
    )
    result = parse_finalization_contract(populated)
    assert len(result) == 1
    assert result[0].path == "x.py"
    # And the prompt block itself still contains usable sentinels for the
    # parser (i.e., we did not break the OPEN/CLOSE tags by editing the
    # surrounding directive prose).
    assert CONTRACT_OPEN_TAG in block
    assert CONTRACT_CLOSE_TAG in block


# ---------------------------------------------------------------------------
# The file-receipt line tracks who reads the reply
# ---------------------------------------------------------------------------
#
# The contract block is appended AFTER the persona, so it is the last thing in
# the leader's system prompt. Its receipt line told every tenant to reference
# artifacts by name and point at the workspace — correct for a reader with a
# file browser, and for a chat-only reader an instruction to answer with a path.
# ``leader_system.j2`` already gates its own twin of this line on the same flag;
# this site was missed, and being last it is the one a persona demanding the
# opposite loses to.

_RECEIPT_RU = "упоминай артефакты по имени и сошлись на workspace"
_RECEIPT_EN = "reference artifacts by name and point at the workspace"


def test_default_block_keeps_the_receipt_line_byte_for_byte() -> None:
    """No caller changes behaviour by upgrading past the flag.

    The default is the pre-flag string exactly, in both mirrors, so a tenant
    whose reader CAN open the workspace is told what it was always told.
    """
    default = build_finalization_contract_block()
    assert default == build_finalization_contract_block(
        workspace_visible_to_user=True
    )
    assert _RECEIPT_RU in default
    assert _RECEIPT_EN in default


def test_chat_only_reader_is_told_the_reply_is_the_deliverable() -> None:
    """With the workspace invisible the line inverts, in BOTH languages.

    Production traffic is RU+EN and the two mirrors are read by the same model;
    a line that flips in one language only leaves the contradiction standing for
    half the prompt, which is the failure this test exists to catch.
    """
    chat_only = build_finalization_contract_block(workspace_visible_to_user=False)
    assert _RECEIPT_RU not in chat_only
    assert _RECEIPT_EN not in chat_only
    assert "пользователь читает ТОЛЬКО финальный ответ" in chat_only
    assert "the user reads ONLY the final reply" in chat_only


def test_the_flag_moves_the_receipt_line_and_nothing_else() -> None:
    """A wording switch, not a restructuring: the gate's own rules survive.

    Everything the block exists to do — declare artifacts, forbid the empty
    contract as a shortcut, keep the sentinels parseable — is about the
    finalization gate and has nothing to do with who reads the reply.
    """
    chat_only = build_finalization_contract_block(workspace_visible_to_user=False)
    assert CONTRACT_OPEN_TAG in chat_only
    assert CONTRACT_CLOSE_TAG in chat_only
    assert "Контракт финализации" in chat_only
    assert "Finalization contract" in chat_only
    assert "declared_deliverables" in chat_only
    assert 'Do NOT use [] as a shortcut' in chat_only
    assert "объяви только реальные пути" in chat_only
    # No placeholder survives into a rendered block in either mode.
    assert "@@RECEIPT" not in chat_only
    assert "@@RECEIPT" not in build_finalization_contract_block()
