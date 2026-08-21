# ruff: noqa: RUF001 — Bilingual RU+EN prompt strings are intentional.
"""Leader ``<finalization_contract>`` block for declared deliverables.

/ A4 prompt scaffolding for the finalization gate.

The leader prompt now carries a small XML-style block that asks the model
to declare, **at run start**, the workspace artifacts it intends to
produce. The block is rendered into ``system_prompt_sections`` by the
executor (`build_finalization_contract_block`) and parsed back into a
list of :class:`DeliverableDeclaration` once the model has answered
(`parse_finalization_contract`).

The contract is intentionally lightweight — the agent emits a JSON
fragment, not free-form prose, so the orchestrator can extract structured
declarations without regex heuristics. The XML wrapper is purely a
visual delimiter for the prompt, not a strict parser invariant: we look
for the JSON object inside it.

Wire format (rendered into the system prompt):

.. code-block:: xml

 <finalization_contract>
 {
 "declared_deliverables": [
 {"path": "site.html", "kind": "file", "required": true,
 "summary": "Rendered HTML for the landing page"}
 ]
 }
 </finalization_contract>

The agent MAY declare zero deliverables for genuine analytic / chat-only
tasks, but only when the assistant reply itself is substantive prose
above ``finalization_empty_contract_min_response_chars`` (default 100 chars). An
empty contract emitted as a gate-bypass shortcut (mutation tool calls
Write/Edit/Bash + empty declared_deliverables, or short response below
the floor with no tool work at all) is detected by the host-side
gate and downgrades the run to ``partial``. Read-only tool chains
(Read/Glob/Grep/TodoRead/TodoWrite) do NOT trip the bypass detector —
they are informational and do not produce verifiable artifacts.
"""

from __future__ import annotations

import json
import re
from typing import Final

from protocore.contracts.attempt_ledger import DeliverableDeclaration

# Block sentinel — the executor renders the opening + closing tags around
# the JSON body so the parser can find it deterministically even when the
# model echoes the block inside a longer message.
CONTRACT_OPEN_TAG: Final[str] = "<finalization_contract>"
CONTRACT_CLOSE_TAG: Final[str] = "</finalization_contract>"


# Pre-compiled DOTALL regex for the lifted block.
_CONTRACT_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    re.escape(CONTRACT_OPEN_TAG) + r"\s*(?P<body>.*?)\s*" + re.escape(CONTRACT_CLOSE_TAG),
    re.DOTALL,
)


# Placeholder paths rendered in the leader's fill-in template. A model that
# echoes the unfilled template verbatim emits ``"path": "..."`` (the literal
# ellipsis sentinel). Such an entry is NOT a real declared deliverable — it is
# the scaffolding the model was asked to replace — so the parser skips it
# rather than minting a phantom required deliverable that can never verify on
# disk. The Unicode single-char ellipsis (``…``) is treated the same way for
# models that render the template with it.
_PLACEHOLDER_PATHS: Final[frozenset[str]] = frozenset({"...", "…"})


def _is_placeholder_path(path: str) -> bool:
    """Return True when ``path`` is an unfilled template placeholder sentinel."""
    return path.strip() in _PLACEHOLDER_PATHS


# The contract's file-receipt line, in both mirrors. It is the twin of the
# ``leader_system.j2`` receipt directive and it assumed the same reader: someone
# who can open what the agent wrote. On a chat-only surface the reply is the only
# thing the user ever sees, so "reference artifacts by name and point at the
# workspace" is followed straight into an empty answer — measured on a live stand
# as an article written to a file and delivered as a 66-word note naming the path
# and listing the sections. The template's twin was already made conditional; this
# site was missed, and it is the one rendered LAST, after the persona, so it is
# also the one a persona demanding the opposite loses to.
_RECEIPT_VISIBLE_RU: Final[str] = (
    "  - НЕ копируй содержимое файлов в финальный ответ — упоминай артефакты "
    "по имени и сошлись на workspace.\n"
)
_RECEIPT_VISIBLE_EN: Final[str] = (
    "  - do NOT inline file contents in the final reply — reference "
    "artifacts by name and point at the workspace.\n"
)
_RECEIPT_CHAT_ONLY_RU: Final[str] = (
    "  - пользователь читает ТОЛЬКО финальный ответ и не может открыть "
    "объявленные артефакты: суть излагай в самом ответе целиком. Артефакт "
    "упомяни одной строкой и не отсылай к workspace — путь с перечнем "
    "разделов для этого читателя пустой ответ.\n"
)
_RECEIPT_CHAT_ONLY_EN: Final[str] = (
    "  - the user reads ONLY the final reply and cannot open the artifacts "
    "you declare: put the substance in the reply itself, in full. Name an "
    "artifact in one line at most and do NOT point at the workspace — a path "
    "with a list of sections is an empty answer to this reader.\n"
)


_LEADER_FINALIZATION_CONTRACT_PROMPT: Final[str] = (
    "Контракт финализации (/ A4).\n"
    "Перед выполнением задачи объяви ожидаемые рабочие артефакты в "
    "JSON-блоке ниже. Используется orchestrator для финальной проверки: "
    "если требуемый артефакт отсутствует — статус run = partial.\n"
    "`finalization_contract` — это НЕ имя инструмента. Когда вы видите его "
    "в шаблоне system-prompt, это формат JSON-блока, который вы пишете "
    "В КОНЦЕ финального сообщения ассистента, ПОСЛЕ выполнения всех "
    "Write / Edit / Bash действий. Если хочется «вызвать» его как "
    "инструмент — это знак, что нужно использовать Write или Edit.\n"
    "Блок контракта — это ФИНАЛЬНЫЙ сигнал — отправляйте его ПОСЛЕ всех "
    "tool calls (Write/Edit/Bash), не как план и не до работы.\n"
    "ОБЯЗАТЕЛЬНО:\n"
    "  - объяви только реальные пути, относительные к workspace (без '/').\n"
    "  - kind ∈ {\"file\", \"directory\", \"artifact_id\"}.\n"
    "  - required=true для критических, false для опциональных.\n"
    "  - чисто аналитические задачи (только текст, без файлов): "
    "оставь \"declared_deliverables\": [] ТОЛЬКО если ты выдал содержательный "
    "аналитический ответ (≥100 символов). Пустой контракт + пустой/короткий "
    "ответ = run будет помечен FAIL. Не используй [] как способ \"мне нечего "
    "было делать\".\n"
    "  - если ты использовал инструменты записи Write/Edit/Bash для "
    "создания/изменения файлов — ОБЯЗАН объявить каждый файл результата в "
    "declared_deliverables (read-only инструменты Read/Glob/Grep "
    "контракт не запускают).\n"
    "@@RECEIPT_RU@@"
    "Finalization contract (/ A4).\n"
    "Before executing the task, declare expected workspace artifacts inside "
    "the JSON block below. The orchestrator uses these for the final gate: "
    "if a required artifact is missing the run is classified as partial.\n"
    "`finalization_contract` is NOT a tool name — when you see it in the "
    "system prompt template, it is the format of the JSON block you write "
    "AT THE END of your final assistant message, AFTER you have completed "
    "all Write / Edit / Bash work. If you find yourself wanting to 'call' "
    "it like a tool, that is a sign you should be using Write or Edit "
    "instead.\n"
    "The contract block is the FINAL signal — emit it AFTER all tool calls "
    "(Write/Edit/Bash) complete, not as a plan or before work.\n"
    "REQUIRED:\n"
    "  - declare only workspace-relative paths (no leading '/').\n"
    "  - kind ∈ {\"file\", \"directory\", \"artifact_id\"}.\n"
    "  - required=true for critical outputs, false for optional ones.\n"
    "  - pure analytic tasks (text-only answer, no files): leave "
    "\"declared_deliverables\": [] ONLY IF you have provided a substantive "
    "analytic response (>=100 chars). An empty contract paired with an "
    "empty/short response will FAIL. Do NOT use [] as a shortcut for "
    "\"I had nothing to do\".\n"
    "  - if you used MUTATION tools (Write/Edit/Bash) to produce or "
    "modify files, you MUST declare each output file in "
    "declared_deliverables (read-only Read/Glob/Grep do NOT require a "
    "contract entry).\n"
    "@@RECEIPT_EN@@"
    "\n"
    "Контракт-шаблон (нужно заполнить):\n"
    f"{CONTRACT_OPEN_TAG}\n"
    "{\n"
    '  "declared_deliverables": [\n'
    '    {"path": "...", "kind": "file", "required": true, "summary": "..."}\n'
    "  ]\n"
    "}\n"
    f"{CONTRACT_CLOSE_TAG}"
)


def build_finalization_contract_block(
    *, workspace_visible_to_user: bool = True
) -> str:
    """Return the leader's finalization-contract prompt section.

    Renders the bilingual RU+EN instruction + an empty JSON template the
    model is expected to fill in.

    ``workspace_visible_to_user`` selects the file-receipt line, exactly as it
    already does for that line's twin in ``leader_system.j2``. It defaults TRUE
    so a caller that does not pass it renders the block byte-identically to the
    pre-flag version.

    The flag has to reach HERE and not only the template because this block is
    appended AFTER the persona — it is the last thing in the system prompt, and
    a persona instructing the opposite loses to it. Leaving one of the two sites
    unconditional recreates the whole contradiction for precisely the chat-only
    tenants the flag exists to serve, and does so from the position closest to
    the answer. Both mirrors switch together for the same reason.
    """

    receipt_ru = _RECEIPT_VISIBLE_RU if workspace_visible_to_user else _RECEIPT_CHAT_ONLY_RU
    receipt_en = _RECEIPT_VISIBLE_EN if workspace_visible_to_user else _RECEIPT_CHAT_ONLY_EN
    return _LEADER_FINALIZATION_CONTRACT_PROMPT.replace(
        "@@RECEIPT_RU@@", receipt_ru
    ).replace("@@RECEIPT_EN@@", receipt_en)


def parse_finalization_contract(text: str) -> list[DeliverableDeclaration]:
    """Extract declarations from a `<finalization_contract>` block.

    Returns an empty list when:
      - The block sentinel is missing.
      - The JSON body fails to parse.
      - The body's ``declared_deliverables`` key is missing or empty.
      - Any entry is malformed (the entry is skipped, not the whole block).
      - An entry's ``path`` is an unfilled template placeholder (the literal
        ellipsis sentinel ``...``/``…`` from the leader's fill-in template);
        the entry is skipped so an echoed template does not mint a phantom
        required deliverable.

    Never raises — a malformed contract degrades to "no declarations",
    which the gate treats as ``unknown`` outcome.
    """
    if not text:
        return []
    match = _CONTRACT_BLOCK_RE.search(text)
    if match is None:
        return []
    body = (match.group("body") or "").strip()
    if not body:
        return []
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    entries = parsed.get("declared_deliverables")
    if not isinstance(entries, list):
        return []
    declarations: list[DeliverableDeclaration] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            declaration = DeliverableDeclaration.model_validate(entry)
        except (ValueError, TypeError):
            # Per the docstring: skip malformed entries; do not abort the
            # whole contract on one bad item.
            continue
        # skip an unfilled template placeholder (``path == "..."``)
        # the model echoed verbatim. ``DeliverableDeclaration.path`` only
        # requires ``min_length=1`` so the ellipsis sentinel validates; left
        # in, it becomes a phantom required deliverable that never verifies
        # on disk and downgrades an otherwise-complete run.
        if _is_placeholder_path(declaration.path):
            continue
        declarations.append(declaration)
    return declarations


__all__ = [
    "CONTRACT_CLOSE_TAG",
    "CONTRACT_OPEN_TAG",
    "build_finalization_contract_block",
    "parse_finalization_contract",
]
