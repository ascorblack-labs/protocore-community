# ruff: noqa: RUF001 — the bilingual texts under test are Cyrillic on purpose.
"""The texts the loop speaks did not change when they became templates.

Each of these was a configuration field holding a block of prompt prose. A
field is the wrong home for prose: there is no template engine behind it, so
an operator serving a third language has nowhere to put the translation, and
the text is reviewed as configuration rather than as the thing the model
reads. They are templates now.

The move is only safe if it is invisible to the model, so the wording as it
stood before the move is written out here in full and compared byte for byte
against what the template renders. A copy of the text is exactly the point:
an assertion that reads the template it is checking asserts nothing.
"""
from __future__ import annotations

import pytest

from protocore.prompts import BUNDLED_TEMPLATES, bundled_prompt_provider

#: The five values the truncation-recovery halves are rendered with. The
#: numbers are arbitrary; only their appearance in the output is checked.
_RECOVERY_CONTEXT = {
    "tool_name": "Write",
    "partial_length": 4096,
    "chunk_bytes": 1024,
    "chunk_bytes_lines": 20,
    "chunk_count_estimate": 10,
}

#: Template name -> (render context, the text as it read before the move).
WORDING_BEFORE_THE_MOVE: dict[str, tuple[dict[str, object], str]] = {
    "tool_call_truncation_recovery_en": (
        _RECOVERY_CONTEXT,
        'Your `{tool_name}` call was TRUNCATED after {partial_length} bytes — the model output budget ran out mid-JSON. Your `content` argument is too large to fit in one call. Action: emit a NEW `Write` call where `content` is at most {chunk_bytes} chars (about {chunk_bytes_lines} lines). Then immediately follow up with one or more `Edit` calls using `replace_all=False` and a unique anchor string at the end of the previous chunk to append the rest. Do NOT retry the same `{tool_name}` call — it will truncate again. Estimate: a 10 KB target needs roughly {chunk_count_estimate} chunked calls.'.format(**_RECOVERY_CONTEXT),
    ),
    "tool_call_truncation_recovery_ru": (
        _RECOVERY_CONTEXT,
        "Ваш вызов `{tool_name}` был ОБРЕЗАН после {partial_length} байт — у модели закончился output budget посередине JSON. Аргумент `content` слишком большой для одного вызова. Действие: эмитируйте НОВЫЙ вызов `Write` где `content` максимум {chunk_bytes} символов (примерно {chunk_bytes_lines} строк). Затем сразу следуют один или более `Edit` вызовов с `replace_all=False` и уникальной anchor-строкой в конце предыдущего chunk'а для дописывания. НЕ повторяйте тот же вызов `{tool_name}` — он опять обрежется. Оценка: ~10 KB цель требует примерно {chunk_count_estimate} chunk'ов.".format(**_RECOVERY_CONTEXT),
    ),
    "tool_call_truncation_resume": (
        {"tool_name": "Write"},
        'Your previous tool call to {tool_name} was truncated by the output token cap before it could finish. Re-issue the tool call with the COMPLETE arguments. Do not summarise or repeat earlier content; emit the call fully. If this is a Write call, ensure the file content is correctly resumed from where you stopped.'.format(tool_name="Write"),
    ),
    "terminal_tool_nudge_write_first": (
        {},
        '[internal control — not part of the reply] If the task asked for a file that has not been written yet, the deliverable must be created with the file-write tool (Write, or AppendFile to continue a chunked file) carrying its full content before finishing with the terminal tool; the file content goes in the written file, not in this note. [внутреннее управление — не часть ответа] Если задача просила файл, который ещё не записан, результат нужно создать инструментом записи (Write, либо AppendFile для продолжения файла по частям) с полным содержимым до завершения терминальным инструментом; содержимое файла идёт в записанный файл, а не в эту заметку.',
    ),
    "finalize_prose_gate_repair": (
        {},
        'Stop. Before you finish, write your final response to the user as a normal assistant message — plain prose, not a tool call. If the deliverable was written to a file or artifact, summarize it and reference its path; do NOT paste the full saved file contents unless the user explicitly asked for them. Only AFTER you have written that response should you call the terminal tool to end the run. Do not put the answer inside the tool; the tool only ends the run. Стоп. Прежде чем завершить, напишите финальный ответ пользователю обычным сообщением ассистента — простым текстом, а не вызовом инструмента. Если результат записан в файл или артефакт, кратко опишите его и укажите путь; НЕ вставляйте полное содержимое сохранённого файла, если пользователь явно об этом не попросил. Только ПОСЛЕ того как вы написали этот ответ, вызовите терминальный инструмент, чтобы завершить выполнение. Не помещайте ответ внутрь инструмента; инструмент лишь завершает выполнение.',
    ),
    "tool_result_interrupted": (
        {},
        'Interrupted',
    ),
    "tool_result_pairing_repair": (
        {},
        '[Tool result missing due to internal error]',
    ),
    "result_eviction": (
        {"tool_call_id": "toolu_7"},
        '[evicted tool result {tool_call_id}; full result persisted]'.format(tool_call_id="toolu_7"),
    ),
    # The terminal-tool nudge body was never a field at all — it was an
    # f-string built inside the loop, which is the same defect wearing
    # different clothes: still prose, still unreachable by a translator.
    "terminal_tool_nudge": (
        {"terminal_tool": "Finalize"},
        '[internal control — not part of the reply] The run is ending and the {tool_name} tool has not been called yet. Finish now by calling {tool_name} with the best supported answer already prepared; the answer text belongs in the reply itself, not in this note.'.format(tool_name="Finalize"),
    ),
}

@pytest.mark.parametrize("name", sorted(WORDING_BEFORE_THE_MOVE))
def test_the_template_renders_what_the_field_used_to_hold(name: str) -> None:
    context, before = WORDING_BEFORE_THE_MOVE[name]
    rendered = bundled_prompt_provider().render(name, dict(context))
    assert rendered == before


def test_every_moved_text_is_registered_as_a_template() -> None:
    """A text with no registry entry is unreachable, however well it renders."""
    missing = sorted(set(WORDING_BEFORE_THE_MOVE) - set(BUNDLED_TEMPLATES))
    assert not missing, f"not in the bundled registry: {missing}"
