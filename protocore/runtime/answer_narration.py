# ruff: noqa: RUF001, RUF002, RUF003 — Cyrillic is the subject matter here: the
# patterns match Russian narration and the examples quote it verbatim.
"""Find the leading process narration on a delegating leader's answer.

A leader that farmed work out to subagents opens its reply by telling the reader
where its own work has got to — «Теперь у меня есть все материалы. Составлю
итоговый обзор.» / "Now I have a good collection of sources. Let me compile the
article." — and only then answers. The narration and the answer are one text
block by construction: the terminal tool carries no answer field, so the reply is
prose, and the first sentence of that prose is the first thing the reader sees.

This module measures the narration, and nothing else acts on it: the caller
splits the block at the returned offset and marks the FIRST half ``collapsed``.
No text is ever dropped.

Reading text to decide visibility is otherwise refused throughout the runtime,
for a good reason stated in :func:`protocore.runtime.query._content_block_visibility`
— a person asking about a config format gets an answer that looks exactly like a
model thinking aloud about one, so a shape rule eats real answers. Three things
keep this narrow enough to be safe anyway:

* The caller applies it ONLY to a run that actually delegated. Without
  delegation the shape is uncommon; with it, it is near-universal.
* The verdict needs BOTH halves of the measured shape in the leading sentences —
  a first-person claim about the WORK MATERIAL the run has gathered, and an
  announcement of intent to produce the deliverable. Either half alone is
  ordinary prose: "У меня есть все данные, которые вы просили" is an answer to
  someone who asked whether the data is complete, and a lone "Составлю обзор" is
  not the shape being measured.
* The scan stops at the first sentence that is not narration, never leaves the
  first paragraph, never reaches past ``scan_chars``, and refuses to fire unless
  a substantial answer survives after the prefix. So the worst case is a leading
  sentence rendered as a chip — never a missing reply.

An opener that frames the answer's own scope is deliberately a different thing
and does not match: «Обзор охватывает пять аспектов…» / "Ниже — полный связный
обзор…" is about the SUBJECT, in the third person, and addresses the reader.
:data:`_PRESENTATIVE` states that distinction directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: End of a sentence: terminal punctuation followed by intra-line whitespace.
#: Newlines are deliberately NOT a separator — the scan never crosses a
#: paragraph break, and a single newline inside a paragraph belongs to the
#: sentence it interrupts.
_SENTENCE_END: Final = re.compile(r"(?<=[.!?…])[ \t]+")

#: The speaker referring to itself. Russian «у меня» / «я», English "I" and the
#: hortative "let me" / "let's" that always introduces the same announcement.
_SELF: Final = re.compile(
    r"(?:^|\W)(?:у\s+меня|я\s|мне\s|мною|i\s|i'll|i'm|i've|let\s+me|let's|we\s)",
    re.IGNORECASE,
)

#: The claim that the gathering is finished — "enough", "all", "the full
#: picture". Narration says the work is complete; an answer says what is true.
_ENOUGH: Final = re.compile(
    r"(?:достаточно|хватает|\bвс(?:е|ё|ю|я|ем|ех|еми|его|ей)\b|\bвесь\b|полн|"
    r"целиком|enough|sufficient|all\s|good\s|comprehensive|rich\s|complete|"
    r"full\s)",
    re.IGNORECASE,
)

#: What was gathered. Raw work material — never the deliverable itself, which is
#: :data:`_DELIVERABLE`. Russian stems are cut short of the fill vowel so they
#: survive inflection (``подбор`` matches подборка / подборок / подборки).
_MATERIAL: Final = re.compile(
    r"(?:материал|источник|данны|информаци|файл|подбор|сниппет|наход|выдерж|"
    r"картин|аспект|раздел|source|material|data|file|finding|snippet|excerpt|"
    r"reference|collection|set\s+of)",
    re.IGNORECASE,
)

#: The same completion claim made without a pronoun — «Все материалы собраны».
_DONE: Final = re.compile(
    r"(?:прочитан|собран|получен|найден|проверен|изучен|просмотрен|gathered|"
    r"collected|retrieved|reviewed)",
    re.IGNORECASE,
)

#: First-person past act opening a sentence — «Получил содержимое четырёх
#: файлов». Anchored at the start because these verb forms are also ordinary
#: narrative prose in the middle of a sentence about somebody else.
_ACT_1SG: Final = re.compile(
    r"^\W*(?:получил|прочитал|собрал|наш[её]л|изучил|просмотрел|проверил|сверил)",
    re.IGNORECASE,
)

#: "Now I will write it" — the announcement half of the shape.
_INTENT_VERB: Final = re.compile(
    r"(?:^|\W)(?:соста[вб][а-яё]*|напиш[уy][а-яё]*|сформулиру[юe][а-яё]*|"
    r"подготовл[юe][а-яё]*|собер[уy][а-яё]*|сформиру[юe][а-яё]*|"
    r"формиру[юe][а-яё]*|приступа[юe][а-яё]*|начина[юe][а-яё]*|начну|"
    r"перейд[уy][а-яё]*|изложу|оформл[юe][а-яё]*|свед[уy]|представл[юe][а-яё]*|"
    r"отвечу|прочита[юe][а-яё]*|проверю|сверю|использую|перенесу|напишем|"
    r"составим|let\s+me\s+\w+|let's\s+\w+|"
    r"i(?:'ll|\s+will|\s+am\s+going\s+to)\s+\w+)",
    re.IGNORECASE,
)

#: The same announcement carried by a purpose phrase instead of a verb —
#: «достаточно данных ДЛЯ ФОРМИРОВАНИЯ итогового ответа».
_PURPOSE: Final = re.compile(
    r"(?:для\s+\w+|чтобы\s+\w+|to\s+(?:write|compile|compose|produce|deliver|"
    r"synthesi[sz]e|summari[sz]e|assemble|answer))",
    re.IGNORECASE,
)

#: The thing being produced, as the narration names it.
_DELIVERABLE: Final = re.compile(
    r"(?:обзор|стать|ответ|текст|подбор|резюме|article|review|answer|text|"
    r"response|summary)",
    re.IGNORECASE,
)

#: «Вот …» / «Ниже — …» / "Here is …". A sentence that hands the answer to the
#: reader is part of the answer however much it resembles an announcement, so it
#: ends the scan rather than being consumed by it.
_PRESENTATIVE: Final = re.compile(r"^\W*(?:вот|ниже|здесь|итак|here|below)\b", re.IGNORECASE)

#: A sentence longer than this cannot be the bare completion claim
#: («Файлы прочитаны.»); at that length it is carrying content.
_MAX_STATE_SENTENCE_CHARS: Final = 90

#: …and the bare announcement is shorter still.
_MAX_INTENT_SENTENCE_CHARS: Final = 60


@dataclass(frozen=True)
class NarrationSpan:
    """How much leading narration the text has, and whether that can still grow.

    ``settled`` is the streaming half of the contract. A caller reading a live
    stream sees a prefix of the final text, and the answer to "how long is the
    narration" can only ever GROW as more sentences arrive — so acting on an
    unsettled span would cut the run of narration in half. ``settled`` is True
    exactly when more text cannot change ``length``: the scan ended on a
    paragraph break, on a sentence that is not narration, or at the scan
    ceiling.
    """

    length: int
    settled: bool

    @property
    def found(self) -> bool:
        return self.length > 0


def _is_state(sentence: str) -> bool:
    """The run claiming its own gathering is complete."""

    if not _MATERIAL.search(sentence):
        return False
    if _SELF.search(sentence) and _ENOUGH.search(sentence):
        return True
    # The pronoun-free form. Kept short and required to name no deliverable so
    # it cannot swallow «Ниже — полный связный обзор, СОБРАННЫЙ из всех
    # ИСТОЧНИКОВ…», where the material appears in a subordinate phrase and the
    # sentence is really about the review it is handing over.
    return (
        len(sentence) <= _MAX_STATE_SENTENCE_CHARS
        and not _DELIVERABLE.search(sentence)
        and bool(_DONE.search(sentence) or _ACT_1SG.search(sentence))
    )


def _is_intent(sentence: str, *, after_state: bool) -> bool:
    """The run announcing that it is about to write the deliverable."""

    if _PRESENTATIVE.search(sentence):
        return False
    if not _DELIVERABLE.search(sentence):
        return False
    if _INTENT_VERB.search(sentence) or _PURPOSE.search(sentence):
        return True
    # Once the state claim has been made, a short sentence naming the
    # deliverable and nothing else IS the announcement, whatever verb it uses —
    # which is what keeps a mistyped one («Собставлю полный обзор.») from
    # leaving half the narration on the reader's surface. Digits disqualify it:
    # a count or a size is a claim about the work, not an intention.
    return (
        after_state
        and len(sentence) <= _MAX_INTENT_SENTENCE_CHARS
        and not re.search(r"\d", sentence)
    )


def leading_narration_span(
    text: str,
    *,
    scan_chars: int,
    complete: bool,
) -> NarrationSpan:
    """Measure the run of process narration at the head of ``text``.

    ``complete`` says whether ``text`` is the whole block. A live stream passes
    ``False``: a trailing fragment with no terminal punctuation is then ignored
    (it may still turn into a narration sentence, or into the answer) and the
    span reports itself unsettled. History passes ``True``, where the last
    sentence may legitimately lack a full stop and nothing can change.

    Returns a zero-length span unless BOTH halves of the shape are present in
    the leading sentences. The caller still has to check that enough answer
    survives after the prefix — this function does not know how long the block
    will turn out to be.
    """

    window = text[:scan_chars]
    paragraph_end = window.find("\n\n")
    truncated = len(text) > scan_chars
    if paragraph_end != -1:
        window = window[:paragraph_end]
        truncated = False
    if not window.strip():
        return NarrationSpan(length=0, settled=complete or truncated)

    sentences = _SENTENCE_END.split(window)
    consumed = 0
    saw_state = False
    saw_intent = False
    stopped_early = False
    for sentence in sentences:
        terminated = sentence.rstrip().endswith((".", "!", "?", "…"))
        if not terminated and not complete:
            # The block is still streaming and this is its open sentence. It has
            # no verdict yet, and guessing one would cut the narration mid-run.
            break
        state = _is_state(sentence)
        intent = _is_intent(sentence, after_state=saw_state)
        if not state and not intent:
            stopped_early = True
            break
        start = text.find(sentence, consumed)
        if start < 0: # pragma: no cover - defensive; window is a prefix of text
            stopped_early = True
            break
        consumed = start + len(sentence)
        saw_state = saw_state or state
        saw_intent = saw_intent or intent

    # Settled once nothing arriving later can lengthen the run: the scan met a
    # non-narration sentence, ran into the paragraph break or the ceiling, or
    # the text is already whole.
    settled = complete or stopped_early or paragraph_end != -1 or truncated
    if not (saw_state and saw_intent):
        return NarrationSpan(length=0, settled=settled)
    return NarrationSpan(length=consumed, settled=settled)
