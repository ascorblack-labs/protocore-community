"""Universal terminal-answer payload normalization (pure core).

Closes a class of terminal-answer misses where the model produced the correct
text but the ``message`` reached a downstream string grader HTML-escaped —
e.g. ``&lt;YES&gt;`` instead of the literal ``<YES>`` an exact-match grader
required.

This module owns ONLY the pure transform + a tiny declarative spec. It is
intentionally tenant-agnostic: it has no tenant- or benchmark-specific symbols
and never inspects a specific tool's argument schema. The host
terminal-tool handler calls :func:`normalize_terminal_text` on the
user-supplied ``message`` BEFORE it builds its provider answer request, gated
by the RuntimeConstants flag ``terminal_answer_entity_normalize_enabled`` (and
optional ``terminal_answer_sentinels``). Keeping the algorithm + RC in core
while the call lives in the host respects the import boundary: core never
imports the tool, the tool imports this pure helper.

Why an unescape (and why it can default ON):

* It is a pure *correctness* normalization. A terminal answer is plain text
  destined for a string grader; HTML/XML entity encoding of that text is
  never semantically intended by the model — it is an artifact of how some
  providers / tool-arg JSON paths round-trip ``<`` ``>`` ``&``. Unescaping
  restores the literal characters the model meant.
* ``html.unescape`` handles named (``&lt;`` ``&gt;`` ``&amp;`` ``&quot;`` …)
  and numeric (``&#60;`` ``&#x3c;``) references in one pass.
* It is gated by an RC so a tenant can disable it; the default reproduces the
  intended behaviour (literal markers) rather than the accidental escape.

Idempotency / double-escape note: ``html.unescape`` is a single pass. The
``message`` we receive is escaped at most once on the way in, so a single
unescape is the right inverse. (A pathological double-escape like
``&amp;lt;`` would unescape to ``&lt;`` in one pass — we deliberately do NOT
loop-unescape, because repeatedly unescaping arbitrary user text could
corrupt a message that legitimately contains an entity-looking substring.)
"""

from __future__ import annotations

import html

__all__ = ["normalize_terminal_text"]


def normalize_terminal_text(
    text: str | None,
    *,
    entity_unescape: bool,
    sentinels: tuple[str, ...] = (),
) -> str | None:
    """Return ``text`` normalized for terminal-answer submission.

    Args:
        text: The model-supplied terminal-answer message. ``None`` / empty is
            returned unchanged so callers can pass the raw field through.
        entity_unescape: When ``True`` (RC
            ``terminal_answer_entity_normalize_enabled``), HTML/XML entity
            references are unescaped in a single pass (``&lt;``→``<``,
            ``&gt;``→``>``, ``&amp;``→``&``, named + numeric refs). When
            ``False`` the function is a no-op pass-through (returns the
            input unchanged).
        sentinels: Optional canonical markers a tenant asserts must appear
            literally in the final message (e.g. ``("<YES>", "<NO>")``). This
            is purely declarative/defensive: after unescaping, the literal
            forms already appear, so this list does not mutate ``text`` — it
            exists so a future strict mode (or a host validator) can
            check presence/canonical form without re-deriving the marker set
            in another repo. Kept as an explicit parameter so the seam is
            visible to the host caller; today it has no rewriting side
            effect beyond the unescape.

    Returns:
        The normalized text, or the original value when ``entity_unescape`` is
        ``False`` or ``text`` is falsy.
    """
    if not entity_unescape or not text:
        return text
    # Single-pass entity unescape. This is the inverse of a single escape on
    # the inbound path and turns ``&lt;YES&gt;`` into the literal ``<YES>``.
    normalized = html.unescape(text)
    # ``sentinels`` is intentionally non-mutating here (see docstring): the
    # unescape already canonicalizes ``&lt;YES&gt;`` → ``<YES>``. We touch the
    # tuple only to keep the parameter live for the documented the host
    # validator seam without a behavioural side effect.
    _ = sentinels
    return normalized
