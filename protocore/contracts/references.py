"""Pure-core evidence-ref canonicalization (RFC-3986-grade, stdlib-only).

Supplies the single canonical :func:`normalize_ref` primitive the host
observed-ref matcher uses at BOTH compare sites (pre-dispatch self-verify +
post-submit terminal-answer validation) AND symmetrically at the record side.
Computing the projection here, once, keeps the matcher DRY and the canonical
form identical on the record-side and the compare-side (the symmetry invariant
that makes normalization behaviorally NEUTRAL — it can only REMOVE a false
veto, never add one).

UNIVERSAL-CORE invariants:

* Pure: stdlib only (``re``, ``string``). No tenant- or benchmark-specific
  string tokens. No RC dependency (RC interpretation is the host policy;
  this module just exposes the projection + its toggles as kwargs).
* :func:`normalize_ref` is a COMPARISON PROJECTION. It NEVER mutates a
  model-submitted or stored ref string — callers compare on the projection and
  keep the original verbatim. (The model's emitted ref is never altered; only
  the membership key is canonicalized.)
* The projection does RFC-3986 path/percent canonicalization ONLY. It does NOT
  HTML-entity-decode: refs compare LITERALLY apart from the standard URI/path
  canonical forms below. A model that emits an entity-encoded ref (``&#47;`` for
  ``/``) therefore keeps that ``&#47;`` literal — it correctly does NOT match a
  real path with a literal ``/`` (that is a malformed ref, not a grounded one),
  and the comparison is symmetric, so two refs that literally contain the same
  entity string still match each other. (Message-body entity artifacts like
  ``&lt;YES&gt;`` are handled by the separate the host answer-text normalizer,
  not by this path membership KEY.)
* The ``strip_extension`` key is framed as UNIVERSAL URI/path canonicalization:
  an extensionless path is a standard canonical membership key; RFC-3986 is
  extension-agnostic — ``index.html`` and ``index`` address one resource. No
  grader/scorer quirk is hardcoded.

Idempotence: ``normalize_ref(normalize_ref(s, **k), **k) == normalize_ref(s, **k)``
for every flag combination (a canonical form is a fixed point).
"""
from __future__ import annotations

import re
import string
from typing import Final

# ONE sub-locator regex, reused by callers that want to strip a trailing
# line/row fragment off the membership key. Matches a trailing URI fragment
# (``#row=12``, ``#L12-30``) OR a trailing ``:line`` / ``:line:col`` suffix
# (``:12``, ``:12:30``). Anchored at end-of-string; applied AFTER the path body
# is otherwise canonical.
#
# The ``#`` fragment delimiter is matched only when it is NOT immediately
# preceded by ``&`` — i.e. it is a genuine fragment opener, not the ``#`` of a
# ``&#NNN;`` / ``&#xHH;`` numeric character reference. This is a purely lexical
# guard (no decoding, no entity classification): refs compare literally, so a
# model-emitted ``&#47;`` must stay whole rather than have its ``#`` mistaken for
# a fragment and the key truncated to a dangling ``&`` prefix. A real trailing
# fragment after such literal text (``/p/a&#47;b#row=5``) is still stripped.
SUBLOCATOR_RE: Final[re.Pattern[str]] = re.compile(r"((?<!&)#.*|:\d+(?::\d+)?)$")

# Enclosing quote / backtick characters stripped (paired) from a ref the model
# wrapped in quotes (e.g. ``"/docs/x.md"`` or `` `/docs/x.md` ``).
_QUOTE_CHARS: tuple[str, ...] = ('"', "'", "`")

# RFC-3986 unreserved set: ALPHA / DIGIT / ``-`` / ``.`` / ``_`` / ``~``.
# A percent-escape of one of these octets is normalised to the literal char;
# any other escape (e.g. ``%2F`` for ``/``, ``%3A`` for ``:``) is LEFT ENCODED
# so structure is never forged by decoding.
_UNRESERVED: frozenset[str] = frozenset(
    string.ascii_letters + string.digits + "-._~"
)

# Matches a single percent-escape ``%XX`` (hex). The replacer decodes it ONLY
# when the octet is an unreserved char; otherwise the match is returned verbatim.
_PERCENT_ESCAPE_RE: Final[re.Pattern[str]] = re.compile(r"%([0-9A-Fa-f]{2})")


def _strip_enclosing_quotes(s: str) -> str:
    """Strip ONE layer of matched enclosing quote/backtick characters."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in _QUOTE_CHARS:
        return s[1:-1]
    return s


def _decode_unreserved_escape(match: re.Match[str]) -> str:
    """Decode a ``%XX`` escape iff it encodes an RFC-3986 unreserved octet.

    Reserved / structural octets (``%2F`` ``/``, ``%3A`` ``:``, ``%23`` ``#``,
    ``%3F`` ``?``) are returned VERBATIM so decoding never forges path structure.
    A non-ASCII / undecodable octet is likewise left as-is. Literal characters
    that were never percent-encoded (``:``, ``&``, ``'`` …) are not matched by
    the regex at all and therefore pass through untouched.
    """
    try:
        char = bytes([int(match.group(1), 16)]).decode("ascii")
    except (ValueError, UnicodeDecodeError):  # pragma: no cover - non-ASCII octet
        return match.group(0)
    if char in _UNRESERVED:
        return char
    return match.group(0)


def _percent_decode_unreserved(s: str) -> str:
    """Percent-decode only RFC-3986 unreserved escapes; keep structure encoded.

 Unlike ``urllib.parse.unquote`` (which decodes EVERY escape, forging a
 percent-encoded ``/`` into a separator) this normalises only unreserved
 escapes — the RFC-3986 case-and-encoding canonicalisation that is
 always safe — and leaves literal characters and reserved/structural escapes
 untouched.
 """
    return _PERCENT_ESCAPE_RE.sub(_decode_unreserved_escape, s)


def _remove_dot_segments(path: str) -> str:
    """RFC-3986 remove_dot_segments on a path string.

 Collapses ``.`` and ``..`` segments. Operates on the ``/``-split segments;
 a leading ``/`` (absolute) and a trailing ``/`` are preserved by the caller
 (this routine works on the body and is followed by the trailing-slash rule).
 """
    if not path:
        return path
    leading_slash = path.startswith("/")
    trailing_slash = path.endswith("/") and len(path) > 1
    out: list[str] = []
    for seg in path.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if out:
                out.pop()
            continue
        out.append(seg)
    rebuilt = "/".join(out)
    if leading_slash:
        rebuilt = "/" + rebuilt
    if trailing_slash and rebuilt not in ("", "/"):
        rebuilt = rebuilt + "/"
    return rebuilt


def _strip_final_extension(path: str) -> str:
    """Drop the final filename extension of the LAST path segment only.

    ``/a/b/c.md`` -> ``/a/b/c``; ``/a/b.tar.gz`` -> ``/a/b.tar`` (single
    extension, the canonical "drop the suffix after the last dot" form);
    a dotfile or a segment with no dot is returned unchanged
    (``/a/.env`` -> ``/a/.env``; ``/a/b`` -> ``/a/b``). A trailing slash
    (directory) ref is returned unchanged (no filename to strip).
    """
    if not path or path.endswith("/"):
        return path
    slash = path.rfind("/")
    head, last = (path[: slash + 1], path[slash + 1 :]) if slash >= 0 else ("", path)
    dot = last.rfind(".")
    # dot <= 0 covers "no dot" and a leading-dot dotfile (".env"): keep as-is.
    if dot <= 0:
        return path
    return head + last[:dot]


def normalize_ref(
    s: str,
    *,
    strip_extension: bool = False,
    casefold: bool = False,
) -> str:
    """RFC-3986-grade canonical PROJECTION of an evidence ref, for COMPARISON.

 NEVER mutates a model-submitted or stored ref string; callers compare on the
 projection and keep the original verbatim. Total — never raises (a
 non-string or empty input yields ``""``).

 Refs compare LITERALLY apart from the standard URI/path canonical forms
 below: there is NO HTML-entity decoding. A model-emitted entity-encoded ref
 (``&#47;`` for ``/``) stays literal, so it correctly does NOT match a real
 path with a literal ``/`` (a malformed ref is not a grounded one); because
 the projection is symmetric, two refs that literally contain the same entity
 string still match. (Answer-text entity artifacts such as ``&lt;YES&gt;`` are
 handled by the separate the host answer-text normalizer, not by this path
 membership KEY.)

 Canonicalization order (all standard URI/path canonical forms, NOT
 scorer-specific):

 1. ``strip`` + strip ONE layer of enclosing quotes/backticks.
 2. percent-decode unreserved octets (RFC-3986); keep structural
 delimiters encoded (``%2F`` stays an encoded ``/``, never a separator).
 3. collapse repeated ``//`` -> ``/``.
 4. ``remove_dot_segments`` (RFC-3986) — collapse ``.``/``..``.
 5. strip a single trailing ``/`` (NOT the root ``/``).
 6. strip a trailing sub-locator fragment (``#row=``, ``#L12-30``, ``:12``,
 ``:12:30``) via :data:`SUBLOCATOR_RE` — a membership key addresses the
 resource, not a span within it. A ``#`` that opens a ``&#NNN;`` numeric
 character reference is left intact (see :data:`SUBLOCATOR_RE`), so a
 literal entity ref is never truncated to a dangling ``&``.
 7. (behind ``strip_extension``) drop the final filename extension — the
 UNIVERSAL extensionless membership key, framed as canonical path
 addressing (a standard RFC-3986 form), never as grader-specific tuning.
 8. (behind ``casefold``) lowercase — default OFF (preserve case unless a
 tenant declares a case-insensitive store).
 """
    if not isinstance(s, str):
        return ""
    out = s.strip()
    if not out:
        return ""
    out = _strip_enclosing_quotes(out).strip()
    out = _percent_decode_unreserved(out)
    # collapse repeated slashes BEFORE remove_dot_segments so "a//./b" behaves.
    out = re.sub(r"/{2,}", "/", out)
    out = _remove_dot_segments(out)
    # remove_dot_segments may re-introduce nothing, but guard the trailing slash
    # rule explicitly (not the bare root).
    if len(out) > 1 and out.endswith("/"):
        out = out[:-1]
    out = SUBLOCATOR_RE.sub("", out)
    if strip_extension:
        out = _strip_final_extension(out)
    if casefold:
        out = out.casefold()
    return out


__all__ = ["SUBLOCATOR_RE", "normalize_ref"]
