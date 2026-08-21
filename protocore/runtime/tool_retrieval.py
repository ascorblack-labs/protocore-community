"""BM25-based multilingual tool retrieval.

Self-contained Okapi BM25 (no external deps), multilingual stopwords
(EN + RU), IDF-based query reduction. Used by the 3-layer tool-surface
filter to keep small-model surfaces focused.

Per-call only — no caching, no module-level state (horizontal scaling).

Architecture:
    - :func:`build_candidate` / :class:`ToolRetrievalCandidate`: pre-built
      searchable form of a tool (name + description).
    - :func:`bm25_score`: per-candidate Okapi BM25 score.
    - :func:`compute_idf` / :func:`compute_avgdl`: corpus aggregates.
    - :func:`reduce_query`: IDF-based query reduction (Elasticsearch MLT
      approach) — used internally by :func:`retrieve_tools` to keep long
      queries focused on rare/distinctive terms.
    - :func:`retrieve_tools`: main entry point — pinned tools + BM25 top-K.
"""
# ruff: noqa: RUF001 — Cyrillic stopword strings are intentional

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Final

# English + Russian stopwords (subset; expanded list keeps retrieval focused).
_STOPWORDS_EN: Final[frozenset[str]] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "for", "in", "on", "at",
        "to", "from", "with", "without", "as", "is", "are", "was", "were",
        "be", "been", "being", "this", "that", "these", "those", "it", "its",
        "by", "if", "then", "else", "do", "does", "did", "have", "has", "had",
        "i", "you", "he", "she", "we", "they", "them", "us", "our", "your",
        "my", "me", "his", "her", "their",
    }
)

_STOPWORDS_RU: Final[frozenset[str]] = frozenset(
    {
        "и", "в", "не", "на", "что", "я", "с", "со", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня",
        "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну",
        "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
        "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
        "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
        "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    }
)

_STOPWORDS: Final[frozenset[str]] = _STOPWORDS_EN | _STOPWORDS_RU

# Unicode-aware word splitter — matches letters/digits, drops punctuation.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# BM25 hyperparameters — frozen here (not RC-tunable: stable retrieval semantics).
_BM25_K1: Final[float] = 1.5
_BM25_B: Final[float] = 0.75


@dataclass(frozen=True, slots=True)
class ToolRetrievalCandidate:
    """A candidate tool with its searchable text + score."""

    name: str
    description: str
    text: str
    """Full searchable corpus (name + description, lowercased)."""


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, drop stopwords."""
    return [w for w in _WORD_RE.findall(text.lower()) if w and w not in _STOPWORDS]


def build_candidate(
    name: str,
    description: str,
    hint: str = "",
) -> ToolRetrievalCandidate:
    """Build a retrieval candidate from a tool name + description.

    ``hint`` is an optional multilingual retrieval hint (the tool's
    ``search_hint`` ClassVar — EN+RU keywords) joined into the searchable
    ``text`` ONLY. It never reaches ``description`` (the LLM-visible wire
    text), so discovery can be multilingual without bloating the schema.
    """
    text = f"{name} {description} {hint}" if hint else f"{name} {description}"
    return ToolRetrievalCandidate(
        name=name,
        description=description,
        text=text.lower(),
    )


def bm25_score(
    query: str,
    candidate: ToolRetrievalCandidate,
    avgdl: float,
    idf: dict[str, float],
) -> float:
    """BM25 score for one candidate against ``query``.

    Caller provides ``avgdl`` (corpus average doc length in tokens) and
    ``idf`` (term → inverse-document-frequency).
    """
    q_terms = _tokenize(query)
    doc_terms = _tokenize(candidate.text)
    if not q_terms or not doc_terms:
        return 0.0
    tf = Counter(doc_terms)
    dl = float(len(doc_terms))
    score = 0.0
    for term in q_terms:
        if term not in tf:
            continue
        term_idf = idf.get(term, 0.0)
        if term_idf == 0.0:
            continue
        numerator = tf[term] * (_BM25_K1 + 1.0)
        denominator = tf[term] + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / avgdl)
        score += term_idf * numerator / denominator
    return score


def compute_idf(candidates: list[ToolRetrievalCandidate]) -> dict[str, float]:
    """Compute inverse-document-frequency over the candidate corpus."""
    if not candidates:
        return {}
    n = len(candidates)
    df: Counter[str] = Counter()
    for cand in candidates:
        df.update(set(_tokenize(cand.text)))
    return {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}


def compute_avgdl(candidates: list[ToolRetrievalCandidate]) -> float:
    """Compute average document length (in tokens) over corpus."""
    if not candidates:
        return 1.0
    total = sum(len(_tokenize(c.text)) for c in candidates)
    return max(1.0, total / len(candidates))


# Default upper bound for query terms before IDF-based reduction kicks in.
# Mirrors v1 ``query_max_terms`` default; not RC-tunable (retrieval semantic
# stability matters more than per-tenant fine-tuning here).
_DEFAULT_MAX_QUERY_TERMS: Final[int] = 25


def reduce_query(
    text: str,
    idf: dict[str, float],
    *,
    max_terms: int = _DEFAULT_MAX_QUERY_TERMS,
) -> list[str]:
    """Reduce a long query to its top-``max_terms`` tokens by IDF weight.

    Mirrors the Elasticsearch ``more_like_this`` approach: tokenise + drop
    stopwords, then rank surviving terms by IDF (rarer = more distinctive)
    and keep the top-K. De-duplicates while preserving the first-seen IDF.

    For long user queries this prevents common terms from drowning the
    retrieval signal — small-model surfaces become much more focused.
    """
    tokens = [t for t in _tokenize(text) if len(t) > 1]
    if len(tokens) <= max_terms:
        return tokens

    seen: dict[str, float] = {}
    for t in tokens:
        if t not in seen:
            seen[t] = idf.get(t, 0.0)

    ranked = sorted(seen, key=seen.__getitem__, reverse=True)
    return ranked[:max_terms]


def retrieve_tools(
    query: str,
    candidates: list[ToolRetrievalCandidate],
    *,
    top_k: int,
    pinned: frozenset[str] = frozenset(),
    query_max_terms: int = _DEFAULT_MAX_QUERY_TERMS,
) -> list[ToolRetrievalCandidate]:
    """Return top-K candidates by BM25 score, with pinned tools always included.

    Empty/whitespace query → returns pinned tools plus the first
    ``top_k - len(pinned)`` other candidates by name order (no retrieval).
    It never returns an empty list when the catalog is non-empty and
    ``top_k > 0`` — a zero-tool surface would leave the model unable to act.

    Long queries (more than ``query_max_terms`` tokens) are reduced via
    :func:`reduce_query` so rare/distinctive terms dominate scoring.
    """
    if not candidates:
        return []
    if top_k <= 0:
        return []

    pinned_candidates = [c for c in candidates if c.name in pinned]
    other_candidates = [c for c in candidates if c.name not in pinned]

    if not other_candidates:
        retrieved: list[ToolRetrievalCandidate] = []
    elif not query.strip():
        # Empty/whitespace query → no retrieval signal. Fall back to the
        # first ``top_k - len(pinned)`` other candidates by name order
        # (deterministic, KV-prefix-cache stable) rather than returning
        # nothing. : the prior ``retrieved = `` handed the model a
        # ZERO-tool surface whenever the clip path fired with no user message
        # yet (autonomous batch / synthetic resume) or a whitespace-only
        # message, contradicting this function's documented contract. Sorting
        # by name (not raw input order) makes the fallback independent of
        # caller ordering, matching ``ToolRegistry.search``'s "first top_k by
        # name order" promise — the clip callers
        # (``ToolRegistry.compute_effective_surface`` / ``assemble_tool_pool``)
        # already feed a name-sorted catalog, so this is a no-op for them.
        retrieved = sorted(other_candidates, key=lambda c: c.name)
    else:
        avgdl = compute_avgdl(other_candidates)
        idf = compute_idf(other_candidates)

        # IDF-based query reduction for long queries (Elasticsearch MLT).
        # ``bm25_score`` re-tokenizes ``query`` internally, so we project the
        # reduced token list back to a synthetic space-joined string.
        query_tokens = _tokenize(query)
        if len(query_tokens) > query_max_terms:
            reduced = reduce_query(query, idf, max_terms=query_max_terms)
            scoring_query = " ".join(reduced)
        else:
            scoring_query = query

        scored = [(bm25_score(scoring_query, c, avgdl, idf), c) for c in other_candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = [c for score, c in scored if score > 0.0]

    # Pinned ALWAYS included; retrieved fills remaining budget.
    remaining = max(0, top_k - len(pinned_candidates))
    return pinned_candidates + retrieved[:remaining]


# ---------------------------------------------------------------------------
# Normalized zero-score fallback (tools-initiative A2 — multilingual mandate)
# ---------------------------------------------------------------------------

# Minimum query-token length considered meaningful for the fallback (shorter
# tokens over-match) and the minimum shared-prefix length for the light
# morphological match (RU inflections share long stems: "память"/"памяти").
_FALLBACK_MIN_TOKEN_LEN: Final[int] = 3
_FALLBACK_MIN_PREFIX_LEN: Final[int] = 4


def _tokens_match(query_token: str, doc_token: str) -> bool:
    """Loose normalized match between one query token and one doc token.

    Containment either way ("workspace" ⊂ "workspacewrite") or a shared
    prefix long enough to absorb inflectional endings ("память"/"памяти"
    share "памят"). The prefix floor scales down for short tokens but never
    below :data:`_FALLBACK_MIN_PREFIX_LEN`.
    """
    if query_token in doc_token or doc_token in query_token:
        return True
    required = max(
        _FALLBACK_MIN_PREFIX_LEN,
        min(len(query_token), len(doc_token)) - 2,
    )
    if len(query_token) < required or len(doc_token) < required:
        return False
    return query_token[:required] == doc_token[:required]


def normalized_fallback_match(
    query: str,
    candidates: list[ToolRetrievalCandidate],
    *,
    top_k: int,
) -> list[ToolRetrievalCandidate]:
    """Zero-score fallback for queries BM25 cannot match exactly.

    BM25 keeps only ``score > 0.0`` candidates, so a query whose tokens have
    no exact overlap with the corpus (an inflected RU form against an EN+hint
    corpus, a partial tool name) returns nothing. This fallback re-ranks the
    SAME candidates with case-folded substring/prefix matching: a candidate
    scores one point per matched query token; zero-point candidates are
    dropped. Deterministic: score DESC, then name ASC; capped at ``top_k``.

    Used by ``ToolRegistry.search`` (the ToolSearch discovery path) only —
    the per-turn surface clip keeps strict BM25 + pin semantics.
    """
    if not candidates or top_k <= 0:
        return []
    q_tokens = [
        t
        for t in _tokenize(query)
        if len(t) >= _FALLBACK_MIN_TOKEN_LEN
    ]
    if not q_tokens:
        return []
    scored: list[tuple[int, ToolRetrievalCandidate]] = []
    for cand in candidates:
        doc_tokens = _WORD_RE.findall(cand.text)
        hits = 0
        for qt in q_tokens:
            if qt in cand.text:
                hits += 1
                continue
            if any(_tokens_match(qt, dt) for dt in doc_tokens):
                hits += 1
        if hits > 0:
            scored.append((hits, cand))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [cand for _, cand in scored[:top_k]]


__all__ = [
    "ToolRetrievalCandidate",
    "bm25_score",
    "build_candidate",
    "compute_avgdl",
    "compute_idf",
    "normalized_fallback_match",
    "reduce_query",
    "retrieve_tools",
]
