"""Tests for :mod:`protocore.runtime.tool_retrieval`."""
from __future__ import annotations

from protocore.runtime.tool_retrieval import (
    build_candidate,
    compute_avgdl,
    compute_idf,
    reduce_query,
    retrieve_tools,
)


def test_retrieval_returns_top_k() -> None:
    candidates = [
        build_candidate("Bash", "Run shell commands."),
        build_candidate("Read", "Read a file from workspace."),
        build_candidate("Write", "Write text to a file."),
        build_candidate("Grep", "Search files via grep."),
    ]
    results = retrieve_tools("read a file", candidates, top_k=2)
    assert len(results) <= 2
    assert any(c.name == "Read" for c in results)


def test_pinned_always_included() -> None:
    candidates = [
        build_candidate("Bash", "Run shell commands."),
        build_candidate("Read", "Read a file from workspace."),
        build_candidate("Write", "Write text to a file."),
    ]
    results = retrieve_tools(
        "search the web",
        candidates,
        top_k=2,
        pinned=frozenset({"Bash"}),
    )
    names = [c.name for c in results]
    assert "Bash" in names


def test_empty_query_returns_first_k() -> None:
    """Empty query falls back to the first ``top_k`` candidates by input order.

 Honors the module docstring contract ("Empty query -> returns first
 ``top_k`` candidates"). Regression for : an empty/whitespace
 query used to return ``[]`` (zero tools advertised to the model) when
 nothing was pinned.
 """
    candidates = [
        build_candidate("A", "alpha"),
        build_candidate("B", "beta"),
        build_candidate("C", "gamma"),
    ]
    results = retrieve_tools("", candidates, top_k=2)
    assert [c.name for c in results] == ["A", "B"]


def test_whitespace_query_returns_first_k() -> None:
    """A whitespace-only query is treated the same as an empty query."""
    candidates = [
        build_candidate("A", "alpha"),
        build_candidate("B", "beta"),
        build_candidate("C", "gamma"),
    ]
    results = retrieve_tools("   \t\n  ", candidates, top_k=2)
    assert [c.name for c in results] == ["A", "B"]


def test_empty_query_catalog_exceeds_top_k_returns_top_k() -> None:
    """Catalog > top_k + empty query must still advertise ``top_k`` tools.

 Regression for : the clip path used by ``compute_effective_surface``
 and ``assemble_tool_pool`` triggers exactly when the policy-filtered
 catalog exceeds ``top_k``. With an empty/whitespace query (no user
 message yet, autonomous batch, synthetic resume) the model must still
 receive ``top_k`` tools, not an empty tools array.
 """
    candidates = [build_candidate(f"tool_{i:02d}", f"desc {i}") for i in range(10)]
    top_k = 3
    results = retrieve_tools("", candidates, top_k=top_k)
    assert len(results) == top_k
    # Deterministic: first ``top_k`` by the candidates' input (name) order.
    assert [c.name for c in results] == ["tool_00", "tool_01", "tool_02"]


def test_empty_query_with_pins_fills_remaining_with_first_k() -> None:
    """Empty query + pins: pinned first, then first non-pinned by input order."""
    candidates = [build_candidate(f"tool_{i:02d}", f"desc {i}") for i in range(10)]
    results = retrieve_tools(
        "",
        candidates,
        top_k=3,
        pinned=frozenset({"tool_07"}),
    )
    names = [c.name for c in results]
    assert "tool_07" in names
    assert len(results) == 3
    # Pinned tool plus the first two non-pinned candidates by input order.
    assert names == ["tool_07", "tool_00", "tool_01"]


def test_empty_corpus() -> None:
    assert retrieve_tools("anything", [], top_k=5) == []


def test_zero_top_k() -> None:
    candidates = [build_candidate("A", "alpha")]
    assert retrieve_tools("alpha", candidates, top_k=0) == []


def test_russian_query_matches_russian_description() -> None:
    candidates = [
        build_candidate("ReadRu", "Прочитать файл из рабочей области."),
        build_candidate("Bash", "Run shell commands."),
    ]
    results = retrieve_tools("прочитать файл", candidates, top_k=1)
    assert results
    assert results[0].name == "ReadRu"


def test_stopwords_filtered() -> None:
    """Verify stopwords don't dominate scoring."""
    candidates = [
        build_candidate("Read", "the file to be read"),
        build_candidate("Write", "to write a text payload"),
    ]
    results = retrieve_tools("read", candidates, top_k=1)
    assert results
    assert results[0].name == "Read"


# --- reduce_query (IDF-based query reduction) ------------------------------


def test_reduce_query_passthrough_when_short() -> None:
    """Queries within ``max_terms`` are returned tokenised, no reduction."""
    candidates = [
        build_candidate("Read", "Read a file from workspace."),
        build_candidate("Write", "Write text to a file."),
    ]
    idf = compute_idf(candidates)
    result = reduce_query("read file", idf, max_terms=10)
    # Short query — all surviving tokens (post-stopword + len > 1) preserved.
    assert "read" in result
    assert "file" in result


def test_reduce_query_truncates_long_query_by_idf() -> None:
    """Long queries are reduced to top-K by IDF, with rarer terms winning."""
    candidates = [
        build_candidate("Search", "search find filter index"),
        build_candidate("Filter", "filter list reduce"),
    ]
    idf = compute_idf(candidates)
    # Long synthetic query; rarer ``index`` should rank above common ``filter``.
    long_query = "search " * 10 + "filter " * 10 + "index"
    result = reduce_query(long_query, idf, max_terms=2)
    assert len(result) == 2
    # The rarest term in the corpus should survive reduction.
    assert "index" in result


def test_reduce_query_drops_short_tokens() -> None:
    """Single-char tokens are dropped before reduction."""
    candidates = [build_candidate("A", "alpha bravo")]
    idf = compute_idf(candidates)
    result = reduce_query("a b alpha", idf, max_terms=5)
    assert "alpha" in result
    assert "a" not in result
    assert "b" not in result


def test_reduce_query_uses_zero_idf_for_unknown_terms() -> None:
    """Unknown terms get zero IDF and rank last in reduction."""
    candidates = [build_candidate("Read", "read write search")]
    idf = compute_idf(candidates)
    # ``read`` is in corpus, ``zzz`` is not.
    long = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega zzz read"
    result = reduce_query(long, idf, max_terms=1)
    # ``read`` is in corpus → has non-zero IDF; ``zzz`` is unknown.
    # Multiple corpus-unknown tokens get zero IDF; top-1 should be a corpus term
    # (``read`` is the only corpus-present token in the query).
    assert "read" in result


def test_retrieve_tools_reduces_long_query() -> None:
    """``retrieve_tools`` applies IDF reduction internally for long queries."""
    candidates = [
        build_candidate("Read", "read file"),
        build_candidate("Write", "write file"),
        build_candidate("Search", "search index find"),
    ]
    # Long noisy query; ``search`` is the most distinctive corpus term.
    long_query = (
        "common common common common common common common common common common "
        "common common common common common common common common common common "
        "common common common common common search"
    )
    results = retrieve_tools(long_query, candidates, top_k=1)
    assert results
    # Reduced query still has signal on ``search``.
    assert results[0].name == "Search"


def test_compute_avgdl_empty_returns_one() -> None:
    """Empty corpus yields ``avgdl=1.0`` (guard division-by-zero)."""
    assert compute_avgdl([]) == 1.0


def test_compute_idf_empty_returns_empty_dict() -> None:
    """Empty corpus → empty IDF map."""
    assert compute_idf([]) == {}
