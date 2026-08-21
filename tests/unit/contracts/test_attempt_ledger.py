"""Unit tests for the AttemptLedger finalization contract.

Covers:
 - ``compute_outcome`` edge cases (B1 vacuous-truth fix, B2 unverified -> failed)
 - Wire-shape coercion (list-of-declarations input)
 - Query helpers — ``required_verified_exists`` / ``required_verified_valid``
 - Query helpers — ``unresolved_required_count`` / ``declarations_by_agent``

Tests integrated onto the canonical core suite.
"""

from __future__ import annotations

from protocore.contracts.attempt_ledger import (
    AttemptLedger,
    AttemptRecord,
    DeliverableDeclaration,
    VerificationRecord,
)


def _make_ledger(run_id: str = "run-1") -> AttemptLedger:
    return AttemptLedger(run_id=run_id)


def _verified_record(
    path: str,
    *,
    exists: bool = True,
    valid_by_schema: bool | None = None,
) -> VerificationRecord:
    return VerificationRecord(
        deliverable_path=path,
        exists=exists,
        valid_by_schema=valid_by_schema,
    )


# ---------------------------------------------------------------------------
# compute_outcome edge cases
# ---------------------------------------------------------------------------


def test_ledger_no_declarations_returns_unknown() -> None:
    ledger = AttemptLedger(run_id="r")
    assert ledger.compute_outcome() == "unknown"


def test_ledger_required_present_returns_completed() -> None:
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    ledger.record_verification(
        VerificationRecord(deliverable_path="site.html", exists=True, size_bytes=1024)
    )
    assert ledger.compute_outcome() == "completed"


def test_ledger_required_missing_returns_failed() -> None:
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    ledger.record_verification(
        VerificationRecord(deliverable_path="site.html", exists=False)
    )
    assert ledger.compute_outcome() == "failed"


def test_ledger_required_unverified_returns_failed() -> None:
    """Reviewer B2: verification never attempted is not the same as
    verified-as-missing, but we treat it as failed for the outcome — the
    operator-visible signal is the verifications log itself (empty).
    """
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    assert ledger.compute_outcome() == "failed"


def test_ledger_all_optional_none_exist_returns_failed() -> None:
    """B1: all-optional + nothing exists should be failed, not completed.

    Vacuous-truth bug: the old code returned True for
    ``required_verified_exists`` on an empty required list, classifying an
    all-empty workspace as completed.
    """
    ledger = AttemptLedger(run_id="r")
    ledger.declare(
        DeliverableDeclaration(
            path="opt.html", declared_by_agent="coder", required=False
        )
    )
    ledger.record_verification(
        VerificationRecord(deliverable_path="opt.html", exists=False)
    )
    assert ledger.compute_outcome() == "failed"


def test_ledger_all_optional_all_exist_returns_completed() -> None:
    """B1: when all-optional declarations exist on disk, we still complete."""
    ledger = AttemptLedger(run_id="r")
    ledger.declare(
        DeliverableDeclaration(
            path="opt.html", declared_by_agent="coder", required=False
        )
    )
    ledger.record_verification(
        VerificationRecord(deliverable_path="opt.html", exists=True)
    )
    assert ledger.compute_outcome() == "completed"


def test_ledger_required_present_optional_missing_returns_completed() -> None:
    """B1: required exists + optional missing -> completed."""
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="req.html", declared_by_agent="coder"))
    ledger.declare(
        DeliverableDeclaration(
            path="opt.html", declared_by_agent="coder", required=False
        )
    )
    ledger.record_verification(
        VerificationRecord(deliverable_path="req.html", exists=True)
    )
    ledger.record_verification(
        VerificationRecord(deliverable_path="opt.html", exists=False)
    )
    assert ledger.compute_outcome() == "completed"


def test_ledger_partial_outcome_self_reported_success() -> None:
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="a.html", declared_by_agent="coder"))
    ledger.declare(DeliverableDeclaration(path="b.html", declared_by_agent="coder"))
    ledger.record_verification(
        VerificationRecord(deliverable_path="a.html", exists=True)
    )
    ledger.record_verification(
        VerificationRecord(deliverable_path="b.html", exists=False)
    )
    ledger.record_attempt(
        AttemptRecord(agent_id="coder", self_reported_status="success")
    )
    assert ledger.compute_outcome() == "partial"


def test_ledger_schema_invalid_returns_failed() -> None:
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="data.json", declared_by_agent="coder"))
    ledger.record_verification(
        VerificationRecord(
            deliverable_path="data.json",
            exists=True,
            valid_by_schema=False,
            schema_kind="json",
        )
    )
    assert ledger.compute_outcome() == "failed"


def test_ledger_serialization_roundtrip() -> None:
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="x.html", declared_by_agent="coder"))
    ledger.record_verification(
        VerificationRecord(deliverable_path="x.html", exists=True)
    )
    dumped = ledger.model_dump()
    restored = AttemptLedger.model_validate(dumped)
    assert restored.compute_outcome() == "completed"


def test_ledger_accepts_list_of_declarations_on_wire() -> None:
    """Subagent sends declared_deliverables as a JSON list; ledger coerces."""
    ledger = AttemptLedger.model_validate(
        {
            "run_id": "r",
            "declared_deliverables": [
                {"path": "site.html", "kind": "file", "declared_by_agent": "coder"}
            ],
        }
    )
    assert "site.html" in ledger.declared_deliverables


# ---------------------------------------------------------------------------
# required_verified_exists
# ---------------------------------------------------------------------------


def test_required_verified_exists_true_when_required_decl_passes_verification() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    led.record_verification(_verified_record("index.html", exists=True))
    assert led.required_verified_exists() is True


def test_required_verified_exists_false_if_required_missing_verification() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    # No verification recorded
    assert led.required_verified_exists() is False


def test_required_verified_exists_false_if_verification_reports_missing() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    led.record_verification(_verified_record("index.html", exists=False))
    assert led.required_verified_exists() is False


def test_optional_declarations_do_not_block_required_verified_exists() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    led.declare(DeliverableDeclaration(path="optional.txt", required=False))
    led.record_verification(_verified_record("index.html", exists=True))
    # Optional unverified is fine for required_verified_exists
    assert led.required_verified_exists() is True


def test_required_verified_exists_false_when_one_of_multiple_required_unverified() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", required=True))
    led.declare(DeliverableDeclaration(path="b.html", required=True))
    led.record_verification(_verified_record("a.html", exists=True))
    # b.html not verified — should be False
    assert led.required_verified_exists() is False


# ---------------------------------------------------------------------------
# required_verified_valid
# ---------------------------------------------------------------------------


def test_required_verified_valid_true_when_no_schema_errors() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    led.record_verification(
        _verified_record("index.html", exists=True, valid_by_schema=None)
    )
    assert led.required_verified_valid() is True


def test_required_verified_valid_false_for_schema_invalid_file() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="data.json", required=True))
    led.record_verification(
        _verified_record("data.json", exists=True, valid_by_schema=False)
    )
    assert led.required_verified_valid() is False


def test_required_verified_valid_false_when_required_missing_verification() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    assert led.required_verified_valid() is False


def test_required_verified_valid_true_for_explicit_true_schema() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="index.html", required=True))
    led.record_verification(
        _verified_record("index.html", exists=True, valid_by_schema=True)
    )
    assert led.required_verified_valid() is True


def test_required_verified_valid_ignores_optional_declarations_when_required_passes() -> None:
    """Optional schema-invalid declarations should not block required_verified_valid."""
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="primary.json", required=True))
    led.declare(DeliverableDeclaration(path="optional.json", required=False))
    led.record_verification(
        _verified_record("primary.json", exists=True, valid_by_schema=True)
    )
    led.record_verification(
        _verified_record("optional.json", exists=True, valid_by_schema=False)
    )
    assert led.required_verified_valid() is True


# ---------------------------------------------------------------------------
# unresolved_required_count
# ---------------------------------------------------------------------------


def test_unresolved_required_count_zero_when_all_required_verified() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", required=True))
    led.declare(DeliverableDeclaration(path="b.html", required=True))
    led.record_verification(_verified_record("a.html", exists=True))
    led.record_verification(_verified_record("b.html", exists=True))
    assert led.unresolved_required_count() == 0


def test_unresolved_required_count_counts_missing_required() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", required=True))
    led.declare(DeliverableDeclaration(path="b.html", required=True))
    led.record_verification(_verified_record("a.html", exists=True))
    led.record_verification(_verified_record("b.html", exists=False))
    assert led.unresolved_required_count() == 1


def test_unresolved_required_count_counts_unverified_required() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", required=True))
    led.declare(DeliverableDeclaration(path="b.html", required=True))
    led.record_verification(_verified_record("a.html", exists=True))
    # b.html never verified
    assert led.unresolved_required_count() == 1


def test_unresolved_required_count_ignores_optional_declarations() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", required=True))
    led.declare(DeliverableDeclaration(path="b.html", required=False))
    led.record_verification(_verified_record("a.html", exists=True))
    # Optional b.html missing must not count
    assert led.unresolved_required_count() == 0


# ---------------------------------------------------------------------------
# declarations_by_agent
# ---------------------------------------------------------------------------


def test_declarations_by_agent_filters_by_agent_id() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", declared_by_agent="coder"))
    led.declare(DeliverableDeclaration(path="b.html", declared_by_agent="reviewer"))
    led.declare(DeliverableDeclaration(path="c.html", declared_by_agent="coder"))
    coder_decls = led.declarations_by_agent("coder")
    assert {d.path for d in coder_decls} == {"a.html", "c.html"}


def test_declarations_by_agent_empty_when_no_match() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="a.html", declared_by_agent="coder"))
    assert led.declarations_by_agent("nonexistent") == []


def test_declarations_by_agent_preserves_insertion_order() -> None:
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="z.html", declared_by_agent="coder"))
    led.declare(DeliverableDeclaration(path="a.html", declared_by_agent="coder"))
    led.declare(DeliverableDeclaration(path="m.html", declared_by_agent="coder"))
    decls = led.declarations_by_agent("coder")
    assert [d.path for d in decls] == ["z.html", "a.html", "m.html"]


# ---------------------------------------------------------------------------
# declare(order=...) — batch-order-deterministic same-path resolution under
# concurrent delegation fan-out (independent of gather completion order)
# ---------------------------------------------------------------------------


def test_declare_without_order_is_last_writer_wins() -> None:
    """Serial / single-child / leader-inferred path is unchanged."""
    led = _make_ledger()
    led.declare(DeliverableDeclaration(path="out.txt", declared_by_agent="first"))
    led.declare(DeliverableDeclaration(path="out.txt", declared_by_agent="second"))
    assert led.declared_deliverables["out.txt"].declared_by_agent == "second"


def test_declare_same_path_resolves_to_batch_order_winner() -> None:
    """Two parallel siblings in ONE group declaring the SAME path resolve to the
    batch-order winner (highest LLM-requested index) NO MATTER which child
    finishes first.

    Simulates both ``asyncio.gather`` completion orders by applying the two
    order-tagged declarations in each sequence; the survivor must be the child
    with the higher batch index (child 1) both times.
    """
    for first_idx, second_idx in ((0, 1), (1, 0)):
        led = _make_ledger()
        led.declare(
            DeliverableDeclaration(
                path="report.html",
                declared_by_agent=f"child-{first_idx}",
                summary=f"from-{first_idx}",
            ),
            order=first_idx,
            group="grp-A",
        )
        led.declare(
            DeliverableDeclaration(
                path="report.html",
                declared_by_agent=f"child-{second_idx}",
                summary=f"from-{second_idx}",
            ),
            order=second_idx,
            group="grp-A",
        )
        survivor = led.declared_deliverables["report.html"]
        assert survivor.declared_by_agent == "child-1", (
            f"completion order ({first_idx}, {second_idx}) must not change the "
            "winner"
        )
        assert survivor.summary == "from-1"


def test_declare_later_group_wins_even_with_lower_indices() -> None:
    """Cross-turn regression guard: two SEQUENTIAL concurrent groups (different
    leader turns) declare the SAME path. The LATER group must win even when its
    batch is NARROWER (lower max index) than the earlier one — matching the
    serial cross-turn last-writer-wins the ``order=None`` path gives. The earlier
    group's high-water mark must NOT freeze out the later group.

    (The ledger persists its transient order map in-memory across turns within a
    hot run, so without per-group scoping the later narrower group would be
    silently dropped — the bug this guards.)
    """
    for first_idx, second_idx in ((0, 1), (1, 0)):
        led = _make_ledger()
        # Turn 1 group: three siblings on X → highest index (2) wins.
        for i in range(3):
            led.declare(
                DeliverableDeclaration(path="X", declared_by_agent=f"g1-c{i}"),
                order=i,
                group="turn1-group",
            )
        assert led.declared_deliverables["X"].declared_by_agent == "g1-c2"
        # Turn 5 group: two siblings on X, indices {0,1} (max 1 < turn-1's 2),
        # applied in each completion order — the later group's highest index wins.
        for i in (first_idx, second_idx):
            led.declare(
                DeliverableDeclaration(path="X", declared_by_agent=f"g2-c{i}"),
                order=i,
                group="turn5-group",
            )
        assert led.declared_deliverables["X"].declared_by_agent == "g2-c1", (
            "later (narrower) group must win despite lower batch indices; "
            f"completion order ({first_idx}, {second_idx})"
        )


def test_declare_order_leaves_distinct_paths_independent() -> None:
    """Order tagging only affects SAME-path conflicts; distinct paths coexist."""
    led = _make_ledger()
    led.declare(
        DeliverableDeclaration(path="a.txt", declared_by_agent="c0"),
        order=0,
        group="grp-A",
    )
    led.declare(
        DeliverableDeclaration(path="b.txt", declared_by_agent="c1"),
        order=1,
        group="grp-A",
    )
    assert set(led.declared_deliverables) == {"a.txt", "b.txt"}
    assert led.declared_deliverables["a.txt"].declared_by_agent == "c0"
    assert led.declared_deliverables["b.txt"].declared_by_agent == "c1"
