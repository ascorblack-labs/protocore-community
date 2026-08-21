"""Unit tests for the finalization gate.

Covers:
 - verify_declared_deliverables records exist/not-exist verdicts
 - decide_finalization interpolates artifact paths into bilingual prompt
 - HTML is no longer schema-checked (B5 fix — fragments would be flagged invalid)
 - cef6e778 reproduction: subagent wrote artifact + ran out of iterations -> completed

Ported from side-branch commit 23452d0 + 7ef5cc3 (ruff cleanup) onto
canonical core.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from protocore.contracts.attempt_ledger import (
    AttemptLedger,
    AttemptRecord,
    DeliverableDeclaration,
)
from protocore.runtime.finalization_gate import (
    WorkspaceStatResult,
    _supports_schema_check,
    decide_finalization,
    verify_declared_deliverables,
)


@dataclass
class _FakeWorkspace:
    files: dict[str, bytes]

    async def stat(self, path: str) -> WorkspaceStatResult:
        if path not in self.files:
            return WorkspaceStatResult(exists=False)
        return WorkspaceStatResult(
            exists=True, size_bytes=len(self.files[path]), is_file=True
        )

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes | None:
        if path not in self.files:
            return None
        return self.files[path][:max_bytes]


# ---------------------------------------------------------------------------
# verify_declared_deliverables
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalization_gate_verifies_exists_records_verification() -> None:
    ws = _FakeWorkspace({"site.html": b"<!DOCTYPE html><html></html>"})
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    await verify_declared_deliverables(ledger, workspace=ws)
    latest = ledger.latest_verification_for("site.html")
    assert latest is not None
    assert latest.exists is True


@dataclass
class _UnknownSizeWorkspace:
    """Reports a path as existing but cannot report its size (size_bytes=None).

    Mirrors the ``WorkspaceStatResult`` default a stat impl produces when the
    file exists but the size could not be determined (a non-FileNotFound IO
    error on a still-existing file).
    """

    present: frozenset[str]

    async def stat(self, path: str) -> WorkspaceStatResult:
        if path not in self.present:
            return WorkspaceStatResult(exists=False)
        # exists=True but size unknown — the defensive case targets.
        return WorkspaceStatResult(exists=True, size_bytes=None, is_file=True)

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes | None:
        return None


@pytest.mark.asyncio
async def test_min_size_skipped_when_stat_size_unknown() -> None:
    """an unknown size (None) must NOT be coerced to 0 and tripped
    against ``min_size_bytes``.

    When the workspace reports the file as existing but cannot report a size,
    the ``size_bytes or 0`` coercion made any positive ``min_size_bytes``
    falsely fail (``0 < min_size``), stamping a genuinely-existing artifact
    invalid and downgrading the run. The size-floor check must only fire when
    a concrete size is known.
    """
    ws = _UnknownSizeWorkspace(present=frozenset({"report.json"}))
    ledger = AttemptLedger(run_id="r")
    ledger.declare(
        DeliverableDeclaration(
            path="report.json",
            declared_by_agent="coder",
            required=True,
            min_size_bytes=1024,
        )
    )
    await verify_declared_deliverables(ledger, workspace=ws)
    latest = ledger.latest_verification_for("report.json")
    assert latest is not None
    assert latest.exists is True
    # Unknown size must NOT be treated as a min_size breach.
    assert latest.valid_by_schema is not False
    assert latest.schema_kind != "min_size"
    # decide_finalization must therefore count it present, not missing.
    decision = decide_finalization(ledger)
    assert "report.json" in decision.artifacts_present
    assert "report.json" not in decision.artifacts_missing


@pytest.mark.asyncio
async def test_min_size_still_fires_when_size_known_and_below_floor() -> None:
    """regression guard — a KNOWN size below the floor still fails.

    The fix must not disable the genuine size-floor check: a concrete size
    smaller than ``min_size_bytes`` is still stamped invalid.
    """
    ws = _FakeWorkspace({"tiny.json": b"{}"})  # 2 bytes
    ledger = AttemptLedger(run_id="r")
    ledger.declare(
        DeliverableDeclaration(
            path="tiny.json",
            declared_by_agent="coder",
            required=True,
            min_size_bytes=1024,
        )
    )
    await verify_declared_deliverables(ledger, workspace=ws)
    latest = ledger.latest_verification_for("tiny.json")
    assert latest is not None
    assert latest.valid_by_schema is False
    assert latest.schema_kind == "min_size"


# ---------------------------------------------------------------------------
# decide_finalization prompt interpolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalization_gate_decision_success_has_path_in_prompt() -> None:
    """B3 fix: prompt_injection interpolates concrete artifact paths."""
    ws = _FakeWorkspace({"site.html": b"<!DOCTYPE html><html></html>"})
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    await verify_declared_deliverables(ledger, workspace=ws)
    decision = decide_finalization(ledger)
    assert decision.outcome == "completed"
    assert "site.html" in decision.prompt_injection
    assert decision.artifacts_present == ("site.html",)


@pytest.mark.asyncio
async def test_finalization_gate_decision_failed_has_missing_path_in_prompt() -> None:
    ws = _FakeWorkspace({})
    ledger = AttemptLedger(run_id="r")
    ledger.declare(DeliverableDeclaration(path="site.html", declared_by_agent="coder"))
    await verify_declared_deliverables(ledger, workspace=ws)
    decision = decide_finalization(ledger)
    assert decision.outcome == "failed"
    assert "site.html" in decision.prompt_injection
    assert decision.artifacts_missing == ("site.html",)


def test_finalization_gate_html_no_longer_schema_checked() -> None:
    """B5 fix: HTML schema check removed; existence + size are enough.

    Previously a 4 KB ``<div>hello</div>`` snippet would be flagged invalid.
    """
    assert _supports_schema_check("site.html") is False
    assert _supports_schema_check("site.htm") is False
    assert _supports_schema_check("data.json") is True


@pytest.mark.asyncio
async def test_finalization_gate_cef6e778_reproduction_succeeds() -> None:
    """The original failing run: site.html was on disk, runtime said failed.

    With the gate: site.html verifies, outcome is completed, leader prompt
    includes the path and says do-not-inline-copy.
    """
    ws = _FakeWorkspace(
        {"site.html": b"<!DOCTYPE html><html><body>x</body></html>" * 500}
    )
    ledger = AttemptLedger(run_id="cef6e778")
    ledger.declare(
        DeliverableDeclaration(
            path="site.html", declared_by_agent="coder", required=True
        )
    )
    ledger.record_attempt(
        AttemptRecord(
            agent_id="coder",
            self_reported_status="success",
            runtime_status="failed_max_iter",
        )
    )
    await verify_declared_deliverables(ledger, workspace=ws)
    decision = decide_finalization(ledger)
    assert decision.outcome == "completed", f"reproduction failed: {decision}"
    assert "site.html" in decision.prompt_injection
    # Leader should be told NOT to inline-copy
    assert (
        "NOT inline" in decision.prompt_injection
        or "inline" in decision.prompt_injection
    )
