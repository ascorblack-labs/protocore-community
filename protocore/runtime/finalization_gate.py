# ruff: noqa: RUF001 — Cyrillic prompt strings are intentional (bilingual RU+EN finalization template)
"""Finalization gate: verify declared deliverables and decide run outcome.

This module closes a finalization gap: a subagent that successfully writes the
user-visible artifact but then runs out of iterations without calling its
terminal tool would otherwise produce a "failed" run, and the leader would
apologize to the user even though the artifact is on disk.

The gate has two responsibilities:

  1. **Verify**: walk the ledger's declared deliverables, stat each one in the
     workspace, and record a VerificationRecord (existence, size, optional
     content hash, optional schema validity).
  2. **Decide**: compute a FinalizationDecision that the leader's final-turn
     LLM call uses: success template (reference the artifact, do NOT inline
     copy it), partial template, or failed template (one-sentence blocker,
     no inline workaround).

The module does **not** import from the orchestrator or from any specific
workspace implementation. Callers inject a WorkspaceFacade implementation; in
production that wraps the same workspace tools the subagent uses.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ..contracts.attempt_ledger import (
    AttemptLedger,
    DeliverableDeclaration,
    LedgerOutcome,
    VerificationRecord,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkspaceStatResult:
    """Minimal stat shape needed by the gate. Maps to WorkspaceStatResult.stat."""

    exists: bool
    size_bytes: int | None = None
    is_file: bool = False
    is_directory: bool = False


class WorkspaceStatProtocol(Protocol):
    """Stat-only workspace facade for use by the finalization gate.

    The concrete implementation lives in the orchestrator and delegates to the
    same workspace runtime the subagent used. A protocol keeps protocore-core
    free of any service-runtime imports.
    """

    async def stat(self, path: str) -> WorkspaceStatResult: ...

    async def read_bytes(self, path: str, *, max_bytes: int) -> bytes | None:
        """Optional: enables content-hash and schema verification. Return
        ``None`` if reading is not supported or path is too large.
        """
        ...


@dataclass(frozen=True, slots=True)
class FinalizationDecision:
    """How the leader's final turn should frame its answer."""

    outcome: LedgerOutcome
    artifacts_present: tuple[str, ...]
    artifacts_missing: tuple[str, ...]
    prompt_injection: str
    """One-paragraph instruction appended to the leader's finalization prompt.

    Tells the leader to reference workspace artifacts (success), explain what
    exists vs. what doesn't (partial), or name the single blocker without
    inline workaround content (failed).
    """


_FINALIZATION_PROMPT_SUCCESS_HEADER = (
    "Финализация: задача завершена успешно. В workspace уже есть готовые "
    "артефакты:\n"
    "{artifacts_present}\n"
    "Перечисли их пользователю коротко: имя, размер, что внутри по существу. "
    "НЕ копируй содержимое файлов в ответ; пользователь видит их через "
    "workspace-браузер. Не извиняйся — артефакт существует.\n"
    "Finalization: task completed. The workspace already contains these "
    "artifacts:\n"
    "{artifacts_present}\n"
    "List them for the user briefly: name, size, what is inside. Do NOT "
    "inline the file contents in the reply; the user can open them through "
    "the workspace browser. Do not apologize — the artifact exists."
)


_FINALIZATION_PROMPT_PARTIAL_HEADER = (
    "Финализация: задача завершена частично.\n"
    "Создано: {artifacts_present}\n"
    "Не создано: {artifacts_missing}\n"
    "Перечисли что уже создано (с именами файлов), что не получилось, и "
    "короткую рекомендацию по продолжению. НЕ давай inline workaround content "
    "для недостающих артефактов.\n"
    "Finalization: task is partial.\n"
    "Present: {artifacts_present}\n"
    "Missing: {artifacts_missing}\n"
    "List what exists (by file name), what is missing, and a short "
    "recommendation. Do NOT provide inline workaround content for the "
    "missing artifacts."
)


_FINALIZATION_PROMPT_FAILED_HEADER = (
    "Финализация: задача не выполнена.\n"
    "Ожидались, но отсутствуют: {artifacts_missing}\n"
    "Назови конкретный блокер в одно предложение и сошлись на конкретный "
    "недостающий путь. НЕ предлагай пользователю inline workaround для "
    "артефакта, который должен был быть в workspace.\n"
    "Finalization: task failed.\n"
    "Expected but missing: {artifacts_missing}\n"
    "Name the specific blocker in one sentence and reference the concrete "
    "missing path. Do NOT offer the user inline workaround content for an "
    "artifact that was supposed to live in the workspace."
)


_FINALIZATION_PROMPT_UNKNOWN = (
    "Финализация: артефакты не объявлены (analytic task). Дай короткий "
    "ответ по существу задачи.\n"
    "Finalization: no artifacts were declared (analytic task). Give a short "
    "substantive answer to the user's question."
)


# How many bytes the gate will read for content-hash / schema validation
# before giving up. Schema checks should never block a deploy on a giant
# artifact; if it's too large to hash, we accept the workspace_stat
# verdict and skip content checks.
DEFAULT_FINALIZATION_CONTENT_VERIFY_MAX_BYTES = 2_000_000


async def verify_declared_deliverables(
    ledger: AttemptLedger,
    *,
    workspace: WorkspaceStatProtocol,
    verifier_id: str = "finalization_gate",
    content_verify_max_bytes: int = DEFAULT_FINALIZATION_CONTENT_VERIFY_MAX_BYTES,
) -> list[VerificationRecord]:
    """Verify each declared deliverable in the ledger.

    Appends one VerificationRecord per declaration to the ledger (replacing
    older verifications is intentionally NOT done — the ledger keeps history,
    and ``latest_verification_for`` finds the most recent one).

    Args:
        ledger: The run's attempt ledger. Mutated in place.
        workspace: A facade that can stat (and optionally read) paths.
        verifier_id: Identifier tagged onto each VerificationRecord.
        content_verify_max_bytes: Skip content-hash + schema checks for files
            larger than this. Stat-only verification still runs.

    Returns:
        The list of VerificationRecords appended (in declaration order).
    """

    new_records: list[VerificationRecord] = []
    for declaration in ledger.declared_deliverables.values():
        record = await _verify_one(
            declaration,
            workspace=workspace,
            verifier_id=verifier_id,
            content_verify_max_bytes=content_verify_max_bytes,
        )
        ledger.record_verification(record)
        new_records.append(record)
    return new_records


async def _verify_one(
    declaration: DeliverableDeclaration,
    *,
    workspace: WorkspaceStatProtocol,
    verifier_id: str,
    content_verify_max_bytes: int,
) -> VerificationRecord:
    when = datetime.now(UTC)
    try:
        stat = await workspace.stat(declaration.path)
    except FileNotFoundError:
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=False,
            verifier_id=verifier_id,
        )
    except Exception as exc:  # pragma: no cover - defensive; stat errors logged
        logger.warning(
            "finalization_gate.stat_failed",
            extra={"path": declaration.path, "error": str(exc)},
        )
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=False,
            verifier_id=verifier_id,
            error=str(exc)[:500],
        )

    if not stat.exists:
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=False,
            size_bytes=stat.size_bytes,
            verifier_id=verifier_id,
        )

    # Stat says it exists. Optionally also check kind, size floor, and content.
    if declaration.kind == "file" and not stat.is_file:
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=False,
            size_bytes=stat.size_bytes,
            verifier_id=verifier_id,
            error="declared kind=file but workspace reports not a file",
        )
    if declaration.kind == "directory" and not stat.is_directory:
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=False,
            size_bytes=stat.size_bytes,
            verifier_id=verifier_id,
            error="declared kind=directory but workspace reports not a directory",
        )

    # an UNKNOWN size (``stat.size_bytes is None``) is not a zero
    # size. Only evaluate the min_size floor when the workspace actually
    # reported a concrete size; otherwise ``size_bytes or 0`` would coerce an
    # unknown size to 0 and falsely fail any positive ``min_size_bytes`` for a
    # genuinely-existing file. ``size_bytes`` below is still ``or 0`` for the
    # downstream content-hash gate (a None there only widens the eligibility
    # cap), but the size FLOOR check keys on the raw ``stat.size_bytes``.
    if (
        stat.size_bytes is not None
        and declaration.min_size_bytes is not None
        and stat.size_bytes < declaration.min_size_bytes
    ):
        return VerificationRecord(
            when=when,
            deliverable_path=declaration.path,
            exists=True,
            size_bytes=stat.size_bytes,
            verifier_id=verifier_id,
            valid_by_schema=False,
            schema_kind="min_size",
            error=(
                f"size {stat.size_bytes} below declared min_size_bytes "
                f"{declaration.min_size_bytes}"
            ),
        )

    size_bytes = stat.size_bytes or 0

    content_hash: str | None = None
    valid_by_schema: bool | None = None
    schema_kind: str | None = None

    if (
        declaration.kind == "file"
        and (declaration.sha256_expected is not None or _supports_schema_check(declaration.path))
        and size_bytes <= content_verify_max_bytes
    ):
        try:
            payload = await workspace.read_bytes(
                declaration.path,
                max_bytes=content_verify_max_bytes,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "finalization_gate.read_failed",
                extra={"path": declaration.path, "error": str(exc)},
            )
            payload = None
        if payload is not None:
            content_hash = hashlib.sha256(payload).hexdigest()
            if (
                declaration.sha256_expected is not None
                and content_hash != declaration.sha256_expected
            ):
                return VerificationRecord(
                    when=when,
                    deliverable_path=declaration.path,
                    exists=True,
                    size_bytes=size_bytes,
                    content_hash=content_hash,
                    valid_by_schema=False,
                    schema_kind="sha256",
                    verifier_id=verifier_id,
                    error="sha256 mismatch with declared expected value",
                )
            schema_kind, valid_by_schema = _schema_check_payload(declaration.path, payload)

    return VerificationRecord(
        when=when,
        deliverable_path=declaration.path,
        exists=True,
        size_bytes=size_bytes,
        content_hash=content_hash,
        valid_by_schema=valid_by_schema,
        schema_kind=schema_kind,
        verifier_id=verifier_id,
    )


def _supports_schema_check(path: str) -> bool:
    """Only formats where parse-failure is unambiguous get schema checks.

    HTML is intentionally excluded. A 4 KB HTML fragment
    (e.g. ``<div>hello</div>``) is legitimate output for embed / snippet
    deliverables and would be flagged invalid by a strict ``<html`` tag
    check. Conversely a JSON file that accidentally contains ``<html`` as
    a substring would pass. Existence + size floor is sufficient for HTML;
    JSON parsing is unambiguous so we keep it.
    """
    lower = path.lower()
    return lower.endswith((".json", ".jsonl"))


def _schema_check_payload(path: str, payload: bytes) -> tuple[str | None, bool | None]:
    """Best-effort content-shape validation; returns (kind, ok).

    Always returns ``None`` rather than False for kinds we cannot evaluate, so
    the ledger's compute_outcome does not penalize unknown formats.
    """
    lower = path.lower()
    if lower.endswith((".json", ".jsonl")):
        import json

        try:
            if lower.endswith(".jsonl"):
                for line in payload.splitlines():
                    if line.strip():
                        json.loads(line)
            else:
                json.loads(payload.decode("utf-8", errors="replace"))
            return "json", True
        except Exception:
            return "json", False
    return None, None


def decide_finalization(ledger: AttemptLedger) -> FinalizationDecision:
    """Compute a FinalizationDecision the leader uses to frame its final turn.

    The prompt_injection text **interpolates the concrete
    artifact paths** (present + missing). A leader prompt with only "list the
    artifacts" — but no path list — would push the LLM to fabricate or revert
    to the apology + inline-HTML anti-pattern. Concrete paths in the prompt
    are load-bearing for the fix.
    """

    outcome = ledger.compute_outcome()
    present: list[str] = []
    missing: list[str] = []
    for declaration in ledger.declared_deliverables.values():
        latest = ledger.latest_verification_for(declaration.path)
        if latest is not None and latest.exists and latest.valid_by_schema is not False:
            present.append(declaration.path)
        elif declaration.required:
            missing.append(declaration.path)

    present_text = _format_paths_for_prompt(present)
    missing_text = _format_paths_for_prompt(missing)
    if outcome == "completed":
        prompt = _FINALIZATION_PROMPT_SUCCESS_HEADER.format(
            artifacts_present=present_text,
        )
    elif outcome == "partial":
        prompt = _FINALIZATION_PROMPT_PARTIAL_HEADER.format(
            artifacts_present=present_text,
            artifacts_missing=missing_text,
        )
    elif outcome == "failed":
        prompt = _FINALIZATION_PROMPT_FAILED_HEADER.format(
            artifacts_missing=missing_text,
        )
    else:
        prompt = _FINALIZATION_PROMPT_UNKNOWN

    return FinalizationDecision(
        outcome=outcome,
        artifacts_present=tuple(present),
        artifacts_missing=tuple(missing),
        prompt_injection=prompt,
    )


def _format_paths_for_prompt(paths: list[str]) -> str:
    if not paths:
        return "  (none)"
    return "\n".join(f"  - {p}" for p in paths)


__all__ = [
    "FinalizationDecision",
    "WorkspaceStatProtocol",
    "WorkspaceStatResult",
    "decide_finalization",
    "verify_declared_deliverables",
]
