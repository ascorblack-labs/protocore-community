"""Features & RC initiative B1 (2026-06-06) — the 2 RC renames.

Old default ``True`` (reasoning forced off)
 ⇔ new default ``False`` (reasoning not enabled by default) — behaviour
 is bit-identical for every scope without an override; the host
 migration 160 inverts existing override VALUES in the same transaction.
* ``sandbox_admission_init_account_main_max`` (a bool that reads like an
 int cap) → ``sandbox_admission_init_account_main_max_enabled``.
 Mechanics + default (``True``) unchanged.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from protocore.contracts.runtime_constants import RuntimeConstants


def test_llm_reasoning_default_enabled_replaces_disabled() -> None:
    fields = RuntimeConstants.model_fields
    assert "llm_reasoning_default_enabled" in fields
    assert "llm_reasoning_default_disabled" not in fields, (
        "old negative-polarity name must be GONE (no alias layer — "
        "one field, one read)"
    )
    # Polarity equivalence: old default True (force-off) == new default
    # False (not enabled). A default-constructed snapshot behaves
    # bit-identically pre/post rename.
    assert RuntimeConstants().llm_reasoning_default_enabled is False


def test_llm_reasoning_old_name_rejected() -> None:
    """``extra='forbid'`` — a stale producer using the old name fails loudly."""

    with pytest.raises(ValidationError):
        RuntimeConstants.model_validate({"llm_reasoning_default_disabled": True})


def test_sandbox_admission_init_account_main_max_enabled_rename() -> None:
    fields = RuntimeConstants.model_fields
    assert "sandbox_admission_init_account_main_max_enabled" in fields
    assert "sandbox_admission_init_account_main_max" not in fields
    # Same mechanics, same default — only the name gains the bool suffix.
    assert RuntimeConstants().sandbox_admission_init_account_main_max_enabled is True


def test_sandbox_admission_old_name_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeConstants.model_validate(
            {"sandbox_admission_init_account_main_max": False}
        )
