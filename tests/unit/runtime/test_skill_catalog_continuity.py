"""A resumed run rebuilds the skill catalog, and says so when it differs.

The catalog block is the head of the cached prompt prefix, and it is not
carried in a snapshot: the process that picks a run up rebuilds it from the
store. If the enabled-skill set moved in between, the rebuilt block differs,
the cached prefix is invalid for every remaining turn, and the only visible
consequence is that the run costs more. The digest the previous process
recorded is the one thing that can tell, so it is compared once the new block
exists.
"""
from __future__ import annotations

import hashlib
import logging

import pytest

from protocore.runtime.query import _ensure_run_skill_catalog
from protocore.runtime.query_engine import QueryEngine


def _digest(block: str) -> str:
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


async def _resume_carrying(engine: QueryEngine, source: QueryEngine, block: str | None) -> None:
    payload = source.snapshot()
    payload["skill_catalog_block_sha256"] = None if block is None else _digest(block)
    await engine.resume_from_snapshot(payload)


async def test_a_rebuilt_catalog_that_differs_is_reported(
    engine_factory, caplog: pytest.LogCaptureFixture
) -> None:
    source = engine_factory()
    resumed = engine_factory()
    await _resume_carrying(resumed, source, "a catalog this process will not rebuild")

    with caplog.at_level(logging.WARNING):
        await _ensure_run_skill_catalog(resumed)

    assert "skill_catalog.rebuilt_differently" in caplog.text


async def test_a_rebuilt_catalog_that_matches_is_silent(
    engine_factory, caplog: pytest.LogCaptureFixture
) -> None:
    source = engine_factory()
    resumed = engine_factory()
    rebuilt = await _ensure_run_skill_catalog(engine_factory())
    await _resume_carrying(resumed, source, rebuilt)

    with caplog.at_level(logging.WARNING):
        await _ensure_run_skill_catalog(resumed)

    assert "skill_catalog.rebuilt_differently" not in caplog.text


async def test_a_run_that_had_built_no_catalog_is_not_compared_against_one(
    engine_factory, caplog: pytest.LogCaptureFixture
) -> None:
    """``None`` means the run had not built a block yet, which is distinct from
    having built an empty one — there is nothing to have drifted from."""
    source = engine_factory()
    resumed = engine_factory()
    await _resume_carrying(resumed, source, None)

    with caplog.at_level(logging.WARNING):
        await _ensure_run_skill_catalog(resumed)

    assert "skill_catalog.rebuilt_differently" not in caplog.text


async def test_the_comparison_happens_once_and_not_every_turn(
    engine_factory, caplog: pytest.LogCaptureFixture
) -> None:
    """The expectation belongs to the pickup, not to the run: a later turn that
    legitimately rebuilds a different block has no earlier process to differ
    from."""
    source = engine_factory()
    resumed = engine_factory()
    await _resume_carrying(resumed, source, "a catalog this process will not rebuild")

    with caplog.at_level(logging.WARNING):
        await _ensure_run_skill_catalog(resumed)
        caplog.clear()
        resumed._skill_catalog_block = None
        await _ensure_run_skill_catalog(resumed)

    assert "skill_catalog.rebuilt_differently" not in caplog.text


async def test_a_run_that_was_never_resumed_compares_against_nothing(
    engine_factory, caplog: pytest.LogCaptureFixture
) -> None:
    engine = engine_factory()

    with caplog.at_level(logging.WARNING):
        await _ensure_run_skill_catalog(engine)

    assert "skill_catalog.rebuilt_differently" not in caplog.text
