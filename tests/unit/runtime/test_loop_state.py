"""Tests for :mod:`protocore.runtime.loop_state`."""
from __future__ import annotations

import pytest

from protocore.runtime.loop_state import (
    TERMINAL_STATES,
    InvalidStateTransitionError,
    LoopState,
    assert_transition,
    is_terminal,
)


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (LoopState.PENDING, LoopState.RUNNING),
        (LoopState.PENDING, LoopState.CANCELLED),
        (LoopState.RUNNING, LoopState.AWAITING),
        (LoopState.RUNNING, LoopState.COMPACTING),
        (LoopState.RUNNING, LoopState.COMPLETED),
        (LoopState.RUNNING, LoopState.FAILED),
        (LoopState.RUNNING, LoopState.CANCELLED),
        (LoopState.AWAITING, LoopState.RUNNING),
        (LoopState.AWAITING, LoopState.CANCELLED),
        (LoopState.COMPACTING, LoopState.RUNNING),
        (LoopState.COMPACTING, LoopState.FAILED),
    ],
)
def test_legal_transitions(from_state: LoopState, to_state: LoopState) -> None:
    assert_transition(from_state, to_state)  # must not raise


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (LoopState.PENDING, LoopState.COMPACTING),
        (LoopState.PENDING, LoopState.AWAITING),
        (LoopState.COMPLETED, LoopState.RUNNING),
        (LoopState.FAILED, LoopState.COMPLETED),
        (LoopState.CANCELLED, LoopState.RUNNING),
        (LoopState.AWAITING, LoopState.COMPACTING),
    ],
)
def test_illegal_transitions_raise(from_state: LoopState, to_state: LoopState) -> None:
    with pytest.raises(InvalidStateTransitionError):
        assert_transition(from_state, to_state)


def test_terminal_set() -> None:
    assert TERMINAL_STATES == {
        LoopState.COMPLETED,
        LoopState.FAILED,
        LoopState.CANCELLED,
    }


def test_is_terminal_predicate() -> None:
    assert is_terminal(LoopState.COMPLETED)
    assert is_terminal(LoopState.FAILED)
    assert is_terminal(LoopState.CANCELLED)
    assert not is_terminal(LoopState.PENDING)
    assert not is_terminal(LoopState.RUNNING)
    assert not is_terminal(LoopState.AWAITING)
    assert not is_terminal(LoopState.COMPACTING)


def test_invalid_transition_error_carries_states() -> None:
    try:
        assert_transition(LoopState.COMPLETED, LoopState.RUNNING)
    except InvalidStateTransitionError as exc:
        assert exc.from_state is LoopState.COMPLETED
        assert exc.to_state is LoopState.RUNNING
    else:  # pragma: no cover
        pytest.fail("expected InvalidStateTransitionError")
