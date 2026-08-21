from protocore.contracts.types import RunStatus


def test_run_status_includes_incomplete_terminal_projection() -> None:
    assert RunStatus("incomplete") is RunStatus.incomplete
