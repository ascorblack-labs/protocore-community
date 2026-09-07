"""pytest configuration for the protocore v2 test suite."""
from __future__ import annotations

import pytest

pytest_plugins: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    """Register markers the suite uses so a bare run stays warning-free."""
    config.addinivalue_line(
        "markers",
        "perf: asserts a wall-clock budget; sensitive to a loaded machine",
    )


@pytest.fixture
def tenant_id() -> str:
    return "test-tenant"
