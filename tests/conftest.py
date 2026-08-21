"""pytest configuration for the protocore v2 test suite."""
from __future__ import annotations

import pytest

pytest_plugins: list[str] = []


@pytest.fixture
def tenant_id() -> str:
    return "test-tenant"
