"""Default :class:`RuntimeConstantsProvider` for tests + smoke runtime.

The host pods supply a
production provider (PG-backed); this is an in-process static-default
provider used by core fixtures and the in-memory runtime helper.
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import RuntimeConstants


def default_runtime_constants(**overrides: object) -> RuntimeConstants:
    """Build a default :class:`RuntimeConstants` with optional overrides.

    Used by :func:`protocore.testing.build_in_memory_runtime` and tests.
    """
    return RuntimeConstants(**overrides)  # type: ignore[arg-type]


class StaticRuntimeConstantsProvider:
    """In-memory :class:`RuntimeConstantsProvider` returning a fixed snapshot.

    Used by tests and the smoke-runtime helper. Production pods substitute
    a PG-backed provider.
    """

    def __init__(self, snapshot: RuntimeConstants | None = None) -> None:
        self._snapshot = snapshot or default_runtime_constants()

    async def get(self, tenant_id: str) -> RuntimeConstants:
        return self._snapshot


__all__ = ["StaticRuntimeConstantsProvider", "default_runtime_constants"]
