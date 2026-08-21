"""Logger factory — WARNING default per .

Returns a module-scoped logger configured to inherit root level. Tests
override via ``caplog`` fixture; production sets root level via env var
``LOG_LEVEL`` interpreted at the host pod entry point — core never
configures handlers, only obtains loggers.
"""
from __future__ import annotations

import logging
from typing import Final

_DEFAULT_LEVEL: Final[int] = logging.WARNING


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with WARNING default.

    Core modules MUST call this; never ``logging.getLogger`` directly.
    Default level is :data:`logging.WARNING`; a host overrides
    via standard logging configuration.
    """
    logger = logging.getLogger(name)
    if logger.level == logging.NOTSET:
        logger.setLevel(_DEFAULT_LEVEL)
    return logger


__all__ = ["get_logger"]
