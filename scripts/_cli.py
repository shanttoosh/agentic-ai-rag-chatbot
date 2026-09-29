"""Shared plumbing for the command-line scripts."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from app.core.config import Settings, load_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging

logger = logging.getLogger("scripts")


def run(main: Callable[[Settings], int]) -> None:
    """Load settings, set up readable logging, and turn AppErrors into exit code 1."""
    try:
        settings = load_settings()
        configure_logging(settings.log_level, "text")
        code = main(settings)
    except AppError as exc:
        logger.debug("script failed", exc_info=True)
        print(f"error: {exc.message}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
