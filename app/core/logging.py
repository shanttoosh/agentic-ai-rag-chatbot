"""Structured logging setup.

Log calls attach context through ``extra``::

    logger.info("chat.answered", extra={"status": "answered", "latency_ms": 812})

The JSON formatter emits every such field; the text formatter appends them as key=value.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has; anything else on the record came from `extra`.
_RESERVED = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {key: value for key, value in vars(record).items() if key not in _RESERVED}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **_extra_fields(record),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = " ".join(f"{key}={value}" for key, value in _extra_fields(record).items())
        return f"{base} {fields}" if fields else base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # provider SDK and HTTP client loggers: warnings and above only
    for noisy in ("httpx", "httpx2", "anthropic", "pinecone", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
