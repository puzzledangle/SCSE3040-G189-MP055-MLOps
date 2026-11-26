"""Structured JSON logging configuration for MP055 FastAPI service.

Emits single-line JSON log events to stdout per prediction.
Does not log personal identifiable information (PII).
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats log records as compact single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "prediction_data"):
            log_obj["data"] = record.prediction_data
        return json.dumps(log_obj, ensure_ascii=False)


def setup_json_logger(name: str = "mp055.api") -> logging.Logger:
    """Initialize a logger with a JSON stdout handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
