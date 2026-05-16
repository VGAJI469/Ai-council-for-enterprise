"""Centralized structured logging utility.

Provides JSON-structured loggers, rotating handlers, and helper getters.
"""
import logging
import os
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
REPORT_DIR = os.path.join(os.getcwd(), "reports", "session_reports")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        # Attach exception info if available
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def setup_logging(level: int = logging.INFO):
    """Configure root logger with JSON formatter and rotating file handlers."""
    _ensure_dirs()

    root = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.setLevel(level)
        return

    root.setLevel(level)

    fmt = JSONFormatter()

    app_handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(fmt)

    error_handler = RotatingFileHandler(os.path.join(LOG_DIR, "error.log"), maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)

    audit_handler = TimedRotatingFileHandler(os.path.join(LOG_DIR, "audit.log"), when="midnight", backupCount=14, encoding="utf-8")
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(fmt)

    # Console handler for development convenience
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    root.addHandler(app_handler)
    root.addHandler(error_handler)
    root.addHandler(audit_handler)
    root.addHandler(console)


def get_logger(name: str = None) -> logging.Logger:
    """Return a structured logger configured by setup_logging()."""
    logger = logging.getLogger(name)
    return logger


# Convenience: auto-configure on import when running under app
try:
    setup_logging()
except Exception:
    # Avoid breaking imports during tests or scripts that reconfigure logging
    pass
