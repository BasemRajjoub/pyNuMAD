"""Centralised logging setup for pyNuMAD.

This module is the single source of truth for how pyNuMAD writes log
records. Modules elsewhere in the package just do::

    from pynumad._logging import get_logger
    logger = get_logger(__name__)
    logger.debug("entered node-pulling", extra={"region_id": rid, "edgeEls": ee})

and the resulting records flow to whatever handlers are configured by the
caller. Two modes are supported:

1. **Default mode** — a NullHandler is attached so pyNuMAD never spams a
   user's stderr unless they explicitly enable logging. Users who want
   normal Python logging can do::

       import logging
       logging.basicConfig(level=logging.INFO)

2. **Debug-sidecar mode** — set ``PYNUMAD_DEBUG_LOG=/path/to/log.jsonl``
   in the environment before importing pyNuMAD (or call
   ``enable_jsonl_sidecar(path)`` explicitly). Every log record at DEBUG
   and above is mirrored to that file, one JSON object per line. The
   schema is::

       {
         "ts": "2026-05-21T14:32:11.123456",
         "level": "DEBUG",
         "logger": "pynumad.mesh_gen.shell_region",
         "msg": "node-pulling fired",
         "module": "shell_region",
         "func": "createShellMesh",
         "line": 127,
         "context": { ... whatever was passed via extra ... }
       }

   Designed for machine consumption: ``jq '.context.region_id'`` works.

Why a custom helper rather than ``logging.basicConfig``? We want the JSONL
sidecar to be additive (it doesn't replace the user's handlers), to be
toggleable by env var (so test runs and ad-hoc scripts can opt in
trivially), and to capture structured ``extra`` fields under a stable
``context`` key (Python's default ``LogRecord`` flattens ``extra`` and
collides with reserved attributes).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging import Logger
from typing import Any

# Reserved attribute names on logging.LogRecord — anything in `extra` whose
# key matches one of these would raise KeyError. We collect everything else
# into a `context` dict.
_LOGRECORD_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }
)

_ROOT_LOGGER_NAME = "pynumad"
_jsonl_handler: logging.Handler | None = None


class JsonlFormatter(logging.Formatter):
    """Format LogRecords as one JSON object per line.

    Captures the standard fields plus any non-reserved keys present on the
    record (those came in via ``extra=``) under ``context``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        # Pick up extras (any key not on a vanilla LogRecord)
        context = {
            k: _to_jsonable(v)
            for k, v in record.__dict__.items()
            if k not in _LOGRECORD_RESERVED and not k.startswith("_")
        }
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _to_jsonable(value: Any) -> Any:
    """Best-effort JSON-friendly conversion of common scientific types."""
    # numpy arrays
    try:
        import numpy as np  # local import; numpy is a hard dep anyway

        if isinstance(value, np.ndarray):
            if value.size > 64:
                # Avoid dumping huge arrays into log records; summarise
                return {
                    "_kind": "ndarray",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(np.nanmin(value)) if value.size else None,
                    "max": float(np.nanmax(value)) if value.size else None,
                }
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except Exception:  # pragma: no cover  — numpy import failures are exotic
        pass
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def get_logger(name: str | None = None) -> Logger:
    """Return a child logger under the ``pynumad`` namespace.

    ``name`` is usually ``__name__`` from the calling module. If it already
    starts with ``pynumad`` it's used as-is; otherwise it's nested under
    ``pynumad.<name>``.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def enable_jsonl_sidecar(
    path: str | os.PathLike,
    level: int = logging.DEBUG,
    truncate: bool = False,
) -> logging.Handler:
    """Attach a JSONL handler that writes every record at ``level`` and
    above to ``path``. Returns the handler so callers can detach it later.

    Idempotent in spirit: calling twice replaces the previous handler.
    """
    global _jsonl_handler
    if _jsonl_handler is not None:
        logging.getLogger(_ROOT_LOGGER_NAME).removeHandler(_jsonl_handler)
        try:
            _jsonl_handler.close()
        except Exception:  # pragma: no cover
            pass
        _jsonl_handler = None

    mode = "w" if truncate else "a"
    handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonlFormatter())

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(min(root.level if root.level else logging.WARNING, level))
    root.addHandler(handler)
    _jsonl_handler = handler
    return handler


def disable_jsonl_sidecar() -> None:
    """Detach and close the JSONL handler if one is active."""
    global _jsonl_handler
    if _jsonl_handler is not None:
        logging.getLogger(_ROOT_LOGGER_NAME).removeHandler(_jsonl_handler)
        try:
            _jsonl_handler.close()
        except Exception:  # pragma: no cover
            pass
        _jsonl_handler = None


def _bootstrap() -> None:
    """Module import-time setup: attach a NullHandler so library use is
    silent by default; auto-enable the JSONL sidecar if ``PYNUMAD_DEBUG_LOG``
    is set in the environment.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root.handlers:
        root.addHandler(logging.NullHandler())
    env_path = os.environ.get("PYNUMAD_DEBUG_LOG")
    if env_path:
        try:
            level_name = os.environ.get("PYNUMAD_DEBUG_LEVEL", "DEBUG").upper()
            level = getattr(logging, level_name, logging.DEBUG)
            enable_jsonl_sidecar(env_path, level=level, truncate=False)
        except Exception as exc:  # pragma: no cover
            # Never let logging setup crash the import chain — surface to stderr
            sys.stderr.write(
                f"pyNuMAD: could not enable JSONL sidecar at {env_path!r}: {exc}\n"
            )


_bootstrap()
