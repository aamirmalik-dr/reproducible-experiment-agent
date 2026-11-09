"""JSONL event traces with sanitization applied before anything is written."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .adapters import ENV_API_KEY

_PATTERNS = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "<redacted-key>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"\b[A-Za-z]:[\\/]+[^\s\"'<>]+"), "<path>"),
    (re.compile(r"(?<![\w.])/(?:home|Users|tmp|var|root|mnt|opt)/[^\s\"'<>]+"), "<path>"),
]
_SECRET_KEYS = {"api_key", "authorization", "token", "password", "secret"}


def sanitize(obj: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Recursively redact key-like strings, absolute paths and known secrets.

    Args:
        obj: Any JSON-compatible value.
        secrets: Exact strings to remove wherever they appear.

    Returns:
        A sanitized copy of ``obj``.
    """
    if isinstance(obj, str):
        for secret in secrets:
            if secret:
                obj = obj.replace(secret, "<redacted-secret>")
        for pattern, repl in _PATTERNS:
            obj = pattern.sub(repl, obj)
        return obj
    if isinstance(obj, dict):
        return {
            k: "<redacted>" if k.lower() in _SECRET_KEYS else sanitize(v, secrets)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, secrets) for v in obj]
    return obj


class TraceWriter:
    """Appends one sanitized JSON object per line.

    Args:
        path: Output file. Parent directories are created.
        record_timing: If False, latency fields are written as null so that
            fixture traces are byte-for-byte reproducible.
    """

    def __init__(self, path: Path, record_timing: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_timing = record_timing
        self._secrets = tuple(s for s in (os.environ.get(ENV_API_KEY, ""),) if len(s) >= 4)

    def write(self, event: dict[str, Any]) -> None:
        """Sanitize and append one event."""
        if not self.record_timing and "latency_s" in event:
            event = {**event, "latency_s": None}
        line = json.dumps(sanitize(event, self._secrets), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load every event from a JSONL trace."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
