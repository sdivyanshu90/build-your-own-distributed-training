"""Rank-0-gated structured (JSON) logging.

What this module does
---------------------
Provides a small logger that (a) emits one structured JSON object per line so the
output is ingestible by Weights & Biases / TensorBoard importers / ``jq``, and
(b) gates info-level logs behind ``rank == 0`` so a 128-GPU job does not print
128 copies of every line. Warnings and errors are *not* gated — a problem on
rank 37 must be visible.

Why structured JSON over free-form prints
-----------------------------------------
At scale you grep logs programmatically. ``{"step": 42, "loss": 3.1, ...}`` is
trivially parsed and plotted; ``"step 42 loss 3.1 ..."`` requires brittle regex.
Each record carries the ``rank`` and a monotonic wall-clock so events across
ranks can be correlated when debugging desync.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any


class RankLogger:
    """A structured logger that gates info logs to rank 0.

    Args:
        rank: This process's global rank.
        run_id: Run identifier embedded in every record.
        stream: Output stream (defaults to stdout); injectable for tests.

    Example:
        >>> import io
        >>> buf = io.StringIO()
        >>> log = RankLogger(rank=0, run_id="demo", stream=buf)
        >>> log.info("step", step=1, loss=2.5)
        >>> "\\"event\\": \\"step\\"" in buf.getvalue()
        True
    """

    def __init__(self, rank: int, run_id: str, stream: Any = None) -> None:
        self.rank = rank
        self.run_id = run_id
        self.stream = stream if stream is not None else sys.stdout
        self._t0 = time.time()

    def _emit(self, level: str, event: str, gated: bool, **fields: Any) -> None:
        if gated and self.rank != 0:
            return
        record: dict[str, Any] = {
            "ts": round(time.time() - self._t0, 4),
            "level": level,
            "rank": self.rank,
            "run_id": self.run_id,
            "event": event,
        }
        for key, value in fields.items():
            record[key] = _jsonable(value)
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()

    def info(self, event: str, **fields: Any) -> None:
        """Log an info record (rank-0 only)."""
        self._emit("INFO", event, gated=True, **fields)

    def metric(self, event: str, **fields: Any) -> None:
        """Log a metrics record (rank-0 only); alias of :meth:`info` for clarity."""
        self._emit("METRIC", event, gated=True, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Log a warning (ALL ranks — not gated)."""
        self._emit("WARNING", event, gated=False, **fields)

    def error(self, event: str, **fields: Any) -> None:
        """Log an error (ALL ranks — not gated)."""
        self._emit("ERROR", event, gated=False, **fields)


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def build_logger(rank: int, run_id: str, stream: Any | None = None) -> RankLogger:
    """Factory for a :class:`RankLogger`."""
    return RankLogger(rank=rank, run_id=run_id, stream=stream)
