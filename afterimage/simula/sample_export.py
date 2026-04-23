"""Append :class:`~afterimage.simula.types.DataPointRecord` rows as JSONL (one object per line)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .types import DataPointRecord


def append_datapoints_jsonl(
    path: Path | str,
    records: Iterable[DataPointRecord],
    *,
    mkdir: bool = True,
) -> int:
    """Append each record as one JSON line. Creates parent directories when ``mkdir`` is true.

    Returns the number of lines written.
    """
    p = Path(path)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
            n += 1
    return n
