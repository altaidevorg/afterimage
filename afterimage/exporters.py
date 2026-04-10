"""Convert AfterImage JSONL datasets to popular training formats.

Supported formats:
- **sharegpt**: ``{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}``
- **alpaca**: ``{"instruction": ..., "input": "", "output": ...}`` (first turn only)
- **messages**: ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ROLE_MAP_SHAREGPT = {"user": "human", "assistant": "gpt"}
_ROLE_MAP_MESSAGES = {"user": "user", "assistant": "assistant"}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning one dict per line."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write rows to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_sharegpt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to ShareGPT format."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        turns = []
        for entry in convs:
            role = entry.get("role", "user")
            turns.append({
                "from": _ROLE_MAP_SHAREGPT.get(role, role),
                "value": entry.get("content", ""),
            })
        out.append({"conversations": turns})
    return out


def to_alpaca(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to Alpaca format (first turn only)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        user_msg = ""
        assistant_msg = ""
        for entry in convs:
            role = entry.get("role", "")
            if role == "user" and not user_msg:
                user_msg = entry.get("content", "")
            elif role == "assistant" and not assistant_msg:
                assistant_msg = entry.get("content", "")
            if user_msg and assistant_msg:
                break
        out.append({
            "instruction": user_msg,
            "input": "",
            "output": assistant_msg,
        })
    return out


def to_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to HuggingFace messages format."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            messages.append({
                "role": _ROLE_MAP_MESSAGES.get(role, role),
                "content": entry.get("content", ""),
            })
        out.append({"messages": messages})
    return out


CONVERTERS = {
    "sharegpt": to_sharegpt,
    "alpaca": to_alpaca,
    "messages": to_messages,
}


def export_dataset(
    input_path: str | Path,
    format_name: str,
    output_path: str | Path | None = None,
) -> Path:
    """Load an AfterImage JSONL dataset and write it in the requested format.

    Args:
        input_path: Path to the source JSONL file.
        format_name: One of ``"sharegpt"``, ``"alpaca"``, ``"messages"``.
        output_path: Destination path. If *None*, derived from *input_path*.

    Returns:
        The resolved output path.
    """
    converter = CONVERTERS.get(format_name)
    if converter is None:
        raise ValueError(f"Unknown format: {format_name!r}. Choose from {list(CONVERTERS)}")

    rows = _load_jsonl(input_path)
    converted = converter(rows)

    if output_path is None:
        inp = Path(input_path)
        output_path = inp.with_name(f"{inp.stem}_{format_name}.jsonl")

    _write_jsonl(converted, output_path)
    return Path(output_path)
