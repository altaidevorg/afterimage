#!/usr/bin/env python3
"""Convert CL-bench/Ctx2Skill context JSONL into Afterimage document JSONL.

The original Ctx2Skill data stores context as chat `messages`. Afterimage's
document pipeline expects one text field per document. This converter preserves
`metadata.context_id` as the Afterimage document id so skill outputs can be
compared context-by-context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def format_messages(messages: list[dict[str, Any]]) -> str:
    parts = []
    for idx, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"## Message {idx}: {role}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def convert(input_path: Path, output_path: Path, max_rows: int | None = None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        progress = tqdm(total=max_rows, desc="Converting contexts", unit="doc")
        try:
            for line in src:
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("metadata") or {}
                context_id = metadata.get("context_id") or metadata.get("task_id")
                if not context_id:
                    context_id = f"row-{count + 1}"
                row = {
                    "id": context_id,
                    "text": format_messages(item.get("messages") or []),
                    "metadata": metadata,
                    "rubrics": item.get("rubrics") or [],
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                progress.update(1)
                if max_rows is not None and count >= max_rows:
                    break
        finally:
            progress.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Ctx2Skill CL-bench context JSONL to Afterimage docs JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    count = convert(args.input, args.output, args.max_rows)
    print(f"Wrote {count} document(s) to {args.output}")


if __name__ == "__main__":
    main()
