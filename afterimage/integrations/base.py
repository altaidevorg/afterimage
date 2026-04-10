"""Base exporter class and ExportResult."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExportResult:
    """Summary of an export operation."""

    format_name: str
    input_path: str
    output_path: str
    total_input: int = 0
    total_output: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)


class BaseExporter(ABC):
    """Abstract base for all format exporters.

    Subclasses implement :meth:`convert_conversation` for one row at a time.
    The base :meth:`export_file` streams through the JSONL line-by-line so
    memory stays constant regardless of dataset size.
    """

    format_name: str = ""
    description: str = ""
    supports_multi_turn: bool = False
    supports_system_prompt: bool = False
    supports_tool_calls: bool = False
    used_by: str = ""

    @abstractmethod
    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert one AfterImage conversation row to target format rows.

        Returns a list because one conversation may produce multiple rows
        (e.g. Alpaca splits multi-turn into separate rows).
        An empty list means the row was skipped.
        """
        ...

    def validate_output(self, row: dict[str, Any]) -> list[str]:
        """Validate a converted row. Return list of warning strings (empty = OK)."""
        return []

    def export_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        system_prompt: str | None = None,
    ) -> ExportResult:
        """Stream-convert an entire JSONL file line by line.

        Never loads the full dataset into memory.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = ExportResult(
            format_name=self.format_name,
            input_path=str(input_path),
            output_path=str(output_path),
        )

        with (
            open(input_path, "r", encoding="utf-8") as fin,
            open(output_path, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                result.total_input += 1

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    result.skipped += 1
                    result.warnings.append(f"Line {result.total_input}: invalid JSON")
                    continue

                try:
                    converted = self.convert_conversation(
                        row, system_prompt=system_prompt
                    )
                except Exception as exc:
                    result.skipped += 1
                    result.warnings.append(f"Line {result.total_input}: {exc}")
                    continue

                if not converted:
                    result.skipped += 1
                    continue

                for out_row in converted:
                    warnings = self.validate_output(out_row)
                    result.warnings.extend(warnings)
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    result.total_output += 1

        return result
