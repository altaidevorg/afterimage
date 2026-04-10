"""Alpaca format exporter.

Output: ``{"instruction": "...", "input": "", "output": "..."}``

Used by Stanford Alpaca and simple fine-tuning setups.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register


@register("alpaca")
class AlpacaExporter(BaseExporter):
    format_name = "alpaca"
    description = "Alpaca instruction format (single-turn)"
    supports_multi_turn = False
    supports_system_prompt = False
    supports_tool_calls = False
    used_by = "Stanford Alpaca, basic SFT"

    def __init__(self, *, split_turns: bool = False):
        self.split_turns = split_turns

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = conversation.get("conversations", [])
        if not entries:
            return []

        # Extract context from metadata if available
        context = conversation.get("instruction_context") or ""

        if self.split_turns:
            return self._split_multi_turn(entries, context)
        return self._first_turn_only(entries, context)

    def _first_turn_only(
        self, entries: list[dict], context: str
    ) -> list[dict[str, Any]]:
        user_msg = ""
        asst_msg = ""
        for entry in entries:
            role = entry.get("role", "")
            if role == "user" and not user_msg:
                user_msg = entry.get("content", "")
            elif role == "assistant" and not asst_msg:
                asst_msg = entry.get("content", "")
            if user_msg and asst_msg:
                break

        if not user_msg or not asst_msg:
            return []

        return [{"instruction": user_msg, "input": context, "output": asst_msg}]

    def _split_multi_turn(
        self, entries: list[dict], context: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        history: list[tuple[str, str]] = []

        i = 0
        while i < len(entries) - 1:
            user_entry = entries[i]
            asst_entry = entries[i + 1]

            if (
                user_entry.get("role") != "user"
                or asst_entry.get("role") != "assistant"
            ):
                i += 1
                continue

            user_msg = user_entry.get("content", "")
            asst_msg = asst_entry.get("content", "")

            input_parts: list[str] = []
            if history:
                hist_lines = []
                for hu, ha in history:
                    hist_lines.append(f"User: {hu}")
                    hist_lines.append(f"Assistant: {ha}")
                input_parts.append("Previous conversation:\n" + "\n".join(hist_lines))
            if context:
                input_parts.append(f"Context: {context}")

            rows.append(
                {
                    "instruction": user_msg,
                    "input": "\n\n".join(input_parts),
                    "output": asst_msg,
                }
            )

            history.append((user_msg, asst_msg))
            i += 2

        return rows

    def validate_output(self, row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        for key in ("instruction", "input", "output"):
            if key not in row:
                warnings.append(f"Alpaca: missing '{key}' field")
        if "input" in row and row["input"] is None:
            warnings.append("Alpaca: 'input' must be string, not null")
        return warnings
