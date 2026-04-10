"""LLaMA-Factory format exporter.

Output::

    {
      "system": "...",
      "instruction": "last user message",
      "input": "context",
      "output": "last assistant response",
      "history": [["user1", "asst1"], ["user2", "asst2"]]
    }

Used by LLaMA-Factory.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register


@register("llama_factory")
class LLaMAFactoryExporter(BaseExporter):
    format_name = "llama_factory"
    description = "LLaMA-Factory instruction + history format"
    supports_multi_turn = True
    supports_system_prompt = True
    supports_tool_calls = False
    used_by = "LLaMA-Factory"

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = conversation.get("conversations", [])
        if not entries:
            return []

        # Collect (user, assistant) pairs
        pairs: list[tuple[str, str]] = []
        i = 0
        while i < len(entries) - 1:
            user_entry = entries[i]
            asst_entry = entries[i + 1]
            if user_entry.get("role") == "user" and asst_entry.get("role") == "assistant":
                pairs.append((
                    user_entry.get("content", ""),
                    asst_entry.get("content", ""),
                ))
                i += 2
            else:
                i += 1

        if not pairs:
            return []

        context = conversation.get("instruction_context") or ""

        # Last pair is instruction/output; rest go to history
        history = [list(p) for p in pairs[:-1]]
        instruction, output = pairs[-1]

        row: dict[str, Any] = {
            "instruction": instruction,
            "input": context,
            "output": output,
            "history": history,
        }
        if system_prompt:
            row["system"] = system_prompt

        return [row]

    def validate_output(self, row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        for key in ("instruction", "output"):
            if key not in row:
                warnings.append(f"LLaMA-Factory: missing '{key}'")
        if not isinstance(row.get("history", []), list):
            warnings.append("LLaMA-Factory: 'history' must be a list")
        return warnings
