"""ShareGPT format exporter.

Output: ``{"conversations": [{"from": "human"|"gpt"|"system", "value": "..."}]}``

Used by Unsloth, Axolotl, and LLaMA-Factory.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register

_ROLE_MAP = {"user": "human", "assistant": "gpt"}


@register("sharegpt")
class ShareGPTExporter(BaseExporter):
    format_name = "sharegpt"
    description = "ShareGPT conversation format"
    supports_multi_turn = True
    supports_system_prompt = True
    supports_tool_calls = False
    used_by = "Unsloth, Axolotl, LLaMA-Factory"

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = conversation.get("conversations", [])
        if not entries:
            return []

        turns: list[dict[str, str]] = []

        if system_prompt:
            turns.append({"from": "system", "value": system_prompt})

        # Merge consecutive same-role messages
        prev_role = None
        for entry in entries:
            role = entry.get("role", "user")
            mapped = _ROLE_MAP.get(role, role)
            content = entry.get("content", "")

            if mapped == prev_role and turns:
                turns[-1]["value"] += "\n" + content
            else:
                turns.append({"from": mapped, "value": content})
                prev_role = mapped

        return [{"conversations": turns}]

    def validate_output(self, row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        convs = row.get("conversations", [])
        if not convs:
            warnings.append("ShareGPT: empty conversations list")
        for entry in convs:
            if "from" not in entry or "value" not in entry:
                warnings.append("ShareGPT: entry missing 'from' or 'value'")
                break
        return warnings
