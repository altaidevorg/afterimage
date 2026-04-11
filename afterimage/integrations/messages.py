"""HuggingFace Messages / ChatML format exporter.

Output: ``{"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}``

Used by TRL SFTTrainer and HuggingFace ecosystem.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register


@register("messages")
class MessagesExporter(BaseExporter):
    format_name = "messages"
    description = "HuggingFace Messages / ChatML format"
    supports_multi_turn = True
    supports_system_prompt = True
    supports_tool_calls = True
    used_by = "TRL SFTTrainer, HuggingFace"

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = conversation.get("conversations", [])
        if not entries:
            return []

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        has_assistant = False
        for entry in entries:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            messages.append({"role": role, "content": content})
            if role == "assistant":
                has_assistant = True

        if not has_assistant:
            return []

        return [{"messages": messages}]

    def validate_output(self, row: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        msgs = row.get("messages", [])
        if not msgs:
            warnings.append("Messages: empty messages list")
        roles = {m.get("role") for m in msgs}
        if "assistant" not in roles:
            warnings.append("Messages: no assistant message")
        return warnings
