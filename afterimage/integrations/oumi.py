"""Oumi format exporter.

Output: ``{"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}``

Structurally identical to HuggingFace Messages format. Used by ``oumi train``.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register


@register("oumi")
class OumiExporter(BaseExporter):
    format_name = "oumi"
    description = "Oumi conversation format"
    supports_multi_turn = True
    supports_system_prompt = True
    supports_tool_calls = True
    used_by = "Oumi"

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

        for entry in entries:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            messages.append({"role": role, "content": content})

        return [{"messages": messages}]
