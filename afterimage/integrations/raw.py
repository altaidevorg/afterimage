"""Raw passthrough exporter — copies input as-is.

Useful for backup, debugging, or piping to custom scripts.
"""

from __future__ import annotations

from typing import Any

from .base import BaseExporter
from .registry import register


@register("raw")
class RawExporter(BaseExporter):
    format_name = "raw"
    description = "AfterImage native format with all metadata preserved"
    supports_multi_turn = True
    supports_system_prompt = True
    supports_tool_calls = True
    used_by = "AfterImage, custom pipelines"

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        return [conversation]
