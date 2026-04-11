"""DPO preference pairs exporter.

Output: ``{"prompt": "...", "chosen": "...", "rejected": "..."}``

Used by TRL DPOTrainer and RLHF pipelines. Requires quality scores.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .base import BaseExporter, ExportResult
from .registry import register


@register("dpo")
class DPOExporter(BaseExporter):
    format_name = "dpo"
    description = "DPO preference pairs (requires quality scores)"
    supports_multi_turn = False
    supports_system_prompt = True
    supports_tool_calls = False
    used_by = "TRL DPOTrainer, RLHF"

    def __init__(self, *, min_score_gap: float = 0.2):
        self.min_score_gap = min_score_gap

    def convert_conversation(
        self,
        conversation: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        # Single-row conversion not meaningful for DPO.
        # DPO requires pairing across conversations.
        # Return the row with score info for batch processing.
        return []

    def export_file(
        self,
        input_path,
        output_path,
        *,
        system_prompt: str | None = None,
    ) -> ExportResult:
        """Override to do cross-conversation pairing."""
        import json
        from pathlib import Path

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = ExportResult(
            format_name=self.format_name,
            input_path=str(input_path),
            output_path=str(output_path),
        )

        # Group conversations by first user message (instruction)
        groups: dict[str, list[tuple[float, str]]] = defaultdict(list)
        has_scores = False

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result.total_input += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    result.skipped += 1
                    continue

                score = row.get("final_score")
                if score is None:
                    result.skipped += 1
                    continue
                has_scores = True

                entries = row.get("conversations", [])
                if len(entries) < 2:
                    result.skipped += 1
                    continue

                # Use first user message as the grouping key
                first_user = ""
                last_assistant = ""
                for entry in entries:
                    if entry.get("role") == "user" and not first_user:
                        first_user = entry.get("content", "")
                    if entry.get("role") == "assistant":
                        last_assistant = entry.get("content", "")

                if first_user and last_assistant:
                    groups[first_user].append((float(score), last_assistant))

        if not has_scores:
            result.warnings.append(
                "DPO export requires quality scores. "
                "Re-generate with quality.auto_improve: true"
            )
            # Write empty file
            output_path.write_text("")
            return result

        # Generate preference pairs
        with open(output_path, "w", encoding="utf-8") as fout:
            for prompt, responses in groups.items():
                if len(responses) < 2:
                    continue
                responses.sort(key=lambda x: x[0], reverse=True)
                best_score, best_resp = responses[0]
                worst_score, worst_resp = responses[-1]

                if best_score - worst_score < self.min_score_gap:
                    continue

                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"

                pair = {
                    "prompt": full_prompt,
                    "chosen": best_resp,
                    "rejected": worst_resp,
                }
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                result.total_output += 1

        if result.total_output == 0 and has_scores:
            result.warnings.append(
                "No instruction pairs found for DPO. "
                "Need multiple responses per prompt."
            )

        return result
