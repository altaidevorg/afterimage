"""Bridge Simula meta-prompts into ConversationGenerator (correspondent side)."""

from __future__ import annotations

import json
from typing import Any

from ...base import BaseInstructionGeneratorCallback
from ...common import GeneratedInstructions


class SimulaInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Yields precomputed Simula scenarios as user instructions (one per call)."""

    def __init__(
        self,
        scenarios: list[tuple[str, dict[str, Any]]],
        *,
        context_prefix: str = "simula",
    ):
        """
        Args:
            scenarios: List of (instruction_text, metadata dict) in dialog order.
            context_prefix: Prefix for context string serialization.
        """
        self._scenarios = list(scenarios)
        self._index = 0
        self._context_prefix = context_prefix
        self.monitor = None

    def set_monitor(self, monitor) -> None:
        self.monitor = monitor

    def reset(self) -> None:
        self._index = 0

    def generate(self, original_prompt: str) -> GeneratedInstructions:
        raise NotImplementedError("Use async generation with ConversationGenerator")

    async def agenerate(self, original_prompt: str) -> GeneratedInstructions:
        if self._index >= len(self._scenarios):
            raise IndexError(
                "SimulaInstructionGeneratorCallback: no more scenarios for this run"
            )
        text, meta = self._scenarios[self._index]
        self._index += 1
        ctx = json.dumps(
            {self._context_prefix: meta, "scenario": text},
            ensure_ascii=False,
        )
        return GeneratedInstructions(
            instructions=[text],
            context=ctx,
            context_id=None,
            context_ids=[],
        )
