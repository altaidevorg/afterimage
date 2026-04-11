from typing import Any, Literal, Optional

from ...key_management import SmartKeyPool
from ...prompts import default_instruction_generation_prompt
from ...monitoring import GenerationMonitor
from .llm_backed import LLMBackedInstructionGeneratorCallback


class SimpleInstructionGeneratorCallback(LLMBackedInstructionGeneratorCallback):
    """Generates instructions from the correspondent prompt only (no document context).

    Omits a ``provider`` attribute so :class:`~afterimage.sampling.SamplingStrategy` does not
    treat this callback as document-backed.

    Args:
        api_key: API key for the generative AI service.
        prompt: System instruction for instruction generation. If None, uses the default.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        n_instructions: Number of instructions to generate in each round.
        safety_settings: Safety settings for the model (mainly for Gemini).
        monitor: Optional :class:`~afterimage.monitoring.GenerationMonitor`.
    """

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal[
            "gemini", "openai", "deepseek", "local"
        ] = "gemini",
        n_instructions: int = 3,
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
        llm_create_extras: dict[str, Any] | None = None,
    ):
        base_prompt = (
            prompt if prompt is not None else default_instruction_generation_prompt
        )
        super().__init__(
            api_key=api_key,
            prompt=base_prompt,
            model_name=model_name,
            model_provider_name=model_provider_name,
            n_instructions=n_instructions,
            safety_settings=safety_settings,
            monitor=monitor,
            llm_create_extras=llm_create_extras,
        )

    def generate(self, original_prompt):
        """Generate instructions using ``original_prompt`` as the user message (no context block)."""
        model = self._create_model()
        return self._execute_generation(
            model=model,
            prompt=original_prompt,
            full_context="",
            context_id=None,
            context_ids=[],
        )

    async def agenerate(self, original_prompt):
        """Async variant of :meth:`generate`."""
        model = self._create_model()
        return await self._aexecute_generation(
            model=model,
            prompt=original_prompt,
            full_context="",
            context_id=None,
            context_ids=[],
        )
