import time
from typing import Literal, Optional

from ...base import BaseInstructionGeneratorCallback
from ...common import GeneratedInstructions, default_model_name, default_safety_settings
from ...key_management import SmartKeyPool
from ...monitoring import GenerationMonitor
from ...prompts import get_correspondent_instruction_generation_prompt
from ...providers.llm_providers import LLMFactory
from ._utils import strip_user_system_prompt_tags, substitute_n_instructions_in_prompt
from .schema import InstructionsSchema


class LLMBackedInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Shared LLM wiring: keys, model config, structured instruction generation, monitoring.

    Subclasses pass a *template* ``prompt``; ``{n_instructions}`` is expanded here when present.
    """

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        prompt: str,
        model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai", "deepseek"] = "gemini",
        n_instructions: int = 3,
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
    ):
        assert api_key is not None, "You need to provide an API key"
        self.monitor = monitor
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.n_instructions = max(n_instructions, 1)
        self.prompt = substitute_n_instructions_in_prompt(
            prompt,
            self.n_instructions,
        )
        self.model_name = model_name if model_name is not None else default_model_name
        self.model_provider_name = model_provider_name
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

    def set_monitor(self, monitor: GenerationMonitor) -> None:
        self.monitor = monitor

    def _create_model(self, system_instruction=None):
        return LLMFactory.create(
            provider=self.model_provider_name,
            model_name=self.model_name,
            api_key=self.key_pool,
            system_instruction=system_instruction or self.prompt,
            safety_settings=self.safety_settings,
        )

    def _execute_generation(
        self,
        model,
        prompt: str,
        full_context: str,
        context_id: str | None,
        context_ids: list[str] | None = None,
        persona: str | None = None,
        persona_generation_depth: int | None = None,
    ) -> GeneratedInstructions:
        start = time.time()
        try:
            output = model.generate_structured(
                prompt=prompt,
                schema=InstructionsSchema,
            )
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start,
                    success=True,
                    prompt_token_count=output.prompt_token_count,
                    completion_token_count=output.completion_token_count,
                    total_token_count=output.total_token_count,
                    finish_reason=output.finish_reason,
                    model_name=output.model_name,
                    metadata={"operation": "instruction_generation"},
                )

            return GeneratedInstructions(
                instructions=output.parsed.instructions,
                context=full_context,
                context_id=context_id,
                context_ids=context_ids or [],
                persona=persona,
                persona_generation_depth=persona_generation_depth,
            )
        except Exception as e:
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "instruction_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            raise e

    async def _aexecute_generation(
        self,
        model,
        prompt: str,
        full_context: str,
        context_id: str | None,
        context_ids: list[str] | None = None,
        persona: str | None = None,
        persona_generation_depth: int | None = None,
    ) -> GeneratedInstructions:
        start = time.time()
        try:
            response = await model.agenerate_structured(
                prompt=prompt,
                schema=InstructionsSchema,
            )
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    finish_reason=response.finish_reason,
                    model_name=response.model_name,
                    metadata={"operation": "instruction_generation"},
                )

            return GeneratedInstructions(
                instructions=response.parsed.instructions,
                context=full_context,
                context_id=context_id,
                context_ids=context_ids or [],
                persona=persona,
                persona_generation_depth=persona_generation_depth,
            )
        except Exception as e:
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "instruction_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            raise e

    def create_correspondent_prompt(self, respondent_prompt: str) -> str:
        start_time = time.time()
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=respondent_prompt
            )
            api_key = self.key_pool.get_next_key()
            model = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = model.generate_content(prompt=prompt, temperature=0.7)
            prompt_text = strip_user_system_prompt_tags(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={"operation": "correspondent_prompt_generation"},
                )
            return prompt_text

        except Exception as e:
            if self.monitor:
                self.monitor.log_error(
                    message="Error while trying to create correspondent prompt in instruction generator callback",
                    error=e,
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            raise

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        start_time = time.time()
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=respondent_prompt
            )
            api_key = await self.key_pool.aget_next_key()
            model = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = await model.agenerate_content(prompt=prompt, temperature=0.7)
            prompt_text = strip_user_system_prompt_tags(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={"operation": "correspondent_prompt_generation"},
                )
            return prompt_text
        except Exception as e:
            if self.monitor:
                self.monitor.log_error(
                    message="Error while trying to create correspondent prompt in instruction generator callback",
                    error=e,
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            raise
