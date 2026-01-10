import asyncio
import traceback
import warnings
from typing import AsyncGenerator, Dict, List, Literal, Optional, Type, TypeVar
import time
import logging

from tqdm.asyncio import tqdm
from pydantic import BaseModel

from .base import BaseGenerator, BaseStoppingCallback
from .callbacks import (
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
    FixedNumberStoppingCallback,
)
from .common import default_model_name, default_safety_settings
from .key_management import SmartKeyPool
from .providers import LLMFactory
from .monitoring import GenerationMonitor
from .storage import BaseStorage, JSONLStorage
from .types import StructuredGenerationRow, GenerationState

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class AsyncStructuredGenerator(BaseGenerator):
    """Generates structured datasets where outputs strictly conform to a Pydantic schema."""

    def __init__(
        self,
        output_schema: Type[T],
        respondent_prompt: str,
        api_key: str | SmartKeyPool,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
        correspondent_prompt: str | None = None,
    ):
        """Initialize the structured generator.

        Args:
            output_schema: Pydantic model class defining the output structure.
            respondent_prompt: System prompt for the respondent (the model generating structured output).
            api_key: API key or SmartKeyPool.
            model_name: Model name to use.
            safety_settings: Safety settings.
            model_provider_name: Provider name ("gemini" or "openai").
            storage: Storage implementation.
            monitor: GenerationMonitor.
            instruction_generator_callback: Callback to generate instructions/inputs.
            respondent_prompt_modifier: Callback to modify the system prompt per instruction.
            correspondent_prompt: The initial prompt for the correspondent, if already known.
        """
        self.output_schema = output_schema
        self.respondent_prompt = respondent_prompt
        self.correspondent_prompt = correspondent_prompt
        self.monitor = monitor
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )
        self.model_provider_name = model_provider_name
        self.model_name = model_name if model_name is not None else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )
        self.storage = storage or JSONLStorage()
        self.instruction_generator_callback = instruction_generator_callback
        self.respondent_prompt_modifier = respondent_prompt_modifier

        if self.instruction_generator_callback is None:
            warnings.warn(
                "No `instruction_generator_callback` provided. You must provide one to drive generation."
            )

    async def create_correspondent_prompt(self, respondent_prompt: str) -> str:
        # Fallback default correspondent prompt if callback doesn't provide one
        return "You are a user asking for assistance."

    async def generate_single(
        self,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ) -> AsyncGenerator[StructuredGenerationRow[T], None]:
        """Generates structured outputs for a single batch of instructions."""

        # ainitialize ensures self.correspondent_prompt is set
        await self.ainitialize(instruction_generator_callback)

        correspondent_prompt = self.correspondent_prompt
        respondent_prompt = self.respondent_prompt

        # Use provided callback or default
        instruction_generator_callback = (
            instruction_generator_callback or self.instruction_generator_callback
        )
        respondent_prompt_modifier = (
            respondent_prompt_modifier or self.respondent_prompt_modifier
        )

        if instruction_generator_callback is None:
            raise ValueError("instruction_generator_callback must be set.")

        if hasattr(instruction_generator_callback, "acall"):
            gen_instructions = await instruction_generator_callback.acall(
                correspondent_prompt
            )
        else:
            gen_instructions = await asyncio.to_thread(
                instruction_generator_callback, correspondent_prompt
            )

        for instruction in gen_instructions.instructions:
            instruction_context = gen_instructions.context

            # Modify prompt if modifier exists
            current_respondent_prompt = respondent_prompt
            if respondent_prompt_modifier:
                if hasattr(respondent_prompt_modifier, "acall"):
                    modified_respondent_prompt = await respondent_prompt_modifier.acall(
                        respondent_prompt,
                        context=instruction_context,
                        instruction=instruction,
                    )
                else:
                    modified_respondent_prompt = await asyncio.to_thread(
                        respondent_prompt_modifier,
                        respondent_prompt,
                        context=instruction_context,
                        instruction=instruction,
                    )
                current_respondent_prompt = modified_respondent_prompt.prompt

            # Combine instruction into the prompt or message
            # For structured generation, we usually just send the prompt.
            # However, the user instruction needs to be part of the request.
            # We'll treat the instruction as the "User Message" and the respondent_prompt as System Prompt.
            full_user_message = instruction
            if instruction_context:
                full_user_message = (
                    f"Context: {instruction_context}\n\nTask: {instruction}"
                )

            start_time = time.time()
            api_key: str | None = None
            try:
                api_key = await self.key_pool.aget_next_key()
                model = LLMFactory.create(
                    self.model_provider_name,
                    self.model_name,
                    api_key=api_key,
                    system_instruction=current_respondent_prompt,
                    safety_settings=self.safety_settings,
                )

                output = await model.agenerate_structured(
                    prompt=full_user_message,
                    schema=self.output_schema,
                    temperature=0.7,  # Default temperature
                )

                if self.monitor:
                    self.monitor.record_metric(
                        "structured_generation_time",
                        time.time() - start_time,
                        metadata={"success": True},
                    )

                yield StructuredGenerationRow(
                    instruction=instruction,
                    context=instruction_context,
                    persona=gen_instructions.persona,
                    output=output,
                    metadata={
                        "context_id": gen_instructions.context_id,
                        "persona_name": gen_instructions.persona,
                    },
                )

            except Exception as e:
                # Log error and continue
                if self.monitor:
                    self.monitor.record_metric(
                        "structured_generation_time",
                        time.time() - start_time,
                        metadata={
                            "success": False,
                            "error": str(e),
                            "error_type": e.__class__.__name__,
                        },
                    )
                if api_key:
                    await self.key_pool.areport_error(api_key)

                traceback.print_exc()
                continue

    async def generate(
        self,
        num_samples: int | None = None,
        max_concurrency: int = 4,
        stopping_criteria: list[BaseStoppingCallback] | None = None,
        instruction_generator_callback: BaseInstructionGeneratorCallback | None = None,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None = None,
    ) -> None:
        """Generates structured samples and saves them to storage.

        Args:
            num_samples: Total number of samples to generate. Defaults to 5.
            max_concurrency: Maximum number of concurrent tasks.
            stopping_criteria: A list of callbacks to determine when to stop generation. If num_dialogs is specified, FixedNumberStoppingCallback is added to this list.
        """
        if instruction_generator_callback is not None:
            warnings.warn(
                "Passing `instruction_generator_callback` to `generate()` is deprecated. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            instruction_generator_callback = self.instruction_generator_callback

        if respondent_prompt_modifier is not None:
            warnings.warn(
                "Passing `respondent_prompt_modifier` to `generate()` is deprecated. "
                "Please pass it to the constructor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            respondent_prompt_modifier = self.respondent_prompt_modifier

        if instruction_generator_callback is None:
            raise ValueError("instruction_generator_callback must be set.")

        if self.correspondent_prompt is None:
            await self.ainitialize(instruction_generator_callback)

        self._ensure_monitor()

        # Handle stopping criteria
        final_stopping_criteria = stopping_criteria or []
        if num_samples is not None:
            final_stopping_criteria.append(FixedNumberStoppingCallback(n=num_samples))

        # Default if nothing specified
        if not final_stopping_criteria:
            num_samples = 5
            final_stopping_criteria.append(FixedNumberStoppingCallback(n=num_samples))

        state = GenerationState(
            num_requested=num_samples or 0,
            monitor=self.monitor,
            stop_event=asyncio.Event(),
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def save_samples(samples: List[StructuredGenerationRow]):
            if samples:
                if hasattr(self.storage, "asave_conversations"):
                    await self.storage.asave_conversations(samples)
                else:
                    await asyncio.to_thread(self.storage.save_conversations, samples)

        async def worker_task():
            while not state.stop_event.is_set():
                async with semaphore:
                    if state.stop_event.is_set():
                        break

                    try:
                        async for sample in self.generate_single(
                            instruction_generator_callback=instruction_generator_callback,
                            respondent_prompt_modifier=respondent_prompt_modifier,
                        ):
                            # Update state and check stopping criteria
                            state.update(sample)

                            for criteria in final_stopping_criteria:
                                if await criteria.should_stop(state):
                                    state.stop_event.set()
                                    break

                            await save_samples([sample])

                            if state.stop_event.is_set():
                                break

                    except Exception as e:
                        logger.error(f"Error in generation: {e}")
                        traceback.print_exc()
                        if self.monitor:
                            self.monitor.record_metric("error_rate", 1.0)
                        continue

        pbar = tqdm(
            total=num_samples, desc="Generating structured data...", unit="sample"
        )
        tasks: list[asyncio.Task] = []

        # Create initial set of tasks
        for _ in range(max_concurrency):
            tasks.append(asyncio.create_task(worker_task()))

        last_count = 0
        while not state.stop_event.is_set() or any(not t.done() for t in tasks):
            # Update progress bar
            if state.num_generated > last_count:
                pbar.update(state.num_generated - last_count)
                last_count = state.num_generated

            if state.stop_event.is_set():
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

            await asyncio.sleep(0.1)

            if all(t.done() for t in tasks):
                break

        pbar.update(state.num_generated - last_count)
        pbar.close()

        # Wait for any remaining tasks to finish/cancel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
