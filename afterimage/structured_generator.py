import asyncio
import traceback
import warnings
from typing import AsyncGenerator, Dict, List, Literal, Optional, Type, TypeVar
import time

from tqdm.asyncio import tqdm
from pydantic import BaseModel

from .base import (
    BaseGenerator,
    BaseInstructionGeneratorCallback,
    BaseRespondentPromptModifierCallback,
)
from .common import default_model_name, default_safety_settings
from .key_management import SmartKeyPool
from .providers import LLMFactory
from .monitoring import GenerationMonitor
from .storage import BaseStorage, JSONLStorage
from .types import StructuredGenerationRow

T = TypeVar("T", bound=BaseModel)


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
        instruction_generator_callback: BaseInstructionGeneratorCallback,
        respondent_prompt_modifier: BaseRespondentPromptModifierCallback | None,
    ) -> AsyncGenerator[StructuredGenerationRow[T], None]:
        """Generates structured outputs for a single batch of instructions."""

        # We pass the correspondent_prompt to the callback.
        # This prompt dictates how the user (correspondent) should behave (e.g. persona).
        # ainitialize() ensures self.correspondent_prompt is set.
        await self.ainitialize(instruction_generator_callback)

        correspondent_prompt = self.correspondent_prompt

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
            current_respondent_prompt = self.respondent_prompt
            if respondent_prompt_modifier:
                if hasattr(respondent_prompt_modifier, "acall"):
                    modified_respondent_prompt = await respondent_prompt_modifier.acall(
                        self.respondent_prompt,
                        context=instruction_context,
                        instruction=instruction,
                    )
                else:
                    modified_respondent_prompt = await asyncio.to_thread(
                        respondent_prompt_modifier,
                        self.respondent_prompt,
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

                # We raise here or swallow? AsyncConversationGenerator raises.
                # But here we are inside a loop over instructions.
                # If one fails, we might want to skip it?
                # AsyncConversationGenerator raises in `generate_single` but catches in `worker_task` wrapper?
                # Actually AsyncConversationGenerator swallows in worker_task loop except for cancellation.
                # Let's log and re-raise to let the worker handle it or skip.
                # If we raise, we stop processing other instructions in this batch.
                # Let's print traceback and continue to next instruction?
                traceback.print_exc()
                continue

    async def generate(
        self,
        num_samples: int = 10,
        max_concurrency: int = 4,
    ) -> None:
        """Generates structured samples and saves them to storage.

        Args:
            num_samples: Total number of samples to generate.
            max_concurrency: Maximum number of concurrent tasks.
        """
        if not self.instruction_generator_callback:
            raise ValueError("instruction_generator_callback must be set.")

        await self.ainitialize(self.instruction_generator_callback)

        pbar = tqdm(
            total=num_samples, desc="Generating structured data...", unit="sample"
        )
        stop = asyncio.Event()
        num_generated = 0
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks: list[asyncio.Task] = []

        # We need a way to continuously fetch instructions until we have enough samples.
        # But `instruction_generator_callback` usually returns a batch.
        # We'll loop calling it.

        async def save_output(output: StructuredGenerationRow[T]):
            if output:
                if hasattr(self.storage, "asave_conversations"):
                    # We reuse asave_conversations which we patched to accept BaseModel (including StructuredGenerationRow)
                    await self.storage.asave_conversations([output])
                else:
                    await asyncio.to_thread(self.storage.save_conversations, [output])

        async def worker_task():
            nonlocal num_generated
            async with semaphore:
                async for output in self.generate_single(
                    self.instruction_generator_callback, self.respondent_prompt_modifier
                ):
                    if stop.is_set():
                        break

                    await save_output(output)
                    num_generated += 1
                    pbar.update(1)

                    if num_generated >= num_samples:
                        stop.set()
                        break

        # Spawn tasks loop
        while not stop.is_set() and num_generated < num_samples:
            # We spawn a new worker which fetches a batch of instructions and processes them
            t = asyncio.create_task(worker_task())
            tasks.append(t)
            await asyncio.sleep(
                0.01
            )  # Small delay to prevent tight loop if callback is fast

        # wait for all spawned tasks to finish/cancel cleanly
        for future in asyncio.as_completed(tasks):
            try:
                await future
                if stop.is_set():
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                # swallow cancellations
                pass
            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"Exception in future: {str(e)}")

        pbar.close()
