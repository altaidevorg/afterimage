"""
Generator factory functions for different generation modes.
"""

import asyncio
from typing import Literal

from schemas import CustomerSupportInteraction, ToolInvocation, AVAILABLE_TOOLS
from storage import CaptureStorage

from afterimage import (
    AsyncConversationGenerator,
    AsyncStructuredGenerator,
    InMemoryDocumentProvider,
    PersonaInstructionGeneratorCallback,
    ToolCallingInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)


GenerationMode = Literal["Structured Generation", "Tool Calling Generation", "Generic Conversation"]


def create_generator(
    mode: GenerationMode,
    api_key: str,
    docs: InMemoryDocumentProvider,
    storage: CaptureStorage,
    respondent_prompt: str,
    model_name: str = "gemini-2.0-flash",
):
    """
    Create a generator based on the specified mode.
    
    Args:
        mode: The generation mode
        api_key: API key for the LLM
        docs: Document provider with context
        storage: Storage for capturing outputs
        respondent_prompt: System prompt for the respondent
        model_name: Name of the model to use
    
    Returns:
        Configured generator instance
    """
    # Default instruction callback (used for Structured and Generic)
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
    )
    
    if mode == "Structured Generation":
        return AsyncStructuredGenerator(
            output_schema=CustomerSupportInteraction,
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            instruction_generator_callback=instruction_callback,
            storage=storage,
        )
    
    elif mode == "Tool Calling Generation":
        tool_instruction_callback = ToolCallingInstructionGeneratorCallback(
            api_key=api_key,
            tools=list(AVAILABLE_TOOLS),
            documents=docs,
            num_random_contexts=1,
        )
        return AsyncStructuredGenerator(
            output_schema=ToolInvocation,
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            instruction_generator_callback=tool_instruction_callback,
            storage=storage,
        )
    
    else:  # Generic Conversation
        return AsyncConversationGenerator(
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            instruction_generator_callback=instruction_callback,
            respondent_prompt_modifier=WithContextRespondentPromptModifier(),
            storage=storage,
        )


def create_generation_task(
    generator,
    num_samples: int,
    mode: GenerationMode,
) -> asyncio.Task:
    """
    Create an async task for generation.
    
    Args:
        generator: The configured generator
        num_samples: Number of samples to generate
        mode: The generation mode (determines which method to call)
    
    Returns:
        An asyncio Task
    """
    if mode == "Generic Conversation":
        return asyncio.create_task(generator.generate(num_dialogs=num_samples))
    else:
        return asyncio.create_task(generator.generate(num_samples=num_samples))
