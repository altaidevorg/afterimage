"""
Generator factory functions for different generation modes.
"""

import asyncio
from typing import List, Literal, Optional

from schemas import CustomerSupportInteraction, ToolInvocation, AVAILABLE_TOOLS
from .storage import CaptureStorage
from .tools_db import get_tools_db
from .function_parser import function_to_openai_schema
from .dynamic_schemas import create_dynamic_tool_invocation_schema

from afterimage import (
    AsyncConversationGenerator,
    AsyncStructuredGenerator,
    InMemoryDocumentProvider,
    PersonaInstructionGeneratorCallback,
    ToolCallingInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)


GenerationMode = Literal[
    "Structured Generation", "Tool Calling Generation", "Generic Conversation"
]


def get_selected_tools(
    tool_names: List[str],
) -> List:
    """
    Get the list of tools to use for generation.

    Args:
        tool_names: List of selected tool names

    Returns:
        List of tool definitions (OpenAI schema dicts)
    """
    tools = []

    if tool_names:
        db = get_tools_db()
        for name in tool_names:
            parsed = db.get_tool(name)
            if parsed:
                tools.append(function_to_openai_schema(parsed.definition))

    return tools


def create_generator(
    mode: GenerationMode,
    api_key: str,
    docs: InMemoryDocumentProvider,
    storage: CaptureStorage,
    respondent_prompt: str,
    model_name: str = "deepseek-chat",
    model_provider_name: str = "deepseek",
    selected_tools: Optional[List] = None,
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
        model_provider_name: Provider to use for LLM
        selected_tools: List of tools to use for Tool Calling mode

    Returns:
        Configured generator instance
    """
    # Default instruction callback (used for Structured and Generic)
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
        model_name=model_name,
        model_provider_name=model_provider_name,
    )

    if mode == "Structured Generation":
        return AsyncStructuredGenerator(
            output_schema=CustomerSupportInteraction,
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            model_provider_name=model_provider_name,
            instruction_generator_callback=instruction_callback,
            storage=storage,
        )

    elif mode == "Tool Calling Generation":
        # Use selected tools or default to all available
        tools_to_use = selected_tools if selected_tools else list(AVAILABLE_TOOLS)

        tool_instruction_callback = ToolCallingInstructionGeneratorCallback(
            api_key=api_key,
            tools=tools_to_use,
            documents=docs,
            num_random_contexts=1,
            model_name=model_name,
            model_provider_name=model_provider_name,
        )

        # Create dynamic schema including custom tools
        dynamic_schema = create_dynamic_tool_invocation_schema(tools_to_use)

        return AsyncStructuredGenerator(
            output_schema=dynamic_schema,
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            model_provider_name=model_provider_name,
            instruction_generator_callback=tool_instruction_callback,
            storage=storage,
        )

    else:  # Generic Conversation
        return AsyncConversationGenerator(
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name=model_name,
            model_provider_name=model_provider_name,
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
