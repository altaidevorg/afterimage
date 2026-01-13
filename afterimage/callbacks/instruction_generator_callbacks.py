import json
import random
import time
from typing import List, Literal, Optional, Type, Union

from pydantic import BaseModel

from ..base import (
    BaseInstructionGeneratorCallback,
)
from ..common import (
    GeneratedInstructions,
    default_model_name,
    default_safety_settings,
)
from ..key_management import SmartKeyPool
from ..monitoring import GenerationMonitor
from ..prompts import (
    default_instruction_generation_prompt,
    default_persona_instruction_generation_prompt,
    default_tool_calling_persona_instruction_generation_prompt,
    get_correspondent_instruction_generation_prompt,
)
from ..providers import DocumentProvider, InMemoryDocumentProvider
from ..providers.llm_providers import LLMFactory
from ..types import (
    Document,
)


class InstructionsSchema(BaseModel):
    instructions: List[str]


class ContextualInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts.

    Args:
        api_key: API key for the generative AI service.
        documents: Either a list of texts or a DocumentProvider instance providing context to ground the instructions.
            For each round of generation `num_random_contexts` documents are sampled from this collection.
        prompt: Prompt that guides the instruction generation. If None, uses the default instruction generation prompt.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        num_random_contexts: Number of contexts to sample for each round of generation.
        n_instructions: Number of instructions to generate in each round of generation.
        separator_text: Separator text for merging contexts if more than one context is sampled.
        safety_settings: Safety settings for the model. Mainly intended for Gemini models.
            Deprecated and may be removed in the future.
        monitor: GenerationMonitor instance to use for tracking.
            If `None`, Conversation and/or structured generators will set their own monitor."""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        documents: Union[list[str], DocumentProvider],
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
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

        # Convert list to provider if needed
        self.provider = (
            documents
            if isinstance(documents, DocumentProvider)
            else InMemoryDocumentProvider(documents)
        )

        self.n_instructions = max(n_instructions, 1)
        self.prompt = (
            prompt if prompt is not None else default_instruction_generation_prompt
        )

        # set the number of instructions to be generated if it has a placeholder for it
        if "{n_instructions}" in self.prompt:
            self.prompt = self.prompt.format(n_instructions=self.n_instructions)

        self.model_name = model_name if model_name is not None else default_model_name
        self.model_provider_name = model_provider_name
        self.num_random_contexts = max(num_random_contexts, 1)
        self.separator_text = separator_text
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

    def set_monitor(self, monitor: GenerationMonitor) -> None:
        self.monitor = monitor

    def _create_model(self, system_instruction=None):
        """Creates and configures the LLM model."""
        return LLMFactory.create(
            provider=self.model_provider_name,
            model_name=self.model_name,
            api_key=self.key_pool,
            system_instruction=system_instruction or self.prompt,
            safety_settings=self.safety_settings,
        )

    def _sample(self) -> list[Document]:
        """Sample random contexts using the document provider."""
        return self.provider.get_documents(self.num_random_contexts)

    def _merge_contexts(self, contexts: list[str]) -> str:
        """Merge multiple contexts into a single string."""
        return self.separator_text.join(contexts)

    def _format_contextual_prompt(self, original_prompt: str, full_context: str) -> str:
        """Format the final prompt with context."""
        return f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

    def _execute_generation(
        self,
        model,
        prompt: str,
        full_context: str,
        context_id: str | None,
        persona: str | None = None,
    ) -> GeneratedInstructions:
        """Execute the generation process with monitoring."""
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
                persona=persona,
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
        persona: str | None = None,
    ) -> GeneratedInstructions:
        """Execute the asynchronous generation process with monitoring."""
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
                persona=persona,
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

    def generate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context.

        Args:
            original_prompt (str): The correspondent prompt if any.

        Returns:
            GeneratedInstructions: The instructions generated along with the context used.
        """
        model = (
            self._create_model()
        )  # it will use self.prompt as the ssystem instruction
        random_contexts = self._sample()
        full_context = self._merge_contexts([c.text for c in random_contexts])
        prompt = self._format_contextual_prompt(original_prompt, full_context)

        # Pick the first document ID as the context_id for the merged context
        context_id = random_contexts[0].id if random_contexts else None

        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context asynchronously."""
        model = (
            self._create_model()
        )  # it will use self.prompt as the system instruction
        random_contexts = self._sample()
        full_context = self._merge_contexts([c.text for c in random_contexts])
        prompt = self._format_contextual_prompt(original_prompt, full_context)

        # Pick the first document ID as the context_id for the merged context
        context_id = random_contexts[0].id if random_contexts else None

        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
        )

    def create_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a correspondent prompt based on the respondent prompt."""
        api_key: str | None = None
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=respondent_prompt
            )
            api_key = self.key_pool.get_next_key()
            model = LLMFactory.create(
                "gemini",
                "gemini-2.5-pro",
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = model.generate_content(prompt=prompt, temperature=0.7)
            return (
                response.text.strip()
                .lstrip("<user_system_prompt>")
                .rstrip("</user_system_prompt>")
                .strip()
            )

        except Exception as e:
            if self.monitor:
                self.monitor.log_error(
                    message="Error while trying to crosspondent prompt in instruction generator callback",
                    error=e,
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a correspondent prompt based on the respondent prompt asynchronously."""
        api_key: str | None = None
        try:
            prompt = get_correspondent_instruction_generation_prompt(
                assistant_prompt=respondent_prompt
            )
            api_key = await self.key_pool.aget_next_key()
            model = LLMFactory.create(
                "gemini",
                "gemini-2.5-pro",
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = await model.agenerate_content(prompt=prompt, temperature=0.7)
            prompt_text = (
                response.text.strip()
                .lstrip("<user_system_prompt>")
                .rstrip("</user_system_prompt>")
                .strip()
            )

            return prompt_text
        except Exception as e:
            if self.monitor:
                self.monitor.log_error(
                    message="Error while trying to crosspondent prompt in instruction generator callback",
                    error=e,
                    metadata={
                        "operation": "correspondent_prompt_generation",
                        "error_type": e.__class__.__name__,
                    },
                )
            raise


class PersonaInstructionGeneratorCallback(ContextualInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts and personas.

    It works very similarly to `~ContextualInstructionGeneratorCallback` but it also samples a persona from the sampled documents.
        This usually results in more diverse yet still contextually relevant instructions.

    Args:
        api_key: API key for the generative AI service.
        documents: Either a list of texts or a DocumentProvider instance providing context to ground the instructions.
            For each round of generation `num_random_contexts` documents are sampled from this collection.
        prompt: Prompt that guides the instruction generation. If None, uses the default instruction generation prompt.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        num_random_contexts: Number of contexts to sample for each round of generation.
        n_instructions: Number of instructions to generate in each round of generation.
        separator_text: Separator text for merging contexts if more than one context is sampled.
        safety_settings: Safety settings for the model. Mainly intended for Gemini models.
            Deprecated and may be removed in the future.
        monitor: GenerationMonitor instance to use for tracking.
            If `None`, Conversation and/or structured generators will set their own monitor"""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        documents: Union[list[str], DocumentProvider],
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
    ):
        super().__init__(
            api_key=api_key,
            documents=documents,
            prompt=prompt
            if prompt is not None
            else default_persona_instruction_generation_prompt,
            model_name=model_name,
            model_provider_name=model_provider_name,
            num_random_contexts=num_random_contexts,
            n_instructions=n_instructions,
            separator_text=separator_text,
            safety_settings=safety_settings,
            monitor=monitor,
        )

    def _sample(self) -> tuple[list[Document], str | None]:
        """Sample random contexts and a persona using the document provider."""
        docs = self.provider.get_documents(self.num_random_contexts)

        # Collect all personas from sampled documents
        all_personas = []
        for doc in docs:
            for persona_entry in doc.personas:
                all_personas.extend(persona_entry.descriptions)

        selected_persona = random.choice(all_personas) if all_personas else None
        return docs, selected_persona

    def generate(self, original_prompt):
        """Generates instructions based on the provided prompt, sampled context and persona.

        Args:
            original_prompt (str): The prompt guiding instruction generation.

        Returns:
            GeneratedInstructions: The instructions generated along with the context and persona used.
        """
        random_contexts, persona = self._sample()
        if persona is None:
            persona = "A curious user"

        # Format the system prompt with persona
        # We use self.prompt which is already formatted for n_instructions but still has a placeholder for persona
        system_prompt = self.prompt
        if "{persona}" in system_prompt:
            system_prompt = system_prompt.format(persona=persona)

        model = self._create_model(system_instruction=system_prompt)
        full_context = self._merge_contexts([c.text for c in random_contexts])

        prompt = self._format_contextual_prompt(original_prompt, full_context)

        # Pick the first document ID as the context_id for the merged context
        context_id = random_contexts[0].id if random_contexts else None

        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            persona=persona,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt, sampled context and persona asynchronously."""
        random_contexts, persona = self._sample()
        if persona is None:
            persona = "A curious user"

        # Format the system prompt with persona
        system_prompt = self.prompt
        if "{persona}" in system_prompt:
            system_prompt = system_prompt.format(persona=persona)

        model = self._create_model(system_instruction=system_prompt)
        full_context = self._merge_contexts([c.text for c in random_contexts])

        prompt = self._format_contextual_prompt(original_prompt, full_context)

        # Pick the first document ID as the context_id for the merged context
        context_id = random_contexts[0].id if random_contexts else None

        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            persona=persona,
        )


class ToolCallingInstructionGeneratorCallback(PersonaInstructionGeneratorCallback):
    """Generates instructions that specifically require calling provided tools, optionally using personas.

    Args:
        api_key: API key for the generative AI service.
        tools: List of tools to use.
            each item of this list should be an OpenAI-style tool description as a dictionary or a pydantic model.
        documents: Either a list of texts or a DocumentProvider instance providing context to ground the instructions.
            For each round of generation `num_random_contexts` documents are sampled from this collection.
        prompt: Prompt that guides the instruction generation. If None, uses the default instruction generation prompt.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        num_random_contexts: Number of contexts to sample for each round of generation.
        n_instructions: Number of instructions to generate in each round of generation.
        num_tools_to_sample: Number of tools to sample as the targets for each round of generation.
        separator_text: Separator text for merging contexts if more than one context is sampled.
        safety_settings: Safety settings for the model. Mainly intended for Gemini models.
            Deprecated and may be removed in the future.
        monitor: GenerationMonitor instance to use for tracking.
            If `None`, Conversation and/or structured generators will set their own monitor"""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        tools: List[Union[dict, Type[BaseModel]]],
        documents: Union[list[str], DocumentProvider],
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        num_tools_to_sample: int = 2,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
    ):
        super().__init__(
            api_key=api_key,
            documents=documents,
            prompt=prompt
            if prompt is not None
            else default_tool_calling_persona_instruction_generation_prompt,
            model_name=model_name,
            model_provider_name=model_provider_name,
            num_random_contexts=num_random_contexts,
            n_instructions=n_instructions,
            separator_text=separator_text,
            safety_settings=safety_settings,
            monitor=monitor,
        )
        self.tools = self._normalize_tools(tools)
        self.num_tools_to_sample = num_tools_to_sample

    def _normalize_tools(self, tools: List[Union[dict, Type[BaseModel]]]) -> List[dict]:
        """Convert Pydantic models to OpenAI function schema if necessary."""
        normalized = []
        for t in tools:
            if isinstance(t, type) and issubclass(t, BaseModel):
                normalized.append(self._tool_model_to_openai_schema(t))
            else:
                normalized.append(t)
        return normalized

    def _tool_model_to_openai_schema(self, tool_model: Type[BaseModel]) -> dict:
        """Convert a Pydantic model into the OpenAI function schema format."""
        # Use existing logic from tool_calling_generator.py
        name = getattr(tool_model, "name", tool_model.__name__.lower())
        if hasattr(tool_model, "model_fields") and "name" in tool_model.model_fields:
            name = tool_model.model_fields["name"].default

        # Assume the model has an 'arguments' field if it's a wrapper,
        # otherwise use the model itself as arguments.
        if (
            hasattr(tool_model, "model_fields")
            and "arguments" in tool_model.model_fields
        ):
            args_model = tool_model.model_fields["arguments"].annotation
        else:
            args_model = tool_model

        args_schema = args_model.model_json_schema()

        params = {
            "type": "object",
            "properties": args_schema.get("properties", {}),
            "required": args_schema.get("required", []),
        }

        # Clean up schema for OpenAI
        for v in params["properties"].values():
            if "title" in v:
                del v["title"]

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool_model.__doc__ or "",
                "parameters": params,
            },
        }

    def _sample_tools(self) -> List[dict]:
        """Sample a subset of tools to focus on."""
        n = min(len(self.tools), self.num_tools_to_sample)
        return random.sample(self.tools, n)

    def _format_tools_context(self, tools: List[dict]) -> str:
        """Format tools into a readable string for the prompt."""
        return json.dumps(tools, indent=2)

    def generate(self, original_prompt):
        """Generates instructions that require tool calls."""
        random_contexts, persona = self._sample()
        if persona is None:
            persona = "A curious user"

        full_context = self._merge_contexts([c.text for c in random_contexts])

        target_tools = self._sample_tools()
        tools_context = self._format_tools_context(target_tools)

        system_prompt = self.prompt.format(
            n_instructions=self.n_instructions,
            tools_context=tools_context,
            context=full_context,
            persona=persona,
        )

        model = self._create_model(system_instruction=system_prompt)

        # We don't really use original_prompt here because we have a very specific system prompt
        # but we follow the interface.
        prompt = "Generate the instructions now."

        context_id = random_contexts[0].id if random_contexts else None

        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            persona=persona,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions that require tool calls asynchronously."""
        random_contexts, persona = self._sample()
        if persona is None:
            persona = "A curious user"

        full_context = self._merge_contexts([c.text for c in random_contexts])

        target_tools = self._sample_tools()
        tools_context = self._format_tools_context(target_tools)

        system_prompt = self.prompt.format(
            n_instructions=self.n_instructions,
            tools_context=tools_context,
            context=full_context,
            persona=persona,
        )

        model = self._create_model(system_instruction=system_prompt)

        prompt = "Generate the instructions now."

        context_id = random_contexts[0].id if random_contexts else None

        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            persona=persona,
        )

    def create_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a prompt for the correspondent."""
        # tool calling instruction generator callback uses a special correspondent prompt.
        # so we don't need to create a new one.
        return self.prompt

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a prompt for the correspondent asynchronously."""
        return self.create_correspondent_prompt(respondent_prompt)
