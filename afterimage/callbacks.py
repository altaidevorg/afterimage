import traceback
from typing import List, Literal, Optional, Union
from .base import BaseInstructionGeneratorCallback, BaseRespondentPromptModifierCallback
from .common import default_model_name, default_safety_settings, GeneratedInstructions
from .key_management import SmartKeyPool
from .prompts import (
    default_instruction_generation_prompt,
    default_respondent_prompt_with_context,
    default_rag_respondent_prompt_with_context,
    default_persona_instruction_generation_prompt,
    get_correspondent_instruction_generation_prompt,
)
from .providers import DocumentProvider, InMemoryDocumentProvider
from .providers.llm_providers import LLMFactory
from .retrievers import ContextRetriever
from .types import GeneratedResponsePrompt, Document
import random

from pydantic import BaseModel


class InstructionsSchema(BaseModel):
    instructions: List[str]


class ContextualInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts."""

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
    ):
        """Initialize the callback with configuration parameters.

        Args:
            api_key: API key for the generative AI service
            documents: Either a list of documents or a DocumentProvider instance
            prompt: Custom instruction generation prompt
            model_name: Model name to use
            model_provider_name: Model provider name to use
            num_random_contexts: Number of contexts to sample
            separator_text: Separator text for merging contexts
            safety_settings: Safety settings for the model
        """
        assert api_key is not None, "You need to provide an API key"

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
        if "{n_instructions}" in self.prompt:
            try:
                self.prompt = self.prompt.format(n_instructions=self.n_instructions)
            except KeyError:
                pass

        self.model_name = model_name if model_name is not None else default_model_name
        self.model_provider_name = model_provider_name
        self.num_random_contexts = max(num_random_contexts, 1)
        self.separator_text = separator_text
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

    def _create_model(self, system_instruction=None):
        """Creates and configures the LLM model."""
        return LLMFactory.create(
            provider=self.model_provider_name,
            model_name=self.model_name,
            api_key=self.key_pool,
            system_instruction=system_instruction or self.prompt,
            safety_settings=self.safety_settings,
            response_mime_type="application/json",
            response_schema=InstructionsSchema,
        )

    def _sample(self) -> list[Document]:
        """Sample random contexts using the document provider."""
        return self.provider.get_documents(self.num_random_contexts)

    def _merge_contexts(self, contexts: list[str]) -> str:
        """Merge multiple contexts into a single string."""
        return self.separator_text.join(contexts)

    def generate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context.

        Args:
            original_prompt (str): The prompt guiding instruction generation.

        Returns:
            GeneratedInstructions: The instructions generated along with the context used.
        """
        model = self._create_model()
        random_contexts = self._sample()
        full_context = self._merge_contexts([c.text for c in random_contexts])
        original_prompt = (
            original_prompt.format(n_instructions=self.n_instructions)
            if "{n_instructions}" in original_prompt
            else original_prompt
        )
        prompt = f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

        instructions = model.generate_content(
            prompt=prompt,
        ).raw_response.parsed.instructions

        return GeneratedInstructions(instructions=instructions, context=full_context)

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context asynchronously."""
        model = self._create_model()
        random_contexts = self._sample()
        full_context = self._merge_contexts([c.text for c in random_contexts])
        original_prompt = (
            original_prompt.format(n_instructions=self.n_instructions)
            if "{n_instructions}" in original_prompt
            else original_prompt
        )
        prompt = f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

        try:
            response = await model.agenerate_content(
                prompt=prompt,
            )
        except Exception:
            traceback.print_exc()
            raise
        instructions = response.raw_response.parsed.instructions
        
        return GeneratedInstructions(instructions=instructions, context=full_context)

    def create_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a correspondent prompt based on the respondent prompt."""
        api_key: str | None = None
        try:
            prompt = get_correspondent_instruction_generation_prompt(assistant_prompt=respondent_prompt)
            api_key = self.key_pool.get_next_key()
            model = LLMFactory.create(
                "gemini",
                "gemini-2.5-pro",
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = model.generate_content(prompt=prompt, temperature=0.7)
            return response.text.strip().lstrip("<user_system_prompt>").rstrip("</user_system_prompt>").strip()

        except Exception:
            if api_key is not None:
                self.key_pool.report_error(api_key)
            raise

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a correspondent prompt based on the respondent prompt asynchronously."""
        api_key: str | None = None
        try:
            prompt = get_correspondent_instruction_generation_prompt(assistant_prompt=respondent_prompt)
            api_key = await self.key_pool.aget_next_key()
            model = LLMFactory.create(
                "gemini",
                "gemini-2.5-pro",
                api_key=api_key,
                safety_settings=self.safety_settings,
            )

            response = await model.agenerate_content(prompt=prompt, temperature=0.7)
            prompt_text = response.text.strip().lstrip("<user_system_prompt>").rstrip("</user_system_prompt>").strip()

            return prompt_text
        except Exception:
            if api_key is not None:
                await self.key_pool.areport_error(api_key)
            raise


class PersonaInstructionGeneratorCallback(ContextualInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts and personas."""

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
        
        # Format the system prompt with persona
        # We use self.prompt which might still have placeholders if __init__ skipped formatting
        system_prompt = self.prompt
        if "{persona}" in system_prompt:
             # We need to handle n_instructions too if it wasn't formatted in __init__
             format_args = {"persona": persona or "A curious user"}
             if "{n_instructions}" in system_prompt:
                 format_args["n_instructions"] = self.n_instructions
             system_prompt = system_prompt.format(**format_args)

        model = self._create_model(system_instruction=system_prompt)
        full_context = self._merge_contexts([c.text for c in random_contexts])
        
        original_prompt = (
            original_prompt.format(n_instructions=self.n_instructions, persona=persona or "A curious user")
            if "{n_instructions}" in original_prompt and "{persona}" in original_prompt
            else original_prompt
        )
        
        prompt = f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

        instructions = model.generate_content(
            prompt=prompt,
        ).raw_response.parsed.instructions

        return GeneratedInstructions(instructions=instructions, context=full_context, persona=persona)

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt, sampled context and persona asynchronously."""
        random_contexts, persona = self._sample()
        
        # Format the system prompt with persona
        system_prompt = self.prompt
        if "{persona}" in system_prompt:
             format_args = {"persona": persona or "A curious user"}
             if "{n_instructions}" in system_prompt:
                 format_args["n_instructions"] = self.n_instructions
             system_prompt = system_prompt.format(**format_args)

        model = self._create_model(system_instruction=system_prompt)
        full_context = self._merge_contexts([c.text for c in random_contexts])
        
        original_prompt = (
            original_prompt.format(n_instructions=self.n_instructions, persona=persona or "A curious user")
            if "{n_instructions}" in original_prompt and "{persona}" in original_prompt
            else original_prompt
        )
        
        prompt = f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

        try:
            response = await model.agenerate_content(
                prompt=prompt,
            )
        except Exception:
            traceback.print_exc()
            raise
        instructions = response.raw_response.parsed.instructions
        
        return GeneratedInstructions(instructions=instructions, context=full_context, persona=persona)



class WithContextRespondentPromptModifier(BaseRespondentPromptModifierCallback):
    """Modifies respondent prompt by adding context."""

    def __init__(self, prompt_template: Optional[str] = None):
        """Initialize the context-aware prompt modifier.

        Args:
            prompt_template: Custom prompt template. If None, uses `default_respondent_prompt_with_context`.
                If t contains {prompt} and/or {context}, they will be replced by the respondent prompt and the context, respectively.
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else default_respondent_prompt_with_context
        )
        self.should_inject_prompt = "{prompt}" in self.prompt_template
        self.should_inject_context = "{context}" in self.prompt_template

    def generate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        """Generates a modified respondent prompt by injecting context and instructions.

        Args:
            respondent_prompt: The original prompt for the respondent
            context: Additional context to be included
            instruction: The instruction associated with the prompt

        Returns:
            GeneratedResponsePrompt containing the modified prompt and context
        """
        additional_context = self._maybe_augment_context(instruction, context)

        if self.should_inject_prompt and self.should_inject_context:
            modified_prompt = self.prompt_template.format(
                prompt=respondent_prompt, context=additional_context
            )
        elif self.should_inject_prompt:
            modified_prompt = self.prompt_template.format(prompt=respondent_prompt)
        elif self.should_inject_context:
            modified_prompt = self.prompt_template.format(context=additional_context)
        else:
            modified_prompt = respondent_prompt

        return GeneratedResponsePrompt(
            prompt=modified_prompt,
            context=additional_context,
        )

    async def agenerate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        """Generates a modified respondent prompt by injecting context and instructions asynchronously."""
        additional_context = self._maybe_augment_context(instruction, context)

        if self.should_inject_prompt and self.should_inject_context:
            modified_prompt = self.prompt_template.format(
                prompt=respondent_prompt, context=additional_context
            )
        elif self.should_inject_prompt:
            modified_prompt = self.prompt_template.format(prompt=respondent_prompt)
        elif self.should_inject_context:
            modified_prompt = self.prompt_template.format(context=additional_context)
        else:
            modified_prompt = respondent_prompt

        return GeneratedResponsePrompt(
            prompt=modified_prompt,
            context=additional_context,
        )


class WithRAGRespondentPromptModifier(WithContextRespondentPromptModifier):
    """Modifies respondent prompt by adding relevant context using a retrieval strategy."""

    def __init__(
        self,
        retriever: ContextRetriever,
        prompt_template: Optional[str] = None,
    ):
        """Initialize the RAG-enhanced prompt modifier.

        Args:
            retriever: Strategy for retrieving relevant context
            prompt_template: Custom prompt template. If None, uses default_rag_respondent_prompt_with_context
        """
        super().__init__(
            prompt_template
            if prompt_template is not None
            else default_rag_respondent_prompt_with_context
        )
        self.retriever = retriever

    def augment_context(self, instruction: str, current_context: str) -> str:
        """Augment existing context with relevant information using the retriever.

        Args:
            instruction: The current instruction/question
            current_context: Any existing context

        Returns:
            str: Combined context from both sources
        """
        rag_context = self.retriever.get_context(instruction)

        if current_context:
            return (
                f"{current_context}\n\nAdditional relevant information:\n{rag_context}"
            )
        return rag_context
