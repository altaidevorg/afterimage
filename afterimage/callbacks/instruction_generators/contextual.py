from typing import Any, Literal, Optional, Union

from ...key_management import SmartKeyPool
from ...prompts import default_instruction_generation_prompt
from ...monitoring import GenerationMonitor
from ...providers import DocumentProvider, InMemoryDocumentProvider
from ...types import Document
from ._utils import context_ids_from_documents
from .llm_backed import LLMBackedInstructionGeneratorCallback


class ContextualInstructionGeneratorCallback(LLMBackedInstructionGeneratorCallback):
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
        model_provider_name: Literal[
            "gemini", "openai", "deepseek", "local"
        ] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
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
        self.provider = (
            documents
            if isinstance(documents, DocumentProvider)
            else InMemoryDocumentProvider(documents)
        )
        self.num_random_contexts = max(num_random_contexts, 1)
        self.separator_text = separator_text

    def _sample(self) -> list[Document]:
        return self.provider.get_documents(self.num_random_contexts)

    def _merge_contexts(self, contexts: list[str]) -> str:
        return self.separator_text.join(contexts)

    def _format_contextual_prompt(self, original_prompt: str, full_context: str) -> str:
        return f"""{original_prompt}
----------------------------

## Context
Ask the questions in the same language as this context.
<context>
{full_context}
</context>
        """

    def _run_contextual_generation(self, original_prompt: str):
        model = self._create_model()
        random_contexts = self._sample()
        full_context = self._merge_contexts([c.text for c in random_contexts])
        prompt = self._format_contextual_prompt(original_prompt, full_context)
        context_id, context_ids = context_ids_from_documents(random_contexts)
        return (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
        )

    def generate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context.

        Args:
            original_prompt (str): The correspondent prompt if any.

        Returns:
            GeneratedInstructions: The instructions generated along with the context used.
        """
        model, prompt, full_context, context_id, context_ids = (
            self._run_contextual_generation(original_prompt)
        )
        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt and sampled context asynchronously."""
        model, prompt, full_context, context_id, context_ids = (
            self._run_contextual_generation(original_prompt)
        )
        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
        )
