from typing import Optional

from ..base import (
    BaseRespondentPromptModifierCallback,
)
from ..prompts import (
    default_rag_respondent_prompt_with_context,
    default_respondent_prompt_with_context,
)
from ..retrievers import (
    ContextRetriever,
    RETRIEVAL_METADATA_KEY,
    RetrievalResult,
    aget_retrieval_result,
    get_retrieval_result_sync,
)
from ..types import (
    GeneratedResponsePrompt,
)


class WithContextRespondentPromptModifier(BaseRespondentPromptModifierCallback):
    """Modifies respondent prompt by adding context.

    Args:
        prompt_template: Custom prompt template. If None, uses `default_respondent_prompt_with_context`.
            If it contains `{prompt}` and/or `{context}`, they will be replaced by the respondent prompt and the context, respectively."""

    def __init__(self, prompt_template: Optional[str] = None):
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else default_respondent_prompt_with_context
        )
        self.should_inject_prompt = "{prompt}" in self.prompt_template
        self.should_inject_context = "{context}" in self.prompt_template

    def _apply_prompt_template(
        self, respondent_prompt: str, additional_context: str
    ) -> str:
        if self.should_inject_prompt and self.should_inject_context:
            return self.prompt_template.format(
                prompt=respondent_prompt, context=additional_context
            )
        if self.should_inject_prompt:
            return self.prompt_template.format(prompt=respondent_prompt)
        if self.should_inject_context:
            return self.prompt_template.format(context=additional_context)
        return respondent_prompt

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
        modified_prompt = self._apply_prompt_template(
            respondent_prompt, additional_context
        )

        return GeneratedResponsePrompt(
            prompt=modified_prompt,
            context=additional_context,
        )

    async def agenerate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        """Generates a modified respondent prompt by injecting context and instructions asynchronously."""
        if hasattr(self, "augment_context_async"):
            additional_context = await self.augment_context_async(instruction, context)
        else:
            additional_context = self._maybe_augment_context(instruction, context)

        modified_prompt = self._apply_prompt_template(
            respondent_prompt, additional_context
        )

        return GeneratedResponsePrompt(
            prompt=modified_prompt,
            context=additional_context,
        )


class WithRAGRespondentPromptModifier(WithContextRespondentPromptModifier):
    """Modifies respondent prompt by adding relevant context using a retrieval strategy.

    Uses :func:`afterimage.retrievers.aget_retrieval_result` / ``get_retrieval_result_sync``
    so retrievers may optionally implement ``*_context_with_metadata`` and attach
    citation-style fields under ``GeneratedResponsePrompt.metadata[RETRIEVAL_METADATA_KEY]``.

    Args:
        retriever: Strategy for retrieving relevant context
        prompt_template: Custom prompt template. If None, uses `default_rag_respondent_prompt_with_context`."""

    def __init__(
        self,
        retriever: ContextRetriever,
        prompt_template: Optional[str] = None,
    ):
        super().__init__(
            prompt_template
            if prompt_template is not None
            else default_rag_respondent_prompt_with_context
        )
        self.retriever = retriever

    @staticmethod
    def _merge_rag_into_instruction_context(rag_text: str, current_context: str) -> str:
        if current_context:
            return f"{current_context}\n\nAdditional relevant information:\n{rag_text}"
        return rag_text

    def _build_from_retrieval(
        self,
        respondent_prompt: str,
        instruction_context: str,
        result: RetrievalResult,
    ) -> GeneratedResponsePrompt:
        additional_context = self._merge_rag_into_instruction_context(
            result.context, instruction_context
        )
        modified_prompt = self._apply_prompt_template(
            respondent_prompt, additional_context
        )
        meta = {RETRIEVAL_METADATA_KEY: result.metadata} if result.metadata else {}
        return GeneratedResponsePrompt(
            prompt=modified_prompt,
            context=additional_context,
            metadata=meta,
        )

    def augment_context(self, instruction: str, current_context: str) -> str:
        """Augment existing context with relevant information using the retriever."""
        result = get_retrieval_result_sync(self.retriever, instruction)
        return self._merge_rag_into_instruction_context(result.context, current_context)

    async def augment_context_async(
        self, instruction: str, current_context: str
    ) -> str:
        """Async RAG augmentation; uses the same resolution as :meth:`agenerate`."""
        result = await aget_retrieval_result(self.retriever, instruction)
        return self._merge_rag_into_instruction_context(result.context, current_context)

    def generate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        result = get_retrieval_result_sync(self.retriever, instruction)
        return self._build_from_retrieval(respondent_prompt, context, result)

    async def agenerate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        result = await aget_retrieval_result(self.retriever, instruction)
        return self._build_from_retrieval(respondent_prompt, context, result)
