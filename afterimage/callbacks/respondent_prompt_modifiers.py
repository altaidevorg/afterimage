from typing import Optional


from ..base import (
    BaseRespondentPromptModifierCallback,
)
from ..prompts import (
    default_rag_respondent_prompt_with_context,
    default_respondent_prompt_with_context,
)
from ..retrievers import ContextRetriever
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
    """Modifies respondent prompt by adding relevant context using a retrieval strategy.

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
