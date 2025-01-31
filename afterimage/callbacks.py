import json
import google.generativeai as genai
from typing import List, TypedDict, Optional, Union
from .base import BaseInstructionGeneratorCallback, BaseRespondentPromptModifierCallback
from .common import default_model_name, default_safety_settings, GeneratedInstructions
from .prompts import (
    default_instruction_generation_prompt,
    default_respondent_prompt_with_context,
    default_rag_respondent_prompt_with_context,
)
from .retrievers import ContextRetriever  # Update import
from .providers import DocumentProvider, InMemoryDocumentProvider  # Update imports


class InstructionsSchema(TypedDict):
    instructions: List[str]


class ContextualInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts."""

    def __init__(
        self,
        api_key: str,
        documents: Union[List[str], DocumentProvider],
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        num_random_contexts: int = 1,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
        safety_settings: Optional[dict] = None,
    ):
        """Initialize the callback with configuration parameters.

        Args:
            api_key: API key for the generative AI service
            documents: Either a list of documents or a DocumentProvider instance
            prompt: Custom instruction generation prompt
            model_name: Model name to use
            num_random_contexts: Number of contexts to sample
            separator_text: Separator text for merging contexts
            safety_settings: Safety settings for the model
        """
        assert api_key is not None, "You need to provide an API key"

        # Convert list to provider if needed
        self.provider = (
            documents
            if isinstance(documents, DocumentProvider)
            else InMemoryDocumentProvider(documents)
        )

        self.prompt = (
            prompt if prompt is not None else default_instruction_generation_prompt
        )
        self.model_name = model_name if model_name is not None else default_model_name
        self.num_random_contexts = max(num_random_contexts, 1)
        self.separator_text = separator_text
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

        genai.configure(api_key=api_key)

    def _create_model(self):
        """Creates and configures the LLM model."""
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.prompt,
            safety_settings=self.safety_settings,
        )

    def _sample(self) -> List[str]:
        """Sample random contexts using the document provider."""
        return self.provider.get_documents(self.num_random_contexts)

    def _merge_contexts(self, contexts: List[str]) -> str:
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
        full_context = self._merge_contexts(random_contexts)
        prompt = f"""{original_prompt}
----------------------------

ask the questions in the same language as this context.

## Context

{full_context}
        """

        instructions_str = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=InstructionsSchema,
            ),
        ).text
        instructions = json.loads(instructions_str)["instructions"]

        return GeneratedInstructions(instructions=instructions, context=full_context)


class WithContextRespondentPromptModifier(BaseRespondentPromptModifierCallback):
    """Modifies a respondent prompt by merging it with additional context."""

    def __init__(self, prompt_template: str | None = None):
        """Initializes the modifier with a template.

        Args:
            prompt_template (str, optional): Template for formatting the respondent prompt.
                Defaults to `default_respondent_prompt_with_context`.
                If the template contains `{prompt}` and/or `{context}`, they will be replaced by the original respondent prompt and the additional context, respectively.
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else default_respondent_prompt_with_context
        )
        self.should_inject_prompt = "{prompt}" in self.prompt_template
        self.should_inject_context = "{context}" in self.prompt_template

    def generate(self, respondent_prompt: str, context: str, instruction: str) -> str:
        """Generates a modified respondent prompt by injecting context and instructions.

        Args:
            respondent_prompt (str): The original prompt for the respondent.
            context (str): Additional context to be included.
            instruction (str): The instruction associated with the prompt.

        Returns:
            str: The modified respondent prompt.
        """
        additional_context = self._maybe_augment_context(instruction, context)

        if self.should_inject_prompt and self.should_inject_context:
            return self.prompt_template.format(
                prompt=respondent_prompt, context=additional_context
            )
        elif self.should_inject_prompt:
            return self.prompt_template.format(prompt=respondent_prompt)
        elif self.should_inject_context:
            return self.prompt_template.format(context=additional_context)
        else:
            return respondent_prompt


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
