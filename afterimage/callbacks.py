import json
import random
import google.generativeai as genai
from typing import List, TypedDict
from .base import BaseInstructionGeneratorCallback, BaseRespondentPromptModifierCallback
from .common import default_model_name, default_safety_settings, GeneratedInstructions
from .prompts import (
    default_instruction_generation_prompt,
    default_respondent_prompt_with_context,
)


class InstructionsSchema(TypedDict):
    instructions: List[str]


class ContextualInstructionGeneratorCallback(BaseInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts from a given set of documents."""

    def __init__(
        self,
        api_key=None,
        docs=None,
        prompt=None,
        model_name=None,
        num_random_contexts=1,
        separator_text="\n" + "-" * 80 + "\n\n",
        safety_settings=None,
    ):
        """Initializes the callback with configuration parameters.

        Args:
            api_key (str): API key for the generative AI service.
            docs (List[str]): List of documents to provide context for instruction generation.
            prompt (str, optional): Instruction generation prompt. Defaults to `default_instruction_generation_prompt`.
            model_name (str, optional): Model name to use. Defaults to `default_model_name`.
            num_random_contexts (int, optional): Number of contexts to sample. Defaults to 1.
            separator_text (str, optional): Separator text for merging contexts. Defaults to a dashed line.
            safety_settings (dict, optional): Safety settings for the model. Defaults to `default_safety_settings`.
        """
        assert api_key is not None, "You need to provide an API key"
        assert (
            isinstance(docs, List) and len(docs) >= 1 and isinstance(docs[0], str)
        ), "`docs` must be a list of strings"
        self.docs = docs
        self.prompt = (
            prompt if prompt is not None else default_instruction_generation_prompt
        )
        self.model_name = model_name if model_name is not None else default_model_name
        self.num_random_contexts = (
            num_random_contexts if num_random_contexts >= 1 else 1
        )
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

    def _sample(self):
        """Samples random contexts from the document set.

        Returns:
            List[str]: A list of sampled document strings.
        """
        return random.sample(self.docs, k=self.num_random_contexts)

    def _merge_contexts(self, contexts):
        """Merges multiple contexts into a single string using the separator.

        Args:
            contexts (List[str]): List of context strings.

        Returns:
            str: A single merged context string.
        """
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
