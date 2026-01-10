import asyncio
import logging

from .common import GeneratedInstructions
from .monitoring import GenerationMonitor
from .types import (
    EvaluatedConversationWithContext,
    GeneratedResponsePrompt,
    GenerationState,
)

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Intended to serve as the base class for all generator classes"""

    def log_correspondent_prompt(self, correspondent_prompt: str | None) -> None:
        """Log the correspondent prompt in a standardized format.

        Args:
            correspondent_prompt: The correspondent prompt to log, or None if not set.
        """
        if correspondent_prompt is None:
            logger.info("Correspondent prompt: Not set (will be generated on demand)")
        else:
            # Log the prompt, truncating if too long for readability
            prompt_preview = (
                correspondent_prompt[:200] + "..."
                if len(correspondent_prompt) > 200
                else correspondent_prompt
            )
            logger.info(
                f"Correspondent prompt initialized (length: {len(correspondent_prompt)}): {prompt_preview}"
            )

    async def ainitialize(self, instruction_generator_callback=None):
        """Initializes the generator by creating the correspondent prompt if it doesn't exist."""
        if self.correspondent_prompt is None:
            # Use provided callback if given, otherwise use instance attribute
            callback = (
                instruction_generator_callback or self.instruction_generator_callback
            )
            # Try to use callback first if available
            if callback is not None:
                if hasattr(callback, "acreate_correspondent_prompt"):
                    created_prompt = await callback.acreate_correspondent_prompt(
                        self.respondent_prompt
                    )
                else:
                    created_prompt = await asyncio.to_thread(
                        callback.create_correspondent_prompt, self.respondent_prompt
                    )
                if created_prompt is not None:
                    self.correspondent_prompt = created_prompt
                    self.log_correspondent_prompt(self.correspondent_prompt)
                    return

            # Fallback to generator's method
            self.correspondent_prompt = await self.create_correspondent_prompt(
                self.respondent_prompt
            )
            self.log_correspondent_prompt(self.correspondent_prompt)

    def initialize(self, instruction_generator_callback=None):
        """Initializes the generator by creating the correspondent prompt if it doesn't exist."""
        if self.correspondent_prompt is None:
            # Use provided callback if given, otherwise use instance attribute
            callback = (
                instruction_generator_callback or self.instruction_generator_callback
            )
            # Try to use callback first if available
            if callback is not None:
                created_prompt = callback.create_correspondent_prompt(
                    self.respondent_prompt
                )
                if created_prompt is not None:
                    self.correspondent_prompt = created_prompt
                    self.log_correspondent_prompt(self.correspondent_prompt)
                    return

            # Fallback to generator's method
            self.correspondent_prompt = self.create_correspondent_prompt(
                self.respondent_prompt
            )
            self.log_correspondent_prompt(self.correspondent_prompt)

    def load_conversations(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[EvaluatedConversationWithContext]:
        return self.storage.load_conversations(limit=limit, offset=offset)

    def _ensure_monitor(self):
        """Ensures that a monitor exists, creating a default one if necessary."""
        if self.monitor is None:
            self.monitor = GenerationMonitor()


class BaseStoppingCallback:
    """Base class for callbacks that decide when to stop generation."""

    async def should_stop(self, state: GenerationState) -> bool:
        """Return True if generation should stop.

        Args:
            state: The current state of the generation process.
        """
        raise NotImplementedError


class BaseInstructionGeneratorCallback:
    """Intended to serve as the base class for all custom instruction generator callbacks"""

    def __call__(self, original_prompt: str) -> GeneratedInstructions:
        instructions = self.generate(original_prompt)
        assert isinstance(instructions, GeneratedInstructions), (
            f".generate() method should return an instance of GeneratedInstructions, but found {type(instructions)}"
        )

        return instructions

    def generate(self, original_prompt) -> GeneratedInstructions:
        raise NotImplementedError

    async def acall(self, original_prompt: str) -> GeneratedInstructions:
        instructions = await self.agenerate(original_prompt)
        assert isinstance(instructions, GeneratedInstructions), (
            f".agenerate() method should return an instance of GeneratedInstructions, but found {type(instructions)}"
        )

        return instructions

    async def agenerate(self, original_prompt) -> GeneratedInstructions:
        raise NotImplementedError

    def create_correspondent_prompt(self, respondent_prompt: str) -> str | None:
        """Create a correspondent prompt based on the respondent prompt.

        This method can be overridden by subclasses to customize correspondent prompt creation.
        By default, returns None, which means the conversation generator should handle it.

        Args:
            respondent_prompt: The prompt for the respondent (assistant)

        Returns:
            str: The correspondent prompt, or None if the generator should handle it
        """
        return None

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        """Create a correspondent prompt based on the respondent prompt asynchronously.

        This method can be overridden by subclasses to customize correspondent prompt creation.
        By default, returns None, which means the conversation generator should handle it.

        Args:
            respondent_prompt: The prompt for the respondent (assistant)

        Returns:
            str: The correspondent prompt, or None if the generator should handle it
        """
        return None


class BaseRespondentPromptModifierCallback:
    """Intended to serve as the base class for all custom respondent prompt modifier callbacks"""

    def __call__(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        modified_prompt = self.generate(respondent_prompt, context, instruction)
        assert isinstance(modified_prompt, GeneratedResponsePrompt), (
            f".generate() method is expected to return an instance of `GeneratedRespondentPrompt`, but found {type(modified_prompt)}"
        )

        return modified_prompt

    def generate(
        self, respondent_prompt, context, instruction
    ) -> GeneratedResponsePrompt:
        raise NotImplementedError

    async def acall(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        modified_prompt = await self.agenerate(respondent_prompt, context, instruction)
        assert isinstance(modified_prompt, GeneratedResponsePrompt), (
            f".agenerate() method is expected to return an instance of `GeneratedRespondentPrompt`, but found {type(modified_prompt)}"
        )

        return modified_prompt

    async def agenerate(
        self, respondent_prompt, context, instruction
    ) -> GeneratedResponsePrompt:
        raise NotImplementedError

    def _maybe_augment_context(self, instruction: str, current_context: str) -> str:
        if hasattr(self, "augment_context"):
            return self.augment_context(instruction, current_context)
        else:
            return current_context
