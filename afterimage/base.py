from .common import GeneratedInstructions


class BaseGenerator:
    """Intended to serve as the base class for all generator classes"""

    pass


class BaseInstructionGeneratorCallback:
    """Intended to serve as the base class for all custom instruction generator callbacks"""

    def __call__(self, original_prompt: str) -> GeneratedInstructions:
        instructions = self.generate(original_prompt)
        assert isinstance(
            instructions, GeneratedInstructions
        ), f".generate() method should return an instance of GeneratedInstructions, but found {type(instructions)}"

        return instructions

    def generate(self, original_prompt) -> GeneratedInstructions:
        raise NotImplementedError


class BaseRespondentPromptModifierCallback:
    """Intended to serve as the base class for all custom respondent prompt modifier callbacks"""

    def __call__(self, respondent_prompt: str, context: str, instruction: str) -> str:
        modified_prompt = self.generate(respondent_prompt, context, instruction)
        assert isinstance(
            modified_prompt, str
        ), f".generate() method is expected to return an str, but found {type(modified_prompt)}"

        return modified_prompt

    def generate(self, respondent_prompt, context, instruction) -> str:
        raise NotImplementedError

    def _maybe_augment_context(self, instruction: str, current_context: str) -> str:
        if hasattr(self, "augment_context"):
            return self.augment_context(instruction, current_context)
        else:
            return current_context
