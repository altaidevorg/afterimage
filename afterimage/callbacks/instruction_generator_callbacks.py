"""Backward-compatible import path for instruction generator callbacks.

Implementations live in :mod:`afterimage.callbacks.instruction_generators`.
"""

from ..providers.llm_providers import LLMFactory  # noqa: F401

from .instruction_generators import (  # noqa: F401
    ContextualInstructionGeneratorCallback,
    InstructionsSchema,
    PersonaCandidate,
    PersonaInstructionGeneratorCallback,
    PersonaSelectionState,
    SimpleInstructionGeneratorCallback,
    ToolCallingInstructionGeneratorCallback,
)
