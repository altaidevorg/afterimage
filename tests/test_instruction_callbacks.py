import unittest
from unittest.mock import MagicMock
import asyncio

from afterimage.callbacks.instruction_generator_callbacks import (
    ContextualInstructionGeneratorCallback,
    PersonaInstructionGeneratorCallback,
)
from afterimage.types import Document


class MockLLM:
    def generate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )

    async def agenerate_structured(self, prompt, schema):
        return MagicMock(
            parsed=MagicMock(instructions=["Test instruction"]),
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock_model",
        )


class TestInstructionGeneratorCallbacks(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MockLLM()
        self.mock_llm_factory = MagicMock()
        self.mock_llm_factory.create.return_value = self.mock_llm

        # Patch LLMFactory
        from afterimage.callbacks.instruction_generator_callbacks import LLMFactory

        self.original_create = LLMFactory.create
        LLMFactory.create = self.mock_llm_factory.create

        self.documents = [
            Document(id="doc1", text="Context 1", personas=[]),
        ]

    def tearDown(self):
        from afterimage.callbacks.instruction_generator_callbacks import LLMFactory

        LLMFactory.create = self.original_create

    def test_contextual_callback_generate(self):
        callback = ContextualInstructionGeneratorCallback(
            api_key="test_key", documents=self.documents, num_random_contexts=1
        )
        result = callback.generate("Test prompt")
        self.assertEqual(result.instructions, ["Test instruction"])
        self.assertIn("Context 1", result.context)
        self.assertEqual(result.context_id, "doc1")

    def test_contextual_callback_agenerate(self):
        async def run_test():
            callback = ContextualInstructionGeneratorCallback(
                api_key="test_key", documents=self.documents, num_random_contexts=1
            )
            result = await callback.agenerate("Test prompt")
            self.assertEqual(result.instructions, ["Test instruction"])
            self.assertIn("Context 1", result.context)
            self.assertEqual(result.context_id, "doc1")

        asyncio.run(run_test())

    def test_persona_callback_generate(self):
        callback = PersonaInstructionGeneratorCallback(
            api_key="test_key",
            documents=self.documents,  # These docs have empty personas, so it should default to "A curious user" or similar if logic dictates, but wait, my doc definition above has empty list.
            # In the code: selected_persona = random.choice(all_personas) if all_personas else None
            # if persona is None: persona = "A curious user"
            num_random_contexts=1,
        )
        result = callback.generate("Test prompt {persona}")
        self.assertEqual(result.instructions, ["Test instruction"])
        self.assertEqual(result.persona, "A curious user")

    def test_persona_callback_agenerate(self):
        async def run_test():
            callback = PersonaInstructionGeneratorCallback(
                api_key="test_key", documents=self.documents, num_random_contexts=1
            )
            result = await callback.agenerate("Test prompt {persona}")
            self.assertEqual(result.instructions, ["Test instruction"])
            self.assertEqual(result.persona, "A curious user")

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
