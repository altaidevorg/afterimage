import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from afterimage.persona_generator import PersonaGenerator
from afterimage.providers import LLMProvider
from afterimage.storage import JSONLStorage
from afterimage.monitoring import GenerationMonitor
from afterimage.types import Document

class TestPersonaGenerator(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        self.llm_provider_mock = MagicMock(spec=LLMProvider)
        self.storage_mock = MagicMock(spec=JSONLStorage)
        self.storage_mock.asave_documents = AsyncMock()
        self.monitor_mock = MagicMock(spec=GenerationMonitor)

        self.persona_generator = PersonaGenerator(
            api_key=self.api_key,
            storage=self.storage_mock,
            monitor=self.monitor_mock
        )

    def test_generate_success(self):
        # Arrange
        test_text = "Sample text"
        mock_response = MagicMock()
        mock_response.text = "Persona 1: A developer.\nPersona 2: A writer."
        self.llm_provider_mock.generate_content.return_value = mock_response

        with unittest.mock.patch('afterimage.persona_generator.LLMFactory') as mock_llm_factory:
            mock_llm = mock_llm_factory.create.return_value
            mock_llm.generate_content.return_value = mock_response

            # Act
            personas = self.persona_generator.generate_from_text(test_text)

            # Assert
            self.assertEqual(len(personas), 2)
            self.assertEqual(personas[0], "A developer.")
            mock_llm.generate_content.assert_called_once()
        self.monitor_mock.track_generation.assert_called_once()
        args, kwargs = self.monitor_mock.track_generation.call_args
        self.assertTrue(kwargs['success'])
        self.assertEqual(kwargs['metadata']['operation'], 'text_to_persona_generation')

    def test_generate_for_documents_batching(self):
        # This is a more complex test to verify concurrency, so we'll simplify
        # by checking the calls were made.
        async def run_test():
            # Arrange
            docs = ["doc1", "doc2", "doc3"]
            mock_response = MagicMock()
            mock_response.text = "Persona 1: A persona."
            
            # Make the async generate_async mockable
            self.persona_generator.agenerate_from_text = AsyncMock(return_value=["A persona."])

            # Act
            await self.persona_generator.generate_from_documents(docs)

            # Assert
            self.assertEqual(self.persona_generator.agenerate_from_text.call_count, 3)
            self.assertEqual(self.storage_mock.asave_documents.call_count, 3)
            # Check if a Document was passed
            args, _ = self.storage_mock.asave_documents.call_args
            self.assertIsInstance(args[0][0], Document)

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
