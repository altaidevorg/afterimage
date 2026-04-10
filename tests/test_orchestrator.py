"""Tests for Orchestrator integration with ConversationGenerator."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from afterimage.conversation_generator import ConversationGenerator
from afterimage.orchestrator import Orchestrator
from afterimage.sampling import SamplingStrategy
from afterimage.quality_gate import QualityGate
from afterimage.common import GeneratedInstructions
from afterimage.providers.llm_providers import (
    LLMResponse,
    ChatSession,
    StructuredLLMResponse,
)
from afterimage.types import ConversationEntry, Role


class MockInstructionCallback:
    """Minimal instruction callback for testing."""

    monitor = None

    def set_monitor(self, monitor):
        self.monitor = monitor

    async def acreate_correspondent_prompt(self, respondent_prompt):
        return "You are a curious user."

    async def acall(self, correspondent_prompt):
        return GeneratedInstructions(
            instructions=["First question?"],
            context="",
            context_id="test",
            persona="A curious user",
            persona_generation_depth=0,
        )


class MockChatSession(ChatSession):
    def __init__(self):
        super().__init__()

    async def asend_message(self, message, temperature=0.7, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="mocked response",
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )


class MockLLMProvider:
    async def agenerate_content(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="mocked",
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )

    async def astart_chat(self, **kwargs) -> ChatSession:
        return MockChatSession()


class TestConversationGeneratorWiring:
    """Test that ConversationGenerator correctly wires the new components."""

    def test_has_orchestrator(self):
        from afterimage.providers import llm_providers

        original = llm_providers.LLMFactory.create
        llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())
        try:
            gen = ConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
                instruction_generator_callback=MockInstructionCallback(),
            )
            assert isinstance(gen._orchestrator, Orchestrator)
        finally:
            llm_providers.LLMFactory.create = original

    def test_has_sampling_strategy(self):
        from afterimage.providers import llm_providers

        original = llm_providers.LLMFactory.create
        llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())
        try:
            gen = ConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
                instruction_generator_callback=MockInstructionCallback(),
            )
            assert isinstance(gen._sampling_strategy, SamplingStrategy)
            assert gen._orchestrator.sampling_strategy is gen._sampling_strategy
        finally:
            llm_providers.LLMFactory.create = original

    def test_has_quality_gate(self):
        from afterimage.providers import llm_providers

        original = llm_providers.LLMFactory.create
        llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())
        try:
            gen = ConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
                instruction_generator_callback=MockInstructionCallback(),
            )
            assert isinstance(gen._quality_gate, QualityGate)
            assert gen._quality_gate.is_enabled is False  # no auto_improve
            assert gen._orchestrator.quality_gate is gen._quality_gate
        finally:
            llm_providers.LLMFactory.create = original

    def test_evaluator_syncs_with_quality_gate(self):
        from afterimage.providers import llm_providers

        original = llm_providers.LLMFactory.create
        llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())
        try:
            gen = ConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
                instruction_generator_callback=MockInstructionCallback(),
            )
            assert gen.evaluator is None
            assert gen._quality_gate._evaluator is None

            # Simulate test-style evaluator injection
            fake_evaluator = MagicMock()
            gen.evaluator = fake_evaluator

            assert gen.evaluator is fake_evaluator
            assert gen._quality_gate._evaluator is fake_evaluator
        finally:
            llm_providers.LLMFactory.create = original

    @pytest.mark.asyncio
    async def test_generate_delegates_to_orchestrator(self):
        from afterimage.providers import llm_providers

        original = llm_providers.LLMFactory.create
        llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())
        try:
            gen = ConversationGenerator(
                respondent_prompt="You are a helpful assistant.",
                api_key="mock_key",
                instruction_generator_callback=MockInstructionCallback(),
            )
            # Should complete without error
            await gen.generate(num_dialogs=1)
            convs = gen.storage.load_conversations()
            assert len(convs) >= 1
        finally:
            llm_providers.LLMFactory.create = original


class TestOrchestratorUnit:
    """Test Orchestrator in isolation."""

    def test_init(self):
        sampling = SamplingStrategy()
        gate = QualityGate()
        orch = Orchestrator(sampling_strategy=sampling, quality_gate=gate)
        assert orch.sampling_strategy is sampling
        assert orch.quality_gate is gate
