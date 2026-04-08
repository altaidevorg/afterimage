"""Tests for AsyncConversationGenerator."""
import pytest
from unittest.mock import MagicMock, AsyncMock

from afterimage.async_conversation_generator import AsyncConversationGenerator
from afterimage.evaluation.evaluators import LLMJudgeStructuredOutput
from afterimage.providers.llm_providers import (
    LLMResponse,
    ChatSession,
    StructuredLLMResponse,
)
from afterimage.common import GeneratedInstructions
from afterimage.types import (
    ConversationEntry,
    EvaluatedConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GeneratedResponsePrompt,
    GradeSchema,
    Role,
)


class MockInstructionCallback:
    """Minimal instruction callback that returns one instruction."""

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
            persona_generation_depth=2,
        )


class MockChatSession(ChatSession):
    def __init__(self):
        super().__init__()
        self.history = []

    async def asend_message(self, message, temperature=0.7, **kwargs) -> LLMResponse:
        self.history.append(message)
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
            text="mocked correspondent prompt",
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )

    async def agenerate_structured(self, prompt, schema, **kwargs):
        parsed = LLMJudgeStructuredOutput(
            scores=[0.9], feedback="ok", needs_improvement=False
        )
        return StructuredLLMResponse(
            text="{}",
            parsed=parsed,
            prompt_token_count=10,
            completion_token_count=5,
            total_token_count=15,
            finish_reason="stop",
            model_name="mock_model",
            raw_response=None,
        )

    async def astart_chat(self, **kwargs) -> ChatSession:
        return MockChatSession()


@pytest.mark.asyncio
async def test_async_conversation_generator_generate():
    from afterimage.providers import llm_providers

    original_create = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

    try:
        generator = AsyncConversationGenerator(
            respondent_prompt="You are a helpful assistant.",
            api_key="mock_key",
            instruction_generator_callback=MockInstructionCallback(),
        )
        await generator.generate(num_dialogs=1)
    finally:
        llm_providers.LLMFactory.create = original_create


@pytest.mark.asyncio
async def test_async_conversation_generator_includes_persona_generation_depth_metadata():
    from afterimage.providers import llm_providers

    original_create = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

    try:
        generator = AsyncConversationGenerator(
            respondent_prompt="You are a helpful assistant.",
            api_key="mock_key",
            instruction_generator_callback=MockInstructionCallback(),
        )

        rows = []
        async for row in generator.generate_single(
            max_turns=1,
            instruction_generator_callback=generator.instruction_generator_callback,
        ):
            rows.append(row)
    finally:
        llm_providers.LLMFactory.create = original_create

    assert rows[0].metadata["persona_generation_depth"] == 2


@pytest.mark.asyncio
async def test_async_conversation_generator_rebuilds_row_after_evaluator_retry():
    from afterimage.providers import llm_providers

    original_create = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

    try:
        generator = AsyncConversationGenerator(
            respondent_prompt="You are a helpful assistant.",
            api_key="mock_key",
            instruction_generator_callback=MockInstructionCallback(),
        )

        first_conversation = [
            ConversationEntry(role=Role.USER, content="first user"),
            ConversationEntry(role=Role.ASSISTANT, content="first assistant"),
        ]
        second_conversation = [
            ConversationEntry(role=Role.USER, content="second user"),
            ConversationEntry(role=Role.ASSISTANT, content="second assistant"),
        ]

        generator.go = AsyncMock(side_effect=[first_conversation, second_conversation])

        class FakeEvaluator:
            def __init__(self):
                self.seen_rows = []

            async def aevaluate_row(self, row):
                self.seen_rows.append(row)
                if len(self.seen_rows) == 1:
                    return EvaluatedConversationWithContext(
                        **row.model_dump(),
                        evaluation=EvaluationSchema(
                            coherence=EvaluationEntrySchema(feedback="bad", score=0.0),
                            factuality=EvaluationEntrySchema(feedback="bad", score=0.0),
                            grounding=EvaluationEntrySchema(feedback="bad", score=0.0),
                            helpfulness=EvaluationEntrySchema(feedback="bad", score=0.0),
                            relevance=EvaluationEntrySchema(feedback="bad", score=0.0),
                            overall_grade=GradeSchema.BAD,
                        ),
                    )
                return EvaluatedConversationWithContext(
                    **row.model_dump(),
                    evaluation=EvaluationSchema(
                        coherence=EvaluationEntrySchema(feedback="ok", score=1.0),
                        factuality=EvaluationEntrySchema(feedback="ok", score=1.0),
                        grounding=EvaluationEntrySchema(feedback="ok", score=1.0),
                        helpfulness=EvaluationEntrySchema(feedback="ok", score=1.0),
                        relevance=EvaluationEntrySchema(feedback="ok", score=1.0),
                        overall_grade=GradeSchema.GOOD,
                    ),
                )

        evaluator = FakeEvaluator()
        generator.evaluator = evaluator

        rows = []
        async for row in generator.generate_single(
            max_turns=1,
            instruction_generator_callback=generator.instruction_generator_callback,
        ):
            rows.append(row)
    finally:
        llm_providers.LLMFactory.create = original_create

    assert generator.go.await_count == 2
    assert evaluator.seen_rows[0].conversations[0].content == "first user"
    assert evaluator.seen_rows[1].conversations[0].content == "second user"
    assert rows[0].conversations[0].content == "second user"


@pytest.mark.asyncio
async def test_async_conversation_generator_retries_with_modified_respondent_prompt():
    from afterimage.providers import llm_providers

    original_create = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=MockLLMProvider())

    try:
        generator = AsyncConversationGenerator(
            respondent_prompt="You are a helpful assistant.",
            api_key="mock_key",
            instruction_generator_callback=MockInstructionCallback(),
        )

        first_conversation = [
            ConversationEntry(role=Role.USER, content="first user"),
            ConversationEntry(role=Role.ASSISTANT, content="first assistant"),
        ]
        second_conversation = [
            ConversationEntry(role=Role.USER, content="second user"),
            ConversationEntry(role=Role.ASSISTANT, content="second assistant"),
        ]
        generator.go = AsyncMock(side_effect=[first_conversation, second_conversation])

        class FakePromptModifier:
            async def acall(self, respondent_prompt, context, instruction):
                return GeneratedResponsePrompt(
                    prompt="MODIFIED PROMPT",
                    context="extra",
                    metadata={},
                )

        class FakeEvaluator:
            def __init__(self):
                self.calls = 0

            async def aevaluate_row(self, row):
                self.calls += 1
                grade = GradeSchema.BAD if self.calls == 1 else GradeSchema.GOOD
                return EvaluatedConversationWithContext(
                    **row.model_dump(),
                    evaluation=EvaluationSchema(
                        coherence=EvaluationEntrySchema(feedback="ok", score=1.0),
                        factuality=EvaluationEntrySchema(feedback="ok", score=1.0),
                        grounding=EvaluationEntrySchema(feedback="ok", score=1.0),
                        helpfulness=EvaluationEntrySchema(feedback="ok", score=1.0),
                        relevance=EvaluationEntrySchema(feedback="ok", score=1.0),
                        overall_grade=grade,
                    ),
                )

        generator.evaluator = FakeEvaluator()

        async for _ in generator.generate_single(
            max_turns=1,
            instruction_generator_callback=generator.instruction_generator_callback,
            respondent_prompt_modifier=FakePromptModifier(),
        ):
            pass
    finally:
        llm_providers.LLMFactory.create = original_create

    first_call = generator.go.await_args_list[0].kwargs["respondent_prompt"]
    second_call = generator.go.await_args_list[1].kwargs["respondent_prompt"]
    assert first_call == "MODIFIED PROMPT"
    assert second_call == "MODIFIED PROMPT"
