"""Tests for ConversationGenerator."""

from unittest.mock import MagicMock

import pytest

from afterimage.common import GeneratedInstructions
from afterimage.conversation_generator import ConversationGenerator
from afterimage.types import (
    ConversationEntry,
    EvaluatedConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GradeSchema,
    Role,
)


class MockInstructionCallback:
    monitor = None

    def set_monitor(self, monitor):
        self.monitor = monitor

    def create_correspondent_prompt(self, respondent_prompt):
        return "You are a curious user."

    def __call__(self, correspondent_prompt):
        return GeneratedInstructions(
            instructions=["First question?"],
            context="",
            context_id="test",
            persona="A curious user",
            persona_generation_depth=2,
        )


def make_evaluation(grade: GradeSchema) -> EvaluationSchema:
    return EvaluationSchema(
        coherence=EvaluationEntrySchema(feedback="ok", score=1.0),
        factuality=EvaluationEntrySchema(feedback="ok", score=1.0),
        grounding=EvaluationEntrySchema(feedback="ok", score=1.0),
        helpfulness=EvaluationEntrySchema(feedback="ok", score=1.0),
        relevance=EvaluationEntrySchema(feedback="ok", score=1.0),
        overall_grade=grade,
    )


@pytest.mark.filterwarnings(
    "ignore:This synchronous implementation is deprecated:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:A correspondent prompt will be automatically created because you did not pass one.:UserWarning"
)
def test_conversation_generator_rebuilds_row_after_evaluator_retry():
    generator = ConversationGenerator(
        respondent_prompt="You are a helpful assistant.",
        api_key="mock_key",
        instruction_generator_callback=MockInstructionCallback(),
        auto_improve=False,
    )

    first_conversation = [
        ConversationEntry(role=Role.USER, content="first user"),
        ConversationEntry(role=Role.ASSISTANT, content="first assistant"),
    ]
    second_conversation = [
        ConversationEntry(role=Role.USER, content="second user"),
        ConversationEntry(role=Role.ASSISTANT, content="second assistant"),
    ]

    generator.go = MagicMock(side_effect=[first_conversation, second_conversation])

    class FakeEvaluator:
        def __init__(self):
            self.seen_rows = []

        def evaluate_row(self, row):
            self.seen_rows.append(row)
            if len(self.seen_rows) == 1:
                return EvaluatedConversationWithContext(
                    **row.model_dump(),
                    evaluation=make_evaluation(GradeSchema.BAD),
                )
            return EvaluatedConversationWithContext(
                **row.model_dump(),
                evaluation=make_evaluation(GradeSchema.GOOD),
            )

    evaluator = FakeEvaluator()
    generator.evaluator = evaluator

    rows = generator.generate_single(
        i=0,
        count=1,
        max_turns=1,
        seed_questions=[],
        add_examples=False,
        num_random_examples=0,
        generation_examples_delay=0,
        check_for_near_duplicates=False,
        instruction_generator_callback=generator.instruction_generator_callback,
        respondent_prompt_modifier=None,
    )

    assert generator.go.call_count == 2
    assert evaluator.seen_rows[0].conversations[0].content == "first user"
    assert evaluator.seen_rows[1].conversations[0].content == "second user"
    assert rows[0].conversations[0].content == "second user"
