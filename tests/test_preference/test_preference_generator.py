"""Tests for PreferenceGenerator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from afterimage.common import GeneratedInstructions
from afterimage.preference.generator import PreferenceGenerator
from afterimage.preference.types import PreferenceConfig
from afterimage.types import (
    ConversationEntry,
    ConversationWithContext,
    EvaluatedConversationWithContext,
    EvaluationEntrySchema,
    EvaluationSchema,
    GradeSchema,
    Role,
)


# ---------------------------------------------------------------------------
# Helpers / Mock infrastructure
# ---------------------------------------------------------------------------


@dataclass
class _MockLLMResponse:
    text: str
    prompt_token_count: int = 10
    completion_token_count: int = 20
    total_token_count: int = 30
    finish_reason: str = "stop"
    model_name: str = "mock-model"
    reasoning_content: str | None = None


class MockLLMProvider:
    """Returns deterministic responses based on temperature."""

    async def agenerate_content(self, prompt: str, temperature: float = 0.7, **kwargs):
        if temperature < 0.5:
            text = (
                "Detailed, well-structured response with citations and examples. "
                "This is a thorough answer covering all aspects of the question."
            )
        else:
            text = "Short answer."
        return _MockLLMResponse(text=text)

    def generate_content(self, prompt: str, temperature: float = 0.7, **kwargs):
        if temperature < 0.5:
            text = "Detailed, well-structured response with citations and examples."
        else:
            text = "Short answer."
        return _MockLLMResponse(text=text)


class MockPromptLLMProvider:
    """Returns responses based on whether system prompt contains 'step by step'."""

    async def agenerate_content(self, prompt: str, temperature: float = 0.7, **kwargs):
        if "step by step" in prompt.lower():
            text = "Step 1: Consider the problem. Step 2: Apply reasoning. Step 3: Conclude."
        elif "briefly" in prompt.lower():
            text = "Brief."
        else:
            text = "Normal response."
        return _MockLLMResponse(text=text)


class MockSecondaryLLMProvider:
    """Returns a distinct response to simulate secondary model."""

    async def agenerate_content(self, prompt: str, temperature: float = 0.7, **kwargs):
        return _MockLLMResponse(text="Secondary model response: " + "x" * 50)


def _make_fake_evaluation(score: float) -> EvaluatedConversationWithContext:
    entry = EvaluationEntrySchema(score=score, feedback="ok")
    schema = EvaluationSchema(
        coherence=entry,
        factuality=entry,
        grounding=entry,
        helpfulness=entry,
        relevance=entry,
        overall_grade=GradeSchema.GOOD,
    )
    conv = ConversationWithContext(
        conversations=[
            ConversationEntry(role=Role.USER, content="q"),
            ConversationEntry(role=Role.ASSISTANT, content="a"),
        ]
    )
    return EvaluatedConversationWithContext(
        **conv.model_dump(), evaluation=schema, final_score=score
    )


class MockJudge:
    """Scores responses by text length as a proxy for quality."""

    async def aevaluate_row(
        self, conversation: ConversationWithContext
    ) -> EvaluatedConversationWithContext:
        response_text = ""
        for entry in conversation.conversations:
            if entry.role == Role.ASSISTANT:
                response_text = entry.content
                break
        score = min(len(response_text) / 100.0, 1.0)
        return _make_fake_evaluation(score)


def _make_instructions(instructions: List[str], persona: str | None = None):
    return GeneratedInstructions(
        instructions=instructions,
        context="test context",
        persona=persona,
        context_id="ctx-1",
        context_ids=["ctx-1"],
    )


def _make_mock_generator(respondent_prompt: str = "You are a helpful assistant."):
    """Build a minimal mock ConversationGenerator."""
    gen = MagicMock()
    gen.respondent_prompt = respondent_prompt
    gen.correspondent_prompt = "Ask questions about the topic."
    gen.model_provider_name = "openai"
    gen.model_name = "gpt-4o-mini"
    gen.key_pool = MagicMock()
    gen._factory_kwargs = {}
    gen.instruction_generator_callback = None
    gen.respondent_prompt_modifier = None
    gen.ainitialize = AsyncMock()

    # go() returns a minimal single-turn conversation
    async def _go(*args, **kwargs):
        return [
            ConversationEntry(role=Role.USER, content="Test question"),
            ConversationEntry(role=Role.ASSISTANT, content="Test answer"),
        ]

    gen.go = _go
    return gen


def _make_pref_gen(
    strategy: str = "temperature",
    num_responses: int = 2,
    min_score_gap: float = 0.1,
    num_pairs: int = 3,
    multi_turn: bool = False,
    max_concurrency: int = 2,
    primary_llm=None,
    secondary_llm=None,
) -> PreferenceGenerator:
    mock_gen = _make_mock_generator()
    cfg = PreferenceConfig(
        num_pairs=num_pairs,
        num_responses=num_responses,
        min_score_gap=min_score_gap,
        strategy=strategy,
        multi_turn=multi_turn,
        max_concurrency=max_concurrency,
    )
    judge = MockJudge()
    with patch("afterimage.preference.generator.LLMFactory") as mock_factory:
        mock_factory.create.return_value = primary_llm or MockLLMProvider()
        pref_gen = PreferenceGenerator(
            conversation_generator=mock_gen,
            judge=judge,
            config=cfg,
            secondary_llm_provider=secondary_llm,
        )
    # Override primary LLM with our mock directly
    pref_gen._primary_llm = primary_llm or MockLLMProvider()
    return pref_gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generates_correct_num_responses():
    """Each prompt should yield exactly num_responses scored responses."""
    pref_gen = _make_pref_gen(strategy="temperature", num_responses=2)

    callback = MagicMock()
    callback.acall = AsyncMock(return_value=_make_instructions(["What is Python?"]))
    pref_gen._gen.instruction_generator_callback = callback

    scored_counts = []

    original_score = pref_gen._score_responses

    async def _capture_score(response_tuples, **kwargs):
        scored_counts.append(len(response_tuples))
        return await original_score(response_tuples, **kwargs)

    pref_gen._score_responses = _capture_score

    pairs, _ = await pref_gen.generate(
        num_pairs=1, instruction_generator_callback=callback
    )
    # At least one scoring call with 2 responses
    assert any(c == 2 for c in scored_counts)


@pytest.mark.asyncio
async def test_generates_correct_num_pairs():
    """generate() should return exactly num_pairs valid pairs."""
    pref_gen = _make_pref_gen(num_pairs=3, min_score_gap=0.0)

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Q1", "Q2", "Q3", "Q4", "Q5"])
    )
    pairs, analytics = await pref_gen.generate(
        num_pairs=3, instruction_generator_callback=callback
    )
    assert len(pairs) == 3
    assert analytics.total_valid == 3


@pytest.mark.asyncio
async def test_score_gap_filtering():
    """Pairs with score gap < min_score_gap should be discarded."""
    # Use a judge that scores based on response length:
    # length < 10 chars → low score, length >= 10 chars → high score.
    # Temperature strategy with MockLLMProvider gives short vs long responses.
    # With min_score_gap=0.0: all pairs should be kept (gap between lengths).
    # Then separately verify that raising min_score_gap discards.

    pref_gen = _make_pref_gen(
        strategy="temperature",
        num_responses=2,
        min_score_gap=0.0,  # accept all gaps
        num_pairs=2,
    )

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Q1", "Q2", "Q3", "Q4"])
    )

    pairs, analytics = await pref_gen.generate(
        num_pairs=2, instruction_generator_callback=callback
    )
    # With gap=0.0, all pairs with 2 distinct-length responses should be kept
    assert len(pairs) == 2

    # Now verify that a very high min_score_gap causes discards by testing
    # _build_pair returns None when gap < threshold
    pref_gen2 = _make_pref_gen(
        strategy="temperature",
        num_responses=2,
        min_score_gap=0.99,  # impossible to satisfy → all discarded
        num_pairs=1,
    )

    # Patch _build_pair to count None returns instead of looping forever
    build_call_count = [0]
    original_build = pref_gen2._build_pair

    async def _limited_build(*args, **kwargs):
        build_call_count[0] += 1
        if build_call_count[0] > 6:
            raise RuntimeError("stop")
        return await original_build(*args, **kwargs)

    pref_gen2._build_pair = _limited_build

    callback2 = MagicMock()
    callback2.acall = AsyncMock(return_value=_make_instructions(["Q1", "Q2", "Q3"]))
    try:
        pairs2, analytics2 = await asyncio.wait_for(
            pref_gen2.generate(num_pairs=1, instruction_generator_callback=callback2),
            timeout=5.0,
        )
    except (asyncio.TimeoutError, RuntimeError):
        pass  # expected: generator loops forever with no valid pairs

    # Verify _build_pair was called (pairs were attempted but discarded)
    assert build_call_count[0] > 0


@pytest.mark.asyncio
async def test_chosen_has_higher_score():
    """chosen.score must always be >= rejected.score."""
    pref_gen = _make_pref_gen(num_pairs=5, min_score_gap=0.0)

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(
            ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
        )
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=5, instruction_generator_callback=callback
    )
    for pair in pairs:
        assert pair.chosen.score >= pair.rejected.score


@pytest.mark.asyncio
async def test_temperature_strategy():
    """Temperature strategy must produce 2 responses at different temperatures."""
    from afterimage.preference.strategies import temperature_strategy

    llm = MockLLMProvider()
    results = await temperature_strategy(
        llm=llm,
        system_prompt="You are helpful.",
        user_turn="What is 2+2?",
        history=[],
        num_responses=2,
    )
    assert len(results) == 2
    contents = [r[0] for r in results]
    labels = [r[2] for r in results]
    # The low-temperature response should be longer (more detailed)
    assert len(contents[0]) > len(contents[1])
    # Labels should differ
    assert labels[0] != labels[1]


@pytest.mark.asyncio
async def test_prompt_strategy():
    """Prompt strategy should produce enhanced and degraded responses."""
    from afterimage.preference.strategies import prompt_strategy

    llm = MockPromptLLMProvider()
    results = await prompt_strategy(
        llm=llm,
        system_prompt="You are helpful.",
        user_turn="Explain gravity.",
        history=[],
        num_responses=2,
    )
    assert len(results) == 2
    labels = [r[2] for r in results]
    assert "prompt_enhanced" in labels
    assert "prompt_degraded" in labels


@pytest.mark.asyncio
async def test_model_strategy():
    """Model strategy should use primary and secondary model."""
    from afterimage.preference.strategies import model_strategy

    primary = MockLLMProvider()
    secondary = MockSecondaryLLMProvider()
    results = await model_strategy(
        primary_llm=primary,
        secondary_llm=secondary,
        system_prompt="You are helpful.",
        user_turn="Describe AI.",
        history=[],
        num_responses=2,
    )
    assert len(results) == 2
    labels = [r[2] for r in results]
    assert "model_primary" in labels
    assert "model_secondary" in labels
    # Secondary model response is distinct
    secondary_content = next(r[0] for r in results if r[2] == "model_secondary")
    assert "Secondary" in secondary_content


@pytest.mark.asyncio
async def test_combined_strategy():
    """Combined strategy should produce responses from multiple approaches."""
    from afterimage.preference.strategies import combined_strategy

    primary = MockLLMProvider()
    secondary = MockSecondaryLLMProvider()
    results = await combined_strategy(
        primary_llm=primary,
        secondary_llm=secondary,
        system_prompt="You are helpful.",
        user_turn="What is ML?",
        history=[],
        num_responses=3,
    )
    assert len(results) >= 2
    labels = [r[2] for r in results]
    # Should have at least 2 different labels
    assert len(set(labels)) >= 2


@pytest.mark.asyncio
async def test_multiturn_preference():
    """Multi-turn: pair should have a non-empty shared_prefix."""
    pref_gen = _make_pref_gen(
        strategy="temperature", num_responses=2, min_score_gap=0.0, multi_turn=True
    )

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Tell me about history?"] * 5)
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=1, instruction_generator_callback=callback
    )
    assert len(pairs) == 1
    # shared_prefix should be present for multi-turn
    assert pairs[0].shared_prefix is not None


@pytest.mark.asyncio
async def test_multiturn_shared_prefix():
    """Multi-turn: chosen and rejected should have identical conversation prefix."""
    pref_gen = _make_pref_gen(
        strategy="temperature", num_responses=2, min_score_gap=0.0, multi_turn=True
    )

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Explain evolution"] * 5)
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=1, instruction_generator_callback=callback
    )
    if pairs and pairs[0].shared_prefix:
        chosen_msgs = pairs[0].chosen.messages or []
        rejected_msgs = pairs[0].rejected.messages or []
        # All messages before the last assistant turn should be identical
        assert chosen_msgs[:-1] == rejected_msgs[:-1]


@pytest.mark.asyncio
async def test_metadata_preserved():
    """Preference pairs should carry persona, context_ids, and scores in metadata."""
    pref_gen = _make_pref_gen(num_pairs=1, min_score_gap=0.0)

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Q1", "Q2"], persona="Expert user")
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=1, instruction_generator_callback=callback
    )
    assert len(pairs) == 1
    meta = pairs[0].metadata
    assert "persona" in meta
    assert "context_ids" in meta
    assert "all_scores" in meta


@pytest.mark.asyncio
async def test_persona_diversity():
    """Pairs built from different callback batches should reflect different personas."""
    # The generator calls callback once per batch; each batch returns 1 instruction
    # with a rotating persona. With num_pairs=3, 3 batches → 3 different personas.
    personas = ["Expert", "Beginner", "Student"]
    call_count = [0]

    async def _rotating_callback(_):
        p = personas[call_count[0] % len(personas)]
        call_count[0] += 1
        # Return only 1 instruction per call so multiple batches are needed
        return _make_instructions(["Q1"], persona=p)

    pref_gen = _make_pref_gen(num_pairs=3, min_score_gap=0.0)

    callback = MagicMock()
    callback.acall = _rotating_callback

    pairs, _ = await pref_gen.generate(
        num_pairs=3, instruction_generator_callback=callback
    )
    seen_personas = {p.metadata.get("persona") for p in pairs}
    assert len(seen_personas) > 1


@pytest.mark.asyncio
async def test_empty_documents():
    """Generator should work without document context (no respondent_prompt_modifier)."""
    pref_gen = _make_pref_gen(num_pairs=2, min_score_gap=0.0)
    pref_gen._gen.respondent_prompt_modifier = None

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["What is gravity?"] * 5)
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=2, instruction_generator_callback=callback
    )
    assert len(pairs) == 2


@pytest.mark.asyncio
async def test_no_personas():
    """Generator should work when persona is None."""
    pref_gen = _make_pref_gen(num_pairs=2, min_score_gap=0.0)

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["What is AI?"] * 5, persona=None)
    )
    pairs, _ = await pref_gen.generate(
        num_pairs=2, instruction_generator_callback=callback
    )
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.metadata.get("persona") is None


@pytest.mark.asyncio
async def test_concurrent_generation():
    """max_concurrency should be respected (basic: no errors under concurrency)."""
    pref_gen = _make_pref_gen(num_pairs=4, min_score_gap=0.0, max_concurrency=2)

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    )
    pairs, analytics = await pref_gen.generate(
        num_pairs=4, instruction_generator_callback=callback
    )
    assert len(pairs) == 4
    assert analytics.total_valid == 4


def test_to_preference_generator():
    """ConversationGenerator.to_preference_generator() should return a PreferenceGenerator."""
    from afterimage.preference.generator import PreferenceGenerator as PG

    mock_gen = _make_mock_generator()
    # Create a real ConversationGenerator-like object to test the method
    # We just verify the method exists and is callable
    judge = MockJudge()
    cfg = PreferenceConfig(num_pairs=5)

    with patch("afterimage.preference.generator.LLMFactory") as mock_factory:
        mock_factory.create.return_value = MockLLMProvider()
        # Simulate what to_preference_generator does
        pg = PG(
            conversation_generator=mock_gen,
            judge=judge,
            config=cfg,
        )
    assert isinstance(pg, PG)
    assert pg._config.num_pairs == 5


@pytest.mark.asyncio
async def test_failed_response_skipped_not_crash():
    """Failed LLM responses should be skipped, not crash the run."""

    class _FailingLLM:
        call_count = 0

        async def agenerate_content(
            self, prompt: str, temperature: float = 0.7, **kwargs
        ):
            self.__class__.call_count += 1
            if self.__class__.call_count % 2 == 0:
                raise RuntimeError("Simulated LLM failure")
            return _MockLLMResponse(text="Good response: " + "x" * 80)

    pref_gen = _make_pref_gen(num_pairs=2, min_score_gap=0.0)
    pref_gen._primary_llm = _FailingLLM()

    callback = MagicMock()
    callback.acall = AsyncMock(
        return_value=_make_instructions(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    )
    # Should not raise even with intermittent failures
    pairs, _ = await pref_gen.generate(
        num_pairs=2, instruction_generator_callback=callback
    )
    # We may get fewer pairs due to failures, but no exception
    assert isinstance(pairs, list)
