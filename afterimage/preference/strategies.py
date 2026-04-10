"""Variation strategies for generating diverse (chosen, rejected) response pairs."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..providers import LLMProvider
    from .types import ScoredResponse

logger = logging.getLogger(__name__)


def _make_messages(system_prompt: str, history: list, user_turn: str) -> list:
    """Build an OpenAI-style messages list."""
    msgs = [{"role": "system", "content": system_prompt}]
    for entry in history:
        msgs.append({"role": entry["role"], "content": entry["content"]})
    msgs.append({"role": "user", "content": user_turn})
    return msgs


async def _generate_single_response(
    llm: "LLMProvider",
    messages: list,
    temperature: float,
) -> str:
    """Call the LLM and return the response text."""
    # Build a single prompt string from messages for providers that need it,
    # but also support providers that accept a messages list via agenerate_content.
    # We use the last user message as the prompt and pass history via system.
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"[System]: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt = "\n\n".join(prompt_parts)

    response = await llm.agenerate_content(prompt=prompt, temperature=temperature)
    return response.text


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


async def temperature_strategy(
    llm: "LLMProvider",
    system_prompt: str,
    user_turn: str,
    history: list,
    num_responses: int,
) -> list:
    """Generate responses at linearly-spaced temperatures.

    Low temperature → focused → usually higher quality (chosen).
    High temperature → creative → more variable quality.

    Returns list of (content, temperature, label) tuples.
    """
    if num_responses < 2:
        num_responses = 2

    # Spread temperatures from 0.1 to 0.9
    step = 0.8 / (num_responses - 1) if num_responses > 1 else 0.0
    temperatures = [0.1 + i * step for i in range(num_responses)]
    labels = [f"temperature_{temp:.2f}" for temp in temperatures]

    messages = _make_messages(system_prompt, history, user_turn)

    tasks = [_generate_single_response(llm, messages, temp) for temp in temperatures]
    contents = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for content, temp, label in zip(contents, temperatures, labels):
        if isinstance(content, Exception):
            logger.warning("Temperature strategy response failed: %s", content)
            continue
        results.append((content, temp, label))
    return results


async def prompt_strategy(
    llm: "LLMProvider",
    system_prompt: str,
    user_turn: str,
    history: list,
    num_responses: int,
) -> list:
    """Generate responses with enhanced vs. degraded system prompts.

    Enhanced: adds "Think step by step. Provide a detailed, well-structured answer."
    Degraded: adds "Answer very briefly. Keep it short."

    Returns list of (content, temperature, label) tuples.
    """
    enhanced_prompt = (
        system_prompt
        + "\n\nThink step by step. Provide a detailed, well-structured answer with examples where helpful."
    )
    degraded_prompt = (
        system_prompt
        + "\n\nAnswer very briefly. Keep your response as short as possible."
    )

    prompts_and_labels = [
        (enhanced_prompt, "prompt_enhanced"),
        (degraded_prompt, "prompt_degraded"),
    ]
    # If more responses needed, alternate
    while len(prompts_and_labels) < num_responses:
        prompts_and_labels.append(
            (enhanced_prompt, f"prompt_enhanced_{len(prompts_and_labels)}")
        )

    tasks = []
    for sys_p, _ in prompts_and_labels[:num_responses]:
        msgs = _make_messages(sys_p, history, user_turn)
        tasks.append(_generate_single_response(llm, msgs, 0.7))

    contents = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for content, (_, label) in zip(contents, prompts_and_labels[:num_responses]):
        if isinstance(content, Exception):
            logger.warning("Prompt strategy response failed: %s", content)
            continue
        results.append((content, 0.7, label))
    return results


async def model_strategy(
    primary_llm: "LLMProvider",
    secondary_llm: "LLMProvider",
    system_prompt: str,
    user_turn: str,
    history: list,
    num_responses: int,
) -> list:
    """Generate responses using different models.

    Alternates between primary and secondary models.

    Returns list of (content, temperature, label) tuples.
    """
    messages = _make_messages(system_prompt, history, user_turn)

    tasks = []
    labels = []
    for i in range(num_responses):
        if i % 2 == 0:
            tasks.append(_generate_single_response(primary_llm, messages, 0.7))
            labels.append("model_primary")
        else:
            tasks.append(_generate_single_response(secondary_llm, messages, 0.7))
            labels.append("model_secondary")

    contents = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for content, label in zip(contents, labels):
        if isinstance(content, Exception):
            logger.warning("Model strategy response failed: %s", content)
            continue
        results.append((content, 0.7, label))
    return results


async def combined_strategy(
    primary_llm: "LLMProvider",
    secondary_llm: Optional["LLMProvider"],
    system_prompt: str,
    user_turn: str,
    history: list,
    num_responses: int,
) -> list:
    """Mix temperature + prompt + model strategies.

    Returns list of (content, temperature, label) tuples.
    """
    enhanced_prompt = (
        system_prompt
        + "\n\nThink step by step. Provide a detailed, well-structured answer."
    )
    degraded_prompt = (
        system_prompt
        + "\n\nAnswer very briefly. Keep your response as short as possible."
    )

    configs = [
        (primary_llm, enhanced_prompt, 0.2, "combined_enhanced_low_temp"),
        (primary_llm, degraded_prompt, 0.9, "combined_degraded_high_temp"),
    ]
    if secondary_llm is not None:
        configs.append((secondary_llm, system_prompt, 0.7, "combined_secondary_model"))
    # Fill up if more needed
    extra_temps = [0.4, 0.6, 0.8]
    idx = 0
    while len(configs) < num_responses:
        temp = extra_temps[idx % len(extra_temps)]
        configs.append(
            (primary_llm, system_prompt, temp, f"combined_temp_{temp:.1f}_{idx}")
        )
        idx += 1

    tasks = []
    for llm, sys_p, temp, _ in configs[:num_responses]:
        msgs = _make_messages(sys_p, history, user_turn)
        tasks.append(_generate_single_response(llm, msgs, temp))

    contents = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for content, (_, _, temp, label) in zip(contents, configs[:num_responses]):
        if isinstance(content, Exception):
            logger.warning("Combined strategy response failed: %s", content)
            continue
        results.append((content, temp, label))
    return results
