"""Tests for optional ConversationTurnHooks in ConversationGenerator.go()."""

import pytest
from unittest.mock import MagicMock

from afterimage.conversation_generator import ConversationGenerator
from afterimage.conversation_turn_hooks import ConversationTurnHooks
from afterimage.providers.llm_providers import LLMResponse, ChatSession
from afterimage.types import Role


class _HookRecorder(ConversationTurnHooks):
    def __init__(self):
        self.events: list[tuple[str, ...]] = []

    async def before_correspondent_completion(self, ctx, correspondent_input):
        self.events.append(
            ("bcc", ctx.respondent_turns_completed, correspondent_input[:24])
        )

    async def after_correspondent_completion(self, ctx, user_message):
        self.events.append(("acc", ctx.respondent_turns_completed, user_message))

    async def before_respondent_completion(self, ctx, user_message):
        self.events.append(("brc", ctx.respondent_turns_completed, user_message))

    async def after_respondent_completion(self, ctx, entry):
        self.events.append(("arc", ctx.respondent_turns_completed, entry.content))


class _SeqChat(ChatSession):
    def __init__(self, replies: list[str]):
        super().__init__()
        self._replies = list(replies)
        self._i = 0

    async def asend_message(self, message, temperature=0.7, **kwargs) -> LLMResponse:
        text = self._replies[self._i]
        self._i += 1
        return LLMResponse(
            text=text,
            prompt_token_count=1,
            completion_token_count=1,
            total_token_count=2,
            finish_reason="stop",
            model_name="mock",
            raw_response=None,
        )


class _SeqLLM:
    """Matches :meth:`~afterimage.conversation_generator.ConversationGenerator.create_model` — two ``astart_chat`` calls in order (correspondent, then respondent); ``system_instruction`` is not passed through to ``astart_chat``."""

    def __init__(self, correspondent_replies: list[str], respondent_replies: list[str]):
        self._corr = _SeqChat(correspondent_replies)
        self._resp = _SeqChat(respondent_replies)
        self._astart_order = 0

    async def agenerate_content(self, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(
            text="ok",
            prompt_token_count=1,
            completion_token_count=1,
            total_token_count=2,
            finish_reason="stop",
            model_name="mock",
            raw_response=None,
        )

    async def astart_chat(self, **kwargs) -> ChatSession:
        self._astart_order += 1
        return self._corr if self._astart_order == 1 else self._resp


@pytest.mark.asyncio
async def test_turn_hooks_order_two_turns():
    from afterimage.providers import llm_providers

    corr_user = ["first user", "second user"]
    asst = ["first asst", "second asst"]
    hooks = _HookRecorder()
    original = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=_SeqLLM(corr_user, asst))
    try:
        gen = ConversationGenerator(
            respondent_prompt="You are assistant.",
            api_key="k",
            correspondent_prompt="You are the simulated user.",
            turn_hooks=hooks,
        )
        conv = await gen.go(turns=2)
    finally:
        llm_providers.LLMFactory.create = original

    assert [e.role for e in conv] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]
    assert hooks.events[0][0] == "bcc"
    assert hooks.events[1][0] == "acc"
    assert hooks.events[2][0] == "brc"
    assert hooks.events[3][0] == "arc"
    assert hooks.events[4][0] == "bcc"
    assert hooks.events[5][0] == "acc"
    assert hooks.events[6][0] == "brc"
    assert hooks.events[7][0] == "arc"


@pytest.mark.asyncio
async def test_first_question_skips_before_first_correspondent_ask():
    from afterimage.providers import llm_providers

    hooks = _HookRecorder()
    original = llm_providers.LLMFactory.create
    llm_providers.LLMFactory.create = MagicMock(return_value=_SeqLLM([], ["only asst"]))
    try:
        gen = ConversationGenerator(
            respondent_prompt="You are assistant.",
            api_key="k",
            correspondent_prompt="You are user.",
            turn_hooks=hooks,
        )
        await gen.go(turns=1, first_question="preset")
    finally:
        llm_providers.LLMFactory.create = original

    kinds = [e[0] for e in hooks.events]
    assert "bcc" not in kinds
    assert kinds[0] == "acc"
    assert kinds[1] == "brc"
    assert kinds[2] == "arc"
