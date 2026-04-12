"""Optional hooks around correspondent / respondent steps in :meth:`~afterimage.conversation_generator.ConversationGenerator.go`."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ConversationEntry


@dataclass(frozen=True)
class ConversationTurnContext:
    """Immutable snapshot for a hook invocation."""

    planned_turns: int
    """``turns`` argument to :meth:`~afterimage.conversation_generator.ConversationGenerator.go`."""
    respondent_turns_completed: int
    """Assistant messages already present in ``conversation`` when the hook runs."""
    conversation: tuple[ConversationEntry, ...]
    respondent_system_prompt: str
    correspondent_system_prompt: str


class ConversationTurnHooks:
    """Base implementation with async no-ops; subclass to observe or intervene."""

    async def before_correspondent_completion(
        self,
        ctx: ConversationTurnContext,
        correspondent_input: str,
    ) -> None:
        """Called immediately before ``ask`` on the correspondent (initial or follow-up)."""

    async def after_correspondent_completion(
        self,
        ctx: ConversationTurnContext,
        user_message: str,
    ) -> None:
        """Called after the correspondent returns the next user message (appended to ``conversation``)."""

    async def before_respondent_completion(
        self,
        ctx: ConversationTurnContext,
        user_message: str,
    ) -> None:
        """Called immediately before ``answer`` on the respondent."""

    async def after_respondent_completion(
        self,
        ctx: ConversationTurnContext,
        entry: ConversationEntry,
    ) -> None:
        """Called after the assistant turn is appended to ``conversation``."""
