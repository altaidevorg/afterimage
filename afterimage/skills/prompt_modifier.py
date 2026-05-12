"""Runtime prompt modifier that injects selected context-specific skills."""

from __future__ import annotations

from ..base import BaseRespondentPromptModifierCallback
from ..types import GeneratedResponsePrompt
from .storage import DirectorySkillStore


class SkillRespondentPromptModifier(BaseRespondentPromptModifierCallback):
    """Wrap a prompt modifier and append a selected skill for the current context."""

    def __init__(
        self,
        skill_store: DirectorySkillStore,
        base_modifier: BaseRespondentPromptModifierCallback | None = None,
    ):
        self.skill_store = skill_store
        self.base_modifier = base_modifier

    def _inject(
        self, prompt: str, skill_content: str, name: str, description: str
    ) -> str:
        return (
            prompt.rstrip()
            + "\n\n## Context-Specific Skill\n\n"
            + "Use this skill only when it is relevant to the current request and context.\n\n"
            + f"### {name}\n"
            + f"When to use: {description}\n\n"
            + skill_content.strip()
        )

    def generate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        base = self._run_base_sync(respondent_prompt, context, instruction)
        skill = self.skill_store.load_selected(context_text=base.context or context)
        if skill is None:
            return base
        metadata = dict(base.metadata)
        metadata["skill"] = {
            "context_id": skill.context_id,
            "version_id": skill.id,
            "name": skill.name,
            "iteration": skill.iteration,
        }
        return GeneratedResponsePrompt(
            prompt=self._inject(
                base.prompt, skill.content, skill.name, skill.description
            ),
            context=base.context,
            metadata=metadata,
        )

    async def agenerate(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        base = await self._run_base_async(respondent_prompt, context, instruction)
        skill = self.skill_store.load_selected(context_text=base.context or context)
        if skill is None:
            return base
        metadata = dict(base.metadata)
        metadata["skill"] = {
            "context_id": skill.context_id,
            "version_id": skill.id,
            "name": skill.name,
            "iteration": skill.iteration,
        }
        return GeneratedResponsePrompt(
            prompt=self._inject(
                base.prompt, skill.content, skill.name, skill.description
            ),
            context=base.context,
            metadata=metadata,
        )

    def _run_base_sync(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        if self.base_modifier is None:
            return GeneratedResponsePrompt(prompt=respondent_prompt, context=context)
        return self.base_modifier.generate(respondent_prompt, context, instruction)

    async def _run_base_async(
        self, respondent_prompt: str, context: str, instruction: str
    ) -> GeneratedResponsePrompt:
        if self.base_modifier is None:
            return GeneratedResponsePrompt(prompt=respondent_prompt, context=context)
        return await self.base_modifier.agenerate(
            respondent_prompt, context, instruction
        )
