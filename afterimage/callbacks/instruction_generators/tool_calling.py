import json
import random
from typing import List, Literal, Optional, Type, Union

from pydantic import BaseModel

from ...key_management import SmartKeyPool
from ...prompts import default_tool_calling_persona_instruction_generation_prompt
from ...monitoring import GenerationMonitor
from ...providers import DocumentProvider
from ._utils import context_ids_from_documents, persona_fields_from_candidate
from .persona import PersonaInstructionGeneratorCallback


class ToolCallingInstructionGeneratorCallback(PersonaInstructionGeneratorCallback):
    """Generates instructions that specifically require calling provided tools, optionally using personas.

    Args:
        api_key: API key for the generative AI service.
        tools: List of tools to use.
            each item of this list should be an OpenAI-style tool description as a dictionary or a pydantic model.
        documents: Either a list of texts or a DocumentProvider instance providing context to ground the instructions.
            For each round of generation `num_random_contexts` documents are sampled from this collection.
        prompt: Prompt that guides the instruction generation. If None, uses the default instruction generation prompt.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        num_random_contexts: Number of contexts to sample for each round of generation.
        n_instructions: Number of instructions to generate in each round of generation.
        num_tools_to_sample: Number of tools to sample as the targets for each round of generation.
        separator_text: Separator text for merging contexts if more than one context is sampled.
        safety_settings: Safety settings for the model. Mainly intended for Gemini models.
            Deprecated and may be removed in the future.
        monitor: GenerationMonitor instance to use for tracking.
            If `None`, Conversation and/or structured generators will set their own monitor"""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        tools: List[Union[dict, Type[BaseModel]]],
        documents: Union[list[str], DocumentProvider],
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal["gemini", "openai", "deepseek"] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        num_tools_to_sample: int = 2,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
    ):
        super().__init__(
            api_key=api_key,
            documents=documents,
            prompt=prompt
            if prompt is not None
            else default_tool_calling_persona_instruction_generation_prompt,
            model_name=model_name,
            model_provider_name=model_provider_name,
            num_random_contexts=num_random_contexts,
            n_instructions=n_instructions,
            separator_text=separator_text,
            safety_settings=safety_settings,
            monitor=monitor,
        )
        self.tools = self._normalize_tools(tools)
        self.num_tools_to_sample = num_tools_to_sample

    def _normalize_tools(self, tools: List[Union[dict, Type[BaseModel]]]) -> List[dict]:
        normalized = []
        for t in tools:
            if isinstance(t, type) and issubclass(t, BaseModel):
                normalized.append(self._tool_model_to_openai_schema(t))
            else:
                normalized.append(t)
        return normalized

    def _tool_model_to_openai_schema(self, tool_model: Type[BaseModel]) -> dict:
        name = getattr(tool_model, "name", tool_model.__name__.lower())
        if hasattr(tool_model, "model_fields") and "name" in tool_model.model_fields:
            name = tool_model.model_fields["name"].default

        if (
            hasattr(tool_model, "model_fields")
            and "arguments" in tool_model.model_fields
        ):
            args_model = tool_model.model_fields["arguments"].annotation
        else:
            args_model = tool_model

        args_schema = args_model.model_json_schema()

        params = {
            "type": "object",
            "properties": args_schema.get("properties", {}),
            "required": args_schema.get("required", []),
        }

        for v in params["properties"].values():
            if "title" in v:
                del v["title"]

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool_model.__doc__ or "",
                "parameters": params,
            },
        }

    def _sample_tools(self) -> List[dict]:
        n = min(len(self.tools), self.num_tools_to_sample)
        return random.sample(self.tools, n)

    def _format_tools_context(self, tools: List[dict]) -> str:
        return json.dumps(tools, indent=2)

    def _run_tool_calling_generation(self):
        random_contexts, persona_candidate = self._sample()
        persona, persona_generation_depth = persona_fields_from_candidate(
            persona_candidate
        )

        full_context = self._merge_contexts([c.text for c in random_contexts])
        target_tools = self._sample_tools()
        tools_context = self._format_tools_context(target_tools)

        system_prompt = self.prompt.format(
            n_instructions=self.n_instructions,
            tools_context=tools_context,
            context=full_context,
            persona=persona,
        )

        model = self._create_model(system_instruction=system_prompt)
        prompt = "Generate the instructions now."
        context_id, context_ids = context_ids_from_documents(random_contexts)
        return (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        )

    def generate(self, original_prompt):
        """Generates instructions that require tool calls."""
        (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        ) = self._run_tool_calling_generation()
        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
            persona=persona,
            persona_generation_depth=persona_generation_depth,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions that require tool calls asynchronously."""
        (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        ) = self._run_tool_calling_generation()
        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
            persona=persona,
            persona_generation_depth=persona_generation_depth,
        )

    def create_correspondent_prompt(self, _respondent_prompt: str) -> str:
        return self.prompt

    async def acreate_correspondent_prompt(self, respondent_prompt: str) -> str:
        return self.create_correspondent_prompt(respondent_prompt)
