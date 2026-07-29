import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

from afterimage.providers.llm_providers import LLMProvider

T = TypeVar("T", bound=BaseModel)


class LLMObservationSynthesizer:
    """Synthesizes realistic tool observations using LLM structured generation (ESAT paper methodology)."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.5-flash-lite",
    ) -> None:
        """Initializes LLMObservationSynthesizer.

        Args:
            llm_provider (LLMProvider): Active LLM provider instance.
            model_name (str): LLM model name for observation synthesis.
        """
        self.llm_provider = llm_provider
        self.model_name = model_name

    async def synthesize_observation(
        self,
        app_name: str,
        action_name: str,
        action_description: str,
        response_model_cls: Type[T],
        parameters: Dict[str, Any],
        initial_context: str = "",
        turns: Optional[List[Dict[str, Any]]] = None,
    ) -> T:
        """Generates a structured, schema-compliant observation model using LLM structured output.

        Args:
            app_name (str): Name of the application domain.
            action_name (str): Tool action endpoint name.
            action_description (str): Endpoint documentation.
            response_model_cls (Type[T]): Target Pydantic response model class.
            parameters (Dict[str, Any]): Arguments passed by agent to tool call.
            initial_context (str): Initial task context payload.
            turns (Optional[List[Dict[str, Any]]]): Trajectory turn history.

        Returns:
            T: Instantiated schema-compliant response model instance.
        """
        schema_json = json.dumps(response_model_cls.model_json_schema(), indent=2)
        params_json = json.dumps(parameters, indent=2)
        turns_summary = ""
        if turns:
            turns_summary = "\n".join(
                [f"Turn {i+1}: {t.get('role', '')} -> {t.get('content', '')[:120]}" for i, t in enumerate(turns[-4:])]
            )

        prompt = (
            f"You are a realistic declarative tool environment simulator.\n"
            f"Synthesize a realistic JSON observation payload for an API tool call.\n\n"
            f"App Domain: {app_name}\n"
            f"Action Name: {action_name}\n"
            f"Description: {action_description}\n"
            f"Parameters Passed by Agent: {params_json}\n\n"
            f"Initial Context: {initial_context}\n"
            f"Recent Turn History:\n{turns_summary}\n\n"
            f"Target Pydantic Model JSON Schema:\n{schema_json}\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. You MUST echo any input parameters (like account_id, user_id, amount) directly in the response payload.\n"
            f"2. Amounts, IDs, and statuses MUST be realistic and consistent with past turn history.\n"
            f"3. Return ONLY a valid JSON object strictly matching the schema above."
        )

        result_instance: Optional[T] = None
        try:
            res = await self.llm_provider.agenerate_structured(
                prompt=prompt,
                schema=response_model_cls,
            )
            if hasattr(res, "parsed") and res.parsed is not None:
                result_instance = res.parsed
            elif isinstance(res, response_model_cls):
                result_instance = res
        except Exception:
            pass

        if result_instance is None:
            # Fallback to agenerate_content with JSON extraction
            resp = await self.llm_provider.agenerate_content(
                prompt=prompt,
                system_instruction="Return ONLY raw JSON matching the requested model schema. No markdown formatting.",
            )
            raw_text = resp.text.strip() if hasattr(resp, "text") else getattr(resp, "content", "").strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            parsed = json.loads(raw_text)
            result_instance = response_model_cls(**parsed)

        # Enforce parameter echoing on LLM-synthesized instance for 100% argument consistency
        for p_name, p_val in parameters.items():
            if p_val is not None and hasattr(result_instance, p_name):
                try:
                    setattr(result_instance, p_name, p_val)
                except Exception:
                    pass

        return result_instance
