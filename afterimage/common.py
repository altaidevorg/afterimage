from pydantic import BaseModel, Field
from typing import List

default_model_name = "gemini-2.0-flash"
default_max_concurrency = 4
deepseek_default_max_concurrency = 16

default_safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]


class GeneratedInstructions(BaseModel):
    instructions: List[str]
    context: str
    persona: str | None = None
    context_id: str | None = None
    context_ids: List[str] = Field(default_factory=list)


def resolve_generation_max_concurrency(
    model_provider_name: str,
    max_concurrency: int | None,
) -> int:
    """Resolve the effective generation concurrency for a provider."""
    if max_concurrency is not None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        return max_concurrency

    if model_provider_name == "deepseek":
        return deepseek_default_max_concurrency

    return default_max_concurrency
