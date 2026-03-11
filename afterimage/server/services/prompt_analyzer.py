"""Analyzes document excerpts to auto-generate context-appropriate system prompt parts."""

from __future__ import annotations

import json

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from ..models import AnalyzeDocumentResponse


class _SystemPromptParts(BaseModel):
    respondent_role: str = Field(
        description="A concise role description for the AI assistant (e.g. 'You are a machine learning researcher specializing in PEFT')."
    )
    correspondent_role: str = Field(
        description="A concise role description for the simulated user (e.g. 'You are a software engineer interested in efficient inference')."
    )
    instruction: str = Field(
        description="Specific instructions on how the assistant should answer in this domain."
    )


class PromptAnalyzer:
    """Calls the LLM once to produce respondent/correspondent roles and an instruction."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model_name = model_name

    async def analyze(
        self, text: str, excerpt_length: int = 4000
    ) -> AnalyzeDocumentResponse:
        excerpt = text[:excerpt_length]
        client = genai.Client(api_key=self._api_key)
        response = await client.aio.models.generate_content(
            model=self._model_name,
            contents=(
                "Analyze the following document excerpt and generate the best system prompt "
                "roles for a synthetic conversation dataset generation.\n\n"
                "The parts should include:\n"
                "1. \"respondent_role\" - describing who the AI assistant is (e.g. \"You are an expert in X\")\n"
                "2. \"correspondent_role\" - describing who the person asking the questions is\n"
                "3. \"instruction\" - specific answering logic for the assistant.\n\n"
                "Make these highly tailored to the document's domain.\n\n"
                f"Document excerpt:\n---\n{excerpt}\n---"
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_SystemPromptParts,
            ),
        )
        data = json.loads(response.text)
        return AnalyzeDocumentResponse(**data)
