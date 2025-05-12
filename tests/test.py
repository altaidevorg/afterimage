from google import genai
from google.genai import types
import os
from pydantic import BaseModel

from typing_extensions import TypedDict


class TurkishFood(BaseModel):
    dish_name: str
    ingredients: list[str]


client = genai.Client()

out = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are an expert in world cultures and cuisines",
        response_mime_type="application/json",
        response_schema=TurkishFood,
    ),
    contents="Recommend me a Turkish food",
)
print(out.text)
