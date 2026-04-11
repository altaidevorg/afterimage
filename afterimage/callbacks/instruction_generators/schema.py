from typing import List

from pydantic import BaseModel


class InstructionsSchema(BaseModel):
    instructions: List[str]
