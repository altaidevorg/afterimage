from enum import Enum
from typing import Any, List, TypedDict, Optional
from pydantic import BaseModel, Field


class GradeSchema(str, Enum):
    PERFECT = "perfect"
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    BAD = "bad"
    NOT_ACCEPTABLE = "not_acceptable"


class EvaluationEntrySchema(TypedDict):
    feedback: str
    score: float


class EvaluationSchema(TypedDict):
    coherence: EvaluationEntrySchema
    factuality: EvaluationEntrySchema
    grounding: EvaluationEntrySchema
    helpfulness: EvaluationEntrySchema
    relevance: EvaluationEntrySchema
    overall_grade: GradeSchema


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationEntry(BaseModel):
    role: Role
    content: str


class Conversation(BaseModel):
    conversations: List[ConversationEntry]
    metadata: Optional[dict[str, Any]] = None


class ConversationWithContext(Conversation):
    instruction_context: Optional[str] = None
    response_context: Optional[str] = None


class EvaluatedConversationWithContext(ConversationWithContext):
    evaluation: EvaluationSchema
    final_score: Optional[float] = 0.0


class GeneratedResponsePrompt(BaseModel):
    """Output of RespondentPromptModifier."""

    prompt: str = Field(..., description="Modified respondent prompt")
    context: Optional[str] = Field(
        None, description="Context used in respondent prompt"
    )
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional metadata about respondent promp generation"
    )
