from enum import Enum
from typing import List, TypedDict, Optional
from pydantic import BaseModel


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


class ConversationWithContext(Conversation):
    context: Optional[str]


class EvaluatedConversationWithContext(ConversationWithContext):
    evaluation: Optional[EvaluationSchema]
    final_score: Optional[float] = 0.0
