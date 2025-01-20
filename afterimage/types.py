from enum import Enum
from typing import List, TypedDict, Optional
from pydantic import BaseModel


class GradeSchema(str, Enum):
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    PERFECT = "perfect"


class EvaluationEntrySchema(TypedDict):
    comment: str
    score: int


class EvaluationSchema(TypedDict):
    relevance: EvaluationEntrySchema
    grounding: EvaluationEntrySchema
    correctness: EvaluationEntrySchema
    completeness: EvaluationEntrySchema
    coherence: EvaluationEntrySchema
    usefulness: EvaluationEntrySchema
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
    final_score: Optional[int]


if __name__ == "__main__":
    c = ConversationWithContext(
        context="abc", conversations=[ConversationEntry(role="user", content="def")]
    )
    print(c)
