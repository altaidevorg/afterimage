# AfterImage JSONL Source Schema

Each line in a generated `.jsonl` file is a JSON object representing one conversation.

## Schema (EvaluatedConversationWithContext)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What is Python?",
      "reasoning_content": null
    },
    {
      "role": "assistant",
      "content": "Python is a high-level programming language...",
      "reasoning_content": null
    }
  ],
  "metadata": {
    "context_id": "doc-abc-123",
    "context_ids": ["doc-abc-123", "doc-def-456"],
    "persona_name": "A curious beginner",
    "persona_generation_depth": 0,
    "instruction_index": 2,
    "batch_id": "batch-xyz",
    "session_id": "sess-001"
  },
  "instruction_context": "Python was created by Guido van Rossum...",
  "response_context": "Python documentation excerpt...",
  "persona": "A curious beginner",
  "evaluation": {
    "coherence": { "score": 0.92, "feedback": "Clear logical flow." },
    "factuality": { "score": 0.85, "feedback": "Accurate." },
    "grounding": { "score": 0.78, "feedback": "Well grounded." },
    "helpfulness": { "score": 0.88, "feedback": "Helpful response." },
    "relevance": { "score": 0.90, "feedback": "Relevant." },
    "overall_grade": "good"
  },
  "final_score": 0.866
}
```

## Field details

| Field | Type | Always present | Description |
|---|---|---|---|
| `conversations` | `list[{role, content, reasoning_content}]` | Yes | Alternating user/assistant turns |
| `conversations[].role` | `"user"` or `"assistant"` | Yes | Speaker role |
| `conversations[].content` | `string` | Yes | Message text |
| `conversations[].reasoning_content` | `string \| null` | Yes | Chain-of-thought (DeepSeek R1 etc.) |
| `metadata` | `dict` | Yes | Generation metadata |
| `metadata.context_id` | `string` | If context used | Source document ID |
| `metadata.context_ids` | `list[string]` | If context used | All source document IDs |
| `metadata.persona_name` | `string \| null` | If personas used | Persona description |
| `metadata.persona_generation_depth` | `int \| null` | If personas used | Depth in persona tree |
| `instruction_context` | `string \| null` | Yes | Context given to instruction generator |
| `response_context` | `string \| null` | Yes | Context given to respondent |
| `persona` | `string \| null` | Yes | Top-level persona shorthand |
| `evaluation` | `object \| null` | When a judge ran (e.g. `auto_improve` or manual evaluation) | Quality evaluation scores; may be absent on unevaluated rows |
| `evaluation.overall_grade` | `string` | With evaluation | One of `perfect`, `good`, `needs_improvement`, `bad`, `not_acceptable` (see `GradeSchema` in `afterimage.types`) |
| `final_score` | `float` | Usually present on stored rows | Composite 0–1 score when evaluated; unevaluated rows may omit the field or serialize with model defaults depending on writer |

## Notes

- The respondent system prompt is NOT stored in each row. It lives in the config or in the ConversationGenerator. Exporters that need a system prompt should accept it as an optional parameter.
- Multi-turn conversations have `len(conversations) > 2` with strictly alternating user/assistant.
- `reasoning_content` is typically null unless using models with chain-of-thought (e.g. DeepSeek R1).
