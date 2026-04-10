"""Shared fixtures for integration tests."""

import json

import pytest

# Realistic mock dataset covering edge cases
MOCK_ROWS = [
    # 1. Simple single-turn, no context, no persona
    {
        "conversations": [
            {"role": "user", "content": "What is Python?", "reasoning_content": None},
            {"role": "assistant", "content": "Python is a high-level programming language.", "reasoning_content": None},
        ],
        "metadata": {},
        "instruction_context": None,
        "response_context": None,
        "persona": None,
    },
    # 2. Single-turn with document context
    {
        "conversations": [
            {"role": "user", "content": "Summarize this document.", "reasoning_content": None},
            {"role": "assistant", "content": "The document discusses machine learning basics.", "reasoning_content": None},
        ],
        "metadata": {"context_id": "doc1", "context_ids": ["doc1"]},
        "instruction_context": "Machine learning is a subset of artificial intelligence.",
        "response_context": "ML overview chapter.",
        "persona": None,
    },
    # 3. Multi-turn (3 turns) with persona
    {
        "conversations": [
            {"role": "user", "content": "What is OOP?", "reasoning_content": None},
            {"role": "assistant", "content": "OOP stands for Object-Oriented Programming.", "reasoning_content": None},
            {"role": "user", "content": "Give me an example.", "reasoning_content": None},
            {"role": "assistant", "content": "A class Dog with methods bark() and fetch().", "reasoning_content": None},
            {"role": "user", "content": "What about inheritance?", "reasoning_content": None},
            {"role": "assistant", "content": "Inheritance lets a class extend another class.", "reasoning_content": None},
        ],
        "metadata": {"persona_name": "CS Student", "persona_generation_depth": 0},
        "instruction_context": "",
        "response_context": None,
        "persona": "CS Student",
    },
    # 4. Single-turn with quality scores (for DPO)
    {
        "conversations": [
            {"role": "user", "content": "Explain REST APIs.", "reasoning_content": None},
            {"role": "assistant", "content": "REST is an architectural style for web services.", "reasoning_content": None},
        ],
        "metadata": {"context_id": "doc2"},
        "instruction_context": "",
        "response_context": None,
        "persona": "Developer",
        "evaluation": {
            "coherence": {"score": 0.9, "feedback": "Clear"},
            "factuality": {"score": 0.85, "feedback": "Accurate"},
            "grounding": {"score": 0.8, "feedback": "OK"},
            "helpfulness": {"score": 0.88, "feedback": "Good"},
            "relevance": {"score": 0.9, "feedback": "Relevant"},
            "overall_grade": "good",
        },
        "final_score": 0.87,
    },
    # 5. Another response for same prompt (different quality — for DPO pairing)
    {
        "conversations": [
            {"role": "user", "content": "Explain REST APIs.", "reasoning_content": None},
            {"role": "assistant", "content": "It's a thing for websites.", "reasoning_content": None},
        ],
        "metadata": {"context_id": "doc2"},
        "instruction_context": "",
        "response_context": None,
        "persona": "Beginner",
        "evaluation": {
            "coherence": {"score": 0.5, "feedback": "Vague"},
            "factuality": {"score": 0.4, "feedback": "Incomplete"},
            "grounding": {"score": 0.3, "feedback": "Poor"},
            "helpfulness": {"score": 0.35, "feedback": "Not helpful"},
            "relevance": {"score": 0.5, "feedback": "Somewhat"},
            "overall_grade": "bad",
        },
        "final_score": 0.41,
    },
    # 6. Edge: empty assistant response
    {
        "conversations": [
            {"role": "user", "content": "Say nothing.", "reasoning_content": None},
            {"role": "assistant", "content": "", "reasoning_content": None},
        ],
        "metadata": {},
        "instruction_context": None,
        "response_context": None,
        "persona": None,
    },
    # 7. Edge: unicode, emoji, special chars
    {
        "conversations": [
            {"role": "user", "content": "Qu'est-ce que c'est? \u00e9\u00e8\u00ea \ud83d\ude80", "reasoning_content": None},
            {"role": "assistant", "content": 'It means "What is it?" in French.\nLine2\n"quotes" & <tags>', "reasoning_content": None},
        ],
        "metadata": {},
        "instruction_context": None,
        "response_context": None,
        "persona": None,
    },
    # 8. Edge: single message only (no assistant) — should be skipped by most exporters
    {
        "conversations": [
            {"role": "user", "content": "Hello?", "reasoning_content": None},
        ],
        "metadata": {},
        "instruction_context": None,
        "response_context": None,
        "persona": None,
    },
]


@pytest.fixture
def mock_rows():
    return [row.copy() for row in MOCK_ROWS]


@pytest.fixture
def mock_jsonl(tmp_path):
    """Write mock dataset to a JSONL file and return the path."""
    p = tmp_path / "dataset.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in MOCK_ROWS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p
