from __future__ import annotations

from .persona_sampling import PersonaCandidate


def substitute_n_instructions_in_prompt(prompt: str, n_instructions: int) -> str:
    n = max(int(n_instructions), 1)
    if "{n_instructions}" in prompt:
        return prompt.replace("{n_instructions}", str(n))
    return prompt


def context_ids_from_documents(docs: list) -> tuple[str | None, list[str]]:
    if not docs:
        return None, []
    return docs[0].id, [d.id for d in docs]


def strip_user_system_prompt_tags(text: str) -> str:
    return (
        text.strip()
        .lstrip("<user_system_prompt>")
        .rstrip("</user_system_prompt>")
        .strip()
    )


def persona_fields_from_candidate(
    candidate: PersonaCandidate | None,
) -> tuple[str, int | None]:
    if candidate is None:
        return "A curious user", None
    return candidate.text, candidate.generation_depth
