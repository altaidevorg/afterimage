from typing import Any, Mapping


def extract_unique_context_ids(metadata: Mapping[str, Any] | None) -> list[str]:
    """Extract ordered, unique context ids from generation metadata."""
    if metadata is None:
        return []

    unique_context_ids: list[str] = []
    seen_context_ids: set[str] = set()

    context_ids = metadata.get("context_ids")
    if isinstance(context_ids, list):
        for context_id in context_ids:
            if (
                isinstance(context_id, str)
                and context_id
                and context_id not in seen_context_ids
            ):
                unique_context_ids.append(context_id)
                seen_context_ids.add(context_id)

    context_id = metadata.get("context_id")
    if (
        isinstance(context_id, str)
        and context_id
        and context_id not in seen_context_ids
    ):
        unique_context_ids.append(context_id)

    return unique_context_ids
