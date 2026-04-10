"""Output format converters for preference pairs (DPO/RLHF)."""

from __future__ import annotations

from typing import Any, Dict, List

from .types import PreferencePair


def format_preference_pairs(
    pairs: List[PreferencePair],
    fmt: str = "dpo",
) -> List[Dict[str, Any]]:
    """Convert a list of PreferencePairs to the requested output format.

    Args:
        pairs: The preference pairs to convert.
        fmt: Output format name. One of: dpo, chat_dpo, ultrafeedback, anthropic_hh, orpo.

    Returns:
        List of JSON-serializable dicts, one per pair.

    Raises:
        ValueError: If fmt is not recognised.
    """
    formatters = {
        "dpo": _to_dpo,
        "chat_dpo": _to_chat_dpo,
        "ultrafeedback": _to_ultrafeedback,
        "anthropic_hh": _to_anthropic_hh,
        "orpo": _to_orpo,
    }
    if fmt not in formatters:
        raise ValueError(
            f"Unknown preference format: {fmt!r}. "
            f"Choose from: {', '.join(formatters)}"
        )
    converter = formatters[fmt]
    return [converter(pair) for pair in pairs]


# ---------------------------------------------------------------------------
# Individual format converters
# ---------------------------------------------------------------------------


def _to_dpo(pair: PreferencePair) -> Dict[str, Any]:
    """Standard DPO format for TRL DPOTrainer.

    Schema::

        {
            "prompt": "<user prompt>",
            "chosen": "<chosen response>",
            "rejected": "<rejected response>"
        }
    """
    return {
        "prompt": pair.prompt,
        "chosen": pair.chosen.content,
        "rejected": pair.rejected.content,
    }


def _to_chat_dpo(pair: PreferencePair) -> Dict[str, Any]:
    """Chat DPO format with message lists (TRL with chat template).

    Schema::

        {
            "prompt": [{"role": "system", ...}, {"role": "user", ...}],
            "chosen":   [{"role": "system", ...}, ..., {"role": "assistant", "content": "<chosen>"}],
            "rejected": [{"role": "system", ...}, ..., {"role": "assistant", "content": "<rejected>"}]
        }

    The ``prompt`` field contains all messages up to (not including) the last
    assistant turn so TRL can build the full conversation.
    """
    chosen_msgs = pair.chosen.messages or []
    rejected_msgs = pair.rejected.messages or []

    # prompt = everything except the last assistant message
    prompt_msgs = chosen_msgs[:-1] if chosen_msgs else []

    return {
        "prompt": prompt_msgs,
        "chosen": chosen_msgs,
        "rejected": rejected_msgs,
    }


def _to_ultrafeedback(pair: PreferencePair) -> Dict[str, Any]:
    """UltraFeedback-style format with all responses and scores.

    Schema::

        {
            "instruction": "<user prompt>",
            "completions": [
                {"response": "...", "score": 0.9, "label": "temperature_0.10"},
                ...
            ],
            "chosen": {"response": "...", "score": 0.9},
            "rejected": {"response": "...", "score": 0.2}
        }
    """
    # Recover all scored responses from metadata if available
    all_scores = pair.metadata.get("all_scores", [])
    completions = [
        {
            "response": s.get("content", ""),
            "score": s.get("score", 0.0),
            "label": s.get("label", ""),
        }
        for s in all_scores
    ]
    # Always include chosen and rejected even if not in all_scores
    if not completions:
        completions = [
            {
                "response": pair.chosen.content,
                "score": pair.chosen.score,
                "label": pair.chosen.variation_label,
            },
            {
                "response": pair.rejected.content,
                "score": pair.rejected.score,
                "label": pair.rejected.variation_label,
            },
        ]

    return {
        "instruction": pair.prompt,
        "completions": completions,
        "chosen": {"response": pair.chosen.content, "score": pair.chosen.score},
        "rejected": {"response": pair.rejected.content, "score": pair.rejected.score},
    }


def _to_anthropic_hh(pair: PreferencePair) -> Dict[str, Any]:
    """Anthropic HH (Helpful & Harmless) format with Human:/Assistant: prefixes.

    Schema::

        {
            "chosen": "Human: <prompt>\n\nAssistant: <chosen>",
            "rejected": "Human: <prompt>\n\nAssistant: <rejected>"
        }

    For multi-turn, the shared_prefix is prepended.
    """
    prefix = ""
    if pair.shared_prefix:
        for turn in pair.shared_prefix:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                prefix += f"\n\nHuman: {content}"
            elif role == "assistant":
                prefix += f"\n\nAssistant: {content}"
        prefix = prefix.strip()

    human_turn = f"Human: {pair.prompt}"
    chosen_text = (
        f"{prefix}\n\n{human_turn}\n\nAssistant: {pair.chosen.content}".strip()
    )
    rejected_text = (
        f"{prefix}\n\n{human_turn}\n\nAssistant: {pair.rejected.content}".strip()
    )

    return {
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def _to_orpo(pair: PreferencePair) -> Dict[str, Any]:
    """ORPO format: DPO schema plus scores.

    Schema::

        {
            "prompt": "<user prompt>",
            "chosen": "<chosen response>",
            "rejected": "<rejected response>",
            "chosen_score": 0.9,
            "rejected_score": 0.2
        }
    """
    return {
        "prompt": pair.prompt,
        "chosen": pair.chosen.content,
        "rejected": pair.rejected.content,
        "chosen_score": pair.chosen.score,
        "rejected_score": pair.rejected.score,
    }
