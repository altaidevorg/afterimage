"""Convert AfterImage JSONL datasets to popular training formats.

Supported formats:
- **sharegpt**: ``{"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}``
- **alpaca**: ``{"instruction": ..., "input": "", "output": ...}`` (first turn only)
- **messages**: ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
- **agent_sft**: ``{"messages": [...], "trajectory_id": ...}``
- **openai_tools**: OpenAI Chat Completions tool calling JSON schema.
- **anthropic_tools**: Anthropic Messages API tool use JSON schema.
- **hermes_tools**: Nous Hermes / Qwen 2.5 XML tool calling format.
- **agent_dpo**: Preference pairs for DPO/ORPO tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ROLE_MAP_SHAREGPT = {"user": "human", "assistant": "gpt"}
_ROLE_MAP_MESSAGES = {"user": "user", "assistant": "assistant"}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning one dict per line."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write rows to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_sharegpt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to ShareGPT format."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        turns = []
        for entry in convs:
            role = entry.get("role", "user")
            turns.append(
                {
                    "from": _ROLE_MAP_SHAREGPT.get(role, role),
                    "value": entry.get("content", ""),
                }
            )
        out.append({"conversations": turns})
    return out


def to_alpaca(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to Alpaca format (first turn only)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        user_msg = ""
        assistant_msg = ""
        for entry in convs:
            role = entry.get("role", "")
            if role == "user" and not user_msg:
                user_msg = entry.get("content", "")
            elif role == "assistant" and not assistant_msg:
                assistant_msg = entry.get("content", "")
            if user_msg and assistant_msg:
                break
        out.append(
            {
                "instruction": user_msg,
                "input": "",
                "output": assistant_msg,
            }
        )
    return out


def to_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage rows to HuggingFace messages format."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            messages.append(
                {
                    "role": _ROLE_MAP_MESSAGES.get(role, role),
                    "content": entry.get("content", ""),
                }
            )
        out.append({"messages": messages})
    return out


def to_agent_sft(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage agent trajectories into structured OpenAI/HuggingFace SFT messages format."""
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            messages.append(
                {
                    "role": _ROLE_MAP_MESSAGES.get(role, role),
                    "content": content,
                }
            )
        out.append(
            {
                "messages": messages,
                "trajectory_id": row.get("metadata", {}).get("trajectory_id"),
            }
        )
    return out


def to_openai_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage agent trajectories into OpenAI Chat Completions tool-calling format.

    Args:
        rows: List of AfterImage dataset rows (from JSONL).

    Returns:
        list[dict[str, Any]]: OpenAI Chat Completions tool calling objects.
    """
    out: list[dict[str, Any]] = []
    call_idx = 0

    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            content = entry.get("content", "")

            if role == "user":
                if content.startswith("Observation:"):
                    obs_content = content.replace("Observation:", "", 1).strip()
                    call_id = f"call_{call_idx}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": obs_content,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if "Action:" in content:
                    call_idx += 1
                    call_id = f"call_{call_idx}"
                    thought_part = content.split("Action:", 1)[0].replace("Thought:", "").strip()
                    action_part = content.split("Action:", 1)[1]
                    action_name = action_part.split("Action Input:", 1)[0].strip().replace(".", "__")
                    args_str = "{}"
                    if "Action Input:" in action_part:
                        args_str = action_part.split("Action Input:", 1)[1].strip()

                    messages.append(
                        {
                            "role": "assistant",
                            "content": thought_part or None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": action_name,
                                        "arguments": args_str,
                                    },
                                }
                            ],
                        }
                    )
                else:
                    text = content.replace("Final Answer:", "").replace("Thought:", "").strip()
                    messages.append({"role": "assistant", "content": text})

        out.append(
            {
                "messages": messages,
                "trajectory_id": row.get("metadata", {}).get("trajectory_id"),
            }
        )
    return out


def to_anthropic_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage agent trajectories into Anthropic Messages API tool format.

    Args:
        rows: List of AfterImage dataset rows.

    Returns:
        list[dict[str, Any]]: Anthropic Messages tool use objects.
    """
    out: list[dict[str, Any]] = []
    call_idx = 0

    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            content = entry.get("content", "")

            if role == "user":
                if content.startswith("Observation:"):
                    obs_content = content.replace("Observation:", "", 1).strip()
                    call_id = f"toolu_{call_idx}"
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": obs_content,
                                }
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if "Action:" in content:
                    call_idx += 1
                    call_id = f"toolu_{call_idx}"
                    thought_part = content.split("Action:", 1)[0].replace("Thought:", "").strip()
                    action_part = content.split("Action:", 1)[1]
                    action_name = action_part.split("Action Input:", 1)[0].strip().replace(".", "__")
                    args_str = "{}"
                    if "Action Input:" in action_part:
                        args_str = action_part.split("Action Input:", 1)[1].strip()

                    try:
                        args_dict = json.loads(args_str)
                    except Exception:
                        args_dict = {}

                    content_blocks = []
                    if thought_part:
                        content_blocks.append({"type": "text", "text": thought_part})
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": action_name,
                            "input": args_dict,
                        }
                    )
                    messages.append({"role": "assistant", "content": content_blocks})
                else:
                    text = content.replace("Final Answer:", "").replace("Thought:", "").strip()
                    messages.append({"role": "assistant", "content": text})

        out.append(
            {
                "messages": messages,
                "trajectory_id": row.get("metadata", {}).get("trajectory_id"),
            }
        )
    return out


def to_hermes_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage agent trajectories into Hermes / Qwen 2.5 tool calling format.

    Args:
        rows: List of AfterImage dataset rows.

    Returns:
        list[dict[str, Any]]: Hermes XML tool call messages.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        messages = []
        for entry in convs:
            role = entry.get("role", "user")
            content = entry.get("content", "")

            if role == "user":
                if content.startswith("Observation:"):
                    obs_content = content.replace("Observation:", "", 1).strip()
                    messages.append({"role": "tool", "content": obs_content})
                else:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if "Action:" in content:
                    thought_part = content.split("Action:", 1)[0].replace("Thought:", "").strip()
                    action_part = content.split("Action:", 1)[1]
                    action_name = action_part.split("Action Input:", 1)[0].strip().replace(".", "__")
                    args_str = "{}"
                    if "Action Input:" in action_part:
                        args_str = action_part.split("Action Input:", 1)[1].strip()

                    try:
                        args_dict = json.loads(args_str)
                    except Exception:
                        args_dict = {}

                    tool_payload = json.dumps({"name": action_name, "arguments": args_dict})
                    text = f"{thought_part}\n<tool_call>{tool_payload}</tool_call>".strip()
                    messages.append({"role": "assistant", "content": text})
                else:
                    text = content.replace("Final Answer:", "").replace("Thought:", "").strip()
                    messages.append({"role": "assistant", "content": text})

        out.append(
            {
                "messages": messages,
                "trajectory_id": row.get("metadata", {}).get("trajectory_id"),
            }
        )
    return out


def to_agent_dpo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AfterImage agent trajectories into DPO / preference pair format.

    Args:
        rows: List of AfterImage dataset rows.

    Returns:
        list[dict[str, Any]]: Preference pairs (prompt, chosen, rejected).
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        convs = row.get("conversations", [])
        if not convs:
            continue
        prompt = convs[0].get("content", "")
        chosen = [c.get("content", "") for c in convs[1:]]
        rejected = chosen[: len(chosen) // 2] if len(chosen) > 1 else ["Unable to fulfill request."]

        out.append(
            {
                "prompt": prompt,
                "chosen": "\n".join(chosen),
                "rejected": "\n".join(rejected),
                "trajectory_id": row.get("metadata", {}).get("trajectory_id"),
            }
        )
    return out


CONVERTERS = {
    "sharegpt": to_sharegpt,
    "alpaca": to_alpaca,
    "messages": to_messages,
    "agent_sft": to_agent_sft,
    "openai_tools": to_openai_tools,
    "anthropic_tools": to_anthropic_tools,
    "hermes_tools": to_hermes_tools,
    "agent_dpo": to_agent_dpo,
}


def export_dataset(
    input_path: str | Path,
    format_name: str,
    output_path: str | Path | None = None,
) -> Path:
    """Load an AfterImage JSONL dataset and write it in the requested format.

    Args:
        input_path: Path to the source JSONL file.
        format_name: One of ``"sharegpt"``, ``"alpaca"``, ``"messages"``, ``"agent_sft"``,
            ``"openai_tools"``, ``"anthropic_tools"``, ``"hermes_tools"``, ``"agent_dpo"``.
        output_path: Destination path. If *None*, derived from *input_path*.

    Returns:
        The resolved output path.
    """
    converter = CONVERTERS.get(format_name)
    if converter is None:
        raise ValueError(
            f"Unknown format: {format_name!r}. Choose from {list(CONVERTERS)}"
        )

    rows = _load_jsonl(input_path)
    converted = converter(rows)

    if output_path is None:
        inp = Path(input_path)
        output_path = inp.with_name(f"{inp.stem}_{format_name}.jsonl")

    _write_jsonl(converted, output_path)
    return Path(output_path)
