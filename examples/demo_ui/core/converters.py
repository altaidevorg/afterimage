"""
Converters for transforming generation output items into display rows.
"""

from typing import Any

from schemas import CustomerSupportInteraction, ToolInvocation
from afterimage.types import ConversationWithContext, EvaluatedConversationWithContext


def customer_support_to_row(item: Any, truncate: bool = False) -> dict:
    """Convert a CustomerSupportInteraction item to a display row."""
    reasoning = item.output.agent_reasoning
    if truncate and reasoning:
        reasoning = str(reasoning)[:100] + "..."
    
    return {
        "Persona": str(item.persona) if item.persona else "N/A",
        "Instruction": item.instruction,
        "Intent": item.output.intent,
        "Urgency": item.output.urgency,
        "Reasoning": reasoning or "",
        "Response": item.output.response,
    }


def tool_invocation_to_row(item: Any, truncate: bool = False) -> dict:
    """Convert a ToolInvocation item to a display row."""
    tool_calls_str = "\n".join(
        [
            f"{tc.function.name}({tc.function.arguments.model_dump_json()})"
            for tc in item.output.tool_calls
        ]
    )
    
    reasoning = item.output.reasoning
    if truncate and reasoning:
        reasoning = str(reasoning)[:100] + "..."
    
    return {
        "Persona": str(item.persona) if item.persona else "N/A",
        "Instruction": item.instruction,
        "Response": item.output.response,
        "Reasoning": reasoning or "",
        "Tool Calls": tool_calls_str,
    }


def conversation_to_row(item: Any, truncate: bool = False) -> dict:
    """Convert a ConversationWithContext item to a display row."""
    conversations = item.conversations
    
    context = item.instruction_context
    if truncate and context:
        context = str(context)[:200] + "..."
    
    return {
        "Instruction": conversations[0].content if conversations else "",
        "Response": conversations[1].content if len(conversations) > 1 else "",
        "Context": context or "",
        "Persona": str(item.persona) if item.persona else "N/A",
    }


def item_to_row(item: Any, truncate: bool = False) -> dict | None:
    """
    Convert any generation output item to a display row.
    
    Args:
        item: The generation output item
        truncate: Whether to truncate long text fields (for live updates)
    
    Returns:
        A dictionary representing the row, or None if item type is unknown
    """
    if hasattr(item, "output") and isinstance(item.output, CustomerSupportInteraction):
        return customer_support_to_row(item, truncate)
    
    elif hasattr(item, "output") and isinstance(item.output, ToolInvocation):
        return tool_invocation_to_row(item, truncate)
    
    elif isinstance(item, (ConversationWithContext, EvaluatedConversationWithContext)):
        return conversation_to_row(item, truncate)
    
    return None


def items_to_dataframe_data(items: list, truncate: bool = False) -> list[dict]:
    """
    Convert a list of items to a list of row dictionaries for DataFrame.
    
    Args:
        items: List of generation output items
        truncate: Whether to truncate long text fields
    
    Returns:
        List of row dictionaries
    """
    data = []
    for item in items:
        try:
            row = item_to_row(item, truncate)
            if row:
                data.append(row)
        except Exception:
            # Skip items that fail to convert
            pass
    return data
