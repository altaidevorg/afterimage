"""
Utilities for dynamically creating Pydantic models from schemas.
"""

from typing import Any, Dict, List, Literal, Optional, Type, Union
from pydantic import BaseModel, ConfigDict, Field, create_model


def create_pydantic_model_from_schema(
    name: str, schema: Dict[str, Any]
) -> Type[BaseModel]:
    """
    Create a Pydantic model from an OpenAI function schema.

    Args:
        name: Name of the model/tool
        schema: OpenAI function schema dictionary

    Returns:
        A Pydantic BaseModel class
    """

    # Extract parameters
    params_schema = schema.get("parameters", {})
    properties = params_schema.get("properties", {})
    required = params_schema.get("required", [])

    # Define fields for the arguments model
    fields = {}
    for param_name, param_info in properties.items():
        # Determine python type
        py_type = str
        json_type = param_info.get("type", "string")

        if json_type == "integer":
            py_type = int
        elif json_type == "number":
            py_type = float
        elif json_type == "boolean":
            py_type = bool
        elif json_type == "array":
            py_type = list
        elif json_type == "object":
            py_type = dict

        # Determine if required
        is_required = param_name in required
        default = (
            param_info.get("default", ...)
            if is_required
            else param_info.get("default", None)
        )

        # Add description
        description = param_info.get("description", "")

        # Create field definition
        if is_required:
            fields[param_name] = (py_type, Field(description=description))
        else:
            fields[param_name] = (
                Optional[py_type],
                Field(default, description=description),
            )

    # Gemini API requires at least one property for OBJECT types
    # If no fields, add a placeholder
    if not fields:
        fields["placeholder"] = (
            Optional[str],
            Field(None, description="No arguments required"),
        )

    # Create arguments model
    args_model_name = f"{name}Args"
    ArgsModel = create_model(
        args_model_name, __config__=ConfigDict(extra="ignore"), **fields
    )

    # Create the main tool model
    # It must look like the built-in ones: 'name' literal + 'arguments'
    tool_fields = {
        "name": (Literal[name], Field(name)),  # type: ignore
        "arguments": (ArgsModel, Field(...)),
    }

    ToolModel = create_model(
        name,
        __doc__=schema.get("description", ""),
        __config__=ConfigDict(extra="ignore"),
        **tool_fields,
    )

    return ToolModel


def create_dynamic_tool_invocation_schema(
    tools: List[Union[Type[BaseModel], Dict[str, Any]]],
) -> Type[BaseModel]:
    """
    Create a dynamic ToolInvocation schema that accepts the provided tools.

    Args:
        tools: List of tool definitions (either Pydantic models or OpenAI schema dicts)

    Returns:
        A custom ToolInvocation Pydantic model
    """
    pydantic_tools = []

    for tool in tools:
        if isinstance(tool, type) and issubclass(tool, BaseModel):
            pydantic_tools.append(tool)
        elif isinstance(tool, dict):
            # It's a schema dict (OpenAI format)
            # Schema structure could be directly the function dict or wrapped in type: function
            func_schema = tool
            if "function" in tool:
                func_schema = tool["function"]

            name = func_schema.get("name", "UnknownTool")
            pydantic_tools.append(create_pydantic_model_from_schema(name, func_schema))

    if not pydantic_tools:
        raise ValueError("At least one tool must be provided.")

    # Create the Union of all tool types
    # We use a constructed Union type
    ToolUnion = Union[tuple(pydantic_tools)]  # type: ignore

    # Define the AnyToolCall wrapper
    # This matches the structure in schemas.py: class AnyToolCall(BaseModel): function: Union[...]
    AnyToolCallValidationModel = create_model(
        "DynamicAnyToolCall", function=(ToolUnion, Field(...))
    )

    # Define the top-level ToolInvocation schema
    DynamicToolInvocation = create_model(
        "DynamicToolInvocation",
        reasoning=(
            str,
            Field(
                description="Chain-of-thought reasoning for selecting the specific tool(s) and arguments."
            ),
        ),
        response=(
            str,
            Field(description="The final response to the user in natural language."),
        ),
        tool_calls=(
            List[AnyToolCallValidationModel],
            Field(description="A list of tool calls to execute."),
        ),
    )

    return DynamicToolInvocation
