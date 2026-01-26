"""
Function parser for extracting FunctionDefinition from Python code.

Uses the `ast` module to parse Python function definitions and extract
metadata for tool calling generation. Also supports creating definitions
from callable objects using inspect.
"""

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints


@dataclass
class FunctionDefinition:
    """Represents a parsed function definition for tool calling."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    callable: Optional[Callable] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDefinition":
        """Create a FunctionDefinition from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            required=data["parameters"].get("required", []),
        )

    @classmethod
    def from_callable(cls, func: Callable) -> "FunctionDefinition":
        """Create a FunctionDefinition from a callable object."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        parameters = {}
        required_params = []

        # Type mapping from Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            datetime: "string",
            date: "string",
        }

        def get_type_info(param_type: Type) -> Dict[str, Any]:
            """Get JSON schema type info for a Python type."""
            # Handle Optional types
            if (
                getattr(param_type, "__origin__", None) is Union
                and type(None) in param_type.__args__
            ):
                actual_type = next(
                    t for t in param_type.__args__ if t is not type(None)
                )
                return get_type_info(actual_type)

            # Handle List types
            if getattr(param_type, "__origin__", None) is list:
                item_type = param_type.__args__[0]
                return {
                    "type": "array",
                    "items": {"type": type_map.get(item_type, "string")},
                }

            # Handle Dict types
            if getattr(param_type, "__origin__", None) is dict:
                return {"type": "object"}

            # Handle basic types
            base_type = type_map.get(param_type, "string")
            type_info = {"type": base_type}

            # Add format for special string types
            if param_type in (datetime, date):
                type_info["format"] = "date-time" if param_type is datetime else "date"

            return type_info

        # Process parameters
        for name, param in sig.parameters.items():
            param_type = type_hints.get(name, str)
            type_info = get_type_info(param_type)

            parameters[name] = type_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required_params.append(name)
            else:
                # Add default value to schema if available
                parameters[name]["default"] = param.default

        # Create the function definition
        return cls(
            name=func.__name__,
            description=func.__doc__ or "No description provided.",
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required_params,
            },
            required=required_params,
            callable=func,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Type hint to JSON Schema type mapping (for AST-based parsing)
TYPE_MAPPING = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "List": "array",
    "Dict": "object",
    "Optional": "string",
    "Any": "string",
}


class FunctionParseError(Exception):
    """Raised when function parsing fails."""
    pass


@dataclass
class ParsedFunction:
    """Container for parsed function with source code."""
    definition: FunctionDefinition
    source_code: str
    category: str = "Uncategorized"


def function_to_openai_schema(func_def: FunctionDefinition) -> dict:
    """Convert FunctionDefinition to OpenAI function calling schema format."""
    return func_def.to_openai_schema()


def function_to_dict(func_def: FunctionDefinition, source_code: str = "") -> dict:
    """Convert FunctionDefinition to dictionary for storage."""
    result = func_def.to_dict()
    result["source_code"] = source_code
    result["required"] = func_def.required
    return result


def function_from_dict(data: dict) -> tuple[FunctionDefinition, str]:
    """Create FunctionDefinition from dictionary."""
    func_def = FunctionDefinition(
        name=data["name"],
        description=data.get("description", ""),
        parameters=data.get("parameters", {}),
        required=data.get("required", []),
    )
    return func_def, data.get("source_code", "")


def _get_type_from_annotation(annotation: ast.expr) -> str:
    """Extract type string from AST annotation node."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    elif isinstance(annotation, ast.Constant):
        return str(annotation.value)
    elif isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id
    elif isinstance(annotation, ast.Attribute):
        return annotation.attr
    return "string"


def _annotation_to_json_type(type_str: str) -> str:
    """Convert Python type annotation to JSON Schema type."""
    return TYPE_MAPPING.get(type_str, "string")


def _get_default_value(default: ast.expr) -> Any:
    """Extract default value from AST node."""
    if isinstance(default, ast.Constant):
        return default.value
    elif isinstance(default, ast.List):
        return []
    elif isinstance(default, ast.Dict):
        return {}
    elif isinstance(default, ast.Name):
        if default.id == "None":
            return None
        elif default.id == "True":
            return True
        elif default.id == "False":
            return False
    return None


def _parse_function_node(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionDefinition:
    """
    Parse an AST function node and extract FunctionDefinition.
    
    Args:
        node: ast.FunctionDef or ast.AsyncFunctionDef node
        
    Returns:
        FunctionDefinition object
    """
    name = node.name
    
    # Extract docstring as description
    description = ""
    if (node.body and 
        isinstance(node.body[0], ast.Expr) and 
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        description = node.body[0].value.value.strip()
    
    # Extract parameters
    parameters: Dict[str, Any] = {}
    required: List[str] = []
    
    args = node.args
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    defaults_offset = num_args - num_defaults
    
    for i, arg in enumerate(args.args):
        param_name = arg.arg
        
        if param_name == "self":
            continue
        
        type_str = "string"
        if arg.annotation:
            type_str = _get_type_from_annotation(arg.annotation)
        
        json_type = _annotation_to_json_type(type_str)
        
        param_info: Dict[str, Any] = {
            "type": json_type,
        }
        
        default_index = i - defaults_offset
        has_default = default_index >= 0 and default_index < num_defaults
        
        if has_default:
            default_value = _get_default_value(args.defaults[default_index])
            if default_value is not None:
                param_info["default"] = default_value
        else:
            required.append(param_name)
        
        param_info["description"] = f"Parameter '{param_name}' of type {type_str}"
        parameters[param_name] = param_info
    
    return FunctionDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": parameters,
            "required": required,
        },
        required=required,
    )


def parse_function(source_code: str) -> ParsedFunction:
    """
    Parse Python function code and extract FunctionDefinition.
    
    Args:
        source_code: Python code containing a function definition
        
    Returns:
        ParsedFunction with FunctionDefinition and source code
        
    Raises:
        FunctionParseError: If parsing fails
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise FunctionParseError(f"Syntax error in code: {e}")
    
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break
    
    if func_node is None:
        raise FunctionParseError("No function definition found in the code")
    
    func_definition = _parse_function_node(func_node)
    
    return ParsedFunction(definition=func_definition, source_code=source_code)


def parse_multiple_functions(source_code: str) -> List[ParsedFunction]:
    """
    Parse multiple function definitions from Python code.
    
    Args:
        source_code: Python code containing one or more function definitions
        
    Returns:
        List of ParsedFunction objects
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise FunctionParseError(f"Syntax error in code: {e}")
    
    definitions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                func_definition = _parse_function_node(node)
                func_source = ast.unparse(node)
                definitions.append(ParsedFunction(definition=func_definition, source_code=func_source))
            except Exception:
                continue
    
    return definitions


def validate_function_code(source_code: str) -> tuple[bool, str]:
    """
    Validate that the provided code is a valid Python function.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not source_code or not source_code.strip():
        return False, "Code is empty"
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    has_function = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_function = True
            break
    
    if not has_function:
        return False, "No function definition found. Code must contain at least one 'def' statement."
    
    return True, ""
