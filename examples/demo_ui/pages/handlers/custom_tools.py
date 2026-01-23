"""
Handlers for the Custom Tools management page.
"""
import json
import gradio as gr
from core.function_parser import (
    FunctionDefinition,
    ParsedFunction,
    FunctionParseError,
    parse_function,
    validate_function_code,
)
from core.tools_db import get_tools_db

# Example parameters JSON
EXAMPLE_PARAMS = {
    "recipient": {"type": "string", "description": "Email address of the recipient"},
    "message": {"type": "string", "description": "Content of the message"},
    "priority": {"type": "integer", "description": "Priority level (1-5)", "default": 3}
}


def get_tool_names():
    """Get list of tool names for dropdown."""
    db = get_tools_db()
    return db.get_tool_names()


def parse_code(code: str):
    """Parse the function code and return preview data."""
    if not code or not code.strip():
        gr.Warning("Please enter some code to parse.")
        return (
            None, "", "", None, None,
            gr.update(interactive=False),
        )
    
    is_valid, error = validate_function_code(code)
    if not is_valid:
        gr.Error(f"Validation Error: {error}")
        return (
            None, "", "", None, None,
            gr.update(interactive=False),
        )
    
    try:
        parsed = parse_function(code)
    except FunctionParseError as e:
        gr.Error(f"Parse Error: {str(e)}")
        return (
            None, "", "", None, None,
            gr.update(interactive=False),
        )
    except Exception as e:
        gr.Error(f"Unexpected Error: {str(e)}")
        return (
            None, "", "", None, None,
            gr.update(interactive=False),
        )
    
    gr.Info("Function parsed successfully!")
    return (
        parsed,
        parsed.definition.name,
        parsed.definition.description or "(No description)",
        parsed.definition.parameters,
        parsed.definition.required,
        gr.update(interactive=True),
    )


def save_from_code(parsed: ParsedFunction):
    """Save the parsed tool to database."""
    if parsed is None:
        gr.Warning("No tool to save. Parse a function first.")
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update()
        )
    
    db = get_tools_db()
    exists = db.tool_exists(parsed.definition.name)
    db.save_tool(parsed)
    
    if exists:
        gr.Info(f"Updated tool '{parsed.definition.name}'")
    else:
        gr.Info(f"Saved new tool '{parsed.definition.name}'")
    
    return (
        "", None,  # Clear code inputs
        "", "", None, None, # Clear preview
        gr.update(interactive=False) # Disable save btn
    )


def clear_code_editor():
    """Clear the code editor and preview."""
    return (
        "", None, "", "", None, None,
        gr.update(interactive=False),
    )


def validate_manual(name: str, desc: str, params_json: str, required_str: str):
    """Validate manual entry fields."""
    if not name or not name.strip():
        gr.Error("Function name is required")
        return
    
    if not name.replace("_", "").isalnum():
        gr.Error("Function name must be alphanumeric with underscores only")
        return
    
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        gr.Error(f"Invalid JSON in parameters: {e}")
        return
    
    if not isinstance(params, dict):
        gr.Error("Parameters must be a JSON object")
        return
    
    gr.Info("Validation passed!")


def save_manual(name: str, desc: str, params_json: str, required_str: str):
    """Save manually entered tool definition."""
    if not name or not name.strip():
        gr.Error("Function name is required")
        return (
            gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        gr.Error(f"Invalid JSON: {e}")
        return (
            gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    required = [r.strip() for r in required_str.split(",") if r.strip()] if required_str else []
    
    func_def = FunctionDefinition(
        name=name.strip(),
        description=desc.strip() if desc else "",
        parameters=params,
        required=required,
    )
    
    parsed = ParsedFunction(definition=func_def, source_code="")
    
    db = get_tools_db()
    exists = db.tool_exists(name.strip())
    db.save_tool(parsed)
    
    if exists:
        gr.Info(f"Updated tool '{name.strip()}'")
    else:
        gr.Info(f"Saved new tool '{name.strip()}'")
    
    return (
        "", "", # Name, Desc
        json.dumps(EXAMPLE_PARAMS, indent=2), # Reset params to example
        "" # Required
    )


def load_tool_for_edit(name: str):
    """Load a tool for editing."""
    if not name:
        gr.Warning("Select a tool first")
        return "", "", "", ""
    
    db = get_tools_db()
    parsed = db.get_tool(name)
    
    if not parsed:
        gr.Error(f"Tool '{name}' not found")
        return "", "", "", ""
    
    required_str = ", ".join(parsed.definition.required) if parsed.definition.required else ""
    params_json = json.dumps(parsed.definition.parameters, indent=2)
    
    gr.Info(f"Loaded '{name}'")
    return (
        parsed.definition.name,
        parsed.definition.description,
        params_json,
        required_str,
    )


def update_tool(name: str, desc: str, params_json: str, required_str: str):
    """Update an existing tool."""
    if not name:
        gr.Error("No tool loaded")
        return
    
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as e:
        gr.Error(f"Invalid JSON: {e}")
        return
    
    required = [r.strip() for r in required_str.split(",") if r.strip()] if required_str else []
    
    func_def = FunctionDefinition(
        name=name,
        description=desc.strip() if desc else "",
        parameters=params,
        required=required,
    )
    
    parsed = ParsedFunction(definition=func_def, source_code="")
    
    db = get_tools_db()
    db.save_tool(parsed)
    
    gr.Info(f"Updated tool '{name}'")


def delete_tool_edit(name: str):
    """Delete the currently loaded tool."""
    if not name:
        gr.Error("No tool selected")
        return
    
    db = get_tools_db()
    if db.delete_tool(name):
        gr.Info(f"Deleted tool '{name}'")
        return
    gr.Error(f"Tool '{name}' not found")


def refresh_edit_dropdown():
    """Refresh the edit dropdown choices."""
    return gr.update(choices=get_tool_names())


def parse_function_code(code: str):
    """Parse function code and return dict for preview (used by tool library)."""
    if not code or not code.strip():
        return {"error": "Please enter some code to parse."}
    
    is_valid, error = validate_function_code(code)
    if not is_valid:
        return {"error": f"Validation Error: {error}"}
    
    try:
        parsed = parse_function(code)
    except FunctionParseError as e:
        return {"error": f"Parse Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}
    
    # Convert to dict for preview
    params_list = []
    params = parsed.definition.parameters
    required = parsed.definition.required or []
    if isinstance(params, dict):
        props = params.get("properties", params)
        for pname, pinfo in props.items():
            params_list.append({
                "name": pname,
                "type": pinfo.get("type", "any"),
                "description": pinfo.get("description", ""),
                "required": pname in required,
            })
    
    return {
        "name": parsed.definition.name,
        "description": parsed.definition.description or "",
        "parameters": params_list,
    }


def save_tool_from_code(code: str):
    """Parse and save tool from Python code (used by tool library)."""
    if not code or not code.strip():
        raise gr.Error("Please enter some code to parse.")
    
    is_valid, error = validate_function_code(code)
    if not is_valid:
        raise gr.Error(f"Validation Error: {error}")
    
    try:
        parsed = parse_function(code)
    except FunctionParseError as e:
        raise gr.Error(f"Parse Error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Unexpected Error: {str(e)}")
    
    db = get_tools_db()
    db.save_tool(parsed)
    gr.Info(f"Saved tool '{parsed.definition.name}'")
