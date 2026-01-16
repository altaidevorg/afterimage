"""
Custom Tools page for defining and managing tool definitions.

Allows users to create tools either by writing Python function code
or by manually entering the function definition fields.
"""

import json
import gradio as gr
import pandas as pd

from core.function_parser import (
    FunctionDefinition,
    ParsedFunction,
    FunctionParseError,
    parse_function,
    validate_function_code,
)
from core.tools_db import get_tools_db


# Example function code for the editor
EXAMPLE_FUNCTION = '''def send_email(to: str, subject: str, body: str, cc: str = ""):
    """Send an email to the specified recipient."""
    pass
'''

# Example parameters JSON
EXAMPLE_PARAMS = {
    "recipient": {"type": "string", "description": "Email address of the recipient"},
    "message": {"type": "string", "description": "Content of the message"},
    "priority": {"type": "integer", "description": "Priority level (1-5)", "default": 3}
}


def create_custom_tools_page():
    """Create the Custom Tools management page."""
    
    with gr.Blocks() as page:
        gr.Markdown("## Custom Tools Manager")
        gr.Markdown(
            "Define custom tools for use in Tool Calling generation. "
            "Create from code or manually enter the definition."
        )
        
        with gr.Tabs():
            # --- TAB 1: From Code ---
            with gr.Tab("From Code"):
                current_parsed = gr.State(None)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Write Function Code")
                        
                        code_input = gr.Code(
                            label="Python Function",
                            language="python",
                            value=EXAMPLE_FUNCTION,
                            lines=12,
                        )
                        
                        with gr.Row():
                            parse_btn = gr.Button("Parse & Preview", variant="secondary")
                            clear_code_btn = gr.Button("Clear", variant="secondary")
                        
                        parse_status = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Preview & Save")
                        
                        preview_name = gr.Textbox(label="Function Name", interactive=False)
                        preview_desc = gr.Textbox(label="Description", interactive=False)
                        preview_params = gr.JSON(label="Parameters Schema")
                        preview_required = gr.JSON(label="Required Parameters")
                        
                        save_code_btn = gr.Button("Save Tool", variant="primary", interactive=False)
                        save_code_status = gr.Markdown("")
            
            # --- TAB 2: Manual Entry ---
            with gr.Tab("Manual Entry"):
                gr.Markdown("### Create Tool Manually")
                gr.Markdown("Enter the function definition fields directly without writing code.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        manual_name = gr.Textbox(
                            label="Function Name",
                            placeholder="e.g., send_notification",
                            info="Unique identifier for the tool (snake_case recommended)"
                        )
                        manual_desc = gr.Textbox(
                            label="Description",
                            placeholder="e.g., Send a push notification to the user",
                            lines=2,
                            info="Describe what this tool does"
                        )
                    
                    with gr.Column(scale=1):
                        manual_params = gr.Code(
                            label="Parameters (JSON)",
                            language="json",
                            value=json.dumps(EXAMPLE_PARAMS, indent=2),
                            lines=8,
                        )
                        manual_required = gr.Textbox(
                            label="Required Parameters",
                            placeholder="e.g., recipient, message",
                            info="Comma-separated list of required parameter names"
                        )
                
                with gr.Row():
                    validate_manual_btn = gr.Button("Validate", variant="secondary")
                    save_manual_btn = gr.Button("Save Tool", variant="primary")
                
                manual_status = gr.Markdown("")
            
            # --- TAB 3: Edit Existing ---
            with gr.Tab("Edit Tool"):
                gr.Markdown("### Edit Existing Tool")
                
                with gr.Row():
                    edit_select = gr.Dropdown(
                        label="Select Tool to Edit",
                        choices=[],
                        interactive=True,
                    )
                    load_edit_btn = gr.Button("Load", variant="secondary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        edit_name = gr.Textbox(label="Function Name", interactive=False)
                        edit_desc = gr.Textbox(label="Description", lines=2)
                    
                    with gr.Column(scale=1):
                        edit_params = gr.Code(
                            label="Parameters (JSON)",
                            language="json",
                            lines=8,
                        )
                        edit_required = gr.Textbox(
                            label="Required Parameters",
                            info="Comma-separated list"
                        )
                
                with gr.Row():
                    update_btn = gr.Button("Update Tool", variant="primary")
                    delete_edit_btn = gr.Button("Delete Tool", variant="stop")
                
                edit_status = gr.Markdown("")
        
        gr.Markdown("---")
        
        # Saved tools section
        gr.Markdown("### Saved Tools")
        
        tools_table = gr.Dataframe(
            headers=["Name", "Description", "Parameters"],
            label="Custom Tools",
            interactive=False,
            wrap=True,
        )
        
        refresh_btn = gr.Button("Refresh List", variant="secondary")
        
        # --- Event Handlers ---
        
        def load_tools_table():
            """Load all tools into the dataframe."""
            db = get_tools_db()
            tools = db.get_all_tools()
            
            if not tools:
                return pd.DataFrame(columns=["Name", "Description", "Parameters"])
            
            data = []
            for parsed in tools:
                param_names = list(parsed.definition.parameters.keys())
                params_str = ", ".join(param_names) if param_names else "(none)"
                desc = parsed.definition.description
                data.append({
                    "Name": parsed.definition.name,
                    "Description": desc[:100] + "..." if len(desc) > 100 else desc,
                    "Parameters": params_str,
                })
            
            return pd.DataFrame(data)
        
        def get_tool_names():
            """Get list of tool names for dropdown."""
            db = get_tools_db()
            return db.get_tool_names()
        
        # --- From Code Handlers ---
        
        def parse_code(code: str):
            """Parse the function code and return preview data."""
            if not code or not code.strip():
                return (
                    None, "", "", None, None,
                    "### Please enter some code",
                    gr.update(interactive=False),
                )
            
            is_valid, error = validate_function_code(code)
            if not is_valid:
                return (
                    None, "", "", None, None,
                    f"### Error: {error}",
                    gr.update(interactive=False),
                )
            
            try:
                parsed = parse_function(code)
            except FunctionParseError as e:
                return (
                    None, "", "", None, None,
                    f"### Parse Error: {str(e)}",
                    gr.update(interactive=False),
                )
            except Exception as e:
                return (
                    None, "", "", None, None,
                    f"### Unexpected Error: {str(e)}",
                    gr.update(interactive=False),
                )
            
            return (
                parsed,
                parsed.definition.name,
                parsed.definition.description or "(No description)",
                parsed.definition.parameters,
                parsed.definition.required,
                "### Parsed successfully!",
                gr.update(interactive=True),
            )
        
        def save_from_code(parsed: ParsedFunction):
            """Save the parsed tool to database."""
            if parsed is None:
                return "### Error: No tool to save. Parse a function first."
            
            db = get_tools_db()
            exists = db.tool_exists(parsed.definition.name)
            db.save_tool(parsed)
            
            if exists:
                return f"### Updated tool '{parsed.definition.name}'"
            return f"### Saved new tool '{parsed.definition.name}'"
        
        def clear_code_editor():
            """Clear the code editor and preview."""
            return (
                "", None, "", "", None, None, "",
                gr.update(interactive=False),
            )
        
        # --- Manual Entry Handlers ---
        
        def validate_manual(name: str, desc: str, params_json: str, required_str: str):
            """Validate manual entry fields."""
            if not name or not name.strip():
                return "### Error: Function name is required"
            
            if not name.replace("_", "").isalnum():
                return "### Error: Function name must be alphanumeric with underscores only"
            
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as e:
                return f"### Error: Invalid JSON in parameters: {e}"
            
            if not isinstance(params, dict):
                return "### Error: Parameters must be a JSON object"
            
            return "### Validation passed!"
        
        def save_manual(name: str, desc: str, params_json: str, required_str: str):
            """Save manually entered tool definition."""
            if not name or not name.strip():
                return "### Error: Function name is required"
            
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as e:
                return f"### Error: Invalid JSON: {e}"
            
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
                return f"### Updated tool '{name.strip()}'"
            return f"### Saved new tool '{name.strip()}'"
        
        # --- Edit Handlers ---
        
        def load_tool_for_edit(name: str):
            """Load a tool for editing."""
            if not name:
                return "", "", "", "", "### Select a tool first"
            
            db = get_tools_db()
            parsed = db.get_tool(name)
            
            if not parsed:
                return "", "", "", "", f"### Tool '{name}' not found"
            
            required_str = ", ".join(parsed.definition.required) if parsed.definition.required else ""
            params_json = json.dumps(parsed.definition.parameters, indent=2)
            
            return (
                parsed.definition.name,
                parsed.definition.description,
                params_json,
                required_str,
                f"### Loaded '{name}'"
            )
        
        def update_tool(name: str, desc: str, params_json: str, required_str: str):
            """Update an existing tool."""
            if not name:
                return "### Error: No tool loaded"
            
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError as e:
                return f"### Error: Invalid JSON: {e}"
            
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
            
            return f"### Updated tool '{name}'"
        
        def delete_tool_edit(name: str):
            """Delete the currently loaded tool."""
            if not name:
                return "### Error: No tool selected"
            
            db = get_tools_db()
            if db.delete_tool(name):
                return f"### Deleted tool '{name}'"
            return f"### Tool '{name}' not found"
        
        def refresh_edit_dropdown():
            """Refresh the edit dropdown choices."""
            return gr.update(choices=get_tool_names())
        
        # --- Wire up events ---
        
        # From Code tab
        parse_btn.click(
            fn=parse_code,
            inputs=[code_input],
            outputs=[
                current_parsed, preview_name, preview_desc,
                preview_params, preview_required, parse_status, save_code_btn,
            ],
        )
        
        save_code_btn.click(
            fn=save_from_code,
            inputs=[current_parsed],
            outputs=[save_code_status],
        ).then(
            fn=load_tools_table,
            outputs=[tools_table],
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )
        
        clear_code_btn.click(
            fn=clear_code_editor,
            outputs=[
                code_input, current_parsed, preview_name, preview_desc,
                preview_params, preview_required, parse_status, save_code_btn,
            ],
        )
        
        # Manual Entry tab
        validate_manual_btn.click(
            fn=validate_manual,
            inputs=[manual_name, manual_desc, manual_params, manual_required],
            outputs=[manual_status],
        )
        
        save_manual_btn.click(
            fn=save_manual,
            inputs=[manual_name, manual_desc, manual_params, manual_required],
            outputs=[manual_status],
        ).then(
            fn=load_tools_table,
            outputs=[tools_table],
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )
        
        # Edit tab
        load_edit_btn.click(
            fn=load_tool_for_edit,
            inputs=[edit_select],
            outputs=[edit_name, edit_desc, edit_params, edit_required, edit_status],
        )
        
        update_btn.click(
            fn=update_tool,
            inputs=[edit_name, edit_desc, edit_params, edit_required],
            outputs=[edit_status],
        ).then(
            fn=load_tools_table,
            outputs=[tools_table],
        )
        
        delete_edit_btn.click(
            fn=delete_tool_edit,
            inputs=[edit_name],
            outputs=[edit_status],
        ).then(
            fn=load_tools_table,
            outputs=[tools_table],
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )
        
        # Refresh button
        refresh_btn.click(
            fn=load_tools_table,
            outputs=[tools_table],
        )
        
        # Load on page load
        page.load(
            fn=load_tools_table,
            outputs=[tools_table],
        )
        
        page.load(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )
    
    return page
