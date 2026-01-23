"""
Tool Library page - View, create, and edit tool definitions.
"""

import json
import gradio as gr

from core.tools_db import get_tools_db
from core.function_parser import FunctionDefinition, ParsedFunction

from .handlers.custom_tools import (
    delete_tool_edit,
    parse_function_code,
    save_tool_from_code,
)


def get_builtin_tool_names():
    """Get set of built-in tool names."""
    builtin_names = set()
    try:
        from schemas import AVAILABLE_TOOLS
        for tool_cls in AVAILABLE_TOOLS:
            schema = tool_cls.model_json_schema()
            name_prop = schema.get("properties", {}).get("name", {})
            name = name_prop.get("default", tool_cls.__name__)
            builtin_names.add(name)
    except Exception:
        pass
    return builtin_names


def create_tool_library_page():
    """Create the Tool Library page."""
    
    with gr.Blocks() as page:
        gr.Markdown("## Tool Library")
        gr.Markdown("Manage your tools for synthetic data generation.")
        
        # States
        params_state = gr.State([])  # List of {name, type, description, required}
        editing_tool_name = gr.State(None)  # Original name when editing
        
        with gr.Row():
            # ========== LEFT: Tool List ==========
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### Tools")
                        refresh_btn = gr.Button("Refresh", size="sm", variant="secondary", scale=0, min_width=60)
                    
                    tool_radio = gr.Radio(
                        choices=[],
                        label=None,
                        show_label=False,
                        elem_id="tool-library-list",
                    )
                    
                    new_tool_btn = gr.Button("+ New Tool", variant="primary")
            
            # ========== RIGHT: Preview / Create / Edit ==========
            with gr.Column(scale=2):
                # Empty state
                with gr.Group(visible=True) as empty_panel:
                    gr.HTML("""
                    <div style="text-align: center; padding: 60px; color: #94a3b8;">
                        <div style="font-size: 16px;">Select a tool to view details</div>
                        <div style="font-size: 14px; margin-top: 8px;">or click "+ New Tool" to create one</div>
                    </div>
                    """)
                
                # Preview panel
                with gr.Group(visible=False) as preview_panel:
                    preview_title = gr.Markdown("### Tool Details")
                    preview_html = gr.HTML()
                    with gr.Row():
                        edit_tool_btn = gr.Button("Edit", variant="secondary")
                        delete_tool_btn = gr.Button("Delete", variant="stop")
                
                # Create/Edit panel  
                with gr.Group(visible=False) as create_panel:
                    create_title = gr.Markdown("### Create New Tool")
                    
                    with gr.Tabs() as create_tabs:
                        # ===== TAB 1: From Code =====
                        with gr.Tab("From Code"):
                            gr.Markdown("Paste a Python function with type hints and docstring:")
                            code_input = gr.Code(
                                language="python",
                                label="Python Function",
                                value='''def send_email(recipient: str, subject: str, body: str, priority: int = 3):
    """
    Send an email to the specified recipient.
    
    Args:
        recipient: Email address of the recipient
        subject: Subject line of the email
        body: Content of the email message
        priority: Priority level from 1 (low) to 5 (high)
    """
    pass''',
                                lines=15,
                            )
                            with gr.Row():
                                parse_btn = gr.Button("Parse & Preview", variant="secondary")
                                save_code_btn = gr.Button("Save Tool", variant="primary")
                            
                            code_preview = gr.HTML(visible=False)
                        
                        # ===== TAB 2: Manual Entry =====
                        with gr.Tab("Manual Entry"):
                            tool_name_input = gr.Textbox(label="Function Name", placeholder="e.g., send_notification")
                            tool_desc_input = gr.Textbox(label="Description", placeholder="What does this tool do?", lines=2)
                            
                            gr.Markdown("#### Parameters")
                            
                            # Dynamic parameters with @gr.render
                            @gr.render(inputs=[params_state])
                            def render_params(params):
                                if not params:
                                    gr.HTML('<div style="color: #94a3b8; padding: 12px; text-align: center;">No parameters yet. Click "Add Parameter" to add one.</div>')
                                else:
                                    for i, p in enumerate(params):
                                        with gr.Row():
                                            gr.Textbox(value=p.get("name", ""), label="Name", scale=2, interactive=False)
                                            gr.Textbox(value=p.get("type", ""), label="Type", scale=1, interactive=False)
                                            gr.Textbox(value=p.get("description", ""), label="Description", scale=3, interactive=False)
                                            gr.Checkbox(value=p.get("required", False), label="Req", scale=0, interactive=False)
                                            
                                            remove_btn = gr.Button("X", size="sm", variant="stop", scale=0, min_width=40)
                                            
                                            def make_remove_handler(idx):
                                                def handler(current_params):
                                                    new_params = current_params.copy()
                                                    if idx < len(new_params):
                                                        new_params.pop(idx)
                                                    return new_params
                                                return handler
                                            
                                            remove_btn.click(
                                                fn=make_remove_handler(i),
                                                inputs=[params_state],
                                                outputs=[params_state]
                                            )
                            
                            # Add parameter section
                            gr.Markdown("---")
                            with gr.Row():
                                new_param_name = gr.Textbox(label="Name", placeholder="param_name", scale=2)
                                new_param_type = gr.Dropdown(label="Type", choices=["string", "integer", "number", "boolean", "array", "object"], value="string", scale=1)
                                new_param_desc = gr.Textbox(label="Description", placeholder="Parameter description", scale=3)
                                new_param_req = gr.Checkbox(label="Required", value=True, scale=0)
                            
                            add_param_btn = gr.Button("+ Add Parameter", variant="secondary")
                            
                            gr.Markdown("---")
                            with gr.Row():
                                save_tool_btn = gr.Button("Save Tool", variant="primary", size="lg")
                    
                    # Cancel button outside tabs
                    cancel_create_btn = gr.Button("Cancel", variant="secondary")

        # ============================================================
        # HELPERS
        # ============================================================
        
        def get_tool_choices():
            db = get_tools_db()
            tools = db.get_all_tools()
            builtin_names = get_builtin_tool_names()
            choices = []
            for tool in tools:
                name = tool.definition.name
                is_builtin = name in builtin_names
                label = f"[built-in] {name}" if is_builtin else name
                choices.append(label)
            return choices
        
        def extract_name(choice):
            if not choice:
                return None
            if choice.startswith("[built-in] "):
                return choice.replace("[built-in] ", "")
            return choice
        
        def load_page():
            return gr.update(choices=get_tool_choices(), value=None)
        
        def show_tool_preview(choice):
            """Show tool details preview."""
            tool_name = extract_name(choice)
            if not tool_name:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "", "",
                )
            
            db = get_tools_db()
            tool = db.get_tool(tool_name)
            if not tool:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "", "",
                )
            
            builtin_names = get_builtin_tool_names()
            is_builtin = tool_name in builtin_names
            
            # Build preview HTML
            params_html = ""
            params = tool.definition.parameters
            if isinstance(params, dict):
                props = params.get("properties", params)
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    params_html += f"""
                    <div style="padding: 8px 12px; background: #f8fafc; border-radius: 6px; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #1e293b;">{param_name}</span>
                        <span style="color: #64748b; font-size: 12px; margin-left: 8px;">({param_type})</span>
                        <div style="font-size: 12px; color: #64748b; margin-top: 2px;">{param_desc}</div>
                    </div>
                    """
            
            if not params_html:
                params_html = '<div style="color: #94a3b8; padding: 12px;">No parameters</div>'
            
            badge = "Built-in" if is_builtin else "Custom"
            badge_color = "#dbeafe" if is_builtin else "#dcfce7"
            badge_text_color = "#1d4ed8" if is_builtin else "#166534"
            
            html = f"""
            <div style="padding: 4px 0;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="background: {badge_color}; color: {badge_text_color}; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500;">{badge}</span>
                </div>
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 13px; color: #64748b; margin-bottom: 4px;">Description</div>
                    <div style="font-size: 14px; color: #1e293b;">{tool.definition.description or "No description"}</div>
                </div>
                <div>
                    <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">Parameters</div>
                    {params_html}
                </div>
            </div>
            """
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                f"### {tool_name}",
                html,
            )
        
        def show_create_panel():
            """Show create new tool panel."""
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                "### Create New Tool",
                "",  # tool_name_input
                "",  # tool_desc_input
                [],  # params_state
                gr.update(value=None),  # tool_radio
                None,  # editing_tool_name - None for new tool
            )
        
        def show_edit_panel(choice):
            """Show edit panel for selected tool."""
            tool_name = extract_name(choice)
            if not tool_name:
                return show_create_panel()
            
            db = get_tools_db()
            tool = db.get_tool(tool_name)
            if not tool:
                return show_create_panel()
            
            # Convert parameters to list format
            params_list = []
            params = tool.definition.parameters
            required_list = tool.definition.required or []
            if isinstance(params, dict):
                props = params.get("properties", params)
                for param_name, param_info in props.items():
                    params_list.append({
                        "name": param_name,
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                        "required": param_name in required_list,
                    })
            
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                f"### Edit: {tool_name}",
                tool.definition.name,
                tool.definition.description or "",
                params_list,
                gr.update(),  # tool_radio unchanged
                tool_name,  # editing_tool_name - original name for edit
            )
        
        def add_parameter(params, name, ptype, desc, req):
            """Add a new parameter."""
            if not name or not name.strip():
                gr.Warning("Parameter name is required")
                return params, "", "string", "", True
            
            new_params = params.copy()
            new_params.append({
                "name": name.strip(),
                "type": ptype,
                "description": desc,
                "required": req,
            })
            return new_params, "", "string", "", True
        
        def save_tool(name, desc, params, original_name):
            """Save the tool to database."""
            if not name or not name.strip():
                raise gr.Error("Function name is required")
            
            name = name.strip()
            db = get_tools_db()
            
            # If editing and name changed, delete old one first
            if original_name and original_name != name:
                db.delete_tool(original_name)
            
            # Build parameters dict
            properties = {}
            required = []
            for p in params:
                properties[p["name"]] = {
                    "type": p["type"],
                    "description": p["description"],
                }
                if p.get("required"):
                    required.append(p["name"])
            
            params_schema = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
            
            func_def = FunctionDefinition(
                name=name,
                description=desc.strip() if desc else "",
                parameters=params_schema,
                required=required,
            )
            
            parsed = ParsedFunction(definition=func_def, source_code="")
            
            db.save_tool(parsed)
            gr.Info(f"Saved tool '{name}'")
        
        def after_save():
            """After save, refresh and show empty."""
            return (
                gr.update(choices=get_tool_choices(), value=None),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "", "",
            )
        
        def cancel_and_back(choice):
            """Cancel and go back."""
            if choice:
                return show_tool_preview(choice)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "", "",
            )
        
        def do_delete(choice):
            tool_name = extract_name(choice)
            if tool_name:
                delete_tool_edit(tool_name)
            return after_save()
        
        def do_parse_code(code):
            """Parse Python code and show preview."""
            result = parse_function_code(code)
            if not result or "error" in result:
                error_msg = result.get("error", "Failed to parse") if result else "Failed to parse"
                return gr.update(visible=True, value=f'<div style="color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 6px;">{error_msg}</div>')
            
            # Build preview HTML
            params_html = ""
            for p in result.get("parameters", []):
                req_badge = '<span style="color: #dc2626; font-size: 10px;">*</span>' if p.get("required") else ""
                params_html += f"""
                <div style="padding: 6px 10px; background: #f8fafc; border-radius: 4px; margin-bottom: 4px;">
                    <span style="font-weight: 500;">{p['name']}</span>{req_badge}
                    <span style="color: #64748b; font-size: 12px; margin-left: 6px;">({p['type']})</span>
                    <span style="color: #64748b; font-size: 12px; margin-left: 8px;">- {p.get('description', '')}</span>
                </div>
                """
            
            html = f"""
            <div style="padding: 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; margin-top: 12px;">
                <div style="font-weight: 600; color: #166534; margin-bottom: 8px;">Parsed Successfully</div>
                <div><strong>Name:</strong> {result['name']}</div>
                <div style="margin: 8px 0;"><strong>Description:</strong> {result.get('description', 'N/A')}</div>
                <div><strong>Parameters:</strong></div>
                {params_html if params_html else '<div style="color: #94a3b8;">No parameters</div>'}
            </div>
            """
            return gr.update(visible=True, value=html)
        
        def do_save_from_code(code):
            """Save tool from Python code."""
            save_tool_from_code(code)

        # ============================================================
        # EVENTS
        # ============================================================
        
        page.load(fn=load_page, outputs=[tool_radio])
        refresh_btn.click(fn=load_page, outputs=[tool_radio])
        
        # Tool selection → show preview
        tool_radio.change(
            fn=show_tool_preview,
            inputs=[tool_radio],
            outputs=[empty_panel, preview_panel, create_panel, preview_title, preview_html],
        )
        
        # New Tool button
        new_tool_btn.click(
            fn=show_create_panel,
            outputs=[empty_panel, preview_panel, create_panel, create_title, tool_name_input, tool_desc_input, params_state, tool_radio, editing_tool_name],
        )
        
        # Edit button
        edit_tool_btn.click(
            fn=show_edit_panel,
            inputs=[tool_radio],
            outputs=[empty_panel, preview_panel, create_panel, create_title, tool_name_input, tool_desc_input, params_state, tool_radio, editing_tool_name],
        )
        
        # Delete button
        delete_tool_btn.click(
            fn=do_delete,
            inputs=[tool_radio],
            outputs=[tool_radio, empty_panel, preview_panel, create_panel, preview_title, preview_html],
        )
        
        # Add parameter
        add_param_btn.click(
            fn=add_parameter,
            inputs=[params_state, new_param_name, new_param_type, new_param_desc, new_param_req],
            outputs=[params_state, new_param_name, new_param_type, new_param_desc, new_param_req],
        )
        
        # Save tool
        save_tool_btn.click(
            fn=save_tool,
            inputs=[tool_name_input, tool_desc_input, params_state, editing_tool_name],
        ).then(
            fn=after_save,
            outputs=[tool_radio, empty_panel, preview_panel, create_panel, preview_title, preview_html],
        )
        
        # Cancel
        cancel_create_btn.click(
            fn=cancel_and_back,
            inputs=[tool_radio],
            outputs=[empty_panel, preview_panel, create_panel, preview_title, preview_html],
        )
        
        # From Code: Parse
        parse_btn.click(
            fn=do_parse_code,
            inputs=[code_input],
            outputs=[code_preview],
        )
        
        # From Code: Save
        save_code_btn.click(
            fn=do_save_from_code,
            inputs=[code_input],
        ).then(
            fn=after_save,
            outputs=[tool_radio, empty_panel, preview_panel, create_panel, preview_title, preview_html],
        )
    
    return page
