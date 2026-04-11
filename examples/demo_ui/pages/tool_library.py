"""
Tool Library page - View, create, and edit tool definitions.
Supports category-based grouping with accordion view.
"""

import json
import gradio as gr

from core.config import MAX_CATEGORIES
from core.tools_db import get_tools_db
from core.function_parser import FunctionDefinition, ParsedFunction

from .handlers.custom_tools import (
    delete_tool_edit,
    parse_function_code,
    save_tool_from_code,
)
from core.mcp_wrapper import (
    MCPClient,
    RemoteMCPClient,
    MCPConfigClient,
    mcp_tool_to_function_def,
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
    """Create the Tool Library page with category support."""

    with gr.Blocks() as page:
        gr.Markdown("## Tool Library")
        gr.Markdown("Manage your tools for synthetic data generation.")

        # States
        params_state = gr.State([])  # List of {name, type, description, required}
        editing_tool_name = gr.State(None)  # Original name when editing
        editing_tool_category = gr.State("Uncategorized")  # Category when editing
        tools_by_category_state = gr.State({})  # Dict of category -> list of tools
        selected_tool_state = gr.State(None)  # Currently selected tool name

        # Pre-load tools for static structure
        db = get_tools_db()
        initial_grouped = db.get_tools_by_category()
        initial_cats = list(initial_grouped.keys())
        builtin_names = get_builtin_tool_names()

        def get_tool_choices(tools):
            choices = []
            for t in tools:
                name = t.definition.name
                label = f"[built-in] {name}" if name in builtin_names else name
                choices.append((label, name))
            return choices

        with gr.Row():
            # ========== LEFT: Tool List (Grouped by Category) ==========
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### Tools")
                        refresh_btn = gr.Button(
                            "Refresh",
                            size="sm",
                            variant="secondary",
                            scale=0,
                            min_width=60,
                        )

                    # Static category slots with Radio buttons
                    cat_accordions = []
                    cat_radios = []

                    for i in range(MAX_CATEGORIES):
                        if i < len(initial_cats):
                            cat = initial_cats[i]
                            tools = initial_grouped[cat]
                            choices = get_tool_choices(tools)
                            with gr.Accordion(
                                f"{cat} ({len(tools)})", open=True
                            ) as acc:
                                radio = gr.Radio(
                                    choices=choices,
                                    value=None,
                                    label=None,
                                    show_label=False,
                                )
                            cat_accordions.append(acc)
                            cat_radios.append(radio)
                        else:
                            with gr.Accordion(
                                "Empty", open=False, visible=False
                            ) as acc:
                                radio = gr.Radio(choices=[], value=None, visible=False)
                            cat_accordions.append(acc)
                            cat_radios.append(radio)

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

                    # Quick category change
                    with gr.Row():
                        preview_category_dropdown = gr.Dropdown(
                            label="Change Category",
                            choices=["Uncategorized"],
                            value="Uncategorized",
                            allow_custom_value=True,
                            scale=2,
                        )
                        change_category_btn = gr.Button(
                            "Save Category", variant="primary", scale=1
                        )

                    with gr.Row():
                        edit_tool_btn = gr.Button("Edit", variant="secondary")
                        delete_tool_btn = gr.Button("Delete", variant="stop")

                # Create/Edit panel
                with gr.Group(visible=False) as create_panel:
                    create_title = gr.Markdown("### Create New Tool")

                    # Category selection (outside tabs, applies to both)
                    with gr.Row():
                        category_dropdown = gr.Dropdown(
                            label="Category",
                            choices=["Uncategorized"],
                            value="Uncategorized",
                            allow_custom_value=True,
                            scale=2,
                            info="Select existing or type new category",
                        )

                    with gr.Tabs() as create_tabs:
                        # ===== TAB 1: Import from MCP =====
                        with gr.Tab("Import from MCP"):
                            gr.Markdown(
                                "Connect to an MCP server and fetch tool definitions."
                            )

                            connection_type = gr.Radio(
                                choices=[
                                    "Local (Command)",
                                    "Remote (URL)",
                                    "Config (JSON)",
                                ],
                                value="Local (Command)",
                                label="Connection Type",
                            )

                            # Local Inputs
                            with gr.Group(visible=True) as local_group:
                                with gr.Row():
                                    mcp_command = gr.Textbox(
                                        label="Server Command",
                                        placeholder="e.g., npx, uvx, python",
                                        scale=1,
                                        value="npx",
                                    )
                                    mcp_args = gr.Textbox(
                                        label="Arguments",
                                        placeholder="e.g., -y @modelcontextprotocol/server-filesystem /path/to/files",
                                        scale=3,
                                        value="-y @modelcontextprotocol/server-filesystem .",
                                    )

                            # Remote Inputs
                            with gr.Group(visible=False) as remote_group:
                                mcp_url = gr.Textbox(
                                    label="Server URL (SSE Endpoint)",
                                    placeholder="e.g., http://localhost:8000/sse",
                                    scale=1,
                                )

                            # Config Inputs
                            with gr.Group(visible=False) as config_group:
                                mcp_config = gr.Code(
                                    label="MCP Config (JSON)",
                                    language="json",
                                    lines=10,
                                    value="""{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}""",
                                )

                            connect_btn = gr.Button(
                                "Connect & Fetch Tools", variant="secondary"
                            )

                            mcp_status = gr.Markdown(visible=False)
                            found_tools_state = gr.State([])

                            # Dynamic checklist for found tools
                            mcp_tools_checkbox = gr.CheckboxGroup(
                                label="Found Tools",
                                choices=[],
                                visible=False,
                                info="Select tools to import",
                            )

                            import_btn = gr.Button(
                                "Import Selected Tools",
                                variant="primary",
                                visible=False,
                            )

                            # Visibility Toggle Handler
                            def toggle_mcp_inputs(conn_type):
                                if conn_type == "Local (Command)":
                                    return (
                                        gr.update(visible=True),
                                        gr.update(visible=False),
                                        gr.update(visible=False),
                                    )
                                elif conn_type == "Remote (URL)":
                                    return (
                                        gr.update(visible=False),
                                        gr.update(visible=True),
                                        gr.update(visible=False),
                                    )
                                else:
                                    return (
                                        gr.update(visible=False),
                                        gr.update(visible=False),
                                        gr.update(visible=True),
                                    )

                            connection_type.change(
                                fn=toggle_mcp_inputs,
                                inputs=[connection_type],
                                outputs=[local_group, remote_group, config_group],
                            )

                        # ===== TAB 2: Manual Entry =====
                        with gr.Tab("Manual Entry"):
                            tool_name_input = gr.Textbox(
                                label="Function Name",
                                placeholder="e.g., send_notification",
                            )
                            tool_desc_input = gr.Textbox(
                                label="Description",
                                placeholder="What does this tool do?",
                                lines=2,
                            )

                            gr.Markdown("#### Parameters")

                            # Dynamic parameters with @gr.render
                            @gr.render(inputs=[params_state])
                            def render_params(params):
                                if not params:
                                    gr.HTML(
                                        '<div style="color: #94a3b8; padding: 12px; text-align: center;">No parameters yet. Click "Add Parameter" to add one.</div>'
                                    )
                                else:
                                    for i, p in enumerate(params):
                                        with gr.Row():
                                            gr.Textbox(
                                                value=p.get("name", ""),
                                                label="Name",
                                                scale=2,
                                                interactive=False,
                                            )
                                            gr.Textbox(
                                                value=p.get("type", ""),
                                                label="Type",
                                                scale=1,
                                                interactive=False,
                                            )
                                            gr.Textbox(
                                                value=p.get("description", ""),
                                                label="Description",
                                                scale=3,
                                                interactive=False,
                                            )
                                            gr.Checkbox(
                                                value=p.get("required", False),
                                                label="Req",
                                                scale=0,
                                                interactive=False,
                                            )

                                            remove_btn = gr.Button(
                                                "X",
                                                size="sm",
                                                variant="stop",
                                                scale=0,
                                                min_width=40,
                                            )

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
                                                outputs=[params_state],
                                            )

                            # Add parameter section
                            gr.Markdown("---")
                            with gr.Row():
                                new_param_name = gr.Textbox(
                                    label="Name", placeholder="param_name", scale=2
                                )
                                new_param_type = gr.Dropdown(
                                    label="Type",
                                    choices=[
                                        "string",
                                        "integer",
                                        "number",
                                        "boolean",
                                        "array",
                                        "object",
                                    ],
                                    value="string",
                                    scale=1,
                                )
                                new_param_desc = gr.Textbox(
                                    label="Description",
                                    placeholder="Parameter description",
                                    scale=3,
                                )
                                new_param_req = gr.Checkbox(
                                    label="Required", value=True, scale=0
                                )

                            add_param_btn = gr.Button(
                                "+ Add Parameter", variant="secondary"
                            )

                            gr.Markdown("---")
                            with gr.Row():
                                save_tool_btn = gr.Button(
                                    "Save Tool", variant="primary", size="lg"
                                )

                        # ===== TAB 3: From Code =====
                        with gr.Tab("From Code"):
                            gr.Markdown(
                                "Paste a Python function with type hints and docstring:"
                            )
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
                                parse_btn = gr.Button(
                                    "Parse & Preview", variant="secondary"
                                )
                                save_code_btn = gr.Button(
                                    "Save Tool", variant="primary"
                                )

                            code_preview = gr.HTML(visible=False)

                    # Cancel button outside tabs
                    cancel_create_btn = gr.Button("Cancel", variant="secondary")

        # ============================================================
        # HELPERS
        # ============================================================

        def load_tools_grouped():
            """Load tools grouped by category."""
            db = get_tools_db()
            return db.get_tools_by_category()

        def get_category_choices():
            """Get list of categories for dropdown."""
            db = get_tools_db()
            categories = db.get_categories()
            if "Uncategorized" not in categories:
                categories.append("Uncategorized")
            return categories

        def extract_name(choice):
            if not choice:
                return None
            if choice.startswith("[built-in] "):
                return choice.replace("[built-in] ", "")
            return choice

        def load_page():
            """Load page: update accordions, radios, and category dropdown."""
            grouped = load_tools_grouped()
            cats = list(grouped.keys())
            categories = get_category_choices()

            results = []
            for i in range(MAX_CATEGORIES):
                if i < len(cats):
                    cat = cats[i]
                    tools = grouped[cat]
                    choices = get_tool_choices(tools)
                    results.append(
                        gr.update(
                            label=f"{cat} ({len(tools)})", visible=True, open=True
                        )
                    )
                    results.append(gr.update(choices=choices, value=None, visible=True))
                else:
                    results.append(gr.update(visible=False))
                    results.append(gr.update(choices=[], value=None, visible=False))

            results.append(gr.update(choices=categories))  # category_dropdown
            return results

        def show_tool_preview(tool_name):
            """Show tool details preview."""
            categories = get_category_choices()

            if not tool_name:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "",
                    gr.update(choices=categories, value="Uncategorized"),
                )

            db = get_tools_db()
            tool = db.get_tool(tool_name)
            if not tool:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    "",
                    "",
                    gr.update(choices=categories, value="Uncategorized"),
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
                params_html = (
                    '<div style="color: #94a3b8; padding: 12px;">No parameters</div>'
                )

            badge = "Built-in" if is_builtin else "Custom"
            badge_color = "#dbeafe" if is_builtin else "#dcfce7"
            badge_text_color = "#1d4ed8" if is_builtin else "#166534"

            # Category badge
            category = tool.category or "Uncategorized"

            html = f"""
            <div style="padding: 4px 0;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="background: {badge_color}; color: {badge_text_color}; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500;">{badge}</span>
                    <span style="background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 500;">{category}</span>
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
                gr.update(choices=categories, value=category),
            )

        def show_create_panel():
            """Show create new tool panel."""
            categories = get_category_choices()
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                "### Create New Tool",
                "",  # tool_name_input
                "",  # tool_desc_input
                [],  # params_state
                None,  # selected_tool_state
                None,  # editing_tool_name - None for new tool
                "Uncategorized",  # editing_tool_category
                gr.update(
                    choices=categories, value="Uncategorized"
                ),  # category_dropdown
            )

        def show_edit_panel(tool_name):
            """Show edit panel for selected tool."""
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
                    params_list.append(
                        {
                            "name": param_name,
                            "type": param_info.get("type", "string"),
                            "description": param_info.get("description", ""),
                            "required": param_name in required_list,
                        }
                    )

            category = tool.category or "Uncategorized"
            categories = get_category_choices()
            if category not in categories:
                categories.append(category)

            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                f"### Edit: {tool_name}",
                tool.definition.name,
                tool.definition.description or "",
                params_list,
                gr.update(),  # selected_tool_state unchanged
                tool_name,  # editing_tool_name - original name for edit
                category,  # editing_tool_category
                gr.update(choices=categories, value=category),  # category_dropdown
            )

        def add_parameter(params, name, ptype, desc, req):
            """Add a new parameter."""
            if not name or not name.strip():
                gr.Warning("Parameter name is required")
                return params, "", "string", "", True

            new_params = params.copy()
            new_params.append(
                {
                    "name": name.strip(),
                    "type": ptype,
                    "description": desc,
                    "required": req,
                }
            )
            return new_params, "", "string", "", True

        def save_tool(name, desc, params, original_name, category):
            """Save the tool to database with category."""
            if not name or not name.strip():
                raise gr.Error("Function name is required")

            name = name.strip()
            category = category.strip() if category else "Uncategorized"
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

            parsed = ParsedFunction(
                definition=func_def, source_code="", category=category
            )

            db.save_tool(parsed)
            gr.Info(f"Saved tool '{name}' in category '{category}'")

        def after_save():
            """After save, refresh accordions/radios and show empty."""
            grouped = load_tools_grouped()
            cats = list(grouped.keys())
            categories = get_category_choices()

            results = []
            # Update accordions and radios
            for i in range(MAX_CATEGORIES):
                if i < len(cats):
                    cat = cats[i]
                    tools = grouped[cat]
                    choices = get_tool_choices(tools)
                    results.append(
                        gr.update(
                            label=f"{cat} ({len(tools)})", visible=True, open=True
                        )
                    )
                    results.append(gr.update(choices=choices, value=None, visible=True))
                else:
                    results.append(gr.update(visible=False))
                    results.append(gr.update(choices=[], value=None, visible=False))

            # Panel visibility and other outputs
            results.extend(
                [
                    None,  # selected_tool_state
                    gr.update(visible=True),  # empty_panel
                    gr.update(visible=False),  # preview_panel
                    gr.update(visible=False),  # create_panel
                    "",
                    "",  # preview_title, preview_html
                    gr.update(choices=categories),  # category_dropdown
                ]
            )
            return results

        def cancel_and_back(tool_name):
            """Cancel and go back."""
            categories = get_category_choices()
            if tool_name:
                result = show_tool_preview(tool_name)
                # Unpack all 6 values
                return (
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4],
                    result[5],
                )
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                "",
                gr.update(choices=categories, value="Uncategorized"),
            )

        def do_delete(tool_name):
            tool_name = extract_name(tool_name)
            if tool_name:
                delete_tool_edit(tool_name)
            # Same as after_save
            return after_save()

        def change_tool_category(tool_name, new_category):
            """Change tool's category."""
            if not tool_name:
                raise gr.Error("No tool selected")

            new_category = new_category.strip() if new_category else "Uncategorized"
            db = get_tools_db()
            db.update_tool_category(tool_name, new_category)
            gr.Info(f"Moved '{tool_name}' to '{new_category}'")
            return after_save()

        def do_parse_code(code):
            """Parse Python code and show preview."""
            result = parse_function_code(code)
            if not result or "error" in result:
                error_msg = (
                    result.get("error", "Failed to parse")
                    if result
                    else "Failed to parse"
                )
                return gr.update(
                    visible=True,
                    value=f'<div style="color: #dc2626; padding: 12px; background: #fef2f2; border-radius: 6px;">{error_msg}</div>',
                )

            # Build preview HTML
            params_html = ""
            for p in result.get("parameters", []):
                req_badge = (
                    '<span style="color: #dc2626; font-size: 10px;">*</span>'
                    if p.get("required")
                    else ""
                )
                params_html += f"""
                <div style="padding: 6px 10px; background: #f8fafc; border-radius: 4px; margin-bottom: 4px;">
                    <span style="font-weight: 500;">{p["name"]}</span>{req_badge}
                    <span style="color: #64748b; font-size: 12px; margin-left: 6px;">({p["type"]})</span>
                    <span style="color: #64748b; font-size: 12px; margin-left: 8px;">- {p.get("description", "")}</span>
                </div>
                """

            html = f"""
            <div style="padding: 12px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; margin-top: 12px;">
                <div style="font-weight: 600; color: #166534; margin-bottom: 8px;">Parsed Successfully</div>
                <div><strong>Name:</strong> {result["name"]}</div>
                <div style="margin: 8px 0;"><strong>Description:</strong> {result.get("description", "N/A")}</div>
                <div><strong>Parameters:</strong></div>
                {params_html if params_html else '<div style="color: #94a3b8;">No parameters</div>'}
            </div>
            """
            return gr.update(visible=True, value=html)

        def do_save_from_code(code, category):
            """Save tool from Python code with category."""
            save_tool_from_code(code, category=category)

        def fetch_mcp_tools(conn_type, cmd, args_str, url, config_str):
            """Connect to MCP server (Local or Remote or Config) and fetch tools."""
            try:
                if conn_type == "Local (Command)":
                    if not cmd:
                        raise gr.Error("Command is required for local connection")
                    args = args_str.split() if args_str else []
                    client = MCPClient(cmd, args)
                elif conn_type == "Remote (URL)":
                    if not url:
                        raise gr.Error("URL is required for remote connection")

                    # Remote connection without custom headers in UI
                    client = RemoteMCPClient(url)
                else:  # Config (JSON)
                    if not config_str or not config_str.strip():
                        raise gr.Error("Config JSON is required")
                    try:
                        config = json.loads(config_str)
                    except json.JSONDecodeError as e:
                        raise gr.Error(f"Invalid JSON: {str(e)}")

                    client = MCPConfigClient(config)

                tools = client.fetch_tools()

                if not tools:
                    return (
                        gr.update(visible=True, value="No tools found on this server."),
                        [],
                        gr.update(choices=[], value=[], visible=False),
                        gr.update(visible=False),
                    )

                tool_names = [t["name"] for t in tools]
                tool_defs = [mcp_tool_to_function_def(t) for t in tools]

                msg = f"Found {len(tools)} tools: {', '.join(tool_names)}"

                return (
                    gr.update(visible=True, value=msg),
                    tool_defs,
                    gr.update(choices=tool_names, value=tool_names, visible=True),
                    gr.update(visible=True),
                )

            except Exception as e:
                raise gr.Error(f"MCP Connection Failed: {str(e)}")

        def save_mcp_tools(selected_names, all_tools, category):
            """Save selected MCP tools to database."""
            if not selected_names:
                raise gr.Warning("No tools selected")

            count = 0
            db = get_tools_db()
            category = category or "Uncategorized"

            for tool_def in all_tools:
                if tool_def["name"] in selected_names:
                    # Create FunctionDefinition
                    func_def = FunctionDefinition(
                        name=tool_def["name"],
                        description=tool_def["description"],
                        parameters=tool_def["parameters"],
                        required=tool_def["required"],
                    )

                    parsed = ParsedFunction(
                        definition=func_def,
                        source_code=f"# Imported from MCP Server\n# Tool: {tool_def['name']}",
                        category=category,
                    )

                    if db.save_tool(parsed):
                        count += 1

            gr.Info(f"Successfully imported {count} tools to '{category}'")
            return after_save()

        # ============================================================
        # EVENTS
        # ============================================================

        # Build output lists
        load_outputs = []
        for i in range(MAX_CATEGORIES):
            load_outputs.append(cat_accordions[i])
            load_outputs.append(cat_radios[i])
        load_outputs.append(category_dropdown)

        after_save_outputs = []
        for i in range(MAX_CATEGORIES):
            after_save_outputs.append(cat_accordions[i])
            after_save_outputs.append(cat_radios[i])
        after_save_outputs.extend(
            [
                selected_tool_state,
                empty_panel,
                preview_panel,
                create_panel,
                preview_title,
                preview_html,
                category_dropdown,
            ]
        )

        # Page load
        page.load(fn=load_page, outputs=load_outputs)

        # Refresh button
        refresh_btn.click(fn=load_page, outputs=load_outputs)

        # Radio selection → update selected_tool_state
        for radio in cat_radios:
            radio.change(
                fn=lambda x: x,
                inputs=[radio],
                outputs=[selected_tool_state],
            )

        # Tool selection → show preview (triggered by selected_tool_state change)
        selected_tool_state.change(
            fn=show_tool_preview,
            inputs=[selected_tool_state],
            outputs=[
                empty_panel,
                preview_panel,
                create_panel,
                preview_title,
                preview_html,
                preview_category_dropdown,
            ],
        )

        # Change category button
        change_category_btn.click(
            fn=change_tool_category,
            inputs=[selected_tool_state, preview_category_dropdown],
            outputs=after_save_outputs,
        )

        # New Tool button
        new_tool_btn.click(
            fn=show_create_panel,
            outputs=[
                empty_panel,
                preview_panel,
                create_panel,
                create_title,
                tool_name_input,
                tool_desc_input,
                params_state,
                selected_tool_state,
                editing_tool_name,
                editing_tool_category,
                category_dropdown,
            ],
        )

        # Edit button
        edit_tool_btn.click(
            fn=show_edit_panel,
            inputs=[selected_tool_state],
            outputs=[
                empty_panel,
                preview_panel,
                create_panel,
                create_title,
                tool_name_input,
                tool_desc_input,
                params_state,
                selected_tool_state,
                editing_tool_name,
                editing_tool_category,
                category_dropdown,
            ],
        )

        # Delete button
        delete_tool_btn.click(
            fn=do_delete,
            inputs=[selected_tool_state],
            outputs=after_save_outputs,
        )

        # Add parameter
        add_param_btn.click(
            fn=add_parameter,
            inputs=[
                params_state,
                new_param_name,
                new_param_type,
                new_param_desc,
                new_param_req,
            ],
            outputs=[
                params_state,
                new_param_name,
                new_param_type,
                new_param_desc,
                new_param_req,
            ],
        )

        # Save tool (Manual Entry)
        save_tool_btn.click(
            fn=save_tool,
            inputs=[
                tool_name_input,
                tool_desc_input,
                params_state,
                editing_tool_name,
                category_dropdown,
            ],
        ).then(
            fn=after_save,
            outputs=after_save_outputs,
        )

        # Cancel
        cancel_create_btn.click(
            fn=cancel_and_back,
            inputs=[selected_tool_state],
            outputs=[
                empty_panel,
                preview_panel,
                create_panel,
                preview_title,
                preview_html,
                preview_category_dropdown,
            ],
        )

        # From Code: Parse
        parse_btn.click(
            fn=do_parse_code,
            inputs=[code_input],
            outputs=[code_preview],
        )

        # From Code: Save (with category)
        save_code_btn.click(
            fn=do_save_from_code,
            inputs=[code_input, category_dropdown],
        ).then(
            fn=after_save,
            outputs=after_save_outputs,
        )

        # MCP: Connect
        connect_btn.click(
            fn=fetch_mcp_tools,
            inputs=[connection_type, mcp_command, mcp_args, mcp_url, mcp_config],
            outputs=[mcp_status, found_tools_state, mcp_tools_checkbox, import_btn],
        )

        # MCP: Import
        import_btn.click(
            fn=save_mcp_tools,
            inputs=[mcp_tools_checkbox, found_tools_state, category_dropdown],
        ).then(
            fn=after_save,
            outputs=after_save_outputs,
        )

    return page
