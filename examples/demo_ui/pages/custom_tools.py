"""
Custom Tools page for defining and managing tool definitions.

Allows users to create tools either by writing Python function code
or by manually entering the function definition fields.
"""

import json
import gradio as gr


from .handlers.custom_tools import (
    EXAMPLE_PARAMS,
    parse_code,
    save_from_code,
    clear_code_editor,
    validate_manual,
    save_manual,
    load_tool_for_edit,
    update_tool,
    delete_tool_edit,
    refresh_edit_dropdown,
)


# Example function code for the editor
EXAMPLE_FUNCTION = '''def send_email(to: str, subject: str, body: str, cc: str = ""):
    """Send an email to the specified recipient."""
    pass
'''

# Example parameters JSON


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
                            parse_btn = gr.Button(
                                "Parse & Preview", variant="secondary"
                            )
                            clear_code_btn = gr.Button("Clear", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Preview & Save")

                        preview_name = gr.Textbox(
                            label="Function Name", interactive=False
                        )
                        preview_desc = gr.Textbox(
                            label="Description", interactive=False
                        )
                        preview_params = gr.JSON(label="Parameters Schema")
                        preview_required = gr.JSON(label="Required Parameters")

                        save_code_btn = gr.Button(
                            "Save Tool", variant="primary", interactive=False
                        )

            # --- TAB 2: Manual Entry ---
            with gr.Tab("Manual Entry"):
                gr.Markdown("### Create Tool Manually")
                gr.Markdown(
                    "Enter the function definition fields directly without writing code."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        manual_name = gr.Textbox(
                            label="Function Name",
                            placeholder="e.g., send_notification",
                            info="Unique identifier for the tool (snake_case recommended)",
                        )
                        manual_desc = gr.Textbox(
                            label="Description",
                            placeholder="e.g., Send a push notification to the user",
                            lines=2,
                            info="Describe what this tool does",
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
                            info="Comma-separated list of required parameter names",
                        )

                with gr.Row():
                    validate_manual_btn = gr.Button("Validate", variant="secondary")
                    save_manual_btn = gr.Button("Save Tool", variant="primary")

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
                            label="Required Parameters", info="Comma-separated list"
                        )

                with gr.Row():
                    update_btn = gr.Button("Update Tool", variant="primary")
                    delete_edit_btn = gr.Button("Delete Tool", variant="stop")

        gr.Markdown("---")

        # --- Event Handlers ---

        # All handlers are imported from custom_tools_handlers.py

        # --- Wire up events ---

        # From Code tab
        parse_btn.click(
            fn=parse_code,
            inputs=[code_input],
            outputs=[
                current_parsed,
                preview_name,
                preview_desc,
                preview_params,
                preview_required,
                save_code_btn,
            ],
        )

        save_code_btn.click(
            fn=save_from_code,
            inputs=[current_parsed],
            outputs=[
                code_input,
                current_parsed,
                preview_name,
                preview_desc,
                preview_params,
                preview_required,
                save_code_btn,
            ],
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )

        clear_code_btn.click(
            fn=clear_code_editor,
            outputs=[
                code_input,
                current_parsed,
                preview_name,
                preview_desc,
                preview_params,
                preview_required,
                save_code_btn,
            ],
        )

        # Manual Entry tab
        validate_manual_btn.click(
            fn=validate_manual,
            inputs=[manual_name, manual_desc, manual_params, manual_required],
            outputs=None,  # Pure sidebar effect
        )

        save_manual_btn.click(
            fn=save_manual,
            inputs=[manual_name, manual_desc, manual_params, manual_required],
            outputs=[manual_name, manual_desc, manual_params, manual_required],
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )

        # Edit tab
        load_edit_btn.click(
            fn=load_tool_for_edit,
            inputs=[edit_select],
            outputs=[edit_name, edit_desc, edit_params, edit_required],
        )

        update_btn.click(
            fn=update_tool,
            inputs=[edit_name, edit_desc, edit_params, edit_required],
            outputs=None,
        )

        delete_edit_btn.click(
            fn=delete_tool_edit,
            inputs=[edit_name],
            outputs=None,
        ).then(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )

        # Load on page load
        page.load(
            fn=refresh_edit_dropdown,
            outputs=[edit_select],
        )

    return page
