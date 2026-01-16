import os
import tempfile
import gradio as gr
from pages.base import (
    TOOL_CALLING_RESPONDENT_PROMPT,
    SMART_HOME_CONTEXT_STR,
    create_context_section,
    create_output_section,
)
from core.config import get_training_dir
from core.file_utils import create_model_zip
from core.tools_db import get_tools_db
from schemas import AVAILABLE_TOOLS


# Built-in tool names for display
BUILTIN_TOOL_NAMES = [tool.__name__ for tool in AVAILABLE_TOOLS]


def create_tool_calling_page(start_gen_fn, train_fn=None):
    with gr.Blocks() as page:
        gr.Markdown("## Tool Calling Generation")
        gr.Markdown(
            "Generate data for training models to use tools precisely based on instructions."
        )

        with gr.Row():
            context_ui = create_context_section(SMART_HOME_CONTEXT_STR)

            with gr.Column(scale=1):
                gr.Markdown("### 2. Configuration")
                prompt_input = gr.TextArea(
                    label="Respondent System Prompt",
                    value=TOOL_CALLING_RESPONDENT_PROMPT,
                    lines=6,
                )
                num_samples = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1,
                    label="Number of Samples",
                )
                
                # Tool Selection Section
                gr.Markdown("### 3. Select Tools")
                
                # Built-in tools
                builtin_tools_checkbox = gr.CheckboxGroup(
                    choices=BUILTIN_TOOL_NAMES,
                    value=BUILTIN_TOOL_NAMES,  # All selected by default
                    label="Built-in Tools (Smart Home)",
                    info="Pre-defined tools from schemas.py",
                )
                
                # Custom tools from database
                custom_tools_checkbox = gr.CheckboxGroup(
                    choices=[],  # Will be populated dynamically
                    value=[],
                    label="Custom Tools",
                    info="Tools you defined in Custom Tools page",
                )
                
                refresh_tools_btn = gr.Button("Refresh Custom Tools", variant="secondary", size="sm")
                
                # Train Model checkbox
                train_model_checkbox = gr.Checkbox(
                    label="Train Model after generation",
                    value=False,
                    info="Automatically train a model with generated data"
                )
                
                generate_btn = gr.Button(
                    "Generate Tool Calls", variant="primary", size="lg"
                )

        status_output, results_output, download_output = create_output_section(
            headers=["Persona", "Instruction", "Response", "Reasoning", "Tool Calls"],
            label="Generated Tool Calls",
        )
        
        # Training section (initially hidden)
        gr.Markdown("---")
        training_section = gr.Column(visible=False)
        with training_section:
            gr.Markdown("## Model Training")
            training_status = gr.Markdown("Status: Waiting...")
            training_progress = gr.Code(
                label="",
                language="shell",
                interactive=False,
                lines=10,
            )
        
        # Download button (outside training_section so it stays visible)
        download_model_btn = gr.DownloadButton(
            label="Download Trained Model",
            visible=False,
            variant="primary",
            size="lg",
        )

        # --- Event Handlers ---
        
        def load_custom_tools():
            """Load custom tools from database and update checkbox choices."""
            db = get_tools_db()
            tool_names = db.get_tool_names()
            return gr.update(choices=tool_names, value=tool_names)
        
        def on_generate_complete(status_text, train_enabled):
            """Show training section if training is enabled"""
            if train_enabled and status_text and "Complete" in status_text:
                return gr.update(visible=True)
            return gr.update(visible=False)
        
        async def start_training_if_enabled(train_enabled, file_path):
            """Start training if checkbox is enabled and generation is complete"""
            if not train_enabled or not train_fn:
                yield "Status: Training not enabled", ""
                return
            
            if not file_path:
                yield "Status: No dataset generated", ""
                return
            
            # Start training
            async for status, progress in train_fn(file_path):
                yield status, progress
        
        def make_model_downloadable(status_text):
            """Show download button when training completes"""
            if not status_text:
                return gr.update(visible=False)
            
            # Check if training is complete
            if "Complete" not in status_text and "✓" not in status_text:
                return gr.update(visible=False)
            
            # Path to trained model using centralized config
            training_dir = get_training_dir()
            model_dir = os.path.join(training_dir, "final_model_stable")
            
            # Use shared utility function to create zip
            zip_path = create_model_zip(
                model_dir, 
                output_name=os.path.join(tempfile.gettempdir(), "trained_model_toolcalling.zip")
            )
            
            if zip_path:
                return gr.update(visible=True, value=zip_path)
            
            return gr.update(visible=False)
        
        # Wire up events
        refresh_tools_btn.click(
            fn=load_custom_tools,
            inputs=[],
            outputs=[custom_tools_checkbox],
        )
        
        # Load custom tools on page load
        page.load(
            fn=load_custom_tools,
            inputs=[],
            outputs=[custom_tools_checkbox],
        )
        
        generate_output = generate_btn.click(
            fn=start_gen_fn,
            inputs=[
                context_ui["input"],
                prompt_input,
                num_samples,
                context_ui["source"],
                context_ui["file"],
                context_ui["key"],
                builtin_tools_checkbox,
                custom_tools_checkbox,
            ],
            outputs=[
                results_output,
                status_output,
                download_output,
            ],
        )
        
        # Show training section when generation completes and checkbox is enabled
        train_output = generate_output.then(
            fn=on_generate_complete,
            inputs=[status_output, train_model_checkbox],
            outputs=[training_section],
        ).then(
            fn=start_training_if_enabled,
            inputs=[train_model_checkbox, download_output],
            outputs=[training_status, training_progress],
        )
        
        # Show download button when training completes
        train_output.then(
            fn=make_model_downloadable,
            inputs=[training_status],
            outputs=[download_model_btn],
        )
        
    return page
