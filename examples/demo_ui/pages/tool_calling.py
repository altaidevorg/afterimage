import os
import tempfile
import gradio as gr
from pages.base import (
    TOOL_CALLING_RESPONDENT_PROMPT,
    SMART_HOME_CONTEXT_STR,
    create_context_section,
    create_output_section,
)
from pages.handlers.training import (
    set_download_loading,
    generate_model_zip,
    get_model_download_label,
)

from core.config import get_training_dir
from core.tools_db import get_tools_db
from schemas import AVAILABLE_TOOLS


def create_tool_calling_page(start_gen_fn, train_fn=None):
    with gr.Blocks() as page:
        # State for wizard progress
        current_step = gr.State(1)
        
        # States for tool selection
        tools_by_category_state = gr.State({})  # Dict of category -> list of tool names
        selected_tools_state = gr.State([])  # List of selected tool names
        
        # Header Stepper UI
        @gr.render(inputs=current_step)
        def render_stepper(step):
            steps = [
                (1, "Context Source"),
                (2, "Configuration"),
                (3, "Select Tools"),
                (4, "Generate & Train")
            ]
            
            html = '<div class="wizard-stepper">'
            for s_num, s_name in steps:
                cls = "wizard-step"
                if step == s_num:
                    cls += " active"
                elif step > s_num:
                    cls += " completed"
                html += f'<div class="{cls}">Step {s_num}: {s_name}</div>'
            html += '</div>'
            gr.HTML(html)

        # --- STEP 1: CONTEXT SOURCE ---
        with gr.Group(visible=True) as step_1_group:
            context_ui = create_context_section(SMART_HOME_CONTEXT_STR)
            with gr.Row():
                gr.Button(value="Back", interactive=False, visible=False) # Spacer
                next_btn_1 = gr.Button("Next", variant="primary")

        # --- STEP 2: CONFIGURATION ---
        with gr.Group(visible=False) as step_2_group:
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
            with gr.Row():
                back_btn_2 = gr.Button("Back", variant="secondary")
                next_btn_2 = gr.Button("Next", variant="primary")

        # --- STEP 3: TOOL SELECTION (Category-based) ---
        with gr.Group(visible=False) as step_3_group:
            gr.Markdown("### 3. Select Tools")
            gr.Markdown("Select tools to include in generation. Click category headers to select/deselect all tools in that category.")
            
            # Dynamic category-based tool selection
            @gr.render(inputs=[tools_by_category_state, selected_tools_state])
            def render_tool_selection(grouped, selected):
                if not grouped:
                    gr.HTML("""
                    <div style="text-align: center; padding: 40px; color: #94a3b8;">
                        No tools found. Add tools in the Tool Library first.
                    </div>
                    """)
                    return
                
                selected_set = set(selected) if selected else set()
                
                for category, tools in grouped.items():
                    tool_names = [t.definition.name for t in tools]
                    selected_in_category = [t for t in tool_names if t in selected_set]
                    total_in_category = len(tool_names)
                    selected_count = len(selected_in_category)
                    
                    # Determine checkbox state
                    all_selected = selected_count == total_in_category
                    partial = 0 < selected_count < total_in_category
                    
                    # Label with count
                    if partial:
                        label = f"{category} ({selected_count}/{total_in_category} selected)"
                        elem_classes = ["category-header", "category-partial"]
                    elif all_selected:
                        label = f"{category} ({total_in_category}/{total_in_category}) ✓"
                        elem_classes = ["category-header", "category-all"]
                    else:
                        label = f"{category} (0/{total_in_category})"
                        elem_classes = ["category-header"]
                    
                    with gr.Group():
                        # Category header with select all checkbox
                        cat_checkbox = gr.Checkbox(
                            label=label,
                            value=all_selected,
                            elem_classes=elem_classes,
                        )
                        
                        # Individual tool checkboxes
                        with gr.Column(elem_classes=["category-tools"]):
                            for tool in tools:
                                name = tool.definition.name
                                is_selected = name in selected_set
                                
                                tool_cb = gr.Checkbox(
                                    label=name,
                                    value=is_selected,
                                    elem_classes=["tool-checkbox"],
                                )
                                
                                # Handler for individual tool selection
                                def make_tool_handler(tool_name):
                                    def handler(checked, current_selected):
                                        new_selected = list(current_selected) if current_selected else []
                                        if checked and tool_name not in new_selected:
                                            new_selected.append(tool_name)
                                        elif not checked and tool_name in new_selected:
                                            new_selected.remove(tool_name)
                                        return new_selected
                                    return handler
                                
                                tool_cb.change(
                                    fn=make_tool_handler(name),
                                    inputs=[tool_cb, selected_tools_state],
                                    outputs=[selected_tools_state],
                                )
                        
                        # Handler for category select all
                        def make_category_handler(cat_tool_names):
                            def handler(checked, current_selected):
                                new_selected = list(current_selected) if current_selected else []
                                if checked:
                                    # Add all tools from this category
                                    for t in cat_tool_names:
                                        if t not in new_selected:
                                            new_selected.append(t)
                                else:
                                    # Remove all tools from this category
                                    new_selected = [t for t in new_selected if t not in cat_tool_names]
                                return new_selected
                            return handler
                        
                        cat_checkbox.change(
                            fn=make_category_handler(tool_names),
                            inputs=[cat_checkbox, selected_tools_state],
                            outputs=[selected_tools_state],
                        )
            
            with gr.Row():
                refresh_tools_btn = gr.Button("Refresh Library", variant="secondary", size="sm")
                select_all_btn = gr.Button("Select All", variant="secondary", size="sm")
                deselect_all_btn = gr.Button("Deselect All", variant="secondary", size="sm")
            
            with gr.Row():
                back_btn_3 = gr.Button("Back", variant="secondary")
                next_btn_3 = gr.Button("Next", variant="primary")

        # --- STEP 4: GENERATE & TRAIN ---
        with gr.Group(visible=False) as step_4_group:
            gr.Markdown("### 4. Review & Generate")
            
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
                
                # Download section (shown after training completes)
                download_status = gr.Markdown("", visible=False)
                download_model_btn = gr.DownloadButton(
                    label="Download Trained Model",
                    visible=False,
                    variant="primary",
                    size="lg",
                )

            with gr.Row():
                back_btn_4 = gr.Button("Back", variant="secondary")
                gr.Button(value="Next", interactive=False, visible=False) # Spacer
        
        # --- LOGIC & HANDLERS ---
        
        # Navigation Handlers
        def go_to_step_1(): return 1, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        def go_to_step_2(): return 2, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        def go_to_step_3(): return 3, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        def go_to_step_4(): return 4, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        # Validation for Step 1
        def validate_step_1(source, manual_text, file_obj):
            if source == "Manual Entry" and not manual_text.strip():
                raise gr.Error("Please enter context text.")
            if source == "File Upload" and file_obj is None:
                raise gr.Error("Please upload a file.")
            # If valid, proceed
            return go_to_step_2()

        # Wire Navigation
        # Step 1 -> 2
        next_btn_1.click(
            fn=validate_step_1,
            inputs=[context_ui["source"], context_ui["input"], context_ui["file"]],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )
        
        # Step 2 -> 3
        next_btn_2.click(
            fn=go_to_step_3,
            inputs=[],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )
        back_btn_2.click(
            fn=go_to_step_1,
            inputs=[],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )
        
        # Step 3 -> 4
        next_btn_3.click(
            fn=go_to_step_4,
            inputs=[],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )
        back_btn_3.click(
            fn=go_to_step_2,
            inputs=[],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )
        
        # Step 4 Back
        back_btn_4.click(
            fn=go_to_step_3,
            inputs=[],
            outputs=[current_step, step_1_group, step_2_group, step_3_group, step_4_group]
        )

        # Tool Loading Handlers
        def load_tools_grouped():
            """Load tools from database grouped by category."""
            db = get_tools_db()
            grouped = db.get_tools_by_category()
            # Also return all tool names as initial selection
            all_tools = []
            for tools in grouped.values():
                all_tools.extend([t.definition.name for t in tools])
            return grouped, all_tools
        
        def select_all_tools(grouped):
            """Select all tools."""
            all_tools = []
            for tools in grouped.values():
                all_tools.extend([t.definition.name for t in tools])
            return all_tools
        
        def deselect_all_tools():
            """Deselect all tools."""
            return []
        
        def on_generate_complete(status_text, train_enabled):
            if train_enabled and status_text and "Complete" in status_text:
                return gr.update(visible=True)
            return gr.update(visible=False)
        
        async def start_training_if_enabled(train_enabled, file_path):
            if not train_enabled or not train_fn:
                yield "Status: Training not enabled", ""
                return
            if not file_path:
                yield "Status: No dataset generated", ""
                return
            async for status, progress in train_fn(file_path):
                yield status, progress
        
        def on_train_complete_toolcalling(status_text):
            if not status_text:
                return gr.update(visible=False), gr.update(visible=False)
            if "Complete" not in status_text and "✓" not in status_text:
                return gr.update(visible=False), gr.update(visible=False)
            
            try:
                training_dir = get_training_dir()
                model_dir = os.path.join(training_dir, "final_model_stable")
                total_size_mb = 0
                if os.path.exists(model_dir):
                    total_size_mb = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(model_dir)
                        for filename in filenames
                    ) / (1024 * 1024)
                label = f"Download Model (~{total_size_mb:.1f} MB)"
            except Exception:
                label = "Download Model"
            
            return gr.update(value="Training complete. Click to download.", visible=True), gr.update(visible=True, value=None, label=label, interactive=True)
        
        # Wire Tool Loading
        refresh_tools_btn.click(
            fn=load_tools_grouped, 
            outputs=[tools_by_category_state, selected_tools_state]
        )
        page.load(
            fn=load_tools_grouped, 
            outputs=[tools_by_category_state, selected_tools_state]
        )
        
        # Select/Deselect All
        select_all_btn.click(
            fn=select_all_tools,
            inputs=[tools_by_category_state],
            outputs=[selected_tools_state],
        )
        deselect_all_btn.click(
            fn=deselect_all_tools,
            outputs=[selected_tools_state],
        )
        
        # Wire Generation (uses selected_tools_state instead of tools_checkbox)
        generate_output = generate_btn.click(
            fn=start_gen_fn,
            inputs=[
                context_ui["input"],
                prompt_input,
                num_samples,
                context_ui["source"],
                context_ui["file"],
                context_ui["key"],
                selected_tools_state,  # Use state instead of checkbox
            ],
            outputs=[results_output, status_output, download_output],
        )
        
        # Wire Training
        train_trigger = generate_output.then(
            fn=on_generate_complete,
            inputs=[status_output, train_model_checkbox],
            outputs=[training_section],
        ).then(
            fn=start_training_if_enabled,
            inputs=[train_model_checkbox, download_output],
            outputs=[training_status, training_progress],
        )
        
        # Wire Download
        train_trigger.then(
            fn=on_train_complete_toolcalling,
            inputs=[training_status],
            outputs=[download_status, download_model_btn],
        )
        
        download_model_btn.click(
            fn=set_download_loading,
            outputs=[download_model_btn],
        ).then(
            fn=generate_model_zip,
            outputs=[download_model_btn],
        )
        
    return page
