import gradio as gr

from .train_model_handlers import (
    on_train_complete,
    prepare_model_download,
    load_datasets_list,
    on_dataset_select,
)


def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn):
    """
    Creates a train model page with a 3-step Wizard flow:
    1. Select Dataset
    2. Training (Normal/Dev)
    3. Evaluation
    """
    with gr.Blocks() as page:
        gr.Markdown("# Model Training Wizard")
        
        # State to track current step (0=Dataset, 1=Training, 2=Evaluation)
        step_state = gr.State(0)
        
        # Hidden state to track selected dataset path
        selected_dataset_path = gr.State(None)

        # ============================================================
        # STEP 1: Select Dataset
        # ============================================================
        with gr.Group(visible=True) as step_1_group:
            gr.Markdown("## Step 1/3: Select Datasets")
            gr.Markdown("Select one or more datasets to merge and train on.")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### Dataset Library")
                    dataset_list = gr.CheckboxGroup(
                        choices=[],
                        label="Saved Datasets",
                        info="Select datasets from your library",
                        elem_id="dataset_library",
                    )
                    refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")
                
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### Dataset Overview")
                    dataset_overview = gr.HTML(
                        value="<div style='color: #6b7280; font-style: italic;'>Select datasets to see their details</div>",
                        elem_id="dataset_overview",
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    pass # Spacer
                with gr.Column(scale=1):
                    next_btn_1 = gr.Button("Next: Configure Training", variant="primary", interactive=False, size="lg")

        # ============================================================
        # STEP 2: Training
        # ============================================================
        with gr.Group(visible=False) as step_2_group:
            gr.Markdown("## Step 2/3: Train Model")
            
            with gr.Tabs() as training_tabs:
                # ========== NORMAL MODE TAB ==========
                with gr.Tab("Normal Mode"):
                    train_btn = gr.Button(
                        "Start Training",
                        variant="primary",
                        size="lg",
                        interactive=True, # Always interactive in this step if we got here
                    )
                    
                    status_output = gr.Markdown("Status: Ready to train")
                    
                    logs_output = gr.Code(
                        label="Training Progress",
                        language="shell",
                        interactive=False,
                        lines=15,
                    )
                    
                    download_status = gr.Markdown("", visible=False)
                    download_model_btn = gr.DownloadButton(
                        label="Download Trained Model",
                        visible=False,
                        variant="secondary",
                        size="lg",
                    )
                
                # ========== DEVELOPER MODE TAB ==========
                with gr.Tab("Developer Mode"):
                    gr.Markdown("### Hyperparameters")
                    with gr.Row():
                        with gr.Column():
                            num_epochs = gr.Slider(label="Epochs", minimum=1, maximum=10, step=1, value=3)
                            learning_rate = gr.Number(label="Learning Rate", value=2e-5, precision=6)
                            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=1)
                        with gr.Column():
                            grad_accumulation = gr.Slider(label="Grad Accumulation", minimum=1, maximum=32, step=1, value=16)
                            test_size = gr.Slider(label="Test Split", minimum=0.05, maximum=0.3, step=0.05, value=0.10)
                            logging_steps = gr.Slider(label="Logging Steps", minimum=1, maximum=50, step=1, value=5)
                    
                    train_dev_btn = gr.Button(
                        "Start Training (Dev)",
                        variant="primary",
                        size="lg",
                        interactive=True,
                    )
                    
                    dev_status_output = gr.Markdown("Status: Ready to train")
                    dev_logs_output = gr.Code(label="Logs", language="shell", interactive=False, lines=15)
                    
                    download_status_dev = gr.Markdown("", visible=False)
                    download_model_dev_btn = gr.DownloadButton(
                        label="Download Trained Model",
                        visible=False,
                        variant="secondary",
                        size="lg",
                    )

            with gr.Row():
                back_btn_2 = gr.Button("Back to Datasets", variant="secondary")
                next_btn_2 = gr.Button("Next: Evaluate Model", variant="primary", interactive=False)

        # ============================================================
        # STEP 3: Evaluation
        # ============================================================
        with gr.Group(visible=False) as step_3_group:
            gr.Markdown("## Step 3/3: Evaluate Model")
            
            eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            eval_status = gr.Markdown("Status: Ready to evaluate")
            eval_output = gr.Code(label="Results", language="shell", interactive=False, lines=20)
            
            with gr.Row():
                back_btn_3 = gr.Button("Back to Training", variant="secondary")
                restart_btn = gr.Button("Restart Wizard", variant="secondary")

        # ============================================================
        # NAVIGATION LOGIC
        # ============================================================
        
        def update_step(current, direction):
            new_step = max(0, min(2, current + direction))
            return (
                new_step,
                gr.update(visible=(new_step == 0)), # Step 1
                gr.update(visible=(new_step == 1)), # Step 2
                gr.update(visible=(new_step == 2)), # Step 3
            )
            
        def go_to_step_0(current): return update_step(current, -current) # Reset to 0

        # Next/Back click handlers
        # Returns: [step_state, step_1_group, step_2_group, step_3_group]
        common_outputs = [step_state, step_1_group, step_2_group, step_3_group]
        
        next_btn_1.click(lambda s: update_step(s, 1), inputs=[step_state], outputs=common_outputs)
        
        back_btn_2.click(lambda s: update_step(s, -1), inputs=[step_state], outputs=common_outputs)
        next_btn_2.click(lambda s: update_step(s, 1), inputs=[step_state], outputs=common_outputs)
        
        back_btn_3.click(lambda s: update_step(s, -1), inputs=[step_state], outputs=common_outputs)
        restart_btn.click(lambda s: go_to_step_0(s), inputs=[step_state], outputs=common_outputs)

        # ============================================================
        # EVENT HANDLERS
        # ============================================================
        
        # --- Step 1: Datasets ---
        page.load(load_datasets_list, outputs=[dataset_list])
        refresh_btn.click(load_datasets_list, outputs=[dataset_list])
        
        # Enable Next button when dataset selected
        def on_dataset_change_ui(selected):
            # Also calling the handler to get overview
            path, overview, _, _ = on_dataset_select(selected)
            can_next = path is not None and len(path) > 0
            return path, overview, gr.update(interactive=can_next)

        dataset_list.change(
            on_dataset_change_ui,
            inputs=[dataset_list],
            outputs=[selected_dataset_path, dataset_overview, next_btn_1]
        )

        # --- Step 2: Training ---
        
        # Hidden trigger states for download logic
        download_trigger_normal = gr.State(False)
        download_trigger_dev = gr.State(False)

        # Normal Training
        train_btn.click(
            train_fn, 
            inputs=[selected_dataset_path], 
            outputs=[status_output, logs_output]
        ).then(
            on_train_complete,
            inputs=[status_output, logs_output],
            outputs=[
                eval_btn, eval_status, # We reuse these handlers but might ignore the updates since we control visibility
                download_status, download_model_btn, download_trigger_normal,
                download_status_dev, download_model_dev_btn, download_trigger_dev,
            ]
        ).then(
            # Enable Next button after training
            lambda: gr.update(interactive=True), outputs=[next_btn_2]
        ).then(
            prepare_model_download,
            inputs=[download_trigger_normal],
            outputs=[download_status, download_model_btn, download_status_dev, download_model_dev_btn]
        )

        # Dev Training
        train_dev_btn.click(
            train_dev_fn,
            inputs=[selected_dataset_path, num_epochs, learning_rate, batch_size, grad_accumulation, test_size, logging_steps],
            outputs=[dev_status_output, dev_logs_output],
        ).then(
            on_train_complete,
            inputs=[dev_status_output, dev_logs_output],
            outputs=[
                eval_btn, eval_status,
                download_status, download_model_btn, download_trigger_normal,
                download_status_dev, download_model_dev_btn, download_trigger_dev,
            ]
        ).then(
            lambda: gr.update(interactive=True), outputs=[next_btn_2]
        ).then(
            prepare_model_download,
            inputs=[download_trigger_normal],
            outputs=[download_status, download_model_btn, download_status_dev, download_model_dev_btn]
        )

        # --- Step 3: Evaluation ---
        eval_btn.click(eval_fn, outputs=[eval_status, eval_output])

    return page
