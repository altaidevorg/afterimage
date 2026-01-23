import gradio as gr

from .handlers.training import (
    on_train_complete,
    generate_model_zip,
    load_datasets_list,
    on_dataset_select,
    create_analyze_handler,
    set_download_loading,
    update_action_buttons,
    open_delete_confirm,
    cancel_delete,
    confirm_delete,
    open_rename_dialog,
    cancel_rename,
    confirm_rename,
    merge_datasets,
)



def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn, chat_fn=None):
    """
    Creates a train model page with a 4-step Wizard flow:
    1. Select Dataset
    2. Training (Normal/Dev)
    3. Evaluation
    4. Chat with Model
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
            gr.Markdown("Choose datasets to train on. Select multiple to merge them.")
            
            # States (defined early so they're accessible)
            tool_dist_state = gr.State({})
            filter_config_state = gr.State({})
            
            with gr.Row():
                # ========== LEFT: Dataset Library ==========
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Dataset Library")
                        dataset_list = gr.CheckboxGroup(
                            choices=[],
                            label=None,
                            show_label=False,
                            elem_id="dataset-list-scroll",
                        )
                        with gr.Row():
                            refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")
                            rename_btn = gr.Button("Rename", size="sm", variant="secondary", interactive=False)
                            delete_btn = gr.Button("Delete", size="sm", variant="stop", interactive=False)
                        
                        gr.Markdown("---")
                        merge_name = gr.Textbox(
                            label="Merge selected into",
                            placeholder="merged_dataset.jsonl",
                        )
                        merge_btn = gr.Button("Merge Datasets", variant="primary", interactive=False)
                
                # ========== RIGHT: Stats + Filters ==========
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Overview")
                        dataset_overview = gr.HTML(
                            value="""
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                                <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Datasets</div>
                                    <div style="font-size: 24px; font-weight: 700; color: #334155;">-</div>
                                </div>
                                <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Size</div>
                                    <div style="font-size: 24px; font-weight: 700; color: #334155;">-</div>
                                </div>
                                <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Samples</div>
                                    <div style="font-size: 24px; font-weight: 700; color: #334155;">-</div>
                                </div>
                            </div>
                            """
                        )
                    
                    gr.Markdown("### Filter by Tool")
                    
                    # Scrollable container for sliders
                    with gr.Column(elem_id="tool-sliders-scroll"):
                        @gr.render(inputs=tool_dist_state)
                        def render_tool_sliders(dist):
                            if not dist:
                                gr.HTML("""
                                <div style="text-align: center; padding: 40px; color: #94a3b8;">
                                    Select datasets to see tool distribution
                                </div>
                                """)
                                return
                            
                            sorted_tools = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                            
                            for tool, total in sorted_tools:
                                s = gr.Slider(
                                    minimum=0,
                                    maximum=total,
                                    value=total,
                                    step=1,
                                    label=tool,
                                    info=f"max: {total}",
                                    interactive=True,
                                )
                                
                                def update_config(val, config, t=tool):
                                    if config is None:
                                        config = {}
                                    config[t] = val
                                    return config
                                
                                s.change(
                                    fn=update_config,
                                    inputs=[s, filter_config_state],
                                    outputs=[filter_config_state]
                                )
            
            with gr.Row():
                with gr.Column(scale=2):
                    pass # Spacer
                with gr.Column(scale=1):
                    next_btn_1 = gr.Button("Next: Configure Training", variant="primary", interactive=False, size="lg")

            # State variables
            delete_target_path = gr.State(None)
            rename_target_path = gr.State(None)

            # Delete confirmation dialog
            with gr.Group(visible=False) as delete_confirm_group:
                gr.Markdown("### Confirm Delete")
                gr.Markdown("This will **permanently remove** the dataset file. This action cannot be undone.")
                with gr.Row():
                    confirm_delete_btn = gr.Button("Yes, Delete", variant="stop")
                    cancel_delete_btn = gr.Button("Cancel", variant="secondary")

            # Rename dialog
            with gr.Group(visible=False) as rename_group:
                gr.Markdown("### Rename Dataset")
                rename_input = gr.Textbox(label="New name", placeholder="new_dataset_name.jsonl")
                with gr.Row():
                    confirm_rename_btn = gr.Button("Rename", variant="primary")
                    cancel_rename_btn = gr.Button("Cancel", variant="secondary")

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
                next_btn_3 = gr.Button("Next: Chat with Model", variant="primary", interactive=False)

        # ============================================================
        # STEP 4: Chat with Model
        # ============================================================
        with gr.Group(visible=False) as step_4_group:
            gr.Markdown("## Step 4/4: Chat with Model")
            gr.Markdown("Test your trained model by chatting with it!")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    scale=4,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                back_btn_4 = gr.Button("Back to Evaluation", variant="secondary")
                restart_btn_2 = gr.Button("Restart Wizard", variant="secondary")


        # ============================================================
        # NAVIGATION LOGIC
        # ============================================================
        
        def update_step(current, direction):
            new_step = max(0, min(3, current + direction))  # Now supports 4 steps (0-3)
            return (
                new_step,
                gr.update(visible=(new_step == 0)), # Step 1
                gr.update(visible=(new_step == 1)), # Step 2
                gr.update(visible=(new_step == 2)), # Step 3
                gr.update(visible=(new_step == 3)), # Step 4
            )
            
        def go_to_step_0(current): return update_step(current, -current) # Reset to 0

        # Next/Back click handlers
        # Returns: [step_state, step_1_group, step_2_group, step_3_group, step_4_group]
        common_outputs = [step_state, step_1_group, step_2_group, step_3_group, step_4_group]
        
        next_btn_1.click(lambda s: update_step(s, 1), inputs=[step_state], outputs=common_outputs)
        
        back_btn_2.click(lambda s: update_step(s, -1), inputs=[step_state], outputs=common_outputs)
        next_btn_2.click(lambda s: update_step(s, 1), inputs=[step_state], outputs=common_outputs)
        
        back_btn_3.click(lambda s: update_step(s, -1), inputs=[step_state], outputs=common_outputs)
        next_btn_3.click(lambda s: update_step(s, 1), inputs=[step_state], outputs=common_outputs)
        
        back_btn_4.click(lambda s: update_step(s, -1), inputs=[step_state], outputs=common_outputs)
        restart_btn_2.click(lambda s: go_to_step_0(s), inputs=[step_state], outputs=common_outputs)


        # ============================================================
        # EVENT HANDLERS
        # ============================================================
        
        # --- Step 1: Datasets ---
        page.load(load_datasets_list, outputs=[dataset_list])
        refresh_btn.click(load_datasets_list, outputs=[dataset_list])
        
        # Enable Next button when dataset selected
        def on_dataset_change_ui(selected):
            # on_dataset_select returns: paths, overview, tool_dist, filter_config, train_btn_update, train_dev_btn_update
            path, overview, dist, config, _, _ = on_dataset_select(selected)
            can_next = path is not None and len(path) > 0
            return path, overview, dist, config, gr.update(interactive=can_next)

        dataset_list.change(
            on_dataset_change_ui,
            inputs=[dataset_list],
            outputs=[selected_dataset_path, dataset_overview, tool_dist_state, filter_config_state, next_btn_1]
        ).then(
            fn=update_action_buttons,
            inputs=[dataset_list],
            outputs=[rename_btn, delete_btn, merge_btn],
        )

        # Delete flow
        delete_btn.click(
            fn=open_delete_confirm,
            inputs=[dataset_list],
            outputs=[delete_confirm_group, delete_target_path],
        )
        cancel_delete_btn.click(
            fn=cancel_delete,
            outputs=[delete_confirm_group, delete_target_path],
        )
        confirm_delete_btn.click(
            fn=confirm_delete,
            inputs=[delete_target_path],
            outputs=[delete_confirm_group, delete_target_path],
        ).then(
            fn=load_datasets_list,
            outputs=[dataset_list],
        )

        # Merge flow
        merge_btn.click(
            fn=merge_datasets,
            inputs=[dataset_list, merge_name],
            outputs=None,
        ).then(
            fn=load_datasets_list,
            outputs=[dataset_list],
        )

        # Rename flow
        rename_btn.click(
            fn=open_rename_dialog,
            inputs=[dataset_list],
            outputs=[rename_group, rename_target_path, rename_input],
        )
        cancel_rename_btn.click(
            fn=cancel_rename,
            outputs=[rename_group, rename_target_path],
        )
        confirm_rename_btn.click(
            fn=confirm_rename,
            inputs=[rename_target_path, rename_input],
            outputs=[rename_group, rename_target_path],
        ).then(
            fn=load_datasets_list,
            outputs=[dataset_list],
        )

        # --- Step 2: Training ---
        
        # Normal Training
        train_btn.click(
            train_fn, 
            inputs=[selected_dataset_path, filter_config_state], 
            outputs=[status_output, logs_output]
        ).then(
            on_train_complete,
            inputs=[status_output, logs_output],
            outputs=[
                eval_btn, eval_status,
                download_status, download_model_btn,
                download_status_dev, download_model_dev_btn,
            ]
        ).then(
            # Enable Next button after training
            lambda: gr.update(interactive=True), outputs=[next_btn_2]
        )



        def wire_download_button(btn):
            """Wire up the download button events."""
            btn.click(
                fn=set_download_loading,
                outputs=[btn],
            ).then(
                fn=generate_model_zip,
                outputs=[btn],
            )

        # One-click download: zip and download immediately
        wire_download_button(download_model_btn)
        wire_download_button(download_model_dev_btn)


        # --- Step 3: Evaluation ---
        eval_btn.click(eval_fn, outputs=[eval_status, eval_output]).then(
            lambda: gr.update(interactive=True), outputs=[next_btn_3]
        )

        # --- Step 4: Chat ---
        if chat_fn:
            send_btn.click(
                chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )
            msg_input.submit(  # Allow Enter key to send
                chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
            )

    return page
