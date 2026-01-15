import gradio as gr
import pandas as pd

from .train_model_handlers import (
    create_analyze_handler,
    on_train_complete,
    prepare_model_download,
)


def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn):
    """
    Creates a 4-stage train model page with Normal and Developer modes:
    1. Upload Files
    2. Dataset Expose (with statistics and charts)
    3. Training
    4. Evaluation
    
    Args:
        analyze_fn: Dataset analysis function
        train_fn: Training function (normal mode)
        train_dev_fn: Training function (developer mode with hyperparameters)
        eval_fn: Evaluation function
    """
    with gr.Blocks() as page:
        gr.Markdown("# Train Your Model")
        gr.Markdown(
            "Follow the 4 steps below to train and evaluate your model."
        )

        # ============================================================
        # STAGE 1: Upload Files
        # ============================================================
        gr.Markdown("## 1. Upload Files")
        
        training_file = gr.File(
            label="Training Dataset (JSONL)",
            file_types=[".jsonl"],
            file_count="single",
            type="filepath",
        )
        
        analyze_btn = gr.Button(
            "Analyze Dataset",
            variant="primary",
            size="lg",
        )

        # ============================================================
        # STAGE 2: Dataset Expose
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("## 2. Dataset Overview")
        
        with gr.Row():
            # Statistics column
            with gr.Column(scale=1):
                stats_md = gr.Markdown("Upload and analyze dataset to see statistics")
            
            # Charts column
            with gr.Column(scale=2):
                # Tool distribution bar chart
                tool_chart = gr.BarPlot(
                    x="Tool",
                    y="Count",
                    title="Tool Distribution",
                    x_title="Tool Name",
                    y_title="Number of Samples",
                    height=300,
                    visible=False,
                )
                
                # Train/Test split bar chart
                split_chart = gr.BarPlot(
                    x="Split",
                    y="Count",
                    title="Train/Test Split",
                    x_title="Dataset Split",
                    y_title="Number of Samples",
                    height=250,
                    visible=False,
                )

        # ============================================================
        # STAGE 3: Training (with Normal and Developer tabs)
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("## 3. Training")
        
        with gr.Tabs() as training_tabs:
            # ========== NORMAL MODE TAB ==========
            with gr.Tab("Normal Mode"):
                train_btn = gr.Button(
                    "Start Training",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )
                
                status_output = gr.Markdown(
                    "Status: Waiting for dataset analysis..."
                )
                
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
                    variant="primary",
                    size="lg",
                )
            
            # ========== DEVELOPER MODE TAB ==========
            with gr.Tab("Developer Mode"):
                gr.Markdown("### Hyperparameters")
                gr.Markdown("Configure advanced training parameters")
                
                with gr.Row():
                    with gr.Column():
                        num_epochs = gr.Slider(
                            label="Number of Epochs",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=3,
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=2e-5,
                            precision=6,
                        )
                        batch_size = gr.Slider(
                            label="Batch Size (per device)",
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=1,
                        )
                    
                    with gr.Column():
                        grad_accumulation = gr.Slider(
                            label="Gradient Accumulation Steps",
                            minimum=1,
                            maximum=32,
                            step=1,
                            value=16,
                        )
                        test_size = gr.Slider(
                            label="Test Split Ratio",
                            minimum=0.05,
                            maximum=0.3,
                            step=0.05,
                            value=0.10,
                        )
                        logging_steps = gr.Slider(
                            label="Logging Steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=5,
                        )
                
                train_dev_btn = gr.Button(
                    "Start Training (Developer)",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )
                
                dev_status_output = gr.Markdown(
                    "Status: Waiting for dataset analysis..."
                )
                
                dev_logs_output = gr.Code(
                    label="Raw Technical Logs",
                    language="shell",
                    interactive=False,
                    lines=20,
                )
                
                download_status_dev = gr.Markdown("", visible=False)
                
                download_model_dev_btn = gr.DownloadButton(
                    label="Download Trained Model",
                    visible=False,
                    variant="primary",
                    size="lg",
                )

        # ============================================================
        # STAGE 4: Evaluation
        # ============================================================
        gr.Markdown("---")
        gr.Markdown("## 4. Evaluate Model")
        gr.Markdown("Evaluate the fine-tuned model on test data with detailed analysis.")
        
        eval_btn = gr.Button(
            "Run Evaluation",
            variant="secondary",
            size="lg",
            interactive=False,
        )

        eval_status = gr.Markdown("Status: Complete training first")
        eval_output = gr.Code(
            label="Evaluation Results",
            language="shell",
            interactive=False,
            lines=25,
        )

        # ============================================================
        # Event Handlers (imported from train_model_handlers.py)
        # ============================================================
        on_analyze = create_analyze_handler(analyze_fn)
        
        # Hidden trigger state for download preparation
        download_trigger_normal = gr.State(False)
        download_trigger_dev = gr.State(False)
        
        # Wire up events
        analyze_btn.click(
            fn=on_analyze,
            inputs=[training_file],
            outputs=[stats_md, tool_chart, split_chart, train_btn, train_dev_btn],
        )
        
        # Normal mode training
        train_btn.click(
            fn=train_fn,
            inputs=[training_file],
            outputs=[status_output, logs_output],
        ).then(
            fn=on_train_complete,
            inputs=[status_output, logs_output],
            outputs=[
                eval_btn, eval_status,
                download_status, download_model_btn, download_trigger_normal,
                download_status_dev, download_model_dev_btn, download_trigger_dev,
            ],
        ).then(
            fn=prepare_model_download,
            inputs=[download_trigger_normal],
            outputs=[download_status, download_model_btn, download_status_dev, download_model_dev_btn],
        )
        
        # Developer mode training with hyperparameters
        train_dev_btn.click(
            fn=train_dev_fn,
            inputs=[
                training_file,
                num_epochs,
                learning_rate,
                batch_size,
                grad_accumulation,
                test_size,
                logging_steps,
            ],
            outputs=[dev_status_output, dev_logs_output],
        ).then(
            fn=on_train_complete,
            inputs=[dev_status_output, dev_logs_output],
            outputs=[
                eval_btn, eval_status,
                download_status, download_model_btn, download_trigger_normal,
                download_status_dev, download_model_dev_btn, download_trigger_dev,
            ],
        ).then(
            fn=prepare_model_download,
            inputs=[download_trigger_normal],
            outputs=[download_status, download_model_btn, download_status_dev, download_model_dev_btn],
        )
        
        # Evaluation
        eval_btn.click(
            fn=eval_fn,
            inputs=[],
            outputs=[eval_status, eval_output],
        )

    return page
