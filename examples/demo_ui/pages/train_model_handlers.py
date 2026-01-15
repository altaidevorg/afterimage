"""
Event handlers for the Train Model page.
"""
import os
import sys
import shutil
import tempfile
import time
import pandas as pd
import gradio as gr
from typing import Tuple, Any

from core.config import get_training_dir


def create_analyze_handler(analyze_fn):
    """
    Create the dataset analysis handler.
    
    Args:
        analyze_fn: Function to analyze the dataset
        
    Returns:
        Handler function for dataset analysis
    """
    def on_analyze(training_file):
        """Analyze dataset and show statistics + charts."""
        if not training_file:
            return (
                "Error: Please upload a training dataset first",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),  # train_btn
                gr.update(interactive=False),  # train_dev_btn
            )
        
        try:
            stats, tool_names, tool_counts = analyze_fn(training_file)
        except Exception as e:
            print(f"Analysis error: {e}")
            return (
                f"Error: Analysis failed - {str(e)}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        
        if not stats or 'error' in stats:
            return (
                "Error: Could not analyze dataset",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        
        # Format statistics markdown
        test_pct_of_total = stats['test_samples']/stats['total_samples']*100 if stats['total_samples'] > 0 else 0
        test_pct_of_tool = stats['test_samples']/stats['tool_call_samples']*100 if stats['tool_call_samples'] > 0 else 0
        
        total_samples = stats['total_samples'] if stats['total_samples'] > 0 else 1  # Prevent division by zero
        
        stats_text = f"""
### Dataset Statistics

**Total Samples:** {stats['total_samples']}

**Distribution:**
- With Tool Calls: {stats['tool_call_samples']} ({stats['tool_call_samples']/total_samples*100:.1f}%)
- Without Tool Calls: {stats['no_tool_call_samples']} ({stats['no_tool_call_samples']/total_samples*100:.1f}%)

**Data Split:**
- Training: {stats['train_samples']} samples ({stats['train_samples']/total_samples*100:.1f}% of total)
- Test: {stats['test_samples']} samples ({test_pct_of_tool:.1f}% of tool calls, {test_pct_of_total:.1f}% of total)

**Available Tools:** {len(stats['tool_distribution'])} different tools

**Sample Instructions:**
"""
        for i, instr in enumerate(stats.get('sample_instructions', []), 1):
            stats_text += f"\n{i}. {instr}"
        
        # Prepare chart data - Gradio BarPlot needs pandas DataFrame
        tool_data = pd.DataFrame({
            "Tool": tool_names if tool_names else [],
            "Count": tool_counts if tool_counts else []
        })
        split_data = pd.DataFrame({
            "Split": ["Training", "Test"],
            "Count": [stats['train_samples'], stats['test_samples']]
        })
        
        return (
            stats_text,
            gr.update(value=tool_data, visible=True if tool_names else False),
            gr.update(value=split_data, visible=True),
            gr.update(interactive=True),  # Enable train button (normal)
            gr.update(interactive=True),  # Enable train button (dev)
        )
    
    return on_analyze


def on_train_complete(status, logs):
    """Enable evaluation button and trigger download preparation when training completes."""
    if not status:
        return (
            gr.update(), gr.update(),  # eval
            gr.update(), gr.update(), gr.update(),  # normal mode download
            gr.update(), gr.update(), gr.update(),  # dev mode download
        )
    
    if "Complete" in status or "✓" in status:
        # Enable evaluation button
        eval_btn_update = gr.update(interactive=True)
        eval_status_update = gr.update(value="Status: Ready to evaluate")
        
        # Show download status messages (trigger preparation)
        download_status_update = gr.update(value="⏳ Preparing download...", visible=True)
        download_status_dev_update = gr.update(value="⏳ Preparing download...", visible=True)
        
        # Buttons will be shown after preparation
        download_btn_update = gr.update(visible=False)
        download_dev_btn_update = gr.update(visible=False)
        
        # Trigger download preparation
        trigger = True
        
        return (
            eval_btn_update, eval_status_update,
            download_status_update, download_btn_update, trigger,
            download_status_dev_update, download_dev_btn_update, trigger,
        )
    
    return (
        gr.update(), gr.update(),
        gr.update(), gr.update(), False,
        gr.update(), gr.update(), False,
    )


def prepare_model_download(trigger):
    """Compress and prepare model for download with progress indication."""
    if not trigger:
        return (
            gr.update(),  # download_status
            gr.update(),  # download_model_btn
            gr.update(),  # download_status_dev
            gr.update(),  # download_model_dev_btn
        )
    
    # Get model directory
    training_dir = get_training_dir()
    model_dir = os.path.join(training_dir, "final_model_stable")
    
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        return (
            gr.update(value="❌ Model not found", visible=True),
            gr.update(visible=False),
            gr.update(value="❌ Model not found", visible=True),
            gr.update(visible=False),
        )
    
    try:
        # Calculate estimated time based on directory size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(model_dir)
            for filename in filenames
        ) / (1024 * 1024)  # MB
        
        # Rough estimate: ~100 MB/sec compression
        estimated_seconds = max(2, int(total_size / 100))
        
        status_msg = f"🔄 Compressing model... (estimated: ~{estimated_seconds}s)"
        yield (
            gr.update(value=status_msg, visible=True),
            gr.update(visible=False),
            gr.update(value=status_msg, visible=True),
            gr.update(visible=False),
        )
        
        # Create zip
        zip_base_path = os.path.join(tempfile.gettempdir(), "trained_model")
        if os.path.exists(f"{zip_base_path}.zip"):
            os.remove(f"{zip_base_path}.zip")
        
        start_time = time.time()
        shutil.make_archive(zip_base_path, 'zip', model_dir)
        actual_time = int(time.time() - start_time)
        
        # Show download buttons
        success_msg = f"✅ Ready to download! (compressed in {actual_time}s)"
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True, value=f"{zip_base_path}.zip", label=f"Download Model ({total_size:.1f}MB → zip)"),
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True, value=f"{zip_base_path}.zip", label=f"Download Model ({total_size:.1f}MB → zip)"),
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to create model zip: {e}")
        error_msg = f"❌ Error: {str(e)}"
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
        )
