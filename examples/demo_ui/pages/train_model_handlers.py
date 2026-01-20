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
from typing import Tuple, Any, List
from collections import Counter
from glob import glob

from core.config import get_training_dir, get_datasets_dir


def load_datasets_list():
    """Load list of saved datasets from the datasets folder."""
    datasets_dir = get_datasets_dir()
    jsonl_files = glob(os.path.join(datasets_dir, "*.jsonl"))
    
    # Sort by modification time (newest first)
    jsonl_files.sort(key=os.path.getmtime, reverse=True)
    
    # Return just the filenames for display
    choices = []
    for f in jsonl_files:
        basename = os.path.basename(f)
        size_kb = os.path.getsize(f) / 1024
        choices.append(f"{basename} ({size_kb:.1f} KB)")
    
    return gr.update(choices=choices, value=[choices[0]] if choices else [])


def on_dataset_select(selected_names: List[str] | None):
    """When datasets are selected, return their full paths, aggregated metadata overview, and button updates."""
    if not selected_names:
        return (
            None, 
            "_Select datasets to see their details_",
            gr.update(interactive=False),  # train_btn
            gr.update(interactive=False),  # train_dev_btn
        )
    
    datasets_dir = get_datasets_dir()
    full_paths = []
    
    # Aggregated stats
    total_samples = 0
    total_size_kb = 0
    tools_counter = Counter()
    has_meta = False
    
    for name in selected_names:
        filename = name.split(" (")[0]
        full_path = os.path.join(datasets_dir, filename)
        
        if os.path.exists(full_path):
            full_paths.append(full_path)
            total_size_kb += os.path.getsize(full_path) / 1024
            
            # Try to load metadata
            meta_path = full_path.replace(".jsonl", ".meta.json")
            if os.path.exists(meta_path):
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    has_meta = True
                    # Parse samples
                    samples = meta.get('total_samples', 0)
                    if isinstance(samples, str) and samples.isdigit():
                        samples = int(samples)
                    total_samples += samples if isinstance(samples, int) else 0
                    
                    # Parse tool distribution
                    tool_dist = meta.get('tool_distribution', {})
                    if tool_dist:
                        tools_counter.update(tool_dist)
                    else:
                        tools_used = meta.get('tools_used', [])
                        for t in tools_used:
                            tools_counter[t] += 1
                            
                except Exception:
                    pass
    
    if not full_paths:
        return (
            None, 
            "_Dataset files not found_",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    # Build overview HTML dashboard
    # Summary Cards
    import textwrap
    overview = textwrap.dedent(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px;">
        <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px;">Datasets</div>
            <div style="font-size: 20px; font-weight: 700; color: #111827;">{len(full_paths)}</div>
        </div>
        <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px;">Total Size</div>
            <div style="font-size: 20px; font-weight: 700; color: #111827;">{total_size_kb:.1f} KB</div>
        </div>
        <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #e5e7eb;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px;">Total Samples</div>
            <div style="font-size: 20px; font-weight: 700; color: #111827;">{total_samples if has_meta else "?"}</div>
        </div>
    </div>
    """)
    
    if has_meta and tools_counter:
        overview += textwrap.dedent("""
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0 0 12px 0; color: #374151; font-size: 14px; font-weight: 600;">Tool Distribution</h4>
            <div style="display: flex; flex-direction: column; gap: 10px;">
        """)
        
        max_count = tools_counter.most_common(1)[0][1] if tools_counter else 1
        
        for tool, count in tools_counter.most_common():
            pct = (count / max_count) * 100
            overview += textwrap.dedent(f"""
                <div style="display: flex; align-items: center; font-size: 13px;">
                    <div style="width: 140px; color: #4b5563; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{tool}">{tool}</div>
                    <div style="flex-grow: 1; height: 8px; background: #f3f4f6; border-radius: 4px; margin: 0 12px; overflow: hidden;">
                        <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #6366f1, #4f46e5); border-radius: 4px;"></div>
                    </div>
                    <div style="width: 30px; text-align: right; color: #6b7280; font-variant-numeric: tabular-nums;">{count}</div>
                </div>
            """)
            
        overview += textwrap.dedent("""
            </div>
        </div>
        """)
    elif not has_meta:
        overview += textwrap.dedent("""
        <div style="padding: 12px; border-radius: 6px; background: #fff7ed; border: 1px solid #ffedd5; color: #9a3412; font-size: 13px;">
            Complete metadata is not available for one or more selected datasets.
        </div>
        """)
    
    return (
        full_paths, 
        overview,
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


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
        
        # Prepare chart data as markdown (BarPlot causes freeze issues)
        tool_md = "### Tool Distribution\n\n"
        if tool_names:
            for name, count in zip(tool_names, tool_counts):
                bar = "█" * min(count * 2, 40)  # Simple text bar
                tool_md += f"**{name}**: {bar} ({count})\n\n"
        else:
            tool_md += "_No tool data available_"
        
        split_md = "### Train/Test Split\n\n"
        split_md += f"**Training**: {'█' * min(stats['train_samples'] // 2, 40)} ({stats['train_samples']})\n\n"
        split_md += f"**Test**: {'█' * min(stats['test_samples'] // 2, 40)} ({stats['test_samples']})\n\n"
        
        return (
            stats_text,
            gr.update(value=tool_md, visible=True if tool_names else False),
            gr.update(value=split_md, visible=True),
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
    
    if "Complete" in status:
        # Enable evaluation button
        eval_btn_update = gr.update(interactive=True)
        eval_status_update = gr.update(value="Status: Ready to evaluate")
        
        # Show download status messages (trigger preparation)
        download_status_update = gr.update(value="Preparing download...", visible=True)
        download_status_dev_update = gr.update(value="Preparing download...", visible=True)
        
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
        
        status_msg = f"Compressing model... (estimated: ~{estimated_seconds}s)"
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
        success_msg = f"Ready to download! (compressed in {actual_time}s)"
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True, value=f"{zip_base_path}.zip", label=f"Download Model ({total_size:.1f}MB → zip)"),
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True, value=f"{zip_base_path}.zip", label=f"Download Model ({total_size:.1f}MB → zip)"),
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to create model zip: {e}")
        error_msg = f"Error: {str(e)}"
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
        )
