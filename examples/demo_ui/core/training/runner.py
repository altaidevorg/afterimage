"""
Model training subprocess runner.
"""
import asyncio
import sys
import subprocess
import threading
import queue
import re
from collections import deque
from typing import AsyncGenerator, Tuple

import pandas as pd
import json
import random
import os

def filter_and_prepare_dataset(training_file, tool_filter_config, target_dir) -> str:
    """
    Merge, filter and prepare dataset for training.
    
    Args:
        training_file: Source file(s)
        tool_filter_config: Dataframe/List from UI with [ToolName, Total, UseCount]
        target_dir: Directory to save processed file
        
    Returns:
        Path to the processed dataset file
    """
    # 1. Merge/Copy first to a temporary location
    if isinstance(training_file, list):
        merged_path = merge_dataset_files(training_file, target_dir)
    else:
        merged_path = copy_dataset_file(training_file, target_dir)
        
    # If no filter config, just return the merged path
    if tool_filter_config is None or (isinstance(tool_filter_config, pd.DataFrame) and tool_filter_config.empty) or (isinstance(tool_filter_config, list) and not tool_filter_config):
        return merged_path
        
    # Convert config to dict: {tool_name: use_count}
    limits = {}
    if isinstance(tool_filter_config, dict):
        # Direct dictionary input
        for tool, count in tool_filter_config.items():
            try:
                limits[str(tool)] = int(count)
            except (ValueError, TypeError):
                continue
    elif isinstance(tool_filter_config, pd.DataFrame):
        for _, row in tool_filter_config.iterrows():
            tool = str(row[0])
            try:
                limit = int(row[2])
                limits[tool] = limit
            except (ValueError, TypeError):
                continue
    elif isinstance(tool_filter_config, list):
        # Gradio might pass list of lists
        for row in tool_filter_config:
            if len(row) >= 3:
                tool = str(row[0])
                try:
                    limit = int(row[2])
                    limits[tool] = limit
                except (ValueError, TypeError):
                    continue
                    
    if not limits:
        return merged_path

    # 2. Read, Filter, Shuffle
    print(f"Filtering dataset with limits: {limits}")
    
    data_by_tool = {} # {tool_name: [lines]}
    others = [] # Lines with no tool calls or unknown structure
    
    with open(merged_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                # Heuristic to find tool name. 
                # Afterimage datasets usually have 'tools_used' or look at first tool call
                tools_used = item.get('tools_used', [])
                
                # If multiple tools, we assign to the first one for simplicity 
                # OR we could just treat "Multi-Tool" as a category.
                # For this feature, let's assume single-turn or primary tool focus.
                if tools_used:
                    primary_tool = tools_used[0]
                    if primary_tool not in data_by_tool:
                        data_by_tool[primary_tool] = []
                    data_by_tool[primary_tool].append(line)
                else:
                    others.append(line)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                others.append(line)
    
    # 3. Apply limits and write new file
    output_path = os.path.join(target_dir, "filtered_training_data.jsonl")
    
    final_lines = []
    
    # Add others (no tool calls) - include all or none? 
    # Let's include all 'conversational' samples by default unless we want to filter them too.
    # The requirement is specifically about tools.
    final_lines.extend(others)
    
    for tool, lines in data_by_tool.items():
        limit = limits.get(tool, len(lines)) # Default to all if not in limits
        
        # Shuffle before taking top N
        random.shuffle(lines)
        
        selected = lines[:limit]
        final_lines.extend(selected)
        print(f"Tool {tool}: Keeping {len(selected)}/{len(lines)} samples")
        
    # Shuffle the final mix so tools aren't clustered
    random.shuffle(final_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + "\n")
            
    return output_path

from ..config import get_training_dir, get_data_dir, SPINNERS
from ..file_utils import copy_dataset_file, merge_dataset_files
from .utils import format_time, make_progress_display



async def _execute_training_subprocess(
    cmd: list, cwd: str, is_dev_mode: bool = False
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Common helper to execute training subprocess and stream logs.
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Thread to read stdout
    line_queue = queue.Queue()
    
    def reader_thread():
        for line in iter(process.stdout.readline, ''):
            line_queue.put(line.strip())
        line_queue.put(None)  # Signal end
    
    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()
    
    # State tracking
    training_started = False
    spinner_idx = 0
    current_percent = 0
    current_remaining = ""
    phase = "loading"  # loading, training, done
    all_logs = [] # Used in dev mode
    recent_lines: deque[str] = deque(maxlen=200)  # Always kept for failure diagnostics
    
    while True:
        # Check for new lines (non-blocking)
        try:
            while True:
                line = line_queue.get_nowait()
                if line is None:
                    phase = "done"
                    break
                
                if not line: continue

                recent_lines.append(line)
                
                if is_dev_mode:
                    all_logs.append(line)
                
                # Detect phases
                if "[OK] Model loaded" in line:
                    phase = "loading_done"
                
                if "TRAINING STARTED" in line:
                    training_started = True
                    phase = "training"
                
                if "TRAINING COMPLETED" in line:
                    training_started = False
                    phase = "training_complete"
                
                # Parse training progress
                if training_started and phase == "training":
                    progress_match = re.search(
                        r"(\d+)%\|[█▏▎▍▌▋▊▉ ]+\|\s*\d+/\d+\s*\[[\d:]+<([\d:]+)",
                        line
                    )
                    if progress_match:
                        current_percent = int(progress_match.group(1))
                        current_remaining = progress_match.group(2)
                
                # Detect final completion
                if "[OK] Model saved" in line:
                    phase = "done"
                    break
        except queue.Empty:
            pass
        
        if phase == "done":
            break
        
        # Update display with animated spinner
        spinner_idx += 1
        spinner = SPINNERS[spinner_idx % len(SPINNERS)]
        
        if is_dev_mode:
            # Show logs in dev mode
            display_logs = "\n".join(all_logs[-40:])
            yield f"Status: {spinner} Training in progress...", display_logs
            
        else:
            # Normal Mode UI updates
            if phase == "loading":
                yield f"Status: {spinner} Loading model...", ""
            elif phase == "loading_done":
                yield f"Status: {spinner} Model loaded, starting training...", ""
            elif phase == "training":
                yield (
                    f"Status: Training... {current_percent}%",
                    make_progress_display(current_percent, current_remaining, spinner_idx)
                )
            elif phase == "training_complete":
                yield "Status: ✓ Training complete", "[" + "█" * 40 + "] 100%"
        
        await asyncio.sleep(0.15)
    
    # Final check
    process.wait()
    if process.returncode == 0:
        if is_dev_mode:
             final_logs = "\n".join(all_logs)
             yield "Status: ✓ Training Complete!", final_logs
        else:
            completion_msg = "Training complete!\n\n"
            completion_msg += "[" + "█" * 40 + "] 100%"
            yield "Status: ✓ Complete!", completion_msg
    else:
        if is_dev_mode:
            final_logs = "\n".join(all_logs)
            yield "Status: Error - Training failed", final_logs
        else:
            tail = "\n".join(recent_lines).strip()
            detail = (
                tail
                if tail
                else "No log output captured. Typical causes: missing HF_TOKEN, dataset path mismatch, or model download failure."
            )
            yield (
                "Status: Error - Training failed",
                "Training subprocess exited with an error. Last output:\n\n" + detail,
            )


async def run_training(training_file, tool_filter_config=None) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Run model training with uploaded dataset.
    Shows animated progress display with smooth spinner.
    """
    if not training_file:
        yield "Status: Error - Please upload a training dataset (JSONL)", ""
        return
    
    training_dir = get_training_dir()
    data_dir = get_data_dir()
    
    try:
        # Preparing phase with animation
        for i in range(3):
            spinner = SPINNERS[i % len(SPINNERS)]
            yield f"Status: {spinner} Preparing files...", ""
            await asyncio.sleep(0.2)
        
        # Filter and prepare dataset
        try:
            prepared_dataset_path = filter_and_prepare_dataset(training_file, tool_filter_config, data_dir)
        except Exception as e:
            print(f"Filtering failed: {e}")
            # Fallback to simple copy/merge if filtering fails
            if isinstance(training_file, list):
                prepared_dataset_path = merge_dataset_files(training_file, data_dir)
            else:
                prepared_dataset_path = copy_dataset_file(training_file, data_dir)
        
        yield "Status: ✓ Files ready", ""
        await asyncio.sleep(0.5)
        
        # Run training command (explicit dataset path so filtered merges and UI-prepared files match train.py)
        rel_dataset = os.path.relpath(prepared_dataset_path, training_dir)
        cmd = [sys.executable, "train.py", "--dataset", rel_dataset]
        
        async for status, output in _execute_training_subprocess(cmd, training_dir, is_dev_mode=False):
            yield status, output
            
    except Exception as e:
        yield f"Status: Error - {str(e)}", "An error occurred during training."


async def run_training_developer(
    training_file,
    tool_filter_config=None,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    grad_accumulation: int = 16,
    test_size: float = 0.10,
    logging_steps: int = 5,
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Run model training with custom hyperparameters (Developer Mode).
    Shows raw technical logs without filtering.
    """
    if not training_file:
        yield "Status: Error - Please upload a training dataset (JSONL)", ""
        return
    
    training_dir = get_training_dir()
    data_dir = get_data_dir()
    
    try:
        # Preparing phase
        yield "Status: Preparing files...", ""
        
        # Filter and prepare dataset
        try:
            prepared_dataset_path = filter_and_prepare_dataset(training_file, tool_filter_config, data_dir)
        except Exception as e:
            print(f"Filtering failed: {e}")
            if isinstance(training_file, list):
                prepared_dataset_path = merge_dataset_files(training_file, data_dir)
            else:
                prepared_dataset_path = copy_dataset_file(training_file, data_dir)
        
        yield "Status: ✓ Files ready, starting training subprocess...", ""
        
        rel_dataset = os.path.relpath(prepared_dataset_path, training_dir)
        # Build command with hyperparameters
        cmd = [
            sys.executable, "train.py",
            "--dataset", rel_dataset,
            "--num_epochs", str(num_epochs),
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--grad_accumulation", str(grad_accumulation),
            "--test_size", str(test_size),
            "--logging_steps", str(logging_steps),
        ]
        
        async for status, output in _execute_training_subprocess(cmd, training_dir, is_dev_mode=True):
            yield status, output
            
    except Exception as e:
        yield f"Status: Error - {str(e)}", "An error occurred during training."
