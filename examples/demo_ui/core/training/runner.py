"""
Model training subprocess runner.
"""
import asyncio
import sys
import subprocess
import threading
import queue
import re
from typing import AsyncGenerator, Tuple

from ..config import get_training_dir, get_data_dir, SPINNERS
from ..file_utils import copy_dataset_file, merge_dataset_files
from .utils import format_time, make_progress_display


async def run_training(training_file) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Run model training with uploaded dataset.
    Shows animated progress display with smooth spinner.
    
    Args:
        training_file: File object (or list of files) from Gradio
        
    Yields:
        Tuple of (status_message, progress_display)
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
        
        # Copy or merge dataset file(s)
        if isinstance(training_file, list):
            merge_dataset_files(training_file, data_dir)
        else:
            copy_dataset_file(training_file, data_dir)
        
        yield "Status: ✓ Files ready", ""
        await asyncio.sleep(0.5)
        
        # Run training subprocess
        process = subprocess.Popen(
            [sys.executable, "train.py"],
            cwd=training_dir,
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
        
        while True:
            # Check for new lines (non-blocking)
            try:
                while True:
                    line = line_queue.get_nowait()
                    if line is None:
                        phase = "done"
                        break
                    
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
            
            await asyncio.sleep(0.15)  # Smooth animation ~6.6 fps
        
        # Final check
        process.wait()
        if process.returncode == 0:
            completion_msg = "Training complete!\n\n"
            completion_msg += "[" + "█" * 40 + "] 100%"
            yield "Status: ✓ Complete!", completion_msg
        else:
            yield "Status: Error - Training failed", "Please check your files and try again."
            
    except Exception as e:
        yield f"Status: Error - {str(e)}", "An error occurred during training."


async def run_training_developer(
    training_file,
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
    
    Args:
        training_file: File object from Gradio upload
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size per device
        grad_accumulation: Gradient accumulation steps
        test_size: Test split ratio
        logging_steps: Logging frequency
        
    Yields:
        Tuple of (status_message, raw_logs)
    """
    if not training_file:
        yield "Status: Error - Please upload a training dataset (JSONL)", ""
        return
    
    training_dir = get_training_dir()
    data_dir = get_data_dir()
    
    try:
        # Preparing phase
        yield "Status: Preparing files...", ""
        
        # Copy or merge dataset file(s)
        if isinstance(training_file, list):
            merge_dataset_files(training_file, data_dir)
        else:
            copy_dataset_file(training_file, data_dir)
        
        yield "Status: ✓ Files ready, starting training subprocess...", ""
        
        # Build command with hyperparameters
        cmd = [
            sys.executable, "train.py",
            "--num_epochs", str(num_epochs),
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--grad_accumulation", str(grad_accumulation),
            "--test_size", str(test_size),
            "--logging_steps", str(logging_steps),
        ]
        
        # Run training subprocess
        process = subprocess.Popen(
            cmd,
            cwd=training_dir,
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
        
        # Collect all logs (no filtering in developer mode)
        all_logs = []
        is_done = False
        spinner_idx = 0
        
        while not is_done:
            # Check for new lines
            try:
                while True:
                    line = line_queue.get_nowait()
                    if line is None:
                        is_done = True
                        break
                    
                    if line:  # Only skip empty lines
                        all_logs.append(line)
            except queue.Empty:
                pass
            
            if is_done:
                break
            
            # Update display with spinner and raw logs
            spinner_idx += 1
            spinner = SPINNERS[spinner_idx % len(SPINNERS)]
            
            # Show last 40 lines of raw logs
            display_logs = "\n".join(all_logs[-40:])
            
            yield f"Status: {spinner} Training in progress...", display_logs
            
            await asyncio.sleep(0.15)
        
        # Final check
        process.wait()
        if process.returncode == 0:
            final_logs = "\n".join(all_logs)
            yield "Status: ✓ Training Complete!", final_logs
        else:
            final_logs = "\n".join(all_logs)
            yield "Status: Error - Training failed", final_logs
            
    except Exception as e:
        yield f"Status: Error - {str(e)}", "An error occurred during training."
