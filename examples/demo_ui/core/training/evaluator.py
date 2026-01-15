"""
Model evaluation subprocess runner.
"""
import asyncio
import sys
import subprocess
import threading
import queue
from typing import AsyncGenerator, Tuple

from ..config import get_training_dir, SPINNERS


async def run_evaluation() -> AsyncGenerator[Tuple[str, str], None]:
    """
    Run detailed model evaluation.
    
    Yields:
        Tuple of (status_message, evaluation_output)
    """
    training_dir = get_training_dir()
    
    try:
        yield "Status: Starting evaluation...", ""
        
        # Run evaluation subprocess
        process = subprocess.Popen(
            [sys.executable, "evaluate.py"],
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
            line_queue.put(None)
        
        thread = threading.Thread(target=reader_thread, daemon=True)
        thread.start()
        
        spinner_idx = 0
        output_lines = []
        is_done = False
        
        # Filter patterns for noisy output
        FILTER_PATTERNS = [
            "Skipping import of cpp extensions",
            "Generating train split:",
            "The following generation flags",
            "TRANSFORMERS_VERBOSITY",
        ]
        
        while not is_done:
            # Check for new lines
            try:
                while True:
                    line = line_queue.get_nowait()
                    if line is None:
                        is_done = True
                        break
                    
                    # Filter out noise
                    if line and not any(skip in line for skip in FILTER_PATTERNS):
                        output_lines.append(line)
            except queue.Empty:
                pass
            
            if is_done:
                break
            
            # Update display
            spinner_idx += 1
            spinner = SPINNERS[spinner_idx % len(SPINNERS)]
            
            display = f"{spinner} Evaluating model...\n\n"
            # Show last 30 lines
            display += "\n".join(output_lines[-30:])
            
            yield f"Status: {spinner} Running evaluation...", display
            
            await asyncio.sleep(0.15)
        
        # Final check
        process.wait()
        if process.returncode == 0:
            final_output = "\n".join(output_lines)
            yield "Status: ✓ Evaluation complete!", final_output
        else:
            yield "Status: Error - Evaluation failed", "\n".join(output_lines[-20:])
            
    except Exception as e:
        yield f"Status: Error - {str(e)}", "An error occurred during evaluation."
