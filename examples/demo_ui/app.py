"""
Afterimage Demo UI - Main Application

A Gradio-based UI for demonstrating synthetic data generation capabilities.
"""

import asyncio
import sys
from pathlib import Path

import gradio as gr

# Ensure repository package imports work when launched as:
# `uv run examples/demo_ui/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup environment first
from core.config import setup_environment, get_api_key

setup_environment()

# Get API key (will raise ValueError if not set)
api_key = get_api_key()

# Import core functionality
from core import (
    run_analysis,
    run_training,
    run_training_developer,
    run_evaluation,
)

# Import pages
from pages.base import create_theme, CUSTOM_CSS
from pages.structured_gen import create_structured_gen_page
from pages.generic_conv import create_generic_conv_page
from pages.tool_calling import create_tool_calling_page
from pages.how_it_works import create_how_it_works_page
from pages.train_model import create_train_model_page
from pages.tool_library import create_tool_library_page
from pages.handlers.chat import chat_with_trained_model
from pages.handlers.generation import (
    start_structured_gen,
    start_generic_gen,
    start_tool_gen,
    start_training_from_path,
)


# --- Context Loading ---


# --- Main Application ---

with gr.Blocks(title="Afterimage") as demo:
    create_how_it_works_page()
    gr.Navbar(main_page_name="How it Works")

with demo.route("Generic Conversation", "/conversations"):
    create_generic_conv_page(start_generic_gen)

with demo.route("Tool Calling", "/tools"):
    create_tool_calling_page(start_tool_gen, start_training_from_path)

with demo.route("Structured Generation", "/structured"):
    create_structured_gen_page(start_structured_gen)

with demo.route("Train Model", "/train"):
    create_train_model_page(
        run_analysis,
        run_training,
        run_training_developer,
        run_evaluation,
        chat_with_trained_model,
    )


with demo.route("Tool Library", "/tool-library"):
    create_tool_library_page()


if __name__ == "__main__":
    demo.launch(
        share=True,
        footer_links=["ALTAI"],
        theme=create_theme(),
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=7860,
    )
