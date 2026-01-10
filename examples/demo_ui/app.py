"""
Afterimage Demo UI - Main Application

A Gradio-based UI for demonstrating synthetic data generation capabilities.
"""

import asyncio
import os

import gradio as gr
import pandas as pd

from afterimage import InMemoryDocumentProvider, PersonaGenerator

from storage import CaptureStorage
from converters import items_to_dataframe_data
from generators import create_generator, create_generation_task, GenerationMode
from utils import create_document_provider_from_file

# Import Pages
from pages.base import create_theme
from pages.structured_gen import create_structured_gen_page
from pages.generic_conv import create_generic_conv_page
from pages.tool_calling import create_tool_calling_page
from pages.how_it_works import create_how_it_works_page


# --- Configuration ---

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


# --- Core Generation Logic ---


async def load_context(
    context_source: str,
    context_text: str,
    context_file: str | None,
    context_key: str,
) -> tuple[list[str] | None, str | None]:
    """
    Load context from the specified source.
    
    Returns:
        Tuple of (texts, error_message). If error, texts is None.
    """
    all_texts = []
    
    if context_source == "Manual Entry":
        manual_chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]
        if manual_chunks:
            all_texts.extend(manual_chunks)
    
    elif context_source == "File Upload":
        if not context_file:
            return None, "### Error: No file uploaded"
        try:
            file_provider = create_document_provider_from_file(context_file, context_key)
            file_docs = file_provider.get_all()
            all_texts.extend([d.text for d in file_docs])
        except Exception as e:
            return None, f"### Error loading file: {str(e)}"
    
    if not all_texts:
        return None, "### Error: No context provided"
    
    return all_texts, None


async def run_generation_task(
    context_text: str,
    respondent_prompt: str,
    num_samples: int,
    context_source: str,
    context_file: str | None = None,
    context_key: str = "text",
    generation_mode: GenerationMode = "Structured Generation",
):
    """
    Main generation task that orchestrates the entire generation process.
    
    Yields status updates and results for the Gradio UI.
    """
    if not api_key:
        yield pd.DataFrame(), "### Error: API Key is required", None
        return

    # Load context
    all_texts, error = await load_context(
        context_source, context_text, context_file, context_key
    )
    if error:
        yield pd.DataFrame(), error, None
        return

    docs = InMemoryDocumentProvider(all_texts)
    storage = CaptureStorage()

    try:
        # Generate personas
        yield pd.DataFrame(), "### Status: Generating Personas...", None
        persona_gen = PersonaGenerator(api_key=api_key)
        await persona_gen.generate_from_documents(docs)

        # Create generator
        yield pd.DataFrame(), "### Status: Initializing Generator...", None
        generator = create_generator(
            mode=generation_mode,
            api_key=api_key,
            docs=docs,
            storage=storage,
            respondent_prompt=respondent_prompt,
        )
        
        # Start generation task
        gen_task = create_generation_task(generator, num_samples, generation_mode)

        # Live updates while generating
        while not gen_task.done():
            await asyncio.sleep(0.5)
            data = items_to_dataframe_data(storage.captured_items, truncate=True)
            yield (
                pd.DataFrame(data),
                f"### Status: Generating... ({len(data)}/{num_samples})",
                None,
            )

        await gen_task

        # Final pass without truncation
        data = items_to_dataframe_data(storage.captured_items, truncate=False)
        yield (
            pd.DataFrame(data),
            "### Status: Generation Complete! ✅",
            storage.get_download_path(),
        )

    except Exception as e:
        yield pd.DataFrame(), f"### Error: {str(e)}", None


# --- Page Wrappers ---


async def start_structured_gen(*args):
    async for u in run_generation_task(*args, generation_mode="Structured Generation"):
        yield u


async def start_generic_gen(*args):
    async for u in run_generation_task(*args, generation_mode="Generic Conversation"):
        yield u


async def start_tool_gen(*args):
    async for u in run_generation_task(*args, generation_mode="Tool Calling Generation"):
        yield u


# --- Main App ---

with gr.Blocks(title="Afterimage") as demo:
    create_how_it_works_page()
    gr.Navbar(main_page_name="How it Works")

with demo.route("Generic Conversation", "/conversations"):
    create_generic_conv_page(start_generic_gen)

with demo.route("Tool Calling", "/tools"):
    create_tool_calling_page(start_tool_gen)

with demo.route("Structured Generation", "/structured"):
    create_structured_gen_page(start_structured_gen)


if __name__ == "__main__":
    demo.launch(footer_links=["ALTAI"], theme=create_theme())
