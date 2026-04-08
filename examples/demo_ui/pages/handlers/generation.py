"""
Handlers for generation tasks (Structured, Generic, Tool Calling).
"""
import asyncio
import pandas as pd
import gradio as gr
from afterimage import InMemoryDocumentProvider, PersonaGenerator

from core.config import get_api_key
from core import (
    CaptureStorage,
    items_to_dataframe_data,
    create_generator,
    create_generation_task,
    GenerationMode,
    create_document_provider_from_file,
    get_selected_tools,
    run_training,
)

# --- Context Loading ---

async def load_context(
    context_source: str,
    context_text: str,
    context_file: str | None,
    context_key: str,
) -> tuple[list[str] | None, str | None]:
    """Load context from the specified source."""
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


# --- Generation Task ---

async def run_generation_task(
    context_text: str,
    respondent_prompt: str,
    num_samples: int,
    context_source: str,
    context_file: str | None = None,
    context_key: str = "text",
    generation_mode: GenerationMode = "Structured Generation",
    selected_tools: list | None = None,
    dataset_category: str = "Uncategorized",
):
    """Main generation task that orchestrates the entire generation process."""
    api_key = get_api_key()
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
    storage = CaptureStorage(category=dataset_category)

    try:
        # Generate personas
        yield pd.DataFrame(), "### Status: Generating Personas...", None
        persona_gen = PersonaGenerator(
            api_key=api_key,
            model_provider_name="deepseek",
            model_name="deepseek-chat",
        )
        await persona_gen.generate_from_documents(docs)

        # Create generator
        yield pd.DataFrame(), "### Status: Initializing Generator...", None
        generator = create_generator(
            mode=generation_mode,
            api_key=api_key,
            docs=docs,
            storage=storage,
            respondent_prompt=respondent_prompt,
            selected_tools=selected_tools,
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
            "### Status: Generation Complete!",
            storage.get_download_path(),
        )

    except Exception as e:
        yield pd.DataFrame(), f"### Error: {str(e)}", None


# --- Page-Specific Generation Wrappers ---

async def start_structured_gen(*args):
    async for update in run_generation_task(*args, generation_mode="Structured Generation"):
        yield update


async def start_generic_gen(*args):
    async for update in run_generation_task(*args, generation_mode="Generic Conversation"):
        yield update


async def start_tool_gen(
    context_text: str,
    respondent_prompt: str,
    num_samples: int,
    context_source: str,
    context_file: str | None,
    context_key: str,
    tool_names: list[str],
    dataset_category: str = "Uncategorized",
):
    """Start tool calling generation with selected tools."""
    # Get the actual tool objects from selected names
    selected_tools = get_selected_tools(tool_names)
    
    if not selected_tools:
        yield pd.DataFrame(), "### Error: Please select at least one tool", None
        return
    
    # Default category if empty
    category = dataset_category.strip() if dataset_category else "Uncategorized"
    
    async for data, status, path in run_generation_task(
        context_text,
        respondent_prompt,
        num_samples,
        context_source,
        context_file,
        context_key,
        generation_mode="Tool Calling Generation",
        selected_tools=selected_tools,
        dataset_category=category,
    ):
        if path:
            # Keep the download component hidden but update its value so training can pick it up
            yield data, status, gr.update(value=path, visible=False)
        else:
            yield data, status, None


# --- Training Wrappers ---

async def start_training_from_path(file_path: str):
    """Train model from a generated dataset file path."""
    if not file_path:
        yield "Status: No dataset file provided", ""
        return
    
    async for update in run_training(file_path):
        yield update
