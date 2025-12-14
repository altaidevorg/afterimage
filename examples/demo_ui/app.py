import asyncio
import os
import tempfile
from enum import Enum
from typing import List

import gradio as gr
import pandas as pd
from pydantic import BaseModel, Field

from afterimage import (
    AsyncStructuredGenerator,
    InMemoryDocumentProvider,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
)
from afterimage.storage import BaseStorage, JSONLStorage
from afterimage.types import (
    ConversationWithContext,
    Document,
    EvaluatedConversationWithContext,
)

# --- Schema Definitions (from examples/customer_support_generator.py) ---


class SupportIntent(str, Enum):
    REFUND = "Refund Request"
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING = "Billing Inquiry"
    PRODUCT_INFO = "Product Information"
    WARRANTY = "Warranty Claim"
    COMPLAINT = "General Complaint"
    OTHER = "Other"


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ActionType(str, Enum):
    CLOSE = "Close"
    ESCALATION = "Escalation"
    KEEP_OPEN = "Keep Open"


class ToolCall(str, Enum):
    KNOWLEDGE_BASE_SEARCH = "Knowledge Base Search"
    NONE = "none"


class CustomerSupportInteraction(BaseModel):
    agent_reasoning: str = Field(
        description="Step-by-step reasoning to reach the final response. Explain the diagnosis and decision process."
    )
    intent: str = Field(description="Primary intent of the customer")
    urgency: str = Field(description="Assessed urgency level")
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (Very Negative) to 1.0 (Very Positive)"
    )
    key_entities: List[str] = Field(
        description="Key entities extracted (Product names, Order IDs, Dates)"
    )
    missing_information: List[str] = Field(
        description="Information missing to resolve the query"
    )
    action: ActionType = Field(
        description="The action taken by the agent. Close if it's resolved, escalade if it's urgent, and keep it open if it's pending customer."
    )
    action_reason: str = Field(description="Reason for the action taken by the agent.")
    query: str = Field(
        description="The search query that you would need to run against the knowledge base to resolve the customer request."
    )
    response: str = Field(
        description="The final natural language response to the customer"
    )


# --- Default Data ---

DEFAULT_RESPONDENT_PROMPT = """
You are an advanced AI Customer Support Agent for "TechGadget Inc".
Your goal is to triage incoming queries, analyze them deeply, and provide helpful, accurate responses based *strictly* on the provided context.

- Be empathetic but professional.
- If a user claims a defect, check the warranty policy.
- If a user wants a refund, check the 30-day window.
""".strip()

DEFAULT_CONTEXTS = """
# Refund Policy for TechGadget Inc.
- Standard Refund: Items can be returned within 30 days of purchase for a full refund.
- Defective Items: Defective items have a 1-year warranty. We will ship a replacement immediately upon proof of defect.
- Digital Goods: No refunds on software keys once redeemed.
- Restocking Fee: Open box items (non-defective) are subject to a 15% restocking fee.

# Troubleshooting Guide: TechGadget 3000
- Screen Won't Turn On: Hold the power button for 15 seconds to force reset. Check the charging port for debris.
- Bluetooth Audio Lag: Enhance connection by turning off WiFi on nearby devices (interference). Firmware v2.1 fixes this.
- Battery Draining Fast: Turn off 'Always-On Display' in Settings > Battery. Replace battery if health < 80%.

# Shipping & Delivery Guidelines
- Express Shipping: 1-2 business days. Cost: $15.00.
- Standard Shipping: 3-5 business days. Free for orders over $50.
- International: 7-14 business days. Customs duties are the responsibility of the recipient.
- Lost Packages: Claims must be filed within 48 hours of marked delivery.
""".strip()


# --- Storage Implementation ---


class CaptureStorage(BaseStorage):
    """Storage that captures items in-memory for UI and writes to a temporary file."""

    def __init__(self):
        self.captured_items = []
        # Create a temp file for the JSONL download
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="w+", encoding="utf-8"
        )
        self.jsonl_storage = JSONLStorage(conversations_path=self.temp_file.name)
        self.temp_file.close()  # Close handle, let JSONLStorage manage it

    def save_conversations(
        self,
        conversations: List[
            EvaluatedConversationWithContext | ConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        self.jsonl_storage.save_conversations(conversations)

    async def asave_conversations(
        self,
        conversations: List[
            ConversationWithContext | EvaluatedConversationWithContext | BaseModel
        ],
    ) -> None:
        self.captured_items.extend(conversations)
        await self.jsonl_storage.asave_conversations(conversations)

    def load_conversations(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> List[ConversationWithContext]:
        return []

    def save_documents(self, documents: List[Document]) -> None:
        pass

    async def asave_documents(self, documents: List[Document]) -> None:
        pass

    def get_download_path(self) -> str:
        return self.jsonl_storage.conversations_path.absolute().as_posix()


# --- Logic ---


async def run_generation_task(
    api_key: str,
    respondent_prompt: str,
    context_text: str,
    num_samples: int,
):
    """
    Main Async Task to run the specific afterimage pipeline.
    Yields tuple of (dataframe, status_msg, download_path_or_none)
    """
    if not api_key:
        yield pd.DataFrame(), "### Error: API Key is required", None
        return

    # Split context by blank lines to approximate separate documents
    # or just treat chunks separated by headers as documents?
    # For simplicity, let's split by double newlines as per the input format suggestion
    context_chunks = [c.strip() for c in context_text.split("\n\n") if c.strip()]

    if not context_chunks:
        yield pd.DataFrame(), "### Error: No context provided", None
        return

    docs = InMemoryDocumentProvider(context_chunks)
    storage = CaptureStorage()

    try:
        yield pd.DataFrame(), "### Status: Generating Personas...", None

        # 1. Generate Personas
        persona_gen = PersonaGenerator(api_key=api_key)
        await persona_gen.generate_from_documents(docs)

        yield pd.DataFrame(), "### Status: Initializing Generator...", None

        # 2. Setup Instruction Generator
        instruction_callback = PersonaInstructionGeneratorCallback(
            api_key=api_key,
            documents=docs,
            num_random_contexts=1,
        )

        # 3. Setup Structured Generator
        generator = AsyncStructuredGenerator(
            output_schema=CustomerSupportInteraction,
            respondent_prompt=respondent_prompt,
            api_key=api_key,
            model_name="gemini-2.0-flash",
            instruction_generator_callback=instruction_callback,
            storage=storage,
        )

        # 4. Run Generation
        # We need to run this alongside polling for updates since `generator.generate` awaits until done.
        # But `CaptureStorage` is updated in real-time.
        # So we spawn the generation task and loop to update UI.

        gen_task = asyncio.create_task(generator.generate(num_samples=num_samples))

        while not gen_task.done():
            # Update UI every 0.5 seconds
            await asyncio.sleep(0.5)

            # Construct DataFrame from captured items
            data = []
            for item in storage.captured_items:
                # We need to access .output (the model) and others
                # Relaxed check: just verify it has the expected attributes
                try:
                    # Attempt to get fields, handling both object and dict access or different types
                    persona = getattr(item, "persona", None)
                    if not persona and isinstance(item, dict):
                        persona = item.get("persona")

                    instruction = getattr(item, "instruction", "")

                    output = getattr(item, "output", None)
                    if not output:  # Skip if no output
                        continue

                    # Output fields
                    intent = getattr(
                        output,
                        "intent",
                        getattr(output, "get", lambda x: "N/A")("intent"),
                    )
                    urgency = getattr(
                        output,
                        "urgency",
                        getattr(output, "get", lambda x: "N/A")("urgency"),
                    )
                    reasoning = getattr(
                        output,
                        "agent_reasoning",
                        getattr(output, "get", lambda x: "")("agent_reasoning"),
                    )
                    response = getattr(
                        output,
                        "response",
                        getattr(output, "get", lambda x: "")("response"),
                    )

                    row = {
                        "Persona": str(persona) if persona else "N/A",
                        "Instruction": instruction,
                        "Intent": intent,
                        "Urgency": urgency,
                        "Reasoning": (str(reasoning)[:100] + "...")
                        if reasoning
                        else "",
                        "Response": response,
                    }
                    data.append(row)
                except Exception:
                    # If something goes wrong parsing a row, skip it or add error row
                    data.append(
                        {
                            "Persona": "Error",
                            "Instruction": "Error parsing row",
                            "Intent": "Error",
                            "Urgency": "Error",
                            "Reasoning": "Error",
                            "Response": "Error",
                        }
                    )

            df = pd.DataFrame(data)
            status_msg = f"### Status: Generating... ({len(data)}/{num_samples})"
            yield df, status_msg, None

        # Check for exceptions
        try:
            await gen_task
        except Exception as e:
            yield pd.DataFrame(), f"### Error during generation: {str(e)}", None
            return

        # Final Update
        data = []
        for item in storage.captured_items:
            try:
                persona = getattr(item, "persona", None)
                if not persona and isinstance(item, dict):
                    persona = item.get("persona")

                instruction = getattr(item, "instruction", "")
                output = getattr(item, "output", None)
                if not output:
                    continue

                intent = getattr(
                    output,
                    "intent",
                    getattr(output, "get", lambda x: "N/A")("intent"),
                )
                urgency = getattr(
                    output,
                    "urgency",
                    getattr(output, "get", lambda x: "N/A")("urgency"),
                )
                reasoning = getattr(
                    output,
                    "agent_reasoning",
                    getattr(output, "get", lambda x: "")("agent_reasoning"),
                )
                response = getattr(
                    output,
                    "response",
                    getattr(output, "get", lambda x: "")("response"),
                )

                row = {
                    "Persona": str(persona) if persona else "N/A",
                    "Instruction": instruction,
                    "Intent": intent,
                    "Urgency": urgency,
                    "Reasoning": str(reasoning),
                    "Response": response,
                }
                data.append(row)
            except Exception:
                pass

        df = pd.DataFrame(data)
        yield df, "### Status: Generation Complete! \u2705", storage.get_download_path()

    except Exception as e:
        yield pd.DataFrame(), f"### Error: {str(e)}", None


# Wrapper for Gradio to run async generator
async def start_generation(api_key, context, prompt, num_samples):
    # Use os.environ key if not provided in input
    final_key = api_key or os.environ.get("GEMINI_API_KEY")
    async for update in run_generation_task(
        final_key, prompt, context, int(num_samples)
    ):
        yield update


# --- UI Layout ---

custom_css = """
footer {visibility: hidden}
.gradio-container {min-height: 0px !important}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Afterimage Demo") as demo:
    gr.Markdown(
        """
        # \U0001f916 Afterimage: Synthetic Dataset Generation
        Generate consistent, high-quality synthetic datasets with persona-driven simulations.
        """
    )

    with gr.Tabs():
        with gr.Tab("Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    api_key_input = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="Leave empty if GEMINI_API_KEY env var is set",
                        type="password",
                    )
                    num_samples_input = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=15,
                        step=1,
                        label="Number of Samples",
                    )
                    context_input = gr.TextArea(
                        label="Context Documents (Separate different docs with blank lines)",
                        value=DEFAULT_CONTEXTS,
                        lines=10,
                    )
                    prompt_input = gr.TextArea(
                        label="Respondent System Prompt",
                        value=DEFAULT_RESPONDENT_PROMPT,
                        lines=5,
                    )
                    generate_btn = gr.Button("Generate Dataset", variant="primary")

                with gr.Column(scale=2):
                    status_output = gr.Markdown("### Status: Ready")
                    results_output = gr.DataFrame(
                        label="Generated Interactions",
                        headers=[
                            "Persona",
                            "Instruction",
                            "Intent",
                            "Urgency",
                            "Reasoning",
                            "Response",
                        ],
                        interactive=False,
                        wrap=True,
                    )
                    download_output = gr.File(label="Download JSONL Dataset")

            generate_btn.click(
                fn=start_generation,
                inputs=[api_key_input, context_input, prompt_input, num_samples_input],
                outputs=[results_output, status_output, download_output],
            )

        with gr.Tab("How it Works"):
            gr.Markdown("## The Afterimage Workflow")
            gr.Markdown(
                """
                Afterimage uses a multi-stage pipeline to generate high-quality synthetic data:
                1.  **Context Loading**: Your documents are loaded and indexed.
                2.  **Persona Generation**: Diverse user personas are created based on the context (e.g., "Frustrated User", "Tech-Savvy Student").
                3.  **Instruction Generation**: A "Simulator" agent adopts a persona and generates a specific query/instruction grounded in the context.
                4.  **Structured Generation**: The "Respondent" agent (the AI you want to train/test) answers the query, following a strict output schema.
                """
            )

            gr.Markdown("### Workflow Diagram")
            gr.Markdown(
                """
                ```mermaid
                graph LR
                    D[Documents] --> P[Persona Generator]
                    P --> Pers[Personas]
                    D --> IG[Instruction Generator]
                    Pers --> IG
                    IG --> Inst[Instruction/Query]
                    Inst --> SG[Structured Generator]
                    Sys[System Prompt] --> SG
                    SG --> Out[Structured Output]
                ```
                """
            )

            gr.Markdown(
                """
                ### About the Schema
                This demo generates data matching the `CustomerSupportInteraction` schema, which includes:
                - **Intent**: The classified intent of the user.
                - **Urgency**: How urgent the request is.
                - **Reasoning**: Chain-of-thought for the agent.
                - **Response**: The final reply.
                - **Actions**: Implementation of specific actions (Refund, Escalate, etc).
                """
            )

if __name__ == "__main__":
    demo.launch(footer_links=[])
