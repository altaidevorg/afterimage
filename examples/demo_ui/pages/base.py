import gradio as gr

# Shared Constants
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

TOOL_CALLING_RESPONDENT_PROMPT = """
You are the central logic unit for a Smart Home Assistant.
Map the user's natural language request to the structured tool calls.

- Select the correct tool(s) from the schema.
- Infer arguments where possible.
- If no tool matches, return an empty list.
""".strip()

SMART_HOME_CONTEXT_STR = """
# Smart Home User Manual

You have a smart home system with the following capabilities:

- **Lights**: Control brightness and color for any room (living_room, kitchen, bedroom, etc.).
- **Climate**: Set the thermostat temperature and mode (cool/heat/auto).
- **Music**: Play music by genre throughout the house.
- **Security**: Lock doors (front/back/garage) and check weather.

Users may ask for things in natural language, often implying multiple steps or inferring parameters.
""".strip()

AFTERIMAGE_DOCS_CONTEXT = """
# Afterimage: Synthetic Dataset Generation
Afterimage uses a multi-stage pipeline to generate high-quality synthetic data:
1.  **Context Loading**: Your documents are loaded and indexed.
2.  **Persona Generation**: Diverse user personas are created based on the context (e.g., "Frustrated User", "Tech-Savvy Student").
3.  **Instruction Generation**: A "Simulator" agent adopts a persona and generates a specific query/instruction grounded in the context.
4.  **Structured Generation**: The "Respondent" agent (the AI you want to train/test) answers the query, following a strict output schema.

### About the Schema
This demo generates data matching the `CustomerSupportInteraction` schema, which includes:
- **Intent**: The classified intent of the user.
- **Urgency**: How urgent the request is.
- **Reasoning**: Chain-of-thought for the agent.
- **Response**: The final reply.
- **Actions**: Implementation of specific actions (Refund, Escalate, etc).
""".strip()


# Shared UI Elements
def create_context_section(default_value=DEFAULT_CONTEXTS):
    with gr.Column():
        gr.Markdown("### 1. Context Source")
        context_source_radio = gr.Radio(
            choices=["Manual Entry", "File Upload"],
            value="Manual Entry",
            label="Select Source",
        )

        with gr.Column(visible=True) as manual_group:
            context_input = gr.TextArea(
                label="Context Documents",
                placeholder="Separate different docs with blank lines",
                value=default_value,
                lines=8,
            )

        with gr.Column(visible=False) as file_group:
            context_file_input = gr.File(
                label="Upload .txt, .csv, .tsv, .jsonl, .docx, .rtf, .html",
                file_types=[
                    ".txt",
                    ".csv",
                    ".tsv",
                    ".jsonl",
                    ".docx",
                    ".rtf",
                    ".html",
                ],
            )
            context_key_input = gr.Textbox(
                label="Column/Key for Context",
                value="text",
                placeholder="e.g., text, content, description",
            )

        def toggle_context_inputs(choice):
            if choice == "Manual Entry":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        context_source_radio.change(
            fn=toggle_context_inputs,
            inputs=[context_source_radio],
            outputs=[manual_group, file_group],
        )

    return {
        "source": context_source_radio,
        "input": context_input,
        "file": context_file_input,
        "key": context_key_input,
    }


def create_output_section(headers, label="Generated Data"):
    with gr.Column(scale=2):
        status_output = gr.Markdown("### Status: Ready")
        results_output = gr.DataFrame(
            label=label,
            headers=headers,
            interactive=False,
            wrap=True,
        )
        download_output = gr.File(label="Download JSONL Dataset", interactive=False)

    return status_output, results_output, download_output


# Theme Customization
def create_theme():
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
        text_size="lg",
        spacing_size="lg",
        radius_size="lg",
    ).set(
        body_background_fill="*neutral_50",
        block_border_width="1px",
        block_shadow="*shadow_drop_lg",
    )
