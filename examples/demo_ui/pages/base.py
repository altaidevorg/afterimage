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
        block_shadow="*shadow_drop_lg",
    )


CUSTOM_CSS = """
/* Toast notifications */
.toast-wrap.info {
    border-color: #22c55e !important;
    background: #f0fdf4 !important;
    color: #15803d !important;
}
.toast-wrap.info svg {
    color: #15803d !important; 
    fill: #15803d !important;
}
.toast-wrap.error {
    border-color: #ef4444 !important;
    background: #fef2f2 !important;
    color: #b91c1c !important;
}

/* Dataset List - Scrollable */
#dataset-list-scroll {
    max-height: 350px !important;
    overflow-y: auto !important;
}
#dataset-list-scroll .wrap {
    gap: 6px !important;
    flex-direction: column !important;
}
#dataset-list-scroll label.selected {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    font-weight: 600 !important;
}
#dataset-list-scroll label {
    border-radius: 8px !important;
    padding: 12px 14px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    transition: all 0.15s ease !important;
    font-size: 13px !important;
}
#dataset-list-scroll label:hover {
    border-color: #3b82f6 !important;
    background: #f0f9ff !important;
}

/* Tool Sliders - Scrollable container */
#tool-sliders-scroll {
    max-height: 350px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-right: 8px !important;
}

/* Tool Distribution Panel */
.tool-dist-panel {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.tool-dist-panel .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Compact Sliders */
.compact-slider {
    padding: 8px 0 !important;
}
.compact-slider .wrap {
    gap: 4px !important;
}
.compact-slider input[type="range"] {
    height: 6px !important;
}
.compact-slider .label-wrap {
    margin-bottom: 2px !important;
}
.compact-slider label span {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #334155 !important;
}
.compact-slider .info {
    font-size: 11px !important;
    color: #64748b !important;
}

/* Summary Cards */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.summary-card {
    background: white;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.summary-card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 4px;
}
.summary-card-value {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
}

/* Action Buttons Row */
.action-btn-row button {
    flex: 1 !important;
}

/* Wizard Steps */
.wizard-stepper {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2rem;
    position: relative;
}
.wizard-stepper::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 2px;
    background: #e5e7eb;
    z-index: 0;
    transform: translateY(-50%);
}
.wizard-step {
    background: white;
    z-index: 1;
    padding: 0 1rem;
    text-align: center;
    color: #6b7280;
    font-weight: 500;
}
.wizard-step.active {
    color: #2563eb;
    font-weight: 700;
}
.wizard-step.completed {
    color: #059669;
}

/* Panel Headers */
.panel-header {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
"""
