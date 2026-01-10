import gradio as gr
from pages.base import (
    DEFAULT_RESPONDENT_PROMPT,
    DEFAULT_CONTEXTS,
    create_context_section,
    create_output_section,
)


def create_structured_gen_page(start_gen_fn):
    with gr.Blocks() as page:
        gr.Markdown("## Structured Generation")
        gr.Markdown(
            "Generate consistent, schema-bound synthetic data for training or evaluation."
        )

        with gr.Row():
            context_ui = create_context_section(DEFAULT_CONTEXTS)

            with gr.Column(scale=1):
                gr.Markdown("### 2. Configuration")
                prompt_input = gr.TextArea(
                    label="Respondent System Prompt",
                    value=DEFAULT_RESPONDENT_PROMPT,
                    lines=8,
                )
                num_samples = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1,
                    label="Number of Samples",
                )
                generate_btn = gr.Button(
                    "Generate Structured Data", variant="primary", size="lg"
                )

        status_output, results_output, download_output = create_output_section(
            headers=[
                "Persona",
                "Instruction",
                "Intent",
                "Urgency",
                "Reasoning",
                "Response",
            ],
            label="Generated Interactions",
        )

        generate_btn.click(
            fn=start_gen_fn,
            inputs=[
                context_ui["input"],
                prompt_input,
                num_samples,
                context_ui["source"],
                context_ui["file"],
                context_ui["key"],
            ],
            outputs=[
                results_output,
                status_output,
                download_output,
            ],
        )
    return page
