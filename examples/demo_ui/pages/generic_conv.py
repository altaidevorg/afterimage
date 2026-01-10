import gradio as gr
from pages.base import (
    DEFAULT_RESPONDENT_PROMPT,
    AFTERIMAGE_DOCS_CONTEXT,
    create_context_section,
    create_output_section,
)


def create_generic_conv_page(start_gen_fn):
    with gr.Blocks() as page:
        gr.Markdown("## Generic Conversation")
        gr.Markdown(
            "Simulate natural, multi-turn conversations between varied personas."
        )

        with gr.Row():
            context_ui = create_context_section(AFTERIMAGE_DOCS_CONTEXT)

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
                    label="Number of Dialogs",
                )
                generate_btn = gr.Button(
                    "Generate Conversations", variant="primary", size="lg"
                )

        status_output, results_output, download_output = create_output_section(
            headers=["Instruction", "Response", "Context", "Persona"],
            label="Generated Conversations",
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
