import gradio as gr
from pages.base import (
    TOOL_CALLING_RESPONDENT_PROMPT,
    SMART_HOME_CONTEXT_STR,
    create_context_section,
    create_output_section,
)


def create_tool_calling_page(start_gen_fn):
    with gr.Blocks() as page:
        gr.Markdown("## Tool Calling Generation")
        gr.Markdown(
            "Generate data for training models to use tools precisely based on instructions."
        )

        with gr.Row():
            context_ui = create_context_section(SMART_HOME_CONTEXT_STR)

            with gr.Column(scale=1):
                gr.Markdown("### 2. Configuration")
                prompt_input = gr.TextArea(
                    label="Respondent System Prompt",
                    value=TOOL_CALLING_RESPONDENT_PROMPT,
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
                    "Generate Tool Calls", variant="primary", size="lg"
                )

        status_output, results_output, download_output = create_output_section(
            headers=["Persona", "Instruction", "Response", "Reasoning", "Tool Calls"],
            label="Generated Tool Calls",
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
