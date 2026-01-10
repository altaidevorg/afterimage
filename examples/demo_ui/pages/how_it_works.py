import gradio as gr


def create_how_it_works_page():
    with gr.Blocks() as page:
        gr.Markdown("# How It Works")

        with gr.Row():
            with gr.Column():
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

                gr.Markdown("## Workflow Diagram")
                gr.Markdown(
                    """
                    ```mermaid
                    graph TD
                        D[Documents] --> P[Persona Generator]
                        P --> Pers[Personas]
                        D --> IG[Instruction Generator]
                        Pers --> IG
                        IG --> Inst[Instruction/Query]
                        Inst --> SG[Structured/Conversation Generator]
                        Sys[System Prompt] --> SG
                        SG --> Out[Synthetic Dataset]
                    ```
                    """
                )

            with gr.Column():
                gr.Markdown("## About the Examples in this Demo")
                gr.Markdown(
                    """
                    ### Conversational Generation
                    - Create conversational datasets from your raw documents.
                    - It generates realistic instructions that can be asked about the provided documents and ensure that the responses are grounded on them.
                    - Conversations may span one or more turns.
                    - Used to train a conversational AI that is expert on your data.

                    ### Tool Calling Generation
                    - Create tool calling datasets from your documents.
                    - It allows to  define tools and their arguments through Pydantic models.
                    - You can also provide a context to guide the generation.
                    - In this example, a tool-calling dataset is generated for a smart home assistant with its manual as the context.
                                        
                    ### Structured Generation
                    - Create a structured dataset from a collection of documents and Pydantic-defined output schema.
                    - This example demonstrates how to generate a customer support dataset from a collectionof company policy documents.
                    - The customer support AI fine*tuned on this data does not only answer questions but also populate the fields such as `tags`, `missing_information` and `urgency` etc.
                    """
                )
    return page
