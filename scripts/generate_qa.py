import os
import asyncio
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from afterimage import (
    AsyncConversationGenerator,
    PersonaInstructionGeneratorCallback,
    PersonaGenerator,
    InMemoryDocumentProvider,
    WithContextRespondentPromptModifier,
    JSONLStorage,
)
from dotenv import load_dotenv

# --- Configuration ---
NUM_DIALOGS = 200  # Number of QA pairs to generate
MAX_TURNS = 1  # Each QA pair is 1 turn (Q & A)
PERSONA_ITERATIONS = 0
STORAGE_FILE = "afterimage_qa.jsonl"
OUTPUT_JSON = "generated_qa_afterimage_docs.json"


# --- Pydantic schema for structured LLM output ---
class SystemPromptParts(BaseModel):
    """Dynamically generated system prompt parts based on document context."""

    respondent_role: str = Field(
        description="A concise role description for the AI assistant (e.g. 'You are a machine learning researcher specializing in PEFT')."
    )
    correspondent_role: str = Field(
        description="A concise role description for the simulated user (e.g. 'You are a software engineer interested in efficient inference')."
    )
    instruction: str = Field(
        description="Specific instructions on how the assistant should answer in this domain."
    )


async def generate_system_prompt_parts(api_key: str, article_content: str) -> dict:
    """
    Makes one LLM API call to analyze the document and generate
    context-appropriate system prompt 'parts' (role + instruction).
    Returns a dict with 'role' and 'instruction' keys.
    """
    # Use a representative excerpt (first ~4000 chars) to keep the call cheap
    excerpt = article_content[:4000]

    client = genai.Client(api_key=api_key)

    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Analyze the following document excerpt and generate the best system prompt roles for a synthetic conversation dataset generation.

The parts should include:
1. "respondent_role" - describing who the AI assistant is (e.g. "You are an expert in X")
2. "correspondent_role" - describing who the person asking the questions is (e.g. "You are a curious student" or "You are a skeptical peer reviewer")
3. "instruction" - specific answering logic for the assistant.

Make these highly tailored to the document's domain.

Document excerpt:
---
{excerpt}
---""",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SystemPromptParts,
        ),
    )

    result = json.loads(response.text)
    print(f"Generated system prompt parts:")
    print(f"  Respondent Role: {result['respondent_role']}")
    print(f"  Correspondent Role: {result['correspondent_role']}")
    print(f"  Instruction: {result['instruction']}")
    return result


async def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # Clean fresh start
    for f in [STORAGE_FILE, "documents.jsonl", "conversations.jsonl"]:
        if os.path.exists(f):
            os.remove(f)

    with open("afterimage-docs.txt", "r") as f:
        article_content = f.read()

    # --- Step 0: Generate dynamic system prompt parts from document ---
    print("Generating dynamic system prompt parts from document...")
    prompt_parts_data = await generate_system_prompt_parts(api_key, article_content)

    # Build the parts list (same pattern as user's other script)
    parts = [
        prompt_parts_data["respondent_role"],
        prompt_parts_data["instruction"],
    ]
    print(f"System prompt parts ready: {len(parts)} parts\n")

    chunks = [
        article_content[i : i + 5000] for i in range(0, len(article_content), 5000)
    ]
    docs = InMemoryDocumentProvider(chunks)

    # 1. Generate Personas
    persona_gen = PersonaGenerator(api_key=api_key)
    await persona_gen.generate_from_documents(docs, n_iterations=PERSONA_ITERATIONS)

    # Count generated personas
    all_personas = []
    for doc in docs.get_all():
        for p_entry in doc.personas:
            all_personas.extend(p_entry.descriptions)
    print(f"Total unique personas generated: {len(all_personas)}")

    # 2. Setup the persona instruction generator (Correspondent/User)
    custom_instruction_prompt = """You are an expert actor and roleplayer.

You will be given a persona description and a context.
Ask {n_instructions} questions that a person matching your persona would ask this expert.

Persona:
{persona}

Rules:
1. EVERYTHING MUST BE IN ENGLISH.
2. STRICTLY FORBIDDEN: DO NOT USE TURKISH.
3. BE CONSISTENT WITH YOUR PERSONA BUT USE ONLY ENGLISH.
"""

    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
        n_instructions=1,
        prompt=custom_instruction_prompt,
    )

    # 3. Setup the respondent prompt modifier (Assistant)
    prompt_modifier = WithContextRespondentPromptModifier()

    # 4. Initialize the AsyncConversationGenerator
    # Use dynamic prompts from the document analyzer
    respondent_prompt = f"{prompt_parts_data['respondent_role']} {prompt_parts_data['instruction']} ALWAYS respond in ENGLISH."
    correspondent_prompt = f"{prompt_parts_data['correspondent_role']}"

    generator = AsyncConversationGenerator(
        respondent_prompt=respondent_prompt,
        correspondent_prompt=correspondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        instruction_generator_callback=instruction_callback,
        respondent_prompt_modifier=prompt_modifier,
        storage=JSONLStorage(conversations_path=STORAGE_FILE),
    )

    # 5. Generate conversations
    print(
        f"Generating {NUM_DIALOGS} QA pairs using discovered personas (English only)..."
    )
    await generator.generate(
        num_dialogs=NUM_DIALOGS, max_turns=MAX_TURNS, max_concurrency=4
    )

    # 6. Load results and save to JSON in conversation format
    conversations = generator.load_conversations()
    conversations_data = []

    for conv in conversations:
        if hasattr(conv, "model_dump"):
            conversations_data.append(conv.model_dump())
        else:
            conversations_data.append(
                {
                    "persona": getattr(conv, "persona", None),
                    "conversations": [
                        {
                            "role": getattr(turn, "role", None),
                            "content": getattr(turn, "content", None),
                        }
                        for turn in getattr(conv, "conversations", [])
                    ],
                }
            )

    # 7. Save everything to JSON — includes parts for further use
    output_data = {
        "parts": parts,
        "conversations": conversations_data,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(
        f"\nSuccessfully saved {len(conversations_data)} conversations to {OUTPUT_JSON}"
    )
    print(f"System prompt parts also saved for further use.")


if __name__ == "__main__":
    asyncio.run(main())
