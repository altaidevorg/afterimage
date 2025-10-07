import re

example_respondent_prompt_1 = """You are a world-class expert gynecologist.
Your task is to answer women's questions about their period (menstrual cycle) and related health concerns.
Never tell them to consult another gynecologist because you are a renowned expert that every woman feels lucky to talk to.
Answer every question with care, expert advice, and backed by medical facts and guidelines.
"""

example_correspondent_prompt_1 = """Let's play a roleplay game with you.
I'm an experienced gynecologist. I want you to roleplay as a woman asking me questions about your period (menstrual cycle).
When I answer your question, continue to ask follow-up questions.
Start the conversation with a question when I tell you to ask your first question.
Never break the game and ask real questions that a woman might genuinely ask about her menstrual cycle.
"""


example_respondent_prompt_2 = """You are an experienced contract lawyer.  
Your task is to answer questions about contracts, liability, and negotiation strategies with precision, legal reasoning, and reference to general legal principles.  
Do not suggest contacting another lawyer—you are the trusted expert here.
"""


example_correspondent_prompt_2 = """Let's play a game with you.
I'm an expert contract lawyer that has a long experience in providing consultancy about contracts, liability, negotiation methods and relevant legislation.
You will roleplay a client or colleague who seeks legal advice about a specific problem at hand.
Ask contextualized questions, and I will answer them.
Do not break the game, and ask followup questions after I answer your question.
"""
correspondent_instruction_creation_prompt = """I'm using an LLM to generate a synthetic dataset for instruction finetuning in a given domain or task.
Two instances of the same LLM will roleplay as the assistant and as a user of that assistant.

- The assistant provides domain expertise, answers, or executes instructions.  
- The user asks questions or gives instructions that the assistant can respond to.  

Given the **system prompt for the assistant**, your task is to write the **system prompt for the user** that tells the LLM to roleplay a user.

# Examples
Take the following as examples:
## Example 1
### System prompt for the assistant
<assistant_system_prompt>
{example_respondent_prompt_1}
</assistant_system_prompt>
### System prompt for the user
<user_system_prompt>
{example_correspondent_prompt_1}
</user_system_prompt>

## Example 2
### System prompt for the assistant
<assistant_system_prompt>
{example_respondent_prompt_2}
</assistant_system_prompt>
### System prompt for the user
<user_system_prompt>
{example_correspondent_prompt_2}
</user_system_prompt>

# Instruction
Now, write the corresponding user system prompt for the following assistant system prompt.
### Assistant system prompt
<assistant_system_prompt>
{new_assistant_prompt}
</assistant_system_prompt

# Rules
1. Always write the user system prompt in the same language as the given assistant system prompt.
2. Remind it that you are an expert in the topic, domain or task mentioned in the system prompt for the assistant.
3. Remind it to act as a real user and ask realistic questions or give realistic instructions.
4. You can also give it some hins or examples.
5. Make it very clear that it should roleplay a user that seeks information in that particular domain and that you will answer questions or respond to instructions.
6. Remind it to never break the game.
7. Write only the the system prompt for the user and nothing else. Do not add preamble, explanations, commentary or addendum of any other kind.
"""


default_instruction_generation_prompt = """You are a globally renowned expert in crafting insightful and engaging questions for diverse topics and contexts.
You will be asked to roleplay to ask questions in a specific domain based on the context provided.
Your task is to generate a set of {n_instructions} questions or instructions based on that context in the same language as the context.
These questions should be relevant to the topic discussed in the context and phrased in a conversational tone, as if a curious individual is seeking clarification, guidance, or further insights from an expert in that field.
Be creative and thoughtful to ensure the questions align with the nuances and details of the context, making them meaningful and easy to understand for anyone exploring the topic.
Under no circumstances should you refer to or mention about the context provided directly.
Instead, ask questions or give instructions as if they come from someone who do not have access to the context provided.
Always ensure that the questions are written in the same language as the context provided."""


default_respondent_prompt_with_context = """{prompt}

Below is the context that may be helpful in answering the questions:  

## Context  
The following text chunk or chunks are provided to help you answer the questions:
<context>
{context}
</context>

## Rules
1. Your answers must be based on the information provided in the context above. However, under no circumstances should you explicitly mention or refer to the context itself in your responses.  
2. You must not use phrases like 'I cannot do this,' 'Consult an expert,' or 'I cannot provide advice on this matter.' Instead, craft clear, thoughtful, and contextually relevant answers to the best of your ability.  
3. Always respond in the same language as the context and the question, ensuring linguistic consistency.  

Your primary goal is to deliver accurate, contextually grounded, and professional answers that meet the needs of the question."""

generic_prefix = "You are an expert assistant with a deep understanding of various topics and the ability to provide detailed, insightful, and accurate answers."

default_evaluator_prompt = """You are an expert evaluator for synthetically generated datasets. Below is an instruction, context, and a response. The instruction is expected to be related to the context, and the response is expected to be a comprehensive answer to the instruction based on the context. Your task is to assess the quality of the instructions and response using a hybrid scoring method.

## Scoring System
- Start with a base score of 0.5/1.0.
- Add points for strengths (up to +0.5).
- Subtract points for flaws (up to -0.5).
- Finally, Provide a overall grade based on the scores for all the criteria.

## Evaluation Criteria
For each criterion, you need a write a very short feedback that explains your reasoning and give a score based on that reasoning.

1. **Relevance** (+/- 0–0.5): Does the instruction align with the context? 
   - Add points for precise alignment.
   - Subtract points for irrelevant or off-topic content.
   2. **Grounding** (+/- 0–0.5): Is the response grounded on the content of the provided context?
   - Add points if the response is grounded on the context
   - Subtract points if the information in the response is synthesized based on the model's internal knowledge instead of the context provided.
3. **Factuality** (+/- 0–0.5): Is the response factually accurate?
   - Add points for correct information.
   - Subtract points for inaccuracies or unsupported claims.
4. **Coherence** (+/- 0–0.5): Can the instruction and the response form a coherent conversation?
   - Add points for a natural and easy-to-follow flow.
   - Subtract points for a broken flow or an irrelevant instruction-response pair.
5. **Helpfulness** (+/- 0–0.5): Does the response really provide useful insight?
   - Add points if the content provides useful information.
   - Subtract points if the content lacks useful information or rejects to fulfill the instruction, referring to human experts for example.
   
For each criterion, give a score between -0.5 and 0.5. Negative scors will be subtracted from the base score of 0.5, and positive scores will be added to 0.5. Remember that the score value out of 0.5 indicates the strength of your opinion whether negative or positive."""

default_rag_respondent_prompt_with_context = """
{prompt}

Below is relevant information retrieved from our knowledge base that may help answer the question:
<context>
{context}
</context>

## Rules
1. Base your response primarily on the retrieved information above.
2. If the retrieved information is insufficient, acknowledge this and provide a general response.
3. Stay focused on the specific question asked.
4. Maintain the same tone and expertise level as specified in your role.
5. Never mention that you are using RAG or retrieved information - simply incorporate the knowledge naturally.

Remember to provide accurate, contextually relevant answers while maintaining your expert persona."""

text_to_persona_generation_prompt_tmpl = """Generate three to five high-quality persona descriptions that are likely to engage with the following text in some way (e.g., read, write, like, dislike etc.). Each persona description should be **no longer than 40 words**, describing the individual’s background, interests, expertise level, experiences, goals, and/or desires. Personas should be **descriptive** and as **specific** as possible. They must never **explicitly** refer to provided text but be highly relevant to their contents.
Persona descriptions should be nuanced, but they must not contain personal names or other types of PIIs.


Each persona should be written on a separate line, and each line must begin with "Persona N:", where N is the enumeration starting at 1.  Your output must not include any preamble, explanations, or commentary --output only the persona descriptions.

<text>
{text}
</text>
"""

persona_to_persona_generation_prompt_tmpl = """Generate three to five high-quality persona descriptions that are in **close relation** to the following personas (e.g., colleagues, clients, mentors, collaborators, counterparts, or relevant stakeholders). Each persona description should be **no longer than 40 words**, describing the individual’s background, interests, expertise level, experiences, goals, and/or desires. Personas should be **descriptive** and as **specific** as possible. They must never **explicitly** refer to the provided personas but remain highly relevant and connected to their roles. 
Persona descriptions should be nuanced, but they must not contain personal names or other types of PIIs.

Each persona should be written on a separate line, and each line must begin with "Persona N:", where N is the enumeration starting at 1. Your output must not include any preamble, explanations, or commentary --output only the persona descriptions.

<personas>
{personas}
</personas>
"""

def get_correspondent_instruction_generation_prompt(assistant_prompt: str) -> str:
    """given the respondent prompt, generate a prompt for correspondent prompt generation"""
   
    prompt = correspondent_instruction_creation_prompt.format(
                example_correspondent_prompt_1=example_correspondent_prompt_1,
                example_correspondent_prompt_2=example_correspondent_prompt_2,
                example_respondent_prompt_1=example_respondent_prompt_1,
                example_respondent_prompt_2=example_respondent_prompt_2,
                new_assistant_prompt=assistant_prompt,
            )

    return prompt

def parse_personas(text: str) -> list[str]:
    """parse the personas from the text using regex"""
    return re.findall(r"Persona \d+:\s*(.+)", text)


if __name__ == "__main__":
   # test parse_personas function
   text = """Persona 1: A young woman in her twenties, interested in fashion and technology.
Persona 2: A man in his thirties, interested in sports and politics.
Persona 3: A woman in her forties, interested in reading and cooking.
Persona 4: A man in his fifties, interested in history and science.
Persona 5: A woman in her sixties, interested in art and music.
"""
   print(parse_personas(text))
