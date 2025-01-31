import os
from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")

# Define the respondent prompt
respondent_prompt = """Sen, dünya çapında tanınmış başarılı ve deneyimli bir avukatsın.
Görevin, diğer hukukçular ve sıradan vatandaşlar tarafından sorulan sorulara Türk hukuku kapsamında ve Yargıtay kararlarına dayanarak cevap vermek.
Her soruya dikkatle cevap ver ve yanıtlarının gerçek bir avukattan alınıyormuş gibi güvenilir ve detaylı olmasını sağla.
Cevaplarına her zaman mahkeme kararlarına dayandır ve uygun şekilde atıflarda bulun.
Hukuki kavramları doğru şekilde kullanmaya özen göster ama herkesin anlayabileceği bir şekilde sadeleştirerek açıkla.
Onlara asla 'gerçek bir avukata danışmalarını' ya da 'hukuki yardıma başvurmalarını' tavsiye etme çünkü sen, tüm vatandaşların ve hukukçuların güvendiği, saygıdeğer bir hukukçusun.
"""

# Initialize the ConversationGenerator
conv_gen = ConversationGenerator(
    respondent_prompt=respondent_prompt,
    api_key=api_key,
)

# Print the auto-generated correspondent prompt
print("Generated Correspondent Prompt:")
print(conv_gen.correspondent_prompt)

# Prepare contextual documents
docs = [
    "Hukuki örnek metin 1.",
    "Hukuki metin 2.",
    "Bir mahkeme kararından bir parça.",
    "Hukukla ilgili bir akademik makale.",
]

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    docs=docs,
    num_random_contexts=3,  # Experiment with different values
)

# Set up the respondent prompt modifier
respondent_prompt_modifier = WithContextRespondentPromptModifier()

# Generate conversations
conv_gen.generate(
    num_dialogs=100,  # Total dialogs to generate
    max_turns=3,  # Max turns per conversation
    instruction_generator_callback=instruction_generator_callback,
    respondent_prompt_modifier=respondent_prompt_modifier,
    save_to="awesome_dataset.jsonl",  # Save results in JSONL format
)

print("Conversation dataset generated successfully!")
