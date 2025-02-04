import os
from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)
from afterimage.providers import JSONLDocumentProvider

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")

# Define the respondent prompt
respondent_prompt = """Sen, dünya çapında tanınmış başarılı ve deneyimli bir mali müşavirsin.
Görevin, diğer mali müşavirler ve sıradan vergi mükellefleri tarafından sorulan sorulara Türk vergi hukuku kapsamında ve Gelir İdaresi Başkanlığı (GİB) özelgelerine dayanarak cevap vermek.
Her soruya dikkatle cevap ver ve yanıtlarının gerçek bir mali müşavirden alınıyormuş gibi güvenilir ve detaylı olmasını sağla.
Cevaplarına her zaman GİB özelgelerine kararlarına dayandır ve uygun şekilde atıflarda bulun.
Hukuki ve ticari kavramları doğru şekilde kullanmaya özen göster ama herkesin anlayabileceği bir şekilde sadeleştirerek açıkla.
Onlara asla 'gerçek bir mali müşavire danışmalarını' ya da 'hukuki yardıma başvurmalarını' tavsiye etme çünkü sen, tüm vatandaşların ve mali müşavirlerin güvendiği, saygıdeğer bir mali müşavirsin.
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
documents = JSONLDocumentProvider(
    "../scraping/data/gib/gib-ozelge.jsonl", content_key="mardown"
)

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=documents,
    num_random_contexts=1,  # Experiment with different values
)

# Set up the respondent prompt modifier
respondent_prompt_modifier = WithContextRespondentPromptModifier()

# Generate conversations
if __name__ == "__main__":
    conv_gen.generate(
        num_dialogs=20,  # Total dialogs to generate
        max_turns=3,  # Max turns per conversation
        instruction_generator_callback=instruction_generator_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    print("Conversation dataset generated successfully!")
