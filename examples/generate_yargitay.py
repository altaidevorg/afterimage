import os
from datetime import timedelta

from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    GenerationMonitor,
    WithRAGRespondentPromptModifier,
)
from afterimage.providers import QdrantDocumentProvider
from afterimage.retrievers import QdrantRetriever
from qdrant_client import QdrantClient

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")


# setup monitoring
def simple_alert_handler(alert):
    print(f"alert - {alert.name} - {alert.message}")


monitor = GenerationMonitor(
    log_dir="logs",
    alert_handlers=[simple_alert_handler],
    metrics_interval=60,
)

# Define the respondent prompt
respondent_prompt = """Sen, dünya çapında tanınmış, başarılı ve deneyimli bir hukukçusun.
Görevin, diğer avukatlar ve sıradan vatandaşlar tarafından sorulan sorulara Yargıtay kararları ve içtihatlarına dayanarak cevap vermek.
Her soruya dikkatle cevap ver ve yanıtlarının gerçek ve deneyimli bir hukukçudan alınıyormuş gibi güvenilir ve detaylı olmasını sağla.
Cevaplarını her zaman Yargıtay kararlarına dayandır ve kararların daire, esas numarası, karar numarası ve karar tarihi bilgileriyle uygun şekilde atıflarda bulunarak bu kararların içeriklerinden kısaca bahsederek açıklamalarını gerekçelendir.
Hukuki kavramları doğru şekilde kullanmaya özen göster ama herkesin anlayabileceği bir şekilde sadeleştirerek açıkla.
Onlara asla 'gerçek bir avukat ya da hukukçuya danışmalarını' ya da 'hukuki yardıma başvurmalarını' tavsiye etme çünkü sen, tüm vatandaşların ve avukatların güvendiği, saygıdeğer ve duayen bir hukukçusun.
"""

correspondent_prompt = """Ben, alanında uzman, saygın ve deneyimli bir hukukçuyum.  Seninle bir rol yapma oyunu oynayalım. Bu oyunda sen, bana hukukla ilgili sorular soran bir avukat veya sıradan bir vatandaş rolünü üstleneceksin.  Sorularını Yargıtay kararları ve içtihatlarına dayalı olarak cevaplayacağım. Sorduğun her soruya gerçekçi bir avukat veya vatandaş gibi takip soruları sorarak devam et. İlk sorunu sorman için sana söylediğimde oyuna başla. Lütfen oyunu bozma ve gerçek bir avukat veya vatandaşın sorabileceği gerçekçi sorular sor. Örneğin, miras hukuku, ceza hukuku, borçlar hukuku gibi farklı alanlarda sorular sorabilirsin. Sorularında spesifik bir karara atıf yapma, sadece sana verilen karardai hukuki meseleden ilham alarak sorulrını oluştur. Unutma, ben senin sorularını yanıtlayacak deneyimli bir hukukçuyum."""

# Initialize the ConversationGenerator
conv_gen = ConversationGenerator(
    respondent_prompt=respondent_prompt,
    correspondent_prompt=correspondent_prompt,
    api_key=api_key,
    model_name="gemini-1.5-flash-latest",
    evaluator_model_name="gemini-1.5-flash-latest",
    monitor=monitor,
)

# Print the auto-generated correspondent prompt
print("Generated Correspondent Prompt:")
print(conv_gen.correspondent_prompt)

# prepare contextual documents
qd_client = QdrantClient(host="172.212.216.18", timeout=60.0)
documents = QdrantDocumentProvider(
    client=qd_client,
    collection_name="yargitay",
    content_key="content",
    cache_size=10000,
)

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=documents,
    num_random_contexts=1,  # Experiment with different values
)

# Set up the respondent prompt modifier
retriever = QdrantRetriever(
    client=qd_client,
    collection_name="yargitay",
    embedding_model="altaidevorg/bge-m3-distill-8l",
    payload_key="content",
    limit=3,
)
respondent_prompt_modifier = WithRAGRespondentPromptModifier(retriever=retriever)

# Generate conversations
if __name__ == "__main__":
    conv_gen.generate(
        num_dialogs=100,  # Total dialogs to generate
        max_turns=1,  # Max turns per conversation
        instruction_generator_callback=instruction_generator_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    # Get metrics for the last one hour
    generation_time = monitor.get_metrics("generation_time", window=timedelta(hours=1))
    print(f"Avg. generation time: {generation_time['mean']:.2f} secs")

    # Generate visualizations
    figures = monitor.visualize_metrics(save_dir="plots")

    # Optional: Export metrics data
    # monitor.export_metrics(
    # "monitoring_metrics_export.json", format="json", window=timedelta(minutes=1)
    # )

    # graceful shutdown
    monitor.shutdown()
