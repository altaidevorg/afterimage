import json
import os
from datetime import timedelta

from afterimage import (
    ConversationGenerator,
    ContextualInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
)
from afterimage.monitoring import GenerationMonitor

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set the GEMINI_API_KEY environment variable!")


# setup monitoring
def simple_alert_handler(alert):
    print(f"alert - {alert.name} - {alert.message}")


monitor = GenerationMonitor(
    log_dir="monitoring_logs",
    alert_handlers=[simple_alert_handler],
    metrics_interval=60,
)

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
    model_name="gemini-1.5-pro-latest",
    evaluator_model_name="gemini-1.5-flash-latest",
    monitor=monitor,
)

# Print the auto-generated correspondent prompt
print("Generated Correspondent Prompt:")
print(conv_gen.correspondent_prompt)

# Prepare contextual documents
with open("../scraping/data/gib/gib-ozelge.jsonl", encoding="utf8") as f:
    docs = [json.loads(line)["markdown"] for line in f]

# Set up the instruction generator callback
instruction_generator_callback = ContextualInstructionGeneratorCallback(
    api_key=api_key,
    documents=docs,
    num_random_contexts=1,  # Experiment with different values
)

# Set up the respondent prompt modifier
respondent_prompt_modifier = WithContextRespondentPromptModifier()

# Generate conversations
if __name__ == "__main__":
    conv_gen.generate(
        num_dialogs=60,  # Total dialogs to generate
        max_turns=1,  # Max turns per conversation
        instruction_generator_callback=instruction_generator_callback,
        respondent_prompt_modifier=respondent_prompt_modifier,
    )

    # Get metrics for the last one hour
    success_rate = monitor.get_metrics("success_rate", window=timedelta(hours=1))
    generation_time = monitor.get_metrics("generation_time", window=timedelta(hours=1))
    print(f"Success rate: {success_rate['mean']:.1%}")
    print(f"Avg. generation time: {generation_time['mean']:.2f} secs")

    # Generate visualizations
    figures = monitor.visualize_metrics(save_dir="monitoring_plots")

    # Optional: Export metrics data
    # monitor.export_metrics(
    # "monitoring_metrics_export.json", format="json", window=timedelta(minutes=1)
    # )

    # graceful shutdown
    monitor.shutdown()
