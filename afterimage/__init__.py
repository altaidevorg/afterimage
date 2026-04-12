import importlib.metadata

from afterimage.async_conversation_generator import AsyncConversationGenerator  # noqa
from afterimage.callbacks import (
    AndStoppingCallback,  # noqa
    BudgetStoppingCallback,  # noqa
    ContextCoverageStoppingCallback,  # noqa
    ContextualInstructionGeneratorCallback,  # noqa
    FixedNumberStoppingCallback,  # noqa
    PersonaInstructionGeneratorCallback,  # noqa
    PersonaUsageStoppingCallback,  # noqa
    RateLimitStoppingCallback,  # noqa
    SimpleInstructionGeneratorCallback,  # noqa
    ToolCallingInstructionGeneratorCallback,  # noqa
    WithContextRespondentPromptModifier,  # noqa
    WithRAGRespondentPromptModifier,  # noqa
)
from afterimage.conversation_generator import ConversationGenerator  # noqa
from afterimage.conversation_turn_hooks import (  # noqa
    ConversationTurnContext,
    ConversationTurnHooks,
)
from afterimage.evaluator import (
    ConversationJudge,  # noqa
    ConversationJudgeConfig,  # noqa
    default_embedding_provider_config,  # noqa
)
from afterimage.key_management import SmartKeyPool  # noqa
from afterimage.orchestrator import Orchestrator  # noqa
from afterimage.quality_gate import QualityGate, QualityResult  # noqa
from afterimage.sampling import SamplingStrategy  # noqa
from afterimage.monitoring import (
    GenerationMonitor,  # noqa
    ModelTokenUsage,  # noqa
    TokenUsageReport,  # noqa
)
from afterimage.persona_generator import PersonaGenerator  # noqa
from afterimage.providers import (
    ChatSession,  # noqa
    DirectoryDocumentProvider,  # noqa
    DocumentProvider,  # noqa
    EmbeddingProvider,  # noqa
    EmbeddingProviderFactory,  # noqa
    FileSystemDocumentProvider,  # noqa
    GeminiChatSession,  # noqa
    GeminiEmbeddingProvider,  # noqa
    InMemoryDocumentProvider,  # noqa
    JSONLDocumentProvider,  # noqa
    LLMFactory,  # noqa
    LLMProvider,  # noqa
    OpenAIEmbeddingProvider,  # noqa
    ProcessEmbeddingProvider,  # noqa
    QdrantDocumentProvider,  # noqa
)
from afterimage.storage import BaseStorage, JSONLStorage, SQLStorage  # noqa
from afterimage.structured_generator import (
    AsyncStructuredGenerator as AsyncStructuredGenerator,
    StructuredGenerator as StructuredGenerator,
)
from afterimage.types import (  # noqa
    MODEL_PROVIDER_NAMES,
    Document,
    ModelProviderName,
    PersonaEntry,
)

try:
    __version__ = importlib.metadata.version("afterimage")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # fallback
