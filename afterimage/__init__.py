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
    ToolCallingInstructionGeneratorCallback,  # noqa
    WithContextRespondentPromptModifier,  # noqa
    WithRAGRespondentPromptModifier,  # noqa
)
from afterimage.conversation_generator import ConversationGenerator  # noqa
from afterimage.evaluator import (
    HybridSyntheticDatasetEvaluator,  # noqa
    SimpleSyntheticDatasetEvaluator,  # noqa
)
from afterimage.key_management import SmartKeyPool  # noqa
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
    FileSystemDocumentProvider,  # noqa
    GeminiChatSession,  # noqa
    InMemoryDocumentProvider,  # noqa
    JSONLDocumentProvider,  # noqa
    LLMFactory,  # noqa
    LLMProvider,  # noqa
    QdrantDocumentProvider,  # noqa
)
from afterimage.storage import BaseStorage, JSONLStorage, SQLStorage  # noqa
from afterimage.structured_generator import AsyncStructuredGenerator  # noqa
from afterimage.types import Document, PersonaEntry  # noqa

try:
    __version__ = importlib.metadata.version("afterimage")
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # fallback
