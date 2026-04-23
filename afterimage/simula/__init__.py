"""OpenSimula: open implementation of Simula-style synthetic data mechanism design.

Experimental API; subject to change. See :class:`~afterimage.simula.pipeline.OpenSimula`.
"""

from .checkpoint import (
    Checkpointer,
    OpenSimulaManifest,
    OpenSimulaRunConfig,
    SimulaCheckpoint,
    load_checkpoint,
    opensimula_dir,
    pull_checkpoint_from_hub,
    push_checkpoint_to_hub,
    save_checkpoint,
)
from .cli_logging import configure_example_console, silence_noisy_third_party_loggers
from .pipeline import OpenSimula
from .sample_export import append_datapoints_jsonl
from .tasks import SimulaInstructionGeneratorCallback
from .types import (
    DatasetBatch,
    DataPointLineage,
    DataPointRecord,
    DocumentProvenance,
    DoubleCritiqueVerdict,
    ExpansionStepTrace,
    FactorTaxonomy,
    MCQRow,
    MetaPrompt,
    Mix,
    MixEntry,
    RequirementCritiqueVerdict,
    SamplingStrategySpec,
    SimulaFactor,
    SingleQARow,
    StrategyMixRule,
    TaxonomyBundle,
    TaxonomyNode,
    digest_documents_for_bundle,
    sha256_text,
    validate_factor_taxonomy,
)

__all__ = [
    "Checkpointer",
    "OpenSimulaManifest",
    "OpenSimulaRunConfig",
    "SimulaCheckpoint",
    "load_checkpoint",
    "opensimula_dir",
    "pull_checkpoint_from_hub",
    "push_checkpoint_to_hub",
    "save_checkpoint",
    "append_datapoints_jsonl",
    "OpenSimula",
    "SimulaInstructionGeneratorCallback",
    "configure_example_console",
    "silence_noisy_third_party_loggers",
    "DatasetBatch",
    "DataPointLineage",
    "DataPointRecord",
    "DocumentProvenance",
    "DoubleCritiqueVerdict",
    "ExpansionStepTrace",
    "FactorTaxonomy",
    "MCQRow",
    "MetaPrompt",
    "Mix",
    "MixEntry",
    "RequirementCritiqueVerdict",
    "SamplingStrategySpec",
    "SimulaFactor",
    "SingleQARow",
    "StrategyMixRule",
    "TaxonomyBundle",
    "TaxonomyNode",
    "digest_documents_for_bundle",
    "sha256_text",
    "validate_factor_taxonomy",
]
