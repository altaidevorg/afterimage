"""OpenSimula: open implementation of Simula-style synthetic data mechanism design.

Experimental API; subject to change. See :class:`~afterimage.simula.pipeline.OpenSimula`.
"""

from .cli_logging import configure_example_console, silence_noisy_third_party_loggers
from .pipeline import OpenSimula
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
