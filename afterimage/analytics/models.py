"""Data models for dataset analytics reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SummaryStats:
    """High-level dataset statistics."""

    total_conversations: int = 0
    total_turns: int = 0
    avg_turns_per_conversation: float = 0.0
    total_words: int = 0
    avg_words_per_turn: float = 0.0
    unique_personas: int = 0
    unique_contexts: int = 0


@dataclass
class PersonaStats:
    """Per-persona conversation counts."""

    persona_counts: Dict[str, int] = field(default_factory=dict)
    depth_distribution: Dict[int, int] = field(default_factory=dict)


@dataclass
class CoverageStats:
    """Context coverage metrics."""

    context_counts: Dict[str, int] = field(default_factory=dict)
    contexts_used_once: int = 0
    contexts_used_multiple: int = 0


@dataclass
class QualityStats:
    """Quality evaluation metrics (empty if no evaluations present)."""

    has_evaluations: bool = False
    grade_counts: Dict[str, int] = field(default_factory=dict)
    avg_scores: Dict[str, float] = field(default_factory=dict)
    score_histogram: List[int] = field(default_factory=list)
    score_bins: List[str] = field(default_factory=list)


@dataclass
class DiversityStats:
    """Text diversity metrics."""

    vocabulary_size: int = 0
    type_token_ratio: float = 0.0
    bigram_repetition_rate: float = 0.0
    shannon_entropy: float = 0.0


@dataclass
class LengthStats:
    """Message length distributions."""

    user_lengths: List[int] = field(default_factory=list)
    assistant_lengths: List[int] = field(default_factory=list)
    avg_user_length: float = 0.0
    avg_assistant_length: float = 0.0
    user_length_histogram: List[int] = field(default_factory=list)
    assistant_length_histogram: List[int] = field(default_factory=list)
    length_bins: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Complete analytics report for a generated dataset."""

    summary: SummaryStats = field(default_factory=SummaryStats)
    personas: PersonaStats = field(default_factory=PersonaStats)
    coverage: CoverageStats = field(default_factory=CoverageStats)
    quality: QualityStats = field(default_factory=QualityStats)
    diversity: DiversityStats = field(default_factory=DiversityStats)
    lengths: LengthStats = field(default_factory=LengthStats)
    dataset_path: str = ""
