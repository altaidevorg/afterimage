"""Compute analytics over an AfterImage JSONL dataset."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    CoverageStats,
    DatasetReport,
    DiversityStats,
    LengthStats,
    PersonaStats,
    QualityStats,
    SummaryStats,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HISTOGRAM_BINS = 10


def _word_tokenize(text: str) -> List[str]:
    """Cheap whitespace+regex tokenizer (no NLTK dependency)."""
    return _WORD_RE.findall(text.lower())


def _make_histogram(
    values: List[int | float], n_bins: int = _HISTOGRAM_BINS
) -> tuple[List[int], List[str]]:
    """Bucket *values* into *n_bins* equal-width bins. Returns (counts, labels)."""
    if not values:
        return [], []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [len(values)], [str(lo)]
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1)
        counts[idx] += 1
    labels = [f"{lo + i * width:.0f}" for i in range(n_bins)]
    return counts, labels


class DatasetAnalyzer:
    """Analyse an AfterImage JSONL dataset and produce a :class:`DatasetReport`.

    Usage::

        report = DatasetAnalyzer.from_jsonl("output.jsonl")
    """

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    @classmethod
    def from_jsonl(cls, path: str | Path) -> DatasetReport:
        """Load a JSONL file and return a fully computed :class:`DatasetReport`."""
        path = Path(path)
        rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        analyzer = cls(rows)
        report = analyzer.analyze()
        report.dataset_path = str(path)
        return report

    def analyze(self) -> DatasetReport:
        """Run all analyses and return the report."""
        return DatasetReport(
            summary=self._summary(),
            personas=self._personas(),
            coverage=self._coverage(),
            quality=self._quality(),
            diversity=self._diversity(),
            lengths=self._lengths(),
        )

    # ------------------------------------------------------------------
    # Individual analyses
    # ------------------------------------------------------------------

    def _summary(self) -> SummaryStats:
        total = len(self._rows)
        total_turns = 0
        total_words = 0
        personas: set[str] = set()
        contexts: set[str] = set()

        for row in self._rows:
            convs = row.get("conversations", [])
            total_turns += len(convs)
            for msg in convs:
                total_words += len(_word_tokenize(msg.get("content", "")))

            meta = row.get("metadata") or {}
            persona = row.get("persona") or meta.get("persona_name")
            if persona:
                personas.add(persona)

            ctx_id = meta.get("context_id")
            if ctx_id:
                contexts.add(ctx_id)
            for cid in meta.get("context_ids", []):
                if cid:
                    contexts.add(cid)

        return SummaryStats(
            total_conversations=total,
            total_turns=total_turns,
            avg_turns_per_conversation=total_turns / total if total else 0,
            total_words=total_words,
            avg_words_per_turn=total_words / total_turns if total_turns else 0,
            unique_personas=len(personas),
            unique_contexts=len(contexts),
        )

    def _personas(self) -> PersonaStats:
        persona_counter: Counter[str] = Counter()
        depth_counter: Counter[int] = Counter()

        for row in self._rows:
            meta = row.get("metadata") or {}
            persona = row.get("persona") or meta.get("persona_name")
            if persona:
                persona_counter[persona] += 1
            depth = meta.get("persona_generation_depth")
            if depth is not None:
                depth_counter[int(depth)] += 1

        return PersonaStats(
            persona_counts=dict(persona_counter.most_common()),
            depth_distribution=dict(sorted(depth_counter.items())),
        )

    def _coverage(self) -> CoverageStats:
        ctx_counter: Counter[str] = Counter()

        for row in self._rows:
            meta = row.get("metadata") or {}
            ctx_id = meta.get("context_id")
            if ctx_id:
                ctx_counter[ctx_id] += 1
            for cid in meta.get("context_ids", []):
                if cid:
                    ctx_counter[cid] += 1

        once = sum(1 for c in ctx_counter.values() if c == 1)
        multi = sum(1 for c in ctx_counter.values() if c > 1)
        return CoverageStats(
            context_counts=dict(ctx_counter.most_common()),
            contexts_used_once=once,
            contexts_used_multiple=multi,
        )

    def _quality(self) -> QualityStats:
        grade_counter: Counter[str] = Counter()
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        scores: list[float] = []

        for row in self._rows:
            ev = row.get("evaluation")
            if ev is None:
                continue

            grade = ev.get("overall_grade")
            if grade:
                grade_counter[grade] += 1

            for metric in (
                "coherence",
                "factuality",
                "grounding",
                "helpfulness",
                "relevance",
            ):
                entry = ev.get(metric)
                if entry and "score" in entry:
                    s = float(entry["score"])
                    metric_sums[metric] = metric_sums.get(metric, 0.0) + s
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1

            fs = row.get("final_score")
            if fs is not None:
                scores.append(float(fs))

        if not grade_counter:
            return QualityStats()

        avg_scores = {
            m: metric_sums[m] / metric_counts[m]
            for m in metric_sums
            if metric_counts.get(m, 0) > 0
        }
        hist, bins = _make_histogram(scores)

        return QualityStats(
            has_evaluations=True,
            grade_counts=dict(grade_counter.most_common()),
            avg_scores=avg_scores,
            score_histogram=hist,
            score_bins=bins,
        )

    def _diversity(self) -> DiversityStats:
        all_words: list[str] = []
        for row in self._rows:
            for msg in row.get("conversations", []):
                all_words.extend(_word_tokenize(msg.get("content", "")))

        if not all_words:
            return DiversityStats()

        vocab = set(all_words)
        ttr = len(vocab) / len(all_words)

        # Shannon entropy
        freq = Counter(all_words)
        total = len(all_words)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in freq.values() if c > 0
        )

        # Bigram repetition rate
        bigrams = [(all_words[i], all_words[i + 1]) for i in range(len(all_words) - 1)]
        if bigrams:
            bigram_freq = Counter(bigrams)
            repeated = sum(c for c in bigram_freq.values() if c > 1)
            bigram_rep = repeated / len(bigrams)
        else:
            bigram_rep = 0.0

        return DiversityStats(
            vocabulary_size=len(vocab),
            type_token_ratio=ttr,
            bigram_repetition_rate=bigram_rep,
            shannon_entropy=entropy,
        )

    def _lengths(self) -> LengthStats:
        user_lens: list[int] = []
        asst_lens: list[int] = []

        for row in self._rows:
            for msg in row.get("conversations", []):
                wc = len(_word_tokenize(msg.get("content", "")))
                role = msg.get("role", "")
                if role == "user":
                    user_lens.append(wc)
                elif role == "assistant":
                    asst_lens.append(wc)

        avg_u = sum(user_lens) / len(user_lens) if user_lens else 0
        avg_a = sum(asst_lens) / len(asst_lens) if asst_lens else 0

        u_hist, u_bins = _make_histogram(user_lens)
        a_hist, _ = _make_histogram(asst_lens)

        # Use combined bins for fair comparison
        all_lens = user_lens + asst_lens
        combined_hist_u, combined_bins = _make_histogram(user_lens)
        combined_hist_a, _ = _make_histogram(asst_lens)

        return LengthStats(
            user_lengths=user_lens,
            assistant_lengths=asst_lens,
            avg_user_length=avg_u,
            avg_assistant_length=avg_a,
            user_length_histogram=u_hist,
            assistant_length_histogram=a_hist,
            length_bins=u_bins or combined_bins,
        )
