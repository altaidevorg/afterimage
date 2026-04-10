"""Analytics computation for preference generation runs."""

from __future__ import annotations

from typing import List

from .types import PreferenceAnalytics, PreferencePair

# Threshold: if one variation label wins more than this fraction of pairs, warn.
_DOMINANCE_THRESHOLD = 0.75


def compute_analytics(
    pairs: List[PreferencePair],
    total_attempted: int,
) -> PreferenceAnalytics:
    """Compute analytics over a completed preference generation run.

    Args:
        pairs: The valid (non-discarded) PreferencePairs.
        total_attempted: Total prompts attempted, including discarded ones.

    Returns:
        Populated :class:`PreferenceAnalytics` instance.
    """
    analytics = PreferenceAnalytics()
    analytics.total_attempted = total_attempted
    analytics.total_valid = len(pairs)
    analytics.total_discarded = total_attempted - len(pairs)

    if total_attempted > 0:
        analytics.discard_rate = analytics.total_discarded / total_attempted

    if not pairs:
        analytics.warnings.append(
            "No valid pairs generated. Consider lowering min_score_gap."
        )
        return analytics

    # Score statistics
    chosen_scores = [p.chosen.score for p in pairs]
    rejected_scores = [p.rejected.score for p in pairs]
    gaps = [p.chosen.score - p.rejected.score for p in pairs]

    analytics.avg_chosen_score = sum(chosen_scores) / len(chosen_scores)
    analytics.avg_rejected_score = sum(rejected_scores) / len(rejected_scores)
    analytics.avg_score_gap = sum(gaps) / len(gaps)

    # Strategy distribution (based on chosen label)
    for pair in pairs:
        label = pair.chosen.variation_label
        analytics.strategy_distribution[label] = (
            analytics.strategy_distribution.get(label, 0) + 1
        )

    # Dominance warning
    total_pairs = len(pairs)
    for label, count in analytics.strategy_distribution.items():
        fraction = count / total_pairs
        if fraction >= _DOMINANCE_THRESHOLD:
            analytics.warnings.append(
                f"Variation '{label}' dominates chosen responses "
                f"({count}/{total_pairs} = {fraction:.0%}). "
                "Consider switching or adding more variation strategies."
            )

    # High discard rate warning
    if analytics.discard_rate > 0.4:
        analytics.warnings.append(
            f"High discard rate ({analytics.discard_rate:.0%}). "
            "Consider lowering min_score_gap or using a stronger variation strategy."
        )

    return analytics
