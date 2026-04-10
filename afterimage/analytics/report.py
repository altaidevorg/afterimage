"""Generate a self-contained HTML analytics report from a DatasetReport."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from .charts import bar_chart, donut_chart, histogram, metric_card
from .models import DatasetReport

_CSS = """
:root {
  --bg: #f8fafc;
  --card-bg: #ffffff;
  --text: #1e293b;
  --text-muted: #64748b;
  --accent: #4f8cff;
  --accent2: #34d399;
  --border: #e2e8f0;
  --section-bg: #ffffff;
}
[data-theme="dark"] {
  --bg: #0f172a;
  --card-bg: #1e293b;
  --text: #e2e8f0;
  --text-muted: #94a3b8;
  --accent: #60a5fa;
  --accent2: #34d399;
  --border: #334155;
  --section-bg: #1e293b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  padding: 24px;
  max-width: 960px;
  margin: 0 auto;
}
h1 { font-size: 1.6rem; margin-bottom: 4px; }
h2 { font-size: 1.15rem; margin-bottom: 12px; color: var(--text); }
.subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 24px; }
.metrics-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.section {
  background: var(--section-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
}
.chart-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.chart-row > * { flex: 1; min-width: 280px; }
.toggle-btn {
  position: fixed; top: 12px; right: 16px;
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 12px; cursor: pointer;
  font-size: 0.8rem; color: var(--text-muted);
}
.toggle-btn:hover { background: var(--border); }
svg text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.empty-note { color: var(--text-muted); font-style: italic; font-size: 0.85rem; }
@media (max-width: 640px) {
  body { padding: 12px; }
  .chart-row { flex-direction: column; }
}
"""

_JS = """
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
  localStorage.setItem('ai-theme', html.getAttribute('data-theme'));
}
(function() {
  const saved = localStorage.getItem('ai-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  else if (window.matchMedia('(prefers-color-scheme: dark)').matches)
    document.documentElement.setAttribute('data-theme', 'dark');
})();
"""


def generate_report(
    report: DatasetReport,
    output_path: Optional[str | Path] = None,
) -> str:
    """Render *report* as a self-contained HTML string.

    If *output_path* is given the HTML is also written to that file.
    Returns the HTML string.
    """
    sections: list[str] = []

    # -- Header --
    sections.append(f'<h1>AfterImage Dataset Report</h1>')
    sections.append(f'<p class="subtitle">{html.escape(report.dataset_path)}</p>')

    # -- Summary metrics --
    s = report.summary
    sections.append('<div class="metrics-row">')
    sections.append(metric_card("Conversations", f"{s.total_conversations:,}"))
    sections.append(metric_card("Total Turns", f"{s.total_turns:,}"))
    sections.append(metric_card("Avg Turns", f"{s.avg_turns_per_conversation:.1f}"))
    sections.append(metric_card("Total Words", f"{s.total_words:,}"))
    sections.append(metric_card("Avg Words/Turn", f"{s.avg_words_per_turn:.1f}"))
    sections.append(metric_card("Personas", f"{s.unique_personas:,}"))
    sections.append(metric_card("Contexts", f"{s.unique_contexts:,}"))
    sections.append('</div>')

    # -- Personas --
    if report.personas.persona_counts:
        sections.append('<div class="section">')
        sections.append('<h2>Persona Distribution</h2>')
        sections.append('<div class="chart-row">')
        sections.append(bar_chart(report.personas.persona_counts, title="Conversations per Persona"))
        if report.personas.depth_distribution:
            sections.append(donut_chart(report.personas.depth_distribution, title="Persona Depth"))
        sections.append('</div>')
        sections.append('</div>')

    # -- Coverage --
    if report.coverage.context_counts:
        sections.append('<div class="section">')
        sections.append('<h2>Context Coverage</h2>')
        sections.append('<div class="metrics-row">')
        sections.append(metric_card("Unique Contexts", str(len(report.coverage.context_counts))))
        sections.append(metric_card("Used Once", str(report.coverage.contexts_used_once)))
        sections.append(metric_card("Used 2+", str(report.coverage.contexts_used_multiple)))
        sections.append('</div>')
        # Show top contexts
        top = dict(list(report.coverage.context_counts.items())[:15])
        sections.append(bar_chart(top, title="Top Contexts by Usage", color="var(--accent2)"))
        sections.append('</div>')

    # -- Quality --
    q = report.quality
    if q.has_evaluations:
        sections.append('<div class="section">')
        sections.append('<h2>Quality Evaluation</h2>')
        sections.append('<div class="chart-row">')
        sections.append(donut_chart(q.grade_counts, title="Grade Distribution"))
        if q.avg_scores:
            sections.append(bar_chart(
                {k: round(v, 3) for k, v in q.avg_scores.items()},
                title="Average Metric Scores",
            ))
        sections.append('</div>')
        if q.score_histogram and q.score_bins:
            sections.append(histogram(q.score_histogram, q.score_bins, title="Score Distribution"))
        sections.append('</div>')
    else:
        sections.append('<div class="section">')
        sections.append('<h2>Quality Evaluation</h2>')
        sections.append('<p class="empty-note">No evaluations found. Enable auto_improve to get quality metrics.</p>')
        sections.append('</div>')

    # -- Diversity --
    d = report.diversity
    if d.vocabulary_size > 0:
        sections.append('<div class="section">')
        sections.append('<h2>Text Diversity</h2>')
        sections.append('<div class="metrics-row">')
        sections.append(metric_card("Vocabulary", f"{d.vocabulary_size:,}"))
        sections.append(metric_card("Type-Token", f"{d.type_token_ratio:.3f}"))
        sections.append(metric_card("Entropy", f"{d.shannon_entropy:.2f}", "bits"))
        sections.append(metric_card("Bigram Rep.", f"{d.bigram_repetition_rate:.3f}"))
        sections.append('</div>')
        sections.append('</div>')

    # -- Lengths --
    le = report.lengths
    if le.user_lengths or le.assistant_lengths:
        sections.append('<div class="section">')
        sections.append('<h2>Message Lengths (words)</h2>')
        sections.append('<div class="metrics-row">')
        sections.append(metric_card("Avg User", f"{le.avg_user_length:.1f}"))
        sections.append(metric_card("Avg Assistant", f"{le.avg_assistant_length:.1f}"))
        sections.append('</div>')
        sections.append('<div class="chart-row">')
        if le.user_length_histogram and le.length_bins:
            sections.append(histogram(le.user_length_histogram, le.length_bins, title="User Message Lengths"))
        if le.assistant_length_histogram and le.length_bins:
            sections.append(histogram(le.assistant_length_histogram, le.length_bins, title="Assistant Message Lengths", color="var(--accent2)"))
        sections.append('</div>')
        sections.append('</div>')

    body = '\n'.join(sections)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AfterImage Dataset Report</title>
<style>{_CSS}</style>
</head>
<body>
<button class="toggle-btn" onclick="toggleTheme()">Toggle theme</button>
{body}
<script>{_JS}</script>
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(doc, encoding="utf-8")

    return doc
