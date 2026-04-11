"""Pure-Python inline SVG chart generators. No external dependencies."""

from __future__ import annotations

import html
import math
from typing import Dict, List


def _esc(text: str) -> str:
    return html.escape(str(text))


# ------------------------------------------------------------------
# Metric card
# ------------------------------------------------------------------


def metric_card(label: str, value: str, subtitle: str = "") -> str:
    """Render a single metric as an SVG card."""
    sub = (
        f'<text x="60" y="72" text-anchor="middle" font-size="11" fill="var(--text-muted)">{_esc(subtitle)}</text>'
        if subtitle
        else ""
    )
    return (
        f'<svg width="120" height="80" viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="120" height="80" rx="8" fill="var(--card-bg)" stroke="var(--border)" stroke-width="1"/>'
        f'<text x="60" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="var(--accent)">{_esc(value)}</text>'
        f'<text x="60" y="54" text-anchor="middle" font-size="11" fill="var(--text-muted)">{_esc(label)}</text>'
        f"{sub}"
        f"</svg>"
    )


# ------------------------------------------------------------------
# Bar chart
# ------------------------------------------------------------------


def bar_chart(
    data: Dict[str, int | float],
    title: str = "",
    width: int = 500,
    bar_height: int = 24,
    max_bars: int = 15,
    color: str = "var(--accent)",
) -> str:
    """Horizontal bar chart. Labels on left, bars on right."""
    if not data:
        return ""

    items = list(data.items())[:max_bars]
    max_val = max(v for _, v in items) if items else 1
    label_width = 140
    chart_width = width - label_width - 20
    chart_height = len(items) * (bar_height + 6) + 40
    total_height = chart_height + (30 if title else 0)

    lines: list[str] = [
        f'<svg width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}" xmlns="http://www.w3.org/2000/svg">',
    ]
    y_offset = 0
    if title:
        lines.append(
            f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="var(--text)">{_esc(title)}</text>'
        )
        y_offset = 30

    for i, (label, val) in enumerate(items):
        y = y_offset + i * (bar_height + 6) + 20
        bw = (val / max_val) * chart_width if max_val else 0
        truncated = label[:20] + "..." if len(str(label)) > 20 else str(label)
        lines.append(
            f'<text x="{label_width - 5}" y="{y + bar_height // 2 + 4}" '
            f'text-anchor="end" font-size="11" fill="var(--text-muted)">{_esc(truncated)}</text>'
        )
        lines.append(
            f'<rect x="{label_width}" y="{y}" width="{bw:.1f}" height="{bar_height}" '
            f'rx="3" fill="{color}" opacity="0.85"/>'
        )
        lines.append(
            f'<text x="{label_width + bw + 5:.1f}" y="{y + bar_height // 2 + 4}" '
            f'font-size="10" fill="var(--text-muted)">{_format_num(val)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Histogram
# ------------------------------------------------------------------


def histogram(
    counts: List[int],
    bins: List[str],
    title: str = "",
    width: int = 500,
    height: int = 200,
    color: str = "var(--accent)",
) -> str:
    """Vertical bar histogram with bin labels on x-axis."""
    if not counts or not bins:
        return ""

    n = len(counts)
    max_val = max(counts) if counts else 1
    margin_left = 40
    margin_bottom = 40
    margin_top = 30 if title else 10
    chart_w = width - margin_left - 10
    chart_h = height - margin_top - margin_bottom
    bar_w = chart_w / n * 0.8
    gap = chart_w / n * 0.2

    lines: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
    ]

    if title:
        lines.append(
            f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="var(--text)">{_esc(title)}</text>'
        )

    # Bars
    for i, cnt in enumerate(counts):
        bh = (cnt / max_val) * chart_h if max_val else 0
        x = margin_left + i * (bar_w + gap) + gap / 2
        y = margin_top + chart_h - bh
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'rx="2" fill="{color}" opacity="0.85"/>'
        )
        # Count label
        if cnt > 0:
            lines.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 3:.1f}" text-anchor="middle" '
                f'font-size="9" fill="var(--text-muted)">{cnt}</text>'
            )
        # Bin label
        if i % max(1, n // 8) == 0 or i == n - 1:
            lines.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + chart_h + 15}" '
                f'text-anchor="middle" font-size="9" fill="var(--text-muted)">{_esc(bins[i])}</text>'
            )

    # Axis line
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_h}" '
        f'x2="{width - 10}" y2="{margin_top + chart_h}" '
        f'stroke="var(--border)" stroke-width="1"/>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Donut chart
# ------------------------------------------------------------------


def donut_chart(
    data: Dict[str, int],
    title: str = "",
    size: int = 200,
    colors: List[str] | None = None,
) -> str:
    """SVG donut chart with legend."""
    if not data:
        return ""

    default_colors = [
        "#4f8cff",
        "#34d399",
        "#fbbf24",
        "#f87171",
        "#a78bfa",
        "#fb923c",
        "#38bdf8",
        "#e879f9",
    ]
    palette = colors or default_colors
    total = sum(data.values())
    if total == 0:
        return ""

    cx, cy, r = size // 2, size // 2, size // 2 - 20
    inner_r = r * 0.55
    legend_width = 160
    full_width = size + legend_width

    lines: list[str] = [
        f'<svg width="{full_width}" height="{max(size, len(data) * 22 + 40)}" '
        f'viewBox="0 0 {full_width} {max(size, len(data) * 22 + 40)}" xmlns="http://www.w3.org/2000/svg">',
    ]

    if title:
        lines.append(
            f'<text x="{full_width // 2}" y="16" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="var(--text)">{_esc(title)}</text>'
        )

    # Total in center
    lines.append(
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" '
        f'font-size="18" font-weight="bold" fill="var(--text)">{total}</text>'
    )

    angle = -90  # start at top

    for i, (label, val) in enumerate(data.items()):
        if val == 0:
            continue
        sweep = (val / total) * 360
        color = palette[i % len(palette)]

        start_rad = math.radians(angle)
        end_rad = math.radians(angle + sweep)

        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)

        ix1 = cx + inner_r * math.cos(start_rad)
        iy1 = cy + inner_r * math.sin(start_rad)
        ix2 = cx + inner_r * math.cos(end_rad)
        iy2 = cy + inner_r * math.sin(end_rad)

        large = 1 if sweep > 180 else 0

        path = (
            f"M {ix1:.1f},{iy1:.1f} "
            f"L {x1:.1f},{y1:.1f} "
            f"A {r},{r} 0 {large},1 {x2:.1f},{y2:.1f} "
            f"L {ix2:.1f},{iy2:.1f} "
            f"A {inner_r},{inner_r} 0 {large},0 {ix1:.1f},{iy1:.1f} Z"
        )
        lines.append(f'<path d="{path}" fill="{color}" opacity="0.85"/>')
        angle += sweep

        # Legend
        ly = 30 + i * 22
        lines.append(
            f'<rect x="{size + 5}" y="{ly}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        truncated = label[:16] + ".." if len(str(label)) > 16 else str(label)
        lines.append(
            f'<text x="{size + 22}" y="{ly + 10}" font-size="11" fill="var(--text-muted)">'
            f"{_esc(truncated)} ({val})</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _format_num(val: int | float) -> str:
    if isinstance(val, float):
        return f"{val:.2f}"
    if val >= 10_000:
        return f"{val / 1000:.1f}k"
    return str(val)
