#!/usr/bin/env python3
"""Compare Ctx2Skill and Afterimage context-specific skill outputs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from tqdm.auto import tqdm


_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class SkillComparison:
    context_id: str
    ctx2skill_path: str | None
    afterimage_path: str | None
    ctx2skill_words: int
    afterimage_words: int
    token_jaccard: float
    sequence_ratio: float


def strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def read_skill(path: Path) -> str:
    return strip_frontmatter(path.read_text(encoding="utf-8"))


def discover_skills(root: Path) -> dict[str, Path]:
    """Find `<context_id>/SKILL.md` files under a root directory."""
    skills = {}
    for path in sorted(root.rglob("SKILL.md")):
        context_id = path.parent.name
        skills[context_id] = path
    return skills


def compare_texts(left: str, right: str) -> tuple[int, int, float, float]:
    left_words = words(left)
    right_words = words(right)
    left_set = set(left_words)
    right_set = set(right_words)
    union = left_set | right_set
    jaccard = (len(left_set & right_set) / len(union)) if union else 1.0
    seq = SequenceMatcher(None, left, right).ratio() if (left or right) else 1.0
    return len(left_words), len(right_words), jaccard, seq


def compare_roots(ctx2skill_root: Path, afterimage_root: Path) -> list[SkillComparison]:
    ctx_skills = discover_skills(ctx2skill_root)
    ai_skills = discover_skills(afterimage_root)
    context_ids = sorted(set(ctx_skills) | set(ai_skills))
    rows = []
    for context_id in tqdm(context_ids, desc="Comparing skills", unit="context"):
        ctx_path = ctx_skills.get(context_id)
        ai_path = ai_skills.get(context_id)
        ctx_text = read_skill(ctx_path) if ctx_path else ""
        ai_text = read_skill(ai_path) if ai_path else ""
        ctx_words, ai_words, jaccard, seq = compare_texts(ctx_text, ai_text)
        rows.append(
            SkillComparison(
                context_id=context_id,
                ctx2skill_path=str(ctx_path) if ctx_path else None,
                afterimage_path=str(ai_path) if ai_path else None,
                ctx2skill_words=ctx_words,
                afterimage_words=ai_words,
                token_jaccard=jaccard,
                sequence_ratio=seq,
            )
        )
    return rows


def summarize(rows: list[SkillComparison]) -> dict:
    paired = [r for r in rows if r.ctx2skill_path and r.afterimage_path]
    return {
        "total_contexts": len(rows),
        "paired_contexts": len(paired),
        "ctx2skill_only": sum(1 for r in rows if r.ctx2skill_path and not r.afterimage_path),
        "afterimage_only": sum(1 for r in rows if r.afterimage_path and not r.ctx2skill_path),
        "avg_token_jaccard": (
            sum(r.token_jaccard for r in paired) / len(paired) if paired else 0.0
        ),
        "avg_sequence_ratio": (
            sum(r.sequence_ratio for r in paired) / len(paired) if paired else 0.0
        ),
    }


def print_table(rows: list[SkillComparison], limit: int) -> None:
    print(
        "context_id                             ctx_words ai_words "
        "jaccard seq_ratio status"
    )
    print("-" * 86)
    for row in rows[:limit]:
        status = (
            "paired"
            if row.ctx2skill_path and row.afterimage_path
            else "ctx2skill-only"
            if row.ctx2skill_path
            else "afterimage-only"
        )
        print(
            f"{row.context_id[:36]:36} "
            f"{row.ctx2skill_words:9d} {row.afterimage_words:8d} "
            f"{row.token_jaccard:7.3f} {row.sequence_ratio:9.3f} {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare skill directories.")
    parser.add_argument("--ctx2skill-root", required=True, type=Path)
    parser.add_argument("--afterimage-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = compare_roots(args.ctx2skill_root, args.afterimage_root)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print()
    print_table(rows, args.limit)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote comparison report to {args.output}")


if __name__ == "__main__":
    main()
