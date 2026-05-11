#!/usr/bin/env python3
"""Evaluate Ctx2Skill and Afterimage skills on shared rubric tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.auto import tqdm

from afterimage.config import load_config, resolve_api_key
from afterimage.config_to_generator import _llm_create_extras
from afterimage.providers import LLMFactory
from afterimage.skills.judging import RubricJudge
from afterimage.skills.prompts import build_reasoner_prompt
from afterimage.skills.types import SkillProbe, SkillVersion


@dataclass
class EvalTask:
    context_id: str
    source: str
    task_id: str
    task: str
    rubrics: list[str]


@dataclass
class EvalRow:
    context_id: str
    source: str
    task_id: str
    variant: str
    passed: bool
    score: float
    satisfied: int
    total_rubrics: int
    answer: str
    judge_feedback: str


def strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1 :]).strip()
    return text.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_contexts(docs_path: Path) -> dict[str, str]:
    contexts = {}
    for row in read_jsonl(docs_path):
        context_id = str(row.get("id") or (row.get("metadata") or {}).get("context_id"))
        text = row.get("text")
        if context_id and isinstance(text, str):
            contexts[context_id] = text
    return contexts


def discover_skills(root: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in sorted(root.rglob("SKILL.md"))}


def load_tasks_for_context(
    *,
    context_id: str,
    ctx2skill_root: Path,
    afterimage_root: Path,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []

    for idx, row in enumerate(read_jsonl(ctx2skill_root / context_id / "hard_set.jsonl")):
        task = row.get("task")
        rubrics = row.get("rubrics")
        if isinstance(task, str) and isinstance(rubrics, list):
            tasks.append(
                EvalTask(
                    context_id=context_id,
                    source="ctx2skill-hard-set",
                    task_id=f"ctx2skill-{idx + 1}",
                    task=task,
                    rubrics=[str(r) for r in rubrics],
                )
            )

    for row in read_jsonl(afterimage_root / context_id / "probes.jsonl"):
        task = row.get("task")
        rubrics = row.get("rubrics")
        if isinstance(task, str) and isinstance(rubrics, list):
            tasks.append(
                EvalTask(
                    context_id=context_id,
                    source="afterimage-probe",
                    task_id=str(row.get("id") or f"afterimage-{len(tasks) + 1}"),
                    task=task,
                    rubrics=[str(r) for r in rubrics],
                )
            )

    seen = set()
    deduped = []
    for task in tasks:
        key = (task.task, tuple(task.rubrics))
        if key not in seen:
            seen.add(key)
            deduped.append(task)
    return deduped


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    variants = sorted({row.variant for row in rows})
    by_variant = {}
    for variant in variants:
        subset = [row for row in rows if row.variant == variant]
        by_variant[variant] = {
            "tasks": len(subset),
            "passed": sum(1 for row in subset if row.passed),
            "pass_rate": (
                sum(1 for row in subset if row.passed) / len(subset) if subset else 0.0
            ),
            "avg_score": (
                sum(row.score for row in subset) / len(subset) if subset else 0.0
            ),
            "avg_rubric_satisfaction": (
                sum(row.satisfied / row.total_rubrics for row in subset if row.total_rubrics)
                / len(subset)
                if subset
                else 0.0
            ),
        }

    by_source = {}
    for source in sorted({row.source for row in rows}):
        by_source[source] = {}
        for variant in variants:
            subset = [
                row for row in rows if row.source == source and row.variant == variant
            ]
            by_source[source][variant] = {
                "tasks": len(subset),
                "passed": sum(1 for row in subset if row.passed),
                "pass_rate": (
                    sum(1 for row in subset if row.passed) / len(subset)
                    if subset
                    else 0.0
                ),
                "avg_score": (
                    sum(row.score for row in subset) / len(subset) if subset else 0.0
                ),
            }

    return {"by_variant": by_variant, "by_source": by_source}


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary["by_variant"], indent=2))
    print()
    print("source              variant       tasks passed pass_rate avg_score")
    print("-" * 68)
    for source, variants in summary["by_source"].items():
        for variant, stats in variants.items():
            print(
                f"{source[:18]:18} {variant[:12]:12} "
                f"{stats['tasks']:5d} {stats['passed']:6d} "
                f"{stats['pass_rate']:9.3f} {stats['avg_score']:9.3f}"
            )


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    if args.env_file is not None:
        load_dotenv(args.env_file)
    config = load_config(str(args.config))
    if args.api_key_env is not None:
        config.model.api_key_env = args.api_key_env
    api_key = resolve_api_key(config)
    llm = LLMFactory.create(
        provider=config.model.provider,
        model_name=config.model.model_name,
        api_key=api_key,
        **_llm_create_extras(config),
    )
    judge = RubricJudge(llm)

    contexts = load_contexts(args.docs)
    ctx_skills = discover_skills(args.ctx2skill_root)
    ai_skills = discover_skills(args.afterimage_root)
    context_ids = sorted(set(ctx_skills) & set(ai_skills) & set(contexts))
    if args.limit_contexts is not None:
        context_ids = context_ids[: args.limit_contexts]

    tasks_by_context = {}
    for context_id in context_ids:
        tasks = load_tasks_for_context(
            context_id=context_id,
            ctx2skill_root=args.ctx2skill_root,
            afterimage_root=args.afterimage_root,
        )
        if args.limit_tasks is not None:
            tasks = tasks[: args.limit_tasks]
        tasks_by_context[context_id] = tasks

    rows: list[EvalRow] = []
    variants = ("baseline", "ctx2skill", "afterimage")
    total = sum(len(tasks) * len(variants) for tasks in tasks_by_context.values())
    progress = tqdm(total=total, desc="Evaluating skills", unit="variant")
    try:
        for context_id in context_ids:
            context = contexts[context_id]
            tasks = tasks_by_context[context_id]

            skill_texts = {
                "baseline": None,
                "ctx2skill": strip_frontmatter(
                    ctx_skills[context_id].read_text(encoding="utf-8")
                ),
                "afterimage": strip_frontmatter(
                    ai_skills[context_id].read_text(encoding="utf-8")
                ),
            }

            for task in tasks:
                for variant, skill_text in skill_texts.items():
                    progress.set_postfix(
                        context=context_id[:8],
                        source=task.source[:16],
                        variant=variant,
                    )
                    skill = None
                    if skill_text:
                        skill = SkillVersion(
                            id=f"{variant}-{context_id}",
                            context_id=context_id,
                            iteration=0,
                            name=variant,
                            description=variant,
                            content=skill_text,
                        )
                    prompt = build_reasoner_prompt(
                        context=context,
                        respondent_prompt=config.respondent.system_prompt or "",
                        skill=skill,
                        task=task.task,
                    )
                    answer = await llm.agenerate_content(prompt, temperature=0.2)
                    result = await judge.aevaluate(
                        probe=SkillProbe(
                            id=task.task_id,
                            context_id=context_id,
                            task=task.task,
                            rubrics=task.rubrics,
                        ),
                        answer=answer.text,
                        context=context,
                        skill_version_id=skill.id if skill else None,
                    )
                    rows.append(
                        EvalRow(
                            context_id=context_id,
                            source=task.source,
                            task_id=task.task_id,
                            variant=variant,
                            passed=result.passed,
                            score=result.score,
                            satisfied=sum(
                                1 for status in result.rubric_status if status
                            ),
                            total_rubrics=len(result.rubric_status),
                            answer=answer.text,
                            judge_feedback=result.judge_feedback,
                        )
                    )
                    progress.update(1)
    finally:
        progress.close()

    summary = summarize(rows)
    return {"summary": summary, "rows": [asdict(row) for row in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--ctx2skill-root", type=Path, required=True)
    parser.add_argument("--afterimage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()

    payload = asyncio.run(evaluate(args))
    print_summary(payload["summary"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
