"""Prompt builders for context-to-skill discovery."""

from __future__ import annotations

import json

from .types import SkillProbeResult, SkillProposal, SkillVersion


def _clip(text: str, max_chars: int = 12000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...context truncated...]"


def build_probe_generation_prompt(
    *,
    context: str,
    respondent_prompt: str,
    current_skill: SkillVersion | None,
    n_probes: int,
    source_rubrics: list[str] | None = None,
) -> str:
    skill_text = current_skill.content if current_skill else "(none)"
    source_rubrics_text = ""
    if source_rubrics:
        rubrics = "\n".join(f"- {rubric}" for rubric in source_rubrics)
        source_rubrics_text = f"""
Source benchmark rubrics:
{rubrics}

Use these rubrics as strong hints about what the hidden task is trying to test.
Generated tasks should make the relevant source rubrics assessable while still
being answerable from the context.
"""
    return f"""You generate evaluation probes for a domain assistant.

Respondent system prompt:
{respondent_prompt}

Current context-specific skill:
{skill_text}

Context:
<context>
{_clip(context)}
</context>
{source_rubrics_text}

Create exactly {n_probes} diverse, context-grounded tasks that require using the
context. For each task, write strict binary rubrics. The tasks should expose
likely respondent failure modes, not ask generic trivia.

Return structured JSON only."""


def build_reasoner_prompt(
    *,
    context: str,
    respondent_prompt: str,
    skill: SkillVersion | None,
    task: str,
) -> str:
    skill_block = ""
    if skill:
        skill_block = f"""

Context-specific skill:
<skill>
{skill.content}
</skill>
"""
    return f"""{respondent_prompt}

Use the context below to answer the task. Do not invent unsupported facts.
{skill_block}
Context:
<context>
{_clip(context)}
</context>

Task:
{task}
"""


def build_rubric_judge_prompt(
    *,
    context: str | None,
    task: str,
    rubrics: list[str],
    answer: str,
) -> str:
    context_block = (
        f"\nContext:\n<context>\n{_clip(context)}\n</context>\n" if context else ""
    )
    rubrics_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rubrics))
    return f"""You are a strict rubric judge. Grade the answer against every rubric.

{context_block}
Task:
{task}

Rubrics:
{rubrics_text}

Answer:
<answer>
{answer}
</answer>

Return structured JSON. requirement_status must have one boolean per rubric.
overall_score should be 1.0 only when all rubrics are fully satisfied, otherwise
0.0 unless partial credit is explicitly warranted by the rubrics."""


def build_skill_proposal_prompt(
    *,
    context: str,
    respondent_prompt: str,
    current_skill: SkillVersion | None,
    failed_results: list[SkillProbeResult],
    iteration: int,
) -> str:
    failures = [
        {
            "task": r.probe.task,
            "rubrics": r.probe.rubrics,
            "answer": r.answer,
            "score": r.score,
            "rubric_status": r.rubric_status,
            "judge_feedback": r.judge_feedback,
        }
        for r in failed_results
    ]
    return f"""You analyze respondent failures and propose a reusable natural-language skill.

Iteration: {iteration}

Respondent system prompt:
{respondent_prompt}

Current skill:
{current_skill.content if current_skill else "(none)"}

Context:
<context>
{_clip(context)}
</context>

Failed probe results:
{json.dumps(failures, ensure_ascii=False, indent=2)}

Propose one concise procedural skill that would help the respondent avoid these
failure modes on future tasks for this context. Do not memorize the exact probes.
Return structured JSON only."""


def build_skill_generation_prompt(
    *,
    context: str,
    proposal: SkillProposal,
    previous_skill: SkillVersion | None,
) -> str:
    previous = previous_skill.content if previous_skill else "(none)"
    return f"""Turn this skill proposal into a complete context-specific skill.

Previous skill:
{previous}

Proposal:
Name: {proposal.name}
Description: {proposal.description}
Failure modes: {proposal.target_failure_modes}
Guidance:
{proposal.proposed_guidance}

Context:
<context>
{_clip(context)}
</context>

Write concise Markdown. The content should be procedural guidance with clear
"when to use" behavior. Avoid copying long source text. Return structured JSON
with name, description, and content."""


def build_skill_bootstrap_prompt(
    *,
    context: str,
    respondent_prompt: str,
    probe_results: list[SkillProbeResult],
) -> str:
    probes = [
        {
            "task": r.probe.task,
            "rubrics": r.probe.rubrics,
            "answer": r.answer,
            "score": r.score,
            "rubric_status": r.rubric_status,
            "judge_feedback": r.judge_feedback,
        }
        for r in probe_results[:5]
    ]
    return f"""Create a reusable context-specific skill for a respondent.

No failed probe was found, so infer the skill directly from the respondent
system prompt, the context, and the successful probes. Capture constraints that
future answers must preserve, especially persona, formatting, length, refusal,
domain, and source-grounding rules. Do not copy long source text.

Respondent system prompt:
{respondent_prompt}

Context:
<context>
{_clip(context)}
</context>

Successful probe examples:
{json.dumps(probes, ensure_ascii=False, indent=2)}

Write concise Markdown with clear "when to use" behavior. Return structured JSON
with name, description, and content."""
