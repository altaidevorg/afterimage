"""Prompt builders for context-to-skill discovery."""

from __future__ import annotations

import json

from .types import SkillProbeResult, SkillProposal, SkillSide, SkillVersion


def _clip(text: str, max_chars: int = 12000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...context truncated...]"


def build_probe_generation_prompt(
    *,
    context: str,
    respondent_prompt: str,
    challenger_skill: SkillVersion | None,
    n_probes: int,
    source_rubrics: list[str] | None = None,
) -> str:
    skill_text = challenger_skill.content if challenger_skill else "(none)"
    source_rubrics_text = ""
    if source_rubrics:
        rubrics = "\n".join(f"- {rubric}" for rubric in source_rubrics)
        source_rubrics_text = f"""
Source benchmark rubrics:
{rubrics}

Use these rubrics only as optional hints about what the benchmark tends to test.
Do not copy them verbatim into every task unless the context truly supports it.
"""
    return f"""You are the Challenger in a context-to-skill self-play loop.

Respondent system prompt:
{respondent_prompt}

Current challenger skill set:
{skill_text}

Context:
<context>
{_clip(context)}
</context>
{source_rubrics_text}

Create exactly {n_probes} diverse, context-grounded tasks that require the
respondent to induce rules or procedures from the context. For each task, write
strict binary rubrics. The tasks should expose likely respondent failure modes,
not ask generic trivia or surface paraphrases.

Use the challenger skill only to improve task and rubric generation. Do not
assume access to any respondent-side skill.

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
    return f"""You are a neutral Judge in a context-to-skill self-play loop.

Follow this grading process exactly:
1. Requirement analysis: restate what each rubric demands.
2. Per-rubric verification: check the answer against each rubric separately.
3. Self-reflection: verify that no rubric was marked passed without support.

{context_block}
Task:
{task}

Rubrics:
{rubrics_text}

Answer:
<answer>
{answer}
</answer>

Return structured JSON. requirement_status must contain one boolean per rubric.
overall_score must be 1.0 only if every rubric passes; otherwise it must be 0.0.
"""


def build_skill_proposal_prompt(
    *,
    context: str,
    respondent_prompt: str,
    current_skill: SkillVersion | None,
    routed_results: list[SkillProbeResult],
    iteration: int,
    side: SkillSide,
) -> str:
    cases = [
        {
            "task": result.probe.task,
            "rubrics": result.probe.rubrics,
            "answer": result.answer,
            "score": result.score,
            "rubric_status": result.rubric_status,
            "judge_feedback": result.judge_feedback,
        }
        for result in routed_results
    ]
    if side == "reasoner":
        title = "Reasoner Proposer"
        routed_label = "Failed respondent cases"
        objective = (
            "Identify which contextual knowledge, procedures, or constraints the "
            "respondent is missing or misapplying."
        )
    else:
        title = "Challenger Proposer"
        routed_label = "Solved respondent cases"
        objective = (
            "Identify why these tasks were too easy and how future tasks and rubrics "
            "should better expose the respondent's remaining weaknesses."
        )
    return f"""You are the {title} in a context-to-skill self-play loop.

Iteration: {iteration}

Respondent system prompt:
{respondent_prompt}

Current {side} skill set:
{current_skill.content if current_skill else "(none)"}

Context:
<context>
{_clip(context)}
</context>

{routed_label}:
{json.dumps(cases, ensure_ascii=False, indent=2)}

Task:
- {objective}
- Propose one reusable natural-language skill update.
- Do not memorize exact tasks or answers.
- Do not write the final SKILL.md content.

Return structured JSON only."""


def build_skill_generation_prompt(
    *,
    context: str,
    proposal: SkillProposal,
    previous_skill: SkillVersion | None,
    side: SkillSide,
) -> str:
    previous = previous_skill.content if previous_skill else "(none)"
    if side == "reasoner":
        title = "Reasoner Generator"
        instructions = (
            "Write a complete respondent-side skill set that improves future task "
            "answers for this context. The content should be concise procedural "
            "guidance with clear when-to-use behavior."
        )
    else:
        title = "Challenger Generator"
        instructions = (
            "Write a complete challenger-side skill set used only to generate future "
            "tasks and rubrics. Focus on making probes more diagnostic, specific, and "
            "strict. Do not mention or depend on any respondent-side skill."
        )
    return f"""You are the {title} in a context-to-skill self-play loop.

Previous {side} skill set:
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

{instructions}
Avoid copying long source text. Return structured JSON with name, description,
and content."""


def build_skill_bootstrap_prompt(
    *,
    context: str,
    respondent_prompt: str,
    probe_results: list[SkillProbeResult],
) -> str:
    probes = [
        {
            "task": result.probe.task,
            "rubrics": result.probe.rubrics,
            "answer": result.answer,
            "score": result.score,
            "rubric_status": result.rubric_status,
            "judge_feedback": result.judge_feedback,
        }
        for result in probe_results[:5]
    ]
    return f"""Create a reusable respondent-side context-specific skill.

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

Write concise Markdown with clear when-to-use behavior. Return structured JSON
with name, description, and content."""
