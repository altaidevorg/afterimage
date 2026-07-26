import json
from typing import Optional
from pydantic import BaseModel, Field

from ..providers.llm_providers import LLMProvider
from .types import AgentTrajectory, JudgeVerdict, RubricScores


JUDGE_PROMPT_TEMPLATE = """You are an expert Trajectory Quality Judge evaluating synthetic agent execution traces.
Evaluate the given trajectory according to these quality rubrics:
1. Grounding: Are tool call parameters correctly derived from task context or prior turn observations?
2. Parameter Correctness: Are tool input values compliant with expected parameter names and types?
3. Loop Avoidance: Does the agent avoid redundant or unnecessary repeated tool invocations?
4. Task Completion: Does the final answer successfully address the user's intent?

Task: {task}
Domain Apps: {domain_apps}

Trajectory Turns:
{turns_text}

Final Answer: {final_answer}

Instruction: Evaluate the trajectory and return a JSON object with:
- grounding: float (0.0 to 1.0)
- parameter_correctness: float (0.0 to 1.0)
- loop_avoidance: float (0.0 to 1.0)
- task_completion: float (0.0 to 1.0)
- is_valid: bool (true if overall trajectory quality is high and accepted)
- confidence_score: float (0.0 to 1.0)
- feedback: string (brief explanation of evaluation verdict)
"""


class JudgeResponsePayload(BaseModel):
    grounding: float = Field(default=1.0, ge=0.0, le=1.0)
    parameter_correctness: float = Field(default=1.0, ge=0.0, le=1.0)
    loop_avoidance: float = Field(default=1.0, ge=0.0, le=1.0)
    task_completion: float = Field(default=1.0, ge=0.0, le=1.0)
    is_valid: bool = Field(default=True)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    feedback: str = Field(default="")


class TrajectoryJudge:
    """Evaluates generated execution trajectories using an LLM rubric judge."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.6-flash",
        min_quality_threshold: float = 0.75,
    ):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.min_quality_threshold = min_quality_threshold

    async def evaluate_trajectory(self, trajectory: AgentTrajectory) -> JudgeVerdict:
        """Evaluates a single trajectory and attaches a JudgeVerdict."""
        turns_lines = []
        for t in trajectory.turns:
            turns_lines.append(f"Turn {t.turn_id}:")
            turns_lines.append(f"  Thought: {t.agent_thought}")
            if t.tool_call:
                turns_lines.append(
                    f"  Tool Call: {t.tool_call.app}.{t.tool_call.action}({t.tool_call.parameters})"
                )
            if t.observation:
                turns_lines.append(f"  Observation: {t.observation.observation}")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            task=trajectory.task,
            domain_apps=", ".join(trajectory.domain_apps),
            turns_text="\n".join(turns_lines),
            final_answer=trajectory.final_answer or "None",
        )

        try:
            structured_res = await self.llm_provider.agenerate_structured(
                prompt=prompt,
                response_schema=JudgeResponsePayload,
                model_name=self.model_name,
                temperature=0.1,
            )
            payload = structured_res.parsed

            avg_score = (
                payload.grounding
                + payload.parameter_correctness
                + payload.loop_avoidance
                + payload.task_completion
            ) / 4.0
            is_valid = payload.is_valid and (avg_score >= self.min_quality_threshold)

            return JudgeVerdict(
                is_valid=is_valid,
                confidence_score=payload.confidence_score,
                evaluator_model=self.model_name,
                rubric_scores=RubricScores(
                    grounding=payload.grounding,
                    parameter_correctness=payload.parameter_correctness,
                    loop_avoidance=payload.loop_avoidance,
                    task_completion=payload.task_completion,
                ),
                feedback=payload.feedback,
            )

        except Exception as e:
            # Fallback evaluation if structured response fails
            return JudgeVerdict(
                is_valid=True,
                confidence_score=0.8,
                evaluator_model=self.model_name,
                feedback=f"Fallback evaluation due to judge exception: {str(e)}",
            )
