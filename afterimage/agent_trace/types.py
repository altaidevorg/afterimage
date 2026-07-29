import uuid
from typing import Any, Dict, List, Literal, Optional, Type
from pydantic import BaseModel, Field

ObservationMode = Literal["faker", "llm"]


class ToolParameterSpec(BaseModel):
    name: str = Field(..., description="Parameter name")
    type: str = Field(default="str", description="Parameter primitive or schema type")
    description: str = Field(default="", description="Description of the parameter")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(
        default=None, description="Default value if optional"
    )


class ToolActionSpec(BaseModel):
    action_name: str = Field(..., description="Name of the API action/tool endpoint")
    description: str = Field(..., description="Description of what the action does")
    parameters: List[ToolParameterSpec] = Field(
        default_factory=list, description="List of parameters accepted by action"
    )
    response_model_name: str = Field(
        ..., description="Name of the Pydantic V2 response model class"
    )
    response_model_cls: Optional[Type[BaseModel]] = Field(
        default=None,
        exclude=True,
        description="Explicit Pydantic response model class if provided",
    )
    response_schema_hint: Optional[str] = Field(
        default=None,
        description="Schema description hint for SchemaArchitect if explicit class omitted",
    )


class AppDomainSpec(BaseModel):
    app_name: str = Field(..., description="Unique name of the app domain")
    description: str = Field(
        ..., description="High-level description of the app domain"
    )
    actions: List[ToolActionSpec] = Field(
        default_factory=list, description="API endpoints available in this app"
    )
    response_models_code: str = Field(
        default="", description="Python source code containing Pydantic response models"
    )


class ToolCall(BaseModel):
    app: str = Field(..., description="Target app name")
    action: str = Field(..., description="Target tool action name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to tool call"
    )


class ToolObservation(BaseModel):
    observation: Any = Field(
        ..., description="Raw output object or dict returned by tool"
    )
    status: Literal["success", "error"] = Field(
        default="success", description="Execution status"
    )
    latency_ms: float = Field(
        default=0.0, description="Tool execution latency in milliseconds"
    )


class TrajectoryTurn(BaseModel):
    turn_id: int = Field(..., description="1-indexed turn number")
    agent_thought: str = Field(
        ..., description="Agent reasoning before action or output"
    )
    tool_call: Optional[ToolCall] = Field(
        default=None, description="Tool call made in this turn"
    )
    observation: Optional[ToolObservation] = Field(
        default=None, description="Observation returned by tool environment"
    )


class RubricScores(BaseModel):
    grounding: float = Field(default=1.0, ge=0.0, le=1.0)
    parameter_correctness: float = Field(default=1.0, ge=0.0, le=1.0)
    loop_avoidance: float = Field(default=1.0, ge=0.0, le=1.0)
    task_completion: float = Field(default=1.0, ge=0.0, le=1.0)


class JudgeVerdict(BaseModel):
    is_valid: bool = Field(..., description="Whether trajectory is accepted")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evaluator_model: str = Field(default="gemini-3.6-flash")
    rubric_scores: Optional[RubricScores] = Field(default_factory=RubricScores)
    feedback: str = Field(default="")


class AgentTrajectory(BaseModel):
    trajectory_id: str = Field(
        default_factory=lambda: f"traj_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for trajectory",
    )
    task: str = Field(..., description="Natural user task/instruction")
    domain_apps: List[str] = Field(
        default_factory=list, description="List of apps involved"
    )
    turns: List[TrajectoryTurn] = Field(
        default_factory=list, description="Sequence of ReAct execution turns"
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final response returned to user"
    )
    judge_verdict: Optional[JudgeVerdict] = Field(
        default=None, description="Trajectory judge evaluation result"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata"
    )


class GridTaskBucket(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    action_type: Literal["read", "write", "mixed"] = "read"
    task_focus: Literal[
        "constraint_satisfaction", "derivation", "iteration", "open"
    ] = "open"
    num_apps: Literal[1, 2, 3] = 1
