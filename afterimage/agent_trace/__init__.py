"""AfterImage Agent Trace subpackage for environment-free synthetic agent-trace dataset generation.
"""

from .context import (
    BaseContextGenerator,
    CallableContextGenerator,
    CompositeContextGenerator,
    PersonaContextGenerator,
    VirtualUserContextGenerator,
)
from .generator import AsyncAgentTraceGenerator
from .llm_observation_synthesizer import LLMObservationSynthesizer
from .schema_architect import SchemaArchitect
from .simulation_engine import DeclarativeEngine, SimulationContext
from .simula_task_synthesis import SimulaTaskSynthesizer
from .task_synthesis import GridTaskSynthesizer, InverseFrequencySampler
from .tool_environment import DeclarativeEnvironment, DeclarativeTool
from .trajectory_generator import ReActTrajectoryLoop
from .trajectory_judge import TrajectoryJudge
from .types import (
    AgentTrajectory,
    AppDomainSpec,
    GridTaskBucket,
    JudgeVerdict,
    ObservationMode,
    RubricScores,
    ToolActionSpec,
    ToolCall,
    ToolObservation,
    ToolParameterSpec,
    TrajectoryTurn,
)
from .verifier import SchemaVerifier, VerificationReport

__all__ = [
    "AsyncAgentTraceGenerator",
    "BaseContextGenerator",
    "VirtualUserContextGenerator",
    "PersonaContextGenerator",
    "CallableContextGenerator",
    "CompositeContextGenerator",
    "LLMObservationSynthesizer",
    "DeclarativeEngine",
    "SimulationContext",
    "DeclarativeEnvironment",
    "DeclarativeTool",
    "SchemaArchitect",
    "SchemaVerifier",
    "VerificationReport",
    "GridTaskSynthesizer",
    "SimulaTaskSynthesizer",
    "InverseFrequencySampler",
    "ReActTrajectoryLoop",
    "TrajectoryJudge",
    "AgentTrajectory",
    "AppDomainSpec",
    "GridTaskBucket",
    "JudgeVerdict",
    "ObservationMode",
    "RubricScores",
    "ToolActionSpec",
    "ToolCall",
    "ToolObservation",
    "ToolParameterSpec",
    "TrajectoryTurn",
]
