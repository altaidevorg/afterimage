import asyncio
import logging
from typing import List, Optional, Union

from ..key_management import SmartKeyPool
from ..providers.llm_providers import LLMFactory, LLMProvider
from ..storage import BaseStorage, JSONLStorage
from ..types import Conversation, ConversationEntry, Role
from .schema_architect import SchemaArchitect
from .task_synthesis import GridTaskSynthesizer
from .tool_environment import DeclarativeEnvironment
from .trajectory_generator import ReActTrajectoryLoop
from .trajectory_judge import TrajectoryJudge
from .types import AgentTrajectory, AppDomainSpec, ToolActionSpec

logger = logging.getLogger(__name__)


class AsyncAgentTraceGenerator:
    """Async Environment-Free Synthetic Agent-Trace Dataset Generator Facade."""

    def __init__(
        self,
        api_key: Optional[Union[str, List[str], SmartKeyPool]] = None,
        llm_provider: Optional[LLMProvider] = None,
        provider: str = "gemini",
        architect_model: str = "gemini-3.6-flash",
        teacher_model: str = "gemini-3.5-flash-lite",
        judge_model: str = "gemini-3.6-flash",
        storage: Optional[BaseStorage] = None,
    ):
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = LLMFactory.create(
                provider=provider,
                api_key=api_key,
                model_name=architect_model,
            )

        self.architect = SchemaArchitect(
            llm_provider=self.llm_provider,
            model_name=architect_model,
        )
        self.synthesizer = GridTaskSynthesizer(
            llm_provider=self.llm_provider,
            model_name=teacher_model,
        )
        self.teacher_loop = ReActTrajectoryLoop(
            llm_provider=self.llm_provider,
            model_name=teacher_model,
        )
        self.judge = TrajectoryJudge(
            llm_provider=self.llm_provider,
            model_name=judge_model,
        )

        self.environment = DeclarativeEnvironment()
        self.storage = storage or JSONLStorage(
            conversations_path="outputs/agent_trajectories.jsonl"
        )

    async def register_app_domain(
        self, app_name: str, app_description: str, actions: List[ToolActionSpec]
    ) -> AppDomainSpec:
        """Runs SchemaArchitect to generate and register Pydantic response models for an app domain."""
        app_spec, model_classes = await self.architect.generate_app_domain_schema(
            app_name=app_name,
            app_description=app_description,
            actions=actions,
        )
        self.environment.register_app_domain(app_spec, model_classes=model_classes)
        return app_spec

    async def generate_single(self, max_turns: int = 6) -> Optional[AgentTrajectory]:
        """Synthesizes a single agent trajectory (task -> ReAct loop -> judge)."""
        if not self.environment.app_domains:
            raise ValueError(
                "No app domains registered. Call register_app_domain() first."
            )

        # 1. Task synthesis via 360-bucket grid & task rewriter
        task, selected_apps, bucket = await self.synthesizer.synthesize_task(
            app_domains=self.environment.app_domains
        )

        # 2. ReAct teacher trajectory loop against DeclarativeEnvironment (< 1ms tool calls)
        trajectory = await self.teacher_loop.run_trajectory(
            task=task,
            environment=self.environment,
            domain_apps=selected_apps,
        )
        trajectory.metadata["grid_bucket"] = bucket.model_dump()

        # 3. Trajectory Judge Quality Filtering
        verdict = await self.judge.evaluate_trajectory(trajectory)
        trajectory.judge_verdict = verdict

        if verdict.is_valid:
            return trajectory
        return None

    async def generate(
        self,
        num_trajectories: int = 10,
        max_turns: int = 6,
        max_concurrency: int = 4,
    ) -> List[AgentTrajectory]:
        """Generates multiple synthetic agent trajectories concurrently."""
        sem = asyncio.Semaphore(max_concurrency)
        accepted_trajectories: List[AgentTrajectory] = []

        async def _worker() -> Optional[AgentTrajectory]:
            async with sem:
                try:
                    return await self.generate_single(max_turns=max_turns)
                except Exception as e:
                    logger.warning(
                        f"Error during trajectory generation worker: {e}", exc_info=True
                    )
                    return None

        tasks = [_worker() for _ in range(num_trajectories)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        conversations = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Worker encountered unhandled exception: {res}")
                continue
            if isinstance(res, AgentTrajectory):
                accepted_trajectories.append(res)
                conv = self._trajectory_to_conversation(res)
                conversations.append(conv)

        if conversations:
            self.storage.save_conversations(conversations)

        return accepted_trajectories

    def _trajectory_to_conversation(self, traj: AgentTrajectory) -> Conversation:
        """Converts an AgentTrajectory into AfterImage's base Conversation schema."""
        entries: List[ConversationEntry] = [
            ConversationEntry(role=Role.USER, content=traj.task)
        ]
        for t in traj.turns:
            entry_text = f"Thought: {t.agent_thought}"
            if t.tool_call:
                entry_text += f"\nAction: {t.tool_call.app}.{t.tool_call.action}\nAction Input: {t.tool_call.parameters}"
            entries.append(ConversationEntry(role=Role.ASSISTANT, content=entry_text))

            if t.observation:
                entries.append(
                    ConversationEntry(
                        role=Role.USER,
                        content=f"Observation: {t.observation.observation}",
                    )
                )

        if traj.final_answer:
            entries.append(
                ConversationEntry(
                    role=Role.ASSISTANT, content=f"Final Answer: {traj.final_answer}"
                )
            )

        metadata = traj.metadata
        if traj.judge_verdict:
            metadata["judge_verdict"] = traj.judge_verdict.model_dump()
        metadata["trajectory_id"] = traj.trajectory_id

        return Conversation(conversations=entries, metadata=metadata)
