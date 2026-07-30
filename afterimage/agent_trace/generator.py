"""Facade generator for environment-free synthetic agent execution trace datasets.

Coordinates schema architecture, task synthesis (combinatorial grid or Simula taxonomy),
declarative local simulation, ReAct teacher execution loops, trajectory quality judging,
and multi-format exporters.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, List, Literal, Optional, Union

from tqdm.asyncio import tqdm

from ..key_management import SmartKeyPool
from ..monitoring import GenerationMonitor
from ..providers.llm_providers import LLMFactory, LLMProvider
from ..storage import BaseStorage, JSONLStorage
from ..types import Conversation, ConversationEntry, Role
from .context import BaseContextGenerator, VirtualUserContextGenerator
from .schema_architect import SchemaArchitect
from .simula_task_synthesis import SimulaTaskSynthesizer
from .task_synthesis import GridTaskSynthesizer
from .tool_environment import DeclarativeEnvironment
from .trajectory_generator import ReActTrajectoryLoop
from .trajectory_judge import TrajectoryJudge
from .types import AgentTrajectory, AppDomainSpec, ObservationMode, ToolActionSpec

logger = logging.getLogger(__name__)


class AsyncAgentTraceGenerator:
    """Async Environment-Free Synthetic Agent-Trace Dataset Generator Facade.

    Coordinates SchemaArchitect, GridTaskSynthesizer, SimulaTaskSynthesizer,
    ReActTrajectoryLoop, DeclarativeEnvironment, TrajectoryJudge, and GenerationMonitor.

    Args:
        api_key: Optional API key string, list of keys, or :class:`SmartKeyPool`.
        llm_provider: Optional shared LLMProvider instance.
        provider: Provider vendor name (e.g. ``"gemini"``, ``"openai"``). Defaults to ``"gemini"``.
        architect_model: Model name for SchemaArchitect. Defaults to ``"gemini-3.6-flash"``.
        teacher_model: Model name for ReAct teacher loop. Defaults to ``"gemini-3.5-flash-lite"``.
        judge_model: Model name for TrajectoryJudge. Defaults to ``"gemini-3.6-flash"``.
        observation_mode: Tool response synthesis mode (``"llm"`` or ``"faker"``). Defaults to ``"llm"``.
        task_synthesis_mode: Synthesis strategy (``"grid"`` or ``"simula"``). Defaults to ``"grid"``.
        context_generator: Extensible initial context generator. Defaults to :class:`VirtualUserContextGenerator`.
        storage: Optional storage backend. Defaults to :class:`JSONLStorage`.
        monitor: Optional generation monitor instance for real-time tracking and metrics.
        llm_factory_kwargs: Extra keyword arguments for LLM initialization.

    Example:
        >>> generator = AsyncAgentTraceGenerator(api_key="your_api_key")
        >>> await generator.register_app_domain("e_commerce", "Shopping platform", actions=[...])
        >>> trajectories = await generator.generate(num_trajectories=10, show_progress=True)
    """

    def __init__(
        self,
        api_key: Optional[Union[str, List[str], SmartKeyPool]] = None,
        llm_provider: Optional[LLMProvider] = None,
        provider: str = "gemini",
        architect_model: str = "gemini-3.6-flash",
        teacher_model: str = "gemini-3.5-flash-lite",
        judge_model: str = "gemini-3.6-flash",
        observation_mode: ObservationMode = "llm",
        task_synthesis_mode: Literal["grid", "simula"] = "grid",
        context_generator: Optional[BaseContextGenerator] = None,
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        llm_factory_kwargs: Optional[dict] = None,
    ):
        self.observation_mode = observation_mode
        self.task_synthesis_mode = task_synthesis_mode
        self.context_generator = (
            context_generator if context_generator is not None else VirtualUserContextGenerator()
        )
        self.monitor = monitor
        extras = dict(llm_factory_kwargs or {})

        if llm_provider:
            self.llm_provider = llm_provider
            architect_llm = llm_provider
            teacher_llm = llm_provider
            judge_llm = llm_provider
        else:
            key_pool: Optional[Union[str, SmartKeyPool]] = None
            if isinstance(api_key, SmartKeyPool):
                key_pool = api_key
            elif isinstance(api_key, str):
                key_pool = SmartKeyPool.from_single_key(api_key)
            elif isinstance(api_key, list):
                key_pool = SmartKeyPool(api_keys=api_key)

            architect_llm = LLMFactory.create(
                provider=provider,
                api_key=key_pool,
                model_name=architect_model,
                **extras,
            )
            teacher_llm = LLMFactory.create(
                provider=provider,
                api_key=key_pool,
                model_name=teacher_model,
                **extras,
            )
            judge_llm = LLMFactory.create(
                provider=provider,
                api_key=key_pool,
                model_name=judge_model,
                **extras,
            )
            self.llm_provider = architect_llm

        self.architect = SchemaArchitect(
            llm_provider=architect_llm,
            model_name=architect_model,
        )
        self.grid_synthesizer = GridTaskSynthesizer(
            llm_provider=teacher_llm,
            model_name=teacher_model,
            context_generator=self.context_generator,
        )
        self.simula_synthesizer = SimulaTaskSynthesizer(
            llm_provider=teacher_llm,
            context_generator=self.context_generator,
            monitor=self.monitor,
        )
        self.teacher_loop = ReActTrajectoryLoop(
            llm_provider=teacher_llm,
            model_name=teacher_model,
        )
        self.judge = TrajectoryJudge(
            llm_provider=judge_llm,
            model_name=judge_model,
        )

        self.environment = DeclarativeEnvironment(
            observation_mode=self.observation_mode,
            llm_provider=teacher_llm,
        )
        self._global_primary_ids: set[str] = set()
        self.storage = storage or JSONLStorage(
            conversations_path="outputs/agent_trajectories.jsonl"
        )

    async def register_app_domain(
        self, app_name: str, app_description: str, actions: List[ToolActionSpec]
    ) -> AppDomainSpec:
        """Registers an application domain and resolves response model schemas.

        Args:
            app_name: Unique name of the app domain.
            app_description: Detailed description of domain functionality.
            actions: List of tool action endpoint specifications.

        Returns:
            AppDomainSpec: Fully resolved app domain specification.
        """
        from pydantic import BaseModel

        explicit_models: dict[str, type[BaseModel]] = {}
        unresolved_actions = []

        for act in actions:
            if act.response_model_cls:
                explicit_models[act.response_model_name] = act.response_model_cls
            else:
                unresolved_actions.append(act)

        if unresolved_actions:
            (
                app_spec,
                generated_classes,
            ) = await self.architect.generate_app_domain_schema(
                app_name=app_name,
                app_description=app_description,
                actions=unresolved_actions,
                existing_primary_ids=self._global_primary_ids,
            )
            merged_models = {**generated_classes, **explicit_models}
            full_spec = AppDomainSpec(
                app_name=app_name,
                description=app_description,
                actions=actions,
                response_models_code=app_spec.response_models_code,
            )
        else:
            merged_models = explicit_models
            full_spec = AppDomainSpec(
                app_name=app_name,
                description=app_description,
                actions=actions,
                response_models_code="# Explicit response model classes provided",
            )

        self.environment.register_app_domain(full_spec, model_classes=merged_models)
        for act in actions:
            self._global_primary_ids.add(f"{app_name}.{act.action_name}_id")
        return full_spec

    async def generate_single(self, max_turns: int = 6) -> Optional[AgentTrajectory]:
        """Synthesizes a single agent trajectory (task -> ReAct loop -> judge).

        Args:
            max_turns: Maximum reasoning turns per trajectory. Defaults to 6.

        Returns:
            Optional[AgentTrajectory]: Validated trajectory object, or None if rejected by judge.

        Raises:
            ValueError: If no app domains are registered.
        """
        if not self.environment.app_domains:
            raise ValueError(
                "No app domains registered. Call register_app_domain() first."
            )

        start_time = time.perf_counter()

        # 1. Task synthesis via Grid or Simula synthesizer
        if self.task_synthesis_mode == "simula":
            (
                task,
                initial_context,
                selected_apps,
                bucket,
            ) = await self.simula_synthesizer.synthesize_task(
                app_domains=self.environment.app_domains
            )
        else:
            (
                task,
                initial_context,
                selected_apps,
                bucket,
            ) = await self.grid_synthesizer.synthesize_task(
                app_domains=self.environment.app_domains
            )

        # 2. Seed initial context state into environment context store
        self.environment.seed_initial_context(initial_context)

        # 3. ReAct teacher trajectory loop against DeclarativeEnvironment (< 1ms tool calls)
        trajectory = await self.teacher_loop.run_trajectory(
            task=task,
            environment=self.environment,
            domain_apps=selected_apps,
        )
        trajectory.metadata["grid_bucket"] = bucket.model_dump()
        trajectory.metadata["initial_context"] = initial_context

        # 4. Trajectory Judge Quality Filtering
        verdict = await self.judge.evaluate_trajectory(trajectory)
        trajectory.judge_verdict = verdict

        elapsed = time.perf_counter() - start_time

        if self.monitor:
            self.monitor.track_generation(
                duration=elapsed,
                success=verdict.is_valid,
                turns=len(trajectory.turns),
                metadata={"task_synthesis_mode": self.task_synthesis_mode},
            )

        if verdict.is_valid:
            return trajectory
        return None

    async def generate(
        self,
        num_trajectories: int = 10,
        max_turns: int = 6,
        max_concurrency: int = 4,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[AgentTrajectory]:
        """Generates multiple synthetic agent trajectories concurrently with incremental storage & progress tracking.

        Args:
            num_trajectories: Total requested trajectories to generate. Defaults to 10.
            max_turns: Maximum ReAct turns per trajectory. Defaults to 6.
            max_concurrency: Concurrency limit for parallel workers. Defaults to 4.
            show_progress: Whether to display a tqdm terminal progress bar. Defaults to True.
            progress_callback: Optional callback receiving (completed_count, total_count).

        Returns:
            List[AgentTrajectory]: List of accepted synthetic agent trajectories.
        """
        sem = asyncio.Semaphore(max_concurrency)
        accepted_trajectories: List[AgentTrajectory] = []
        completed_count = 0

        async def _worker(pbar: Optional[Any] = None) -> Optional[AgentTrajectory]:
            nonlocal completed_count
            async with sem:
                try:
                    res = await self.generate_single(max_turns=max_turns)
                    completed_count += 1
                    if pbar:
                        pbar.update(1)
                    if progress_callback:
                        progress_callback(completed_count, num_trajectories)

                    if res is not None:
                        accepted_trajectories.append(res)
                        conv = self._trajectory_to_conversation(res)
                        # Incremental storage save so progress/file creation is never lost!
                        self.storage.save_conversations([conv])
                        if self.monitor:
                            self.monitor.log_info(
                                "Generated and stored valid agent trajectory",
                                trajectory_id=res.trajectory_id,
                            )
                    return res
                except Exception as e:
                    completed_count += 1
                    if pbar:
                        pbar.update(1)
                    if progress_callback:
                        progress_callback(completed_count, num_trajectories)
                    logger.error(
                        f"Error during trajectory generation worker: {e}", exc_info=True
                    )
                    if self.monitor:
                        self.monitor.log_error(
                            "Trajectory generation worker encountered exception",
                            error=e,
                        )
                        self.monitor.record_metric("error_rate", 1.0)
                    return None

        if show_progress:
            with tqdm(total=num_trajectories, desc="Generating Agent Traces") as pbar:
                tasks = [_worker(pbar=pbar) for _ in range(num_trajectories)]
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [_worker(pbar=None) for _ in range(num_trajectories)]
            await asyncio.gather(*tasks, return_exceptions=True)

        return accepted_trajectories

    def _trajectory_to_conversation(self, traj: AgentTrajectory) -> Conversation:
        """Converts an AgentTrajectory into AfterImage's base Conversation schema.

        Args:
            traj: Synthesized AgentTrajectory instance.

        Returns:
            Conversation: Converted Conversation object for storage.
        """
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

        metadata = traj.metadata if traj.metadata is not None else {}
        if traj.judge_verdict:
            metadata["judge_verdict"] = traj.judge_verdict.model_dump()
        metadata["trajectory_id"] = traj.trajectory_id

        return Conversation(conversations=entries, metadata=metadata)
