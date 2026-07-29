import time
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from .llm_observation_synthesizer import LLMObservationSynthesizer
from .simulation_engine import DeclarativeEngine, SimulationContext
from .types import AppDomainSpec, ObservationMode, ToolActionSpec, ToolObservation
from afterimage.providers.llm_providers import LLMProvider


class DeclarativeTool:
    """Wraps an API tool specification and Pydantic model into an executable local tool."""

    def __init__(
        self,
        app_name: str,
        action_spec: ToolActionSpec,
        response_model_cls: Optional[Type[BaseModel]] = None,
        engine: Optional[DeclarativeEngine] = None,
        observation_mode: ObservationMode = "faker",
        llm_synthesizer: Optional[LLMObservationSynthesizer] = None,
    ):
        self.app_name = app_name
        self.action_spec = action_spec
        self.response_model_cls = response_model_cls
        self.engine = engine or DeclarativeEngine()
        self.observation_mode = observation_mode
        self.llm_synthesizer = llm_synthesizer

    async def aexecute(
        self,
        initial_context: str = "",
        turns: Optional[List[Dict[str, Any]]] = None,
        **parameters: Any,
    ) -> ToolObservation:
        """Executes the tool asynchronously supporting both faker sub-millisecond mode and LLM observation synthesis."""
        start_time = time.perf_counter()

        try:
            # 1. Record input parameters into SimulationContext as existing entity state
            for k, v in parameters.items():
                if v is not None:
                    self.engine.ctx.record_entity(f"{self.app_name}.{k}", v)
                    self.engine.ctx.record_entity(k, v)

            # 2. Check for stateful transfer action mutations
            if "transfer" in self.action_spec.action_name.lower():
                s_id = parameters.get("sender_id") or parameters.get("sender_account_id")
                r_id = parameters.get("receiver_id") or parameters.get("recipient_id") or parameters.get("recipient_account_id")
                amt = parameters.get("amount")
                if s_id is not None and r_id is not None and amt is not None:
                    self.engine.ctx.apply_transfer(s_id, r_id, amt)

            # 3. Synthesize response payload via LLM or local Faker engine
            if self.observation_mode == "llm" and self.llm_synthesizer and self.response_model_cls:
                result_model = await self.llm_synthesizer.synthesize_observation(
                    app_name=self.app_name,
                    action_name=self.action_spec.action_name,
                    action_description=self.action_spec.description,
                    response_model_cls=self.response_model_cls,
                    parameters=parameters,
                    initial_context=initial_context,
                    turns=turns,
                )
                output_data = result_model.model_dump(mode="json")
            elif self.response_model_cls:
                result_model = self.engine.generate_response(
                    self.response_model_cls, parameters=parameters
                )
                output_data = result_model.model_dump(mode="json")
            else:
                output_data = {
                    "status": "success",
                    "message": f"Action {self.action_spec.action_name} executed successfully",
                }

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolObservation(
                observation=output_data,
                status="success",
                latency_ms=round(elapsed_ms, 3),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolObservation(
                observation={"error": f"Tool execution failed: {str(e)}"},
                status="error",
                latency_ms=round(elapsed_ms, 3),
            )

    def execute(self, **parameters: Any) -> ToolObservation:
        """Synchronous wrapper for local faker execution mode."""
        start_time = time.perf_counter()
        try:
            for k, v in parameters.items():
                if v is not None:
                    self.engine.ctx.record_entity(f"{self.app_name}.{k}", v)
                    self.engine.ctx.record_entity(k, v)

            if "transfer" in self.action_spec.action_name.lower():
                s_id = parameters.get("sender_id") or parameters.get("sender_account_id")
                r_id = parameters.get("receiver_id") or parameters.get("recipient_id") or parameters.get("recipient_account_id")
                amt = parameters.get("amount")
                if s_id is not None and r_id is not None and amt is not None:
                    self.engine.ctx.apply_transfer(s_id, r_id, amt)

            if self.response_model_cls:
                result_model = self.engine.generate_response(
                    self.response_model_cls, parameters=parameters
                )
                output_data = result_model.model_dump(mode="json")
            else:
                output_data = {
                    "status": "success",
                    "message": f"Action {self.action_spec.action_name} executed successfully",
                }

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolObservation(
                observation=output_data,
                status="success",
                latency_ms=round(elapsed_ms, 3),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolObservation(
                observation={"error": f"Tool execution failed: {str(e)}"},
                status="error",
                latency_ms=round(elapsed_ms, 3),
            )


class DeclarativeEnvironment:
    """Manages registered declarative tool apps and executes tool calls across multi-turn trajectories."""

    def __init__(
        self,
        seed: Optional[int] = None,
        observation_mode: ObservationMode = "faker",
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.observation_mode = observation_mode
        self.llm_provider = llm_provider
        self.llm_synthesizer = (
            LLMObservationSynthesizer(llm_provider) if llm_provider else None
        )
        self.context = SimulationContext(seed=seed)
        self.engine = DeclarativeEngine(context=self.context)
        self.tools: Dict[str, DeclarativeTool] = {}
        self.app_domains: Dict[str, AppDomainSpec] = {}

    def seed_initial_context(self, initial_context: Dict[str, Any]) -> None:
        """Seeds initial task context data into context store."""
        if not initial_context:
            return

        for k, v in initial_context.items():
            if isinstance(v, list):
                for item in v:
                    self.context.record_entity(k, item)
            else:
                self.context.record_entity(k, v)

    def register_tool(
        self,
        app_name: str,
        action_spec: ToolActionSpec,
        response_model_cls: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Registers an individual executable tool."""
        full_key = f"{app_name}.{action_spec.action_name}"
        tool = DeclarativeTool(
            app_name=app_name,
            action_spec=action_spec,
            response_model_cls=response_model_cls,
            engine=self.engine,
            observation_mode=self.observation_mode,
            llm_synthesizer=self.llm_synthesizer,
        )
        self.tools[full_key] = tool

    def register_app_domain(
        self,
        app_spec: AppDomainSpec,
        model_classes: Optional[Dict[str, Type[BaseModel]]] = None,
    ) -> None:
        """Registers an entire AppDomainSpec and attaches resolved Pydantic response models."""
        self.app_domains[app_spec.app_name] = app_spec
        models_map = model_classes or {}

        for action in app_spec.actions:
            model_cls = models_map.get(action.response_model_name)
            self.register_tool(app_spec.app_name, action, response_model_cls=model_cls)

    async def aexecute_tool(
        self,
        app: str,
        action: str,
        parameters: Dict[str, Any],
        initial_context: str = "",
        turns: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolObservation:
        """Asynchronously executes a tool call using active observation mode."""
        full_key = f"{app}.{action}"
        target_tool = None
        if full_key in self.tools:
            target_tool = self.tools[full_key]
        else:
            for tool_key, tool in self.tools.items():
                if tool_key.endswith(f".{action}") or tool.action_spec.action_name == action:
                    target_tool = tool
                    break

        if target_tool:
            return await target_tool.aexecute(
                initial_context=initial_context, turns=turns, **parameters
            )

        return ToolObservation(
            observation={"error": f"Tool '{app}.{action}' not found in environment."},
            status="error",
            latency_ms=0.0,
        )

    def execute_tool(
        self, app: str, action: str, parameters: Dict[str, Any]
    ) -> ToolObservation:
        """Executes a tool call synchronously."""
        full_key = f"{app}.{action}"
        if full_key in self.tools:
            return self.tools[full_key].execute(**parameters)

        for tool_key, tool in self.tools.items():
            if tool_key.endswith(f".{action}") or tool.action_spec.action_name == action:
                return tool.execute(**parameters)

        return ToolObservation(
            observation={"error": f"Tool '{app}.{action}' not found in environment."},
            status="error",
            latency_ms=0.0,
        )

    def get_tool_prompt_descriptions(self) -> str:
        """Formats all registered tools into clean API prompt documentation for ReAct teacher agents."""
        lines = []
        for app_name, app_spec in self.app_domains.items():
            lines.append(f"### App: {app_name}")
            lines.append(f"Description: {app_spec.description}\nActions:")
            for action in app_spec.actions:
                params_str = ", ".join(
                    [
                        f"{p.name}: {p.type}"
                        + (" (optional)" if not p.required else "")
                        for p in action.parameters
                    ]
                )
                lines.append(
                    f"- `{app_name}.{action.action_name}({params_str})`: {action.description}"
                )
            lines.append("")
        return "\n".join(lines)
