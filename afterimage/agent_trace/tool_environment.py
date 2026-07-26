import time
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

from .simulation_engine import DeclarativeEngine, SimulationContext
from .types import AppDomainSpec, ToolActionSpec, ToolObservation


class DeclarativeTool:
    """Wraps an API tool specification and Pydantic model into an executable local tool."""

    def __init__(
        self,
        app_name: str,
        action_spec: ToolActionSpec,
        response_model_cls: Optional[Type[BaseModel]] = None,
        engine: Optional[DeclarativeEngine] = None,
    ):
        self.app_name = app_name
        self.action_spec = action_spec
        self.response_model_cls = response_model_cls
        self.engine = engine or DeclarativeEngine()

    def execute(self, **parameters: Any) -> ToolObservation:
        """Executes the tool deterministically in sub-millisecond latency."""
        start_time = time.perf_counter()

        # 1. Record input parameters into SimulationContext as existing entity state
        for k, v in parameters.items():
            if v is not None:
                self.engine.ctx.record_entity(f"{self.app_name}.{k}", v)
                self.engine.ctx.record_entity(k, v)

        # 2. Synthesize response payload
        if self.response_model_cls:
            result_model = self.engine.generate_response(self.response_model_cls)
            output_data = result_model.model_dump(mode="json")
        else:
            # Fallback output dict if no Pydantic model registered
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


class DeclarativeEnvironment:
    """Manages registered declarative tool apps and executes tool calls across multi-turn trajectories."""

    def __init__(self, seed: Optional[int] = None):
        self.context = SimulationContext(seed=seed)
        self.engine = DeclarativeEngine(context=self.context)
        self.tools: Dict[str, DeclarativeTool] = {}  # "app.action" -> DeclarativeTool
        self.app_domains: Dict[str, AppDomainSpec] = {}

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

    def execute_tool(
        self, app: str, action: str, parameters: Dict[str, Any]
    ) -> ToolObservation:
        """Executes a tool call by app name and action name."""
        full_key = f"{app}.{action}"
        if full_key in self.tools:
            return self.tools[full_key].execute(**parameters)

        # Fallback search if app prefix omitted
        for tool_key, tool in self.tools.items():
            if (
                tool_key.endswith(f".{action}")
                or tool.action_spec.action_name == action
            ):
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
