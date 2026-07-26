import json
import re
from typing import List, Optional

from ..providers.llm_providers import LLMProvider
from .tool_environment import DeclarativeEnvironment
from .types import AgentTrajectory, ToolCall, TrajectoryTurn


REACT_TEACHER_SYSTEM_PROMPT = """You are an expert ReAct AI Assistant capable of invoking local declarative API tools to fulfill user requests.

AVAILABLE TOOLS:
{tool_descriptions}

FORMAT INSTRUCTIONS:
To interact with tools, use the following turn structure:
Thought: <Explain your step-by-step reasoning>
Action: <app.action_name>
Action Input: {{"parameter_name": value}}

When you have gathered all necessary information or completed the user request, output:
Thought: <Final reasoning summary>
Final Answer: <Your clear, complete response to the user>
"""


class ReActTrajectoryLoop:
    """ReAct Teacher Agent execution loop running against sub-millisecond local DeclarativeEnvironment."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.5-flash-lite",
        max_turns: int = 6,
    ):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.max_turns = max_turns

    async def run_trajectory(
        self,
        task: str,
        environment: DeclarativeEnvironment,
        domain_apps: Optional[List[str]] = None,
    ) -> AgentTrajectory:
        """Executes a multi-turn ReAct trajectory for the given user task."""
        tool_descriptions = environment.get_tool_prompt_descriptions()
        system_prompt = REACT_TEACHER_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions
        )

        chat_session = self.llm_provider.start_chat_session(
            system_instruction=system_prompt,
            model_name=self.model_name,
        )

        turns: List[TrajectoryTurn] = []
        final_answer: Optional[str] = None
        current_input = f"User Request: {task}"

        try:
            for turn_idx in range(1, self.max_turns + 1):
                response = await chat_session.asend_message(
                    message=current_input,
                    temperature=0.3,
                )
                raw_text = response.text.strip()

                thought, tool_call, is_final, answer = self._parse_react_response(
                    raw_text
                )

                if is_final or answer:
                    final_answer = answer or thought
                    turns.append(
                        TrajectoryTurn(
                            turn_id=turn_idx,
                            agent_thought=thought,
                            tool_call=None,
                            observation=None,
                        )
                    )
                    break

                if tool_call:
                    obs = environment.execute_tool(
                        app=tool_call.app,
                        action=tool_call.action,
                        parameters=tool_call.parameters,
                    )
                    turns.append(
                        TrajectoryTurn(
                            turn_id=turn_idx,
                            agent_thought=thought,
                            tool_call=tool_call,
                            observation=obs,
                        )
                    )
                    current_input = f"Observation: {json.dumps(obs.observation)}"
                else:
                    # If agent didn't output a valid action or final answer, default turn
                    turns.append(
                        TrajectoryTurn(
                            turn_id=turn_idx,
                            agent_thought=thought,
                            tool_call=None,
                            observation=None,
                        )
                    )
                    final_answer = raw_text
                    break
        finally:
            if hasattr(chat_session, "close"):
                chat_session.close()

        return AgentTrajectory(
            task=task,
            domain_apps=domain_apps or list(environment.app_domains.keys()),
            turns=turns,
            final_answer=final_answer,
        )

    def _parse_react_response(
        self, text: str
    ) -> tuple[str, Optional[ToolCall], bool, Optional[str]]:
        """Parses ReAct agent response into thought, tool call, final answer flag, and answer string."""
        thought = ""
        action_str = None
        action_input_dict = {}
        final_answer = None

        if "Final Answer:" in text:
            parts = text.split("Final Answer:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            final_answer = parts[1].strip()
            return thought, None, True, final_answer

        # Parse Thought
        if "Thought:" in text:
            thought_part = text.split("Action:", 1)[0] if "Action:" in text else text
            thought = thought_part.replace("Thought:", "").strip()
        else:
            thought = text

        # Parse Action & Action Input
        action_match = re.search(r"Action:\s*([\w\.\-]+)", text)
        if action_match:
            action_str = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if input_match:
            try:
                action_input_dict = json.loads(input_match.group(1).strip())
            except Exception:
                action_input_dict = {}

        if action_str:
            if "." in action_str:
                app_name, action_name = action_str.split(".", 1)
            else:
                app_name, action_name = "default_app", action_str

            return (
                thought,
                ToolCall(
                    app=app_name, action=action_name, parameters=action_input_dict
                ),
                False,
                None,
            )

        return thought, None, False, None
