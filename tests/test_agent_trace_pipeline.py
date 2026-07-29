import pytest
from unittest.mock import AsyncMock, MagicMock

from afterimage.agent_trace import (
    AsyncAgentTraceGenerator,
    GridTaskSynthesizer,
    ReActTrajectoryLoop,
    SchemaArchitect,
    ToolActionSpec,
    ToolParameterSpec,
    TrajectoryJudge,
)
from afterimage.providers.llm_providers import (
    LLMProvider,
    LLMResponse,
    StructuredLLMResponse,
)


class MockChatSession:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def asend_message(self, message: str, temperature: float = 0.7, **kwargs):
        resp_text = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else "Final Answer: Done."
        )
        self.call_count += 1
        return LLMResponse(
            text=resp_text,
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock-model",
            raw_response=None,
        )

    async def aclose(self):
        pass


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock(spec=LLMProvider)
    llm.model_name = "mock-model"

    # agenerate_content mock
    async def mock_agenerate_content(prompt, temperature=0.7, **kwargs):
        code_text = """```python
from pydantic import BaseModel, Field

class AccountResponse(BaseModel):
    account_id: int = Field(json_schema_extra={'generator': 'id'})
    account_name: str = Field(default='Savings')
```"""
        return LLMResponse(
            text=code_text,
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock-model",
            raw_response=None,
        )

    llm.agenerate_content = AsyncMock(side_effect=mock_agenerate_content)

    # agenerate_structured mock
    async def mock_agenerate_structured(prompt, schema, temperature=0.7, **kwargs):
        parsed = schema(
            grounding=0.9,
            parameter_correctness=0.9,
            loop_avoidance=1.0,
            task_completion=0.9,
            is_valid=True,
            confidence_score=0.95,
            feedback="Great trajectory",
        )
        return StructuredLLMResponse(
            text="{}",
            parsed=parsed,
            prompt_token_count=10,
            completion_token_count=10,
            total_token_count=20,
            finish_reason="stop",
            model_name="mock-model",
            raw_response=None,
        )

    llm.agenerate_structured = AsyncMock(side_effect=mock_agenerate_structured)

    # astart_chat mock
    async def mock_astart_chat(temperature=0.7, **kwargs):
        return MockChatSession(
            [
                'Thought: Need to check account details\nAction: bank.get_account\nAction Input: {"account_id": 1001}',
                "Thought: Account details obtained\nFinal Answer: Account is active.",
            ]
        )

    llm.astart_chat = AsyncMock(side_effect=mock_astart_chat)
    return llm


@pytest.mark.asyncio
async def test_schema_architect_llm_call(mock_llm_provider):
    architect = SchemaArchitect(llm_provider=mock_llm_provider)
    actions = [
        ToolActionSpec(
            action_name="get_account",
            description="Get account details",
            parameters=[ToolParameterSpec(name="account_id", type="int")],
            response_model_name="AccountResponse",
        )
    ]

    app_spec, model_classes = await architect.generate_app_domain_schema(
        app_name="bank",
        app_description="Banking App",
        actions=actions,
    )

    assert app_spec.app_name == "bank"
    assert "AccountResponse" in model_classes
    mock_llm_provider.agenerate_content.assert_called_once()
    call_kwargs = mock_llm_provider.agenerate_content.call_args.kwargs
    assert "model_name" not in call_kwargs


@pytest.mark.asyncio
async def test_grid_task_synthesizer_llm_calls(mock_llm_provider):
    synthesizer = GridTaskSynthesizer(llm_provider=mock_llm_provider)
    actions = [
        ToolActionSpec(
            action_name="get_account",
            description="Get account details",
            response_model_name="AccountResponse",
        )
    ]
    architect = SchemaArchitect(llm_provider=mock_llm_provider)
    app_spec, _ = await architect.generate_app_domain_schema(
        app_name="bank",
        app_description="Banking App",
        actions=actions,
    )

    task, initial_context, selected_apps, bucket = await synthesizer.synthesize_task(
        app_domains={"bank": app_spec}
    )

    assert isinstance(task, str)
    assert isinstance(initial_context, dict)
    assert selected_apps == ["bank"]
    assert (
        mock_llm_provider.agenerate_content.call_count == 3
    )  # 1 architect + 2 synthesizer


@pytest.mark.asyncio
async def test_react_trajectory_loop(mock_llm_provider):
    loop = ReActTrajectoryLoop(llm_provider=mock_llm_provider)
    from afterimage.agent_trace import DeclarativeEnvironment

    env = DeclarativeEnvironment()
    actions = [
        ToolActionSpec(
            action_name="get_account",
            description="Get account details",
            response_model_name="AccountResponse",
        )
    ]
    architect = SchemaArchitect(llm_provider=mock_llm_provider)
    app_spec, model_classes = await architect.generate_app_domain_schema(
        app_name="bank",
        app_description="Banking App",
        actions=actions,
    )
    env.register_app_domain(app_spec, model_classes=model_classes)

    trajectory = await loop.run_trajectory(
        task="Check my account balance",
        environment=env,
    )

    assert trajectory.task == "Check my account balance"
    assert len(trajectory.turns) >= 1
    assert trajectory.final_answer is not None
    mock_llm_provider.astart_chat.assert_called_once()


@pytest.mark.asyncio
async def test_trajectory_judge(mock_llm_provider):
    judge = TrajectoryJudge(llm_provider=mock_llm_provider)
    from afterimage.agent_trace.types import AgentTrajectory, TrajectoryTurn

    trajectory = AgentTrajectory(
        task="Check balance",
        domain_apps=["bank"],
        turns=[TrajectoryTurn(turn_id=1, agent_thought="Checking balance...")],
        final_answer="Balance is $500",
    )

    verdict = await judge.evaluate_trajectory(trajectory)

    assert verdict.is_valid is True
    assert verdict.confidence_score == 0.95
    mock_llm_provider.agenerate_structured.assert_called_once()
    call_kwargs = mock_llm_provider.agenerate_structured.call_args.kwargs
    assert "schema" in call_kwargs
    assert "response_schema" not in call_kwargs
    assert "model_name" not in call_kwargs


@pytest.mark.asyncio
async def test_trajectory_judge_hard_gate_error_rejection(mock_llm_provider):
    judge = TrajectoryJudge(llm_provider=mock_llm_provider)
    from afterimage.agent_trace.types import (
        AgentTrajectory,
        ToolCall,
        ToolObservation,
        TrajectoryTurn,
    )

    trajectory_with_error = AgentTrajectory(
        task="Check balance",
        domain_apps=["bank"],
        turns=[
            TrajectoryTurn(
                turn_id=1,
                agent_thought="Checking balance...",
                tool_call=ToolCall(app="bank", action="get_balance", parameters={}),
                observation=ToolObservation(
                    observation={"error": "Tool execution failed"}, status="error"
                ),
            )
        ],
        final_answer="Failed to check balance",
    )

    verdict = await judge.evaluate_trajectory(trajectory_with_error)

    assert verdict.is_valid is False
    assert verdict.confidence_score == 0.0
    assert "rejected" in verdict.feedback.lower()


@pytest.mark.asyncio
async def test_async_agent_trace_generator_facade(mock_llm_provider):
    generator = AsyncAgentTraceGenerator(llm_provider=mock_llm_provider)

    actions = [
        ToolActionSpec(
            action_name="get_account",
            description="Get account details",
            response_model_name="AccountResponse",
        )
    ]
    await generator.register_app_domain(
        app_name="bank",
        app_description="Banking App",
        actions=actions,
    )

    trajectory = await generator.generate_single()
    assert trajectory is not None
    assert trajectory.judge_verdict.is_valid is True
