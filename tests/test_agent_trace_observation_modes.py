import pytest
from typing import Any, Dict
from pydantic import BaseModel, Field

from afterimage.agent_trace import (
    DeclarativeEngine,
    DeclarativeEnvironment,
    DeclarativeTool,
    LLMObservationSynthesizer,
    SimulationContext,
    ToolActionSpec,
    ToolParameterSpec,
)


class AccountBalanceResponse(BaseModel):
    account_id: int = Field(json_schema_extra={"generator": "param:account_id"})
    total_balance: float = Field(default=1000.0)
    available_balance: float = Field(default=900.0)


class TransferResponse(BaseModel):
    transfer_id: int = Field(json_schema_extra={"generator": "id"})
    status: str = Field(default="completed")
    amount: float = Field(json_schema_extra={"generator": "param:amount"})


class DummyLLMProvider:
    """Mock LLM provider for testing LLM observation mode."""

    def __init__(self, response_dict: Dict[str, Any]):
        self.response_dict = response_dict

    async def agenerate_structured(
        self, prompt: str, schema: type, model_name: str = ""
    ) -> Any:
        return schema(**self.response_dict)

    async def agenerate_content(
        self, prompt: str, model_name: str = "", system_instruction: str = ""
    ) -> Any:
        class DummyContent:
            content = '{"account_id": 9999, "total_balance": 100.0, "available_balance": 90.0}'

        return DummyContent()


def test_faker_mode_parameter_echoing():
    """Tests that DeclarativeEngine echoes incoming parameters into responses cleanly."""
    ctx = SimulationContext(seed=42)
    engine = DeclarativeEngine(context=ctx)

    # 1. Parameter Echoing on get_account_balance
    bal_res = engine.generate_response(AccountBalanceResponse, parameters={"account_id": 68635})
    assert bal_res.account_id == 68635

    # 2. Parameter Echoing on transfer_money
    tr_res = engine.generate_response(TransferResponse, parameters={"sender_id": 68635, "receiver_id": 34337, "amount": 150.0})
    assert tr_res.amount == 150.0


def test_stateful_account_balance_deductions():
    """Tests stateful transfer deductions across balance queries."""
    env = DeclarativeEnvironment(seed=42, observation_mode="faker")
    bal_action = ToolActionSpec(
        action_name="get_account_balance",
        description="Get balance",
        parameters=[ToolParameterSpec(name="account_id", type="int")],
        response_model_name="AccountBalanceResponse",
    )
    tr_action = ToolActionSpec(
        action_name="transfer_money",
        description="Transfer money",
        parameters=[
            ToolParameterSpec(name="sender_id", type="int"),
            ToolParameterSpec(name="receiver_id", type="int"),
            ToolParameterSpec(name="amount", type="float"),
        ],
        response_model_name="TransferResponse",
    )

    env.register_tool("banking_app", bal_action, response_model_cls=AccountBalanceResponse)
    env.register_tool("banking_app", tr_action, response_model_cls=TransferResponse)

    # Turn 1: Initial balance check for account 68635
    obs1 = env.execute_tool("banking_app", "get_account_balance", {"account_id": 68635})
    initial_avail = obs1.observation["available_balance"]

    # Turn 2: Transfer $100 from 68635 to 34337
    obs2 = env.execute_tool(
        "banking_app",
        "transfer_money",
        {"sender_id": 68635, "receiver_id": 34337, "amount": 100.0},
    )
    assert obs2.observation["amount"] == 100.0

    # Turn 3: Post-transfer balance check for account 68635
    obs3 = env.execute_tool("banking_app", "get_account_balance", {"account_id": 68635})
    new_avail = obs3.observation["available_balance"]

    assert round(initial_avail - new_avail, 2) == 100.0


@pytest.mark.asyncio
async def test_llm_observation_mode():
    """Tests LLM observation synthesis mode integration with DeclarativeTool."""
    mock_llm = DummyLLMProvider({"account_id": 77777, "total_balance": 500.0, "available_balance": 450.0})
    env = DeclarativeEnvironment(observation_mode="llm", llm_provider=mock_llm)

    bal_action = ToolActionSpec(
        action_name="get_account_balance",
        description="Get balance",
        parameters=[ToolParameterSpec(name="account_id", type="int")],
        response_model_name="AccountBalanceResponse",
    )
    env.register_tool("banking_app", bal_action, response_model_cls=AccountBalanceResponse)

    obs = await env.aexecute_tool("banking_app", "get_account_balance", {"account_id": 77777})
    assert obs.status == "success"
    assert obs.observation["account_id"] == 77777
