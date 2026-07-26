import pytest
import time
from pydantic import BaseModel, Field, EmailStr

from afterimage.agent_trace.simulation_engine import (
    DeclarativeEngine,
    SimulationContext,
)
from afterimage.agent_trace.tool_environment import (
    DeclarativeEnvironment,
    DeclarativeTool,
)
from afterimage.agent_trace.types import ToolActionSpec, ToolParameterSpec


class UserProfile(BaseModel):
    user_id: int = Field(json_schema_extra={"generator": "id"})
    email: EmailStr
    full_name: str = Field(json_schema_extra={"generator": "faker:name"})
    account_balance: float = Field(json_schema_extra={"generator": "money"})


class TransactionResponse(BaseModel):
    transaction_id: int = Field(json_schema_extra={"generator": "id"})
    sender_id: int = Field(json_schema_extra={"generator": "fk:user_id"})
    receiver_id: int = Field(json_schema_extra={"generator": "fk:user_id"})
    amount: float = Field(json_schema_extra={"generator": "money"})


def test_simulation_context_entity_recording():
    ctx = SimulationContext(seed=42)
    ctx.record_entity("user.user_id", 101)
    ctx.record_entity("user.user_id", 102)

    sampled = ctx.sample_fk("user.user_id")
    assert sampled in [101, 102]


def test_declarative_engine_speed_and_compliance():
    ctx = SimulationContext(seed=123)
    engine = DeclarativeEngine(context=ctx)

    # Warmup first call (loads Faker locale data)
    _ = engine.generate_response(UserProfile)

    start = time.perf_counter()
    N = 20
    for _ in range(N):
        user = engine.generate_response(UserProfile)
    avg_elapsed_ms = ((time.perf_counter() - start) * 1000.0) / N

    assert isinstance(user.user_id, int)
    assert "@" in user.email
    assert isinstance(user.account_balance, float)
    # Average sub-millisecond execution time
    assert avg_elapsed_ms < 5.0


def test_declarative_tool_execution():
    env = DeclarativeEnvironment(seed=99)
    action_spec = ToolActionSpec(
        action_name="get_user",
        description="Fetch user profile",
        parameters=[ToolParameterSpec(name="user_id", type="int")],
        response_model_name="UserProfile",
    )
    env.register_tool("banking_app", action_spec, response_model_cls=UserProfile)

    obs = env.execute_tool("banking_app", "get_user", parameters={"user_id": 500})
    assert obs.status == "success"
    assert "user_id" in obs.observation
    assert "full_name" in obs.observation
    assert obs.latency_ms < 10.0


class ZeroConstraintModel(BaseModel):
    zero_count: int = Field(ge=0, le=0)
    zero_amount: float = Field(ge=0.0, le=0.0)


def test_numeric_zero_constraints():
    engine = DeclarativeEngine()
    res = engine.generate_response(ZeroConstraintModel)
    assert res.zero_count == 0
    assert res.zero_amount == 0.0
