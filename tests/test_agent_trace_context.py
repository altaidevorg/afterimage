import pytest
from afterimage.agent_trace.context import (
    CallableContextGenerator,
    CompositeContextGenerator,
    PersonaContextGenerator,
    VirtualUserContextGenerator,
)


@pytest.mark.asyncio
async def test_virtual_user_context_generator():
    gen = VirtualUserContextGenerator(locale="en_US", seed=42)
    ctx = await gen.generate_context()

    assert "user_id" in ctx
    assert "user_name" in ctx
    assert "user_email" in ctx
    assert "checking_balance" in ctx
    assert isinstance(ctx["user_id"], int)

    snippet = gen.render_prompt_snippet(ctx)
    assert "user_id" in snippet
    assert "user_name" in snippet


@pytest.mark.asyncio
async def test_persona_context_generator():
    persona_data = {"persona_name": "Coffee Enthusiast", "expertise": "expert"}
    gen = PersonaContextGenerator(persona_data)
    ctx = await gen.generate_context()

    assert "persona_context" in ctx
    assert ctx["persona_context"]["persona_name"] == "Coffee Enthusiast"


@pytest.mark.asyncio
async def test_callable_context_generator():
    gen = CallableContextGenerator(lambda: {"order_id": 9991, "status": "shipped"})
    ctx = await gen.generate_context()

    assert ctx["order_id"] == 9991
    assert ctx["status"] == "shipped"


@pytest.mark.asyncio
async def test_composite_context_generator():
    gen1 = VirtualUserContextGenerator(seed=123)
    gen2 = CallableContextGenerator(lambda: {"custom_setting": "dark_mode"})

    composite = CompositeContextGenerator([gen1, gen2])
    ctx = await composite.generate_context()

    assert "user_id" in ctx
    assert ctx["custom_setting"] == "dark_mode"
