from unittest.mock import AsyncMock, MagicMock
import pytest
from afterimage.agent_trace import (
    AsyncAgentTraceGenerator,
    SimulaTaskSynthesizer,
    ToolActionSpec,
    ToolParameterSpec,
    VirtualUserContextGenerator,
)


@pytest.mark.asyncio
async def test_simula_task_synthesizer():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "PROMPT: Transfer Money to account 1002\nINITIAL_CONTEXT: ```json\n{\"user_id\": 101}\n```"
    mock_llm.agenerate_content = AsyncMock(return_value=mock_response)

    synthesizer = SimulaTaskSynthesizer(
        llm_provider=mock_llm,
        context_generator=VirtualUserContextGenerator(seed=42),
    )

    actions = [
        ToolActionSpec(
            action_name="get_account_balance",
            description="Returns balance",
            parameters=[ToolParameterSpec(name="account_id", type="int")],
            response_model_name="AccountBalanceResponse",
        )
    ]
    app_domains = {
        "banking": MagicMock(app_name="banking", description="Banking app", actions=actions)
    }

    # Mock build_taxonomy to return bundle with FactorTaxonomy
    mock_bundle = MagicMock()
    mock_node_root = MagicMock(id="root_1", label="Banking Operations", parent_id=None)
    mock_node_child = MagicMock(id="node_1", label="Transfer Money", parent_id="root_1")
    mock_taxonomy = MagicMock(
        root_id="root_1",
        nodes={"root_1": mock_node_root, "node_1": mock_node_child},
    )
    mock_bundle.taxonomies = [mock_taxonomy]
    synthesizer.simula.build_taxonomy = AsyncMock(return_value=mock_bundle)

    task_text, context, selected_apps, bucket = await synthesizer.synthesize_task(app_domains)

    assert isinstance(task_text, str)
    assert "Transfer Money" in task_text
    assert "user_id" in context
    assert len(selected_apps) >= 1
    assert bucket is not None


@pytest.mark.asyncio
async def test_async_agent_trace_generator_simula_mode():
    mock_llm = MagicMock()
    generator = AsyncAgentTraceGenerator(
        llm_provider=mock_llm,
        task_synthesis_mode="simula",
    )
    assert generator.task_synthesis_mode == "simula"
    assert isinstance(generator.simula_synthesizer, SimulaTaskSynthesizer)
