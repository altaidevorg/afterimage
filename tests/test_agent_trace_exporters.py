import pytest
from afterimage.exporters import (
    to_agent_dpo,
    to_anthropic_tools,
    to_hermes_tools,
    to_openai_tools,
)


@pytest.fixture
def mock_trajectory_rows():
    return [
        {
            "conversations": [
                {"role": "user", "content": "Check balance for account 1001"},
                {
                    "role": "assistant",
                    "content": 'Thought: Need balance.\nAction: banking.get_account_balance\nAction Input: {"account_id": 1001}',
                },
                {
                    "role": "user",
                    "content": 'Observation: {"balance": 1500.0, "status": "active"}',
                },
                {
                    "role": "assistant",
                    "content": "Thought: Done.\nFinal Answer: Your checking balance is $1500.0.",
                },
            ],
            "metadata": {"trajectory_id": "traj_test_123"},
        }
    ]


def test_to_openai_tools(mock_trajectory_rows):
    converted = to_openai_tools(mock_trajectory_rows)
    assert len(converted) == 1
    msgs = converted[0]["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "tool_calls" in msgs[1]
    assert (
        msgs[1]["tool_calls"][0]["function"]["name"] == "banking__get_account_balance"
    )
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]


def test_to_anthropic_tools(mock_trajectory_rows):
    converted = to_anthropic_tools(mock_trajectory_rows)
    assert len(converted) == 1
    msgs = converted[0]["messages"]
    assert msgs[1]["role"] == "assistant"
    blocks = msgs[1]["content"]
    tool_use = [b for b in blocks if b["type"] == "tool_use"][0]
    assert tool_use["name"] == "banking__get_account_balance"
    assert msgs[2]["content"][0]["type"] == "tool_result"


def test_to_hermes_tools(mock_trajectory_rows):
    converted = to_hermes_tools(mock_trajectory_rows)
    assert len(converted) == 1
    msgs = converted[0]["messages"]
    assert "<tool_call>" in msgs[1]["content"]
    assert "banking__get_account_balance" in msgs[1]["content"]


def test_to_agent_dpo(mock_trajectory_rows):
    converted = to_agent_dpo(mock_trajectory_rows)
    assert len(converted) == 1
    assert "prompt" in converted[0]
    assert "chosen" in converted[0]
    assert "rejected" in converted[0]
    assert converted[0]["prompt"] == "Check balance for account 1001"
