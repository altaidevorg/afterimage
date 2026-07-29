"""
Environment-Free Synthetic Agent-Trace Dataset Generation Example.

This script demonstrates using AsyncAgentTraceGenerator (afterimage.agent_trace)
to generate high-quality synthetic multi-turn agent interaction traces grounded
in declarative app tools without requiring executable backends or real databases.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add repository root to python path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field

from afterimage.agent_trace import (
    AsyncAgentTraceGenerator,
    ToolActionSpec,
    ToolParameterSpec,
)
from afterimage.exporters import export_dataset


# --- Explicit Concrete Response Models with Parameter Echoing Annotations ---

class UserProfileResponse(BaseModel):
    user_id: int = Field(json_schema_extra={"generator": "id"})
    full_name: str = Field(json_schema_extra={"generator": "faker:name"})
    email: EmailStr = Field(json_schema_extra={"generator": "faker:email"})
    primary_checking_account_id: int = Field(json_schema_extra={"generator": "id"})
    primary_savings_account_id: int = Field(json_schema_extra={"generator": "id"})


class UserAccountItem(BaseModel):
    account_id: int = Field(json_schema_extra={"generator": "param:account_id"})
    account_type: str = Field(json_schema_extra={"generator": "enum", "values": ["checking", "savings"]})
    balance: float = Field(json_schema_extra={"generator": "money"})


class UserAccountsResponse(BaseModel):
    accounts: List[UserAccountItem] = Field(default_factory=list)


class AccountBalanceResponse(BaseModel):
    account_id: int = Field(json_schema_extra={"generator": "param:account_id"})
    account_type: str = Field(default="checking")
    total_balance: float = Field(json_schema_extra={"generator": "money"})
    available_balance: float = Field(json_schema_extra={"generator": "money"})


class TransferResponse(BaseModel):
    transfer_id: int = Field(json_schema_extra={"generator": "id"})
    status: str = Field(default="completed")
    amount: float = Field(json_schema_extra={"generator": "param:amount"})


class ExpenseRecord(BaseModel):
    expense_id: int = Field(json_schema_extra={"generator": "id"})
    user_id: int = Field(json_schema_extra={"generator": "param:user_id"})
    merchant: str = Field(json_schema_extra={"generator": "faker:company"})
    category: str = Field(json_schema_extra={"generator": "enum", "values": ["office supplies", "travel", "dining"]})
    amount: float = Field(json_schema_extra={"generator": "money"})
    description: str = Field(json_schema_extra={"generator": "faker:sentence"})


class ExpensesListResponse(BaseModel):
    expenses: List[ExpenseRecord] = Field(default_factory=list)


class CommentRecord(BaseModel):
    comment_id: int = Field(json_schema_extra={"generator": "id"})
    expense_id: int = Field(json_schema_extra={"generator": "param:expense_id"})
    author_name: str = Field(json_schema_extra={"generator": "faker:name"})
    comment_text: str = Field(json_schema_extra={"generator": "faker:sentence"})


class ExpenseCommentsResponse(BaseModel):
    comments: List[CommentRecord] = Field(default_factory=list)


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required.")
        sys.exit(1)

    print("=== AfterImage Agent Trace Dataset Generator ===")

    # 1. Initialize AsyncAgentTraceGenerator facade (observation_mode="faker" or "llm")
    generator = AsyncAgentTraceGenerator(
        api_key=api_key,
        architect_model="gemini-3.6-flash",
        teacher_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.6-flash",
        observation_mode="llm",  # Preferred production mode (ESAT paper LLM observation synthesis). Use "faker" for experimental local mode.
    )

    # 2. Define App Domain Endpoints for Banking App (Discovery + Action endpoints)
    banking_actions = [
        ToolActionSpec(
            action_name="get_current_user_profile",
            description="Returns current logged-in user profile details including user_id, name, primary_checking_account_id, and primary_savings_account_id.",
            parameters=[],  # Parameterless self-discovery endpoint!
            response_model_name="UserProfileResponse",
            response_model_cls=UserProfileResponse,
        ),
        ToolActionSpec(
            action_name="list_user_accounts",
            description="Lists checking and savings accounts for a user ID.",
            parameters=[
                ToolParameterSpec(name="user_id", type="int", description="User ID")
            ],
            response_model_name="UserAccountsResponse",
            response_model_cls=UserAccountsResponse,
        ),
        ToolActionSpec(
            action_name="get_account_balance",
            description="Returns total and available balance for a user account.",
            parameters=[
                ToolParameterSpec(name="account_id", type="int", description="User Account ID")
            ],
            response_model_name="AccountBalanceResponse",
            response_model_cls=AccountBalanceResponse,
        ),
        ToolActionSpec(
            action_name="transfer_money",
            description="Transfers funds from sender account to recipient account.",
            parameters=[
                ToolParameterSpec(name="sender_id", type="int", description="Sender Account ID"),
                ToolParameterSpec(name="receiver_id", type="int", description="Receiver Account ID"),
                ToolParameterSpec(name="amount", type="float", description="Amount to transfer"),
            ],
            response_model_name="TransferResponse",
            response_model_cls=TransferResponse,
        ),
    ]

    # 3. Define App Domain Endpoints for Expenses App
    expenses_actions = [
        ToolActionSpec(
            action_name="list_expenses",
            description="Lists recent user expense transactions.",
            parameters=[
                ToolParameterSpec(name="user_id", type="int", description="User ID"),
                ToolParameterSpec(name="category", type="str", description="Expense category filter", required=False),
            ],
            response_model_name="ExpensesListResponse",
            response_model_cls=ExpensesListResponse,
        ),
        ToolActionSpec(
            action_name="get_expense_comments",
            description="Retrieves comments posted on a specific expense entry.",
            parameters=[
                ToolParameterSpec(name="expense_id", type="int", description="Expense ID")
            ],
            response_model_name="ExpenseCommentsResponse",
            response_model_cls=ExpenseCommentsResponse,
        ),
    ]

    # 4. Register App Domains (LLM Schema Architect + Static AST Verifier)
    print("\n[Phase 1 & 2] Running Schema Architect and Static Invariant Verifier...")
    await generator.register_app_domain(
        app_name="banking_app",
        app_description="Personal banking and peer-to-peer transfers application.",
        actions=banking_actions,
    )
    await generator.register_app_domain(
        app_name="expenses_app",
        app_description="Personal expense tracking and team comment app.",
        actions=expenses_actions,
    )

    print("Schemas successfully generated and verified cleanly!")

    # 5. Generate Synthetic Trajectories
    num_trajectories = 4
    output_jsonl = "outputs/agent_trajectories_demo.jsonl"

    print(f"\n[Phase 4] Generating {num_trajectories} agent trajectories in parallel...")
    trajectories = await generator.generate(
        num_trajectories=num_trajectories,
        max_turns=5,
        max_concurrency=4,
    )

    print(f"\nSuccessfully generated {len(trajectories)} accepted trajectories.")
    for idx, traj in enumerate(trajectories, 1):
        print(f"\nTrajectory {idx} ID: {traj.trajectory_id}")
        print(f"  Task: {traj.task}")
        print(f"  Turns Count: {len(traj.turns)}")
        if traj.judge_verdict:
            print(f"  Judge Verdict: Accepted (Confidence: {traj.judge_verdict.confidence_score})")

    # 6. Export Dataset to Agent SFT Format
    output_path = export_dataset(
        input_path="outputs/agent_trajectories.jsonl",
        format_name="agent_sft",
        output_path="outputs/agent_sft_dataset_demo.jsonl",
    )
    print(f"\nExported dataset to SFT messages format: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
