# Environment-Free Agent Traces (`afterimage.agent_trace`)

`afterimage.agent_trace` provides an environment-free synthetic data generation pipeline for training API-calling AI agents. It combines the methodology from the **ESAT** research paper (*Environment-free Synthetic Data Generation for API-Calling Agents*, arXiv:2607.16900) with a sub-millisecond local **Declarative Tool Simulation Framework**.

Instead of setting up complex, executable backend applications or real databases, `afterimage.agent_trace` uses LLMs as offline schema architects and online teacher/judge agents, while delegating tool observation generation to a deterministic, local Python simulation engine.

---

## Key Benefits

- **Sub-Millisecond Execution:** Tool calls complete locally in `< 1 ms` (vs. 1,500 ms – 4,000 ms for LLM-simulated passes).
- **Zero Simulator Hallucinations:** Pydantic V2 response models guarantee 100% schema compliance.
- **Stateful Entity Context:** `SimulationContext` maintains entity pools across multi-turn trajectories so foreign key relationships (`user_id`, `order_id`, etc.) remain consistent.
- **60% Token Cost Reduction:** Eliminates simulator prompting overhead during multi-turn ReAct loops.
- **360-Bucket Combinatorial Grid:** Guarantees task diversity across difficulties, action types, task foci, and application counts.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   LLM Schema Architect │ (gemini-3.6-flash)
                          └───────────┬────────────┘
                                      │ (Generates Pydantic response models)
                                      ▼
                          ┌────────────────────────┐
                          │  Static AST Verifier   │ (6 Structural Invariants)
                          └───────────┬────────────┘
                                      │ (Self-correction feedback loop)
                                      ▼
                          ┌────────────────────────┐
                          │  Declarative Engine    │ (< 1ms local Python simulation)
                          └───────────┬────────────┘
                                      │
     ┌────────────────────────┐       │       ┌────────────────────────┐
     │  360-Bucket Grid Task  ├───────┴───────►   ReAct Teacher Loop   │ (gemini-3.5-flash-lite)
     │       Synthesizer      │               └───────────┬────────────┘
     └────────────────────────┘                           │
                                                          ▼
                                              ┌────────────────────────┐
                                              │    Trajectory Judge    │ (gemini-3.6-flash)
                                              └───────────┬────────────┘
                                                          │
                                                          ▼
                                              ┌────────────────────────┐
                                              │  JSONL / SQL Storage   │
                                              └────────────────────────┘
```

---

## Getting Started Example

```python
import asyncio
import os
from afterimage.agent_trace import (
    AsyncAgentTraceGenerator,
    ToolActionSpec,
    ToolParameterSpec,
)
from afterimage.exporters import export_dataset

async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    # 1. Initialize the Generator with Provider Models
    generator = AsyncAgentTraceGenerator(
        api_key=api_key,
        architect_model="gemini-3.6-flash",
        teacher_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.6-flash",
    )

    # 2. Define App Domain Endpoints
    actions = [
        ToolActionSpec(
            action_name="get_user_profile",
            description="Retrieve profile details for a user.",
            parameters=[
                ToolParameterSpec(name="user_id", type="int", description="User ID")
            ],
            response_model_name="UserProfileResponse",
        ),
        ToolActionSpec(
            action_name="create_order",
            description="Create a new order for a customer.",
            parameters=[
                ToolParameterSpec(name="user_id", type="int", description="Customer User ID"),
                ToolParameterSpec(name="item_name", type="str", description="Item name"),
                ToolParameterSpec(name="price", type="float", description="Order price"),
            ],
            response_model_name="OrderResponse",
        ),
    ]

    # 3. Register Domain Schema (Runs LLM Architect + AST Verification)
    await generator.register_app_domain(
        app_name="e_commerce_app",
        app_description="Online shopping and order management platform.",
        actions=actions,
    )

    # 4. Generate Synthetic Agent Trajectories Concurrently
    trajectories = await generator.generate(
        num_trajectories=10,
        max_turns=5,
        max_concurrency=4,
    )

    print(f"Generated {len(trajectories)} valid synthetic agent trajectories.")

    # 5. Export Trajectories to SFT Messages Format
    export_dataset(
        input_path="outputs/agent_trajectories.jsonl",
        format_name="agent_sft",
        output_path="outputs/agent_sft_dataset.jsonl",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Observation Generation Modes (`llm` vs `faker`)

`afterimage.agent_trace` supports two observation generation modes for synthetic tool responses, configurable via `AsyncAgentTraceGenerator(observation_mode=...)`:

```python
# Mode 1: Preferred production mode (Original ESAT Paper LLM-driven structured observation synthesis)
generator = AsyncAgentTraceGenerator(api_key=api_key, observation_mode="llm")

# Mode 2: Experimental local sub-millisecond declarative engine
generator = AsyncAgentTraceGenerator(api_key=api_key, observation_mode="faker")
```

| Observation Mode | Status | Latency | Token Cost | Mechanism & Best Use Case |
|---|---|---|---|---|
| **`llm`** (Default) | **Preferred / Production** | ~300ms – 800ms per turn | Token cost for LLM synthesis | Uses `LLMObservationSynthesizer` to generate structured, realistic JSON payloads guided by target Pydantic schemas, tool parameters, past turn history, and initial state context. |
| **`faker`** | **Experimental** | Sub-millisecond (`< 1 ms`) | **0 tokens** for tool responses | Uses local Pydantic synthesis, Faker generators, parameter echoing annotations, and stateful `SimulationContext` context stores for zero-cost execution. |

---

## Generator Annotations Protocol (for `faker` Mode)

In `faker` mode, referential integrity and parameter matching are guaranteed using explicit generator annotations inside Pydantic field schemas (`json_schema_extra={"generator": "..."}`):

```python
class AccountBalanceResponse(BaseModel):
    # Echoes the input 'account_id' argument directly into the response field
    account_id: int = Field(json_schema_extra={"generator": "param:account_id"})
    account_type: str = Field(default="checking")
    total_balance: float = Field(json_schema_extra={"generator": "money"})
    available_balance: float = Field(json_schema_extra={"generator": "money"})

class TransferResponse(BaseModel):
    transfer_id: int = Field(json_schema_extra={"generator": "id"})
    status: str = Field(default="completed")
    # Echoes the input 'amount' argument directly into the response field
    amount: float = Field(json_schema_extra={"generator": "param:amount"})
```

### Supported Annotation Tags

- **`param:<param_name>` / `echo`**: Echoes input argument values (`account_id`, `amount`, `user_id`, `expense_id`) directly into response fields.
- **`state:account_balance`**: Interacts with persistent account balance tables in `SimulationContext` (e.g. deducting transfer amounts).
- **`fk:<entity>.<field>`**: Samples foreign key identifiers from state lookup pools across trajectory turns.
- **`id`**: Generates a primary integer ID and records it into `SimulationContext`.
- **`money`**: Generates realistic monetary float amounts.
- **`enum`**: Selects a value from `json_schema_extra={"values": [...]}`.
- **`faker:<method>`**: Generates Faker strings (e.g., `faker:name`, `faker:email`, `faker:company`).

---

## Explicit Response Models (`response_model_cls`)

Pass explicit Pydantic response models directly into `ToolActionSpec` to eliminate LLM schema generation overhead:

```python
ToolActionSpec(
    action_name="get_account_balance",
    description="Returns total and available balance for a user account.",
    parameters=[ToolParameterSpec(name="account_id", type="int", description="Account ID")],
    response_model_name="AccountBalanceResponse",
    response_model_cls=AccountBalanceResponse,  # Explicit class attached!
)
```

---

## Recommended Model Configuration

| Component | Default Model | Purpose |
|---|---|---|
| **Schema Architect** | `gemini-3.6-flash` | High-quality Pydantic response code generation with metadata tags. |
| **Task Synthesizer & Rewriter** | `gemini-3.5-flash-lite` | Ultra-fast combinatorial grid task synthesis & natural language rewriter. |
| **ReAct Teacher Agent** | `gemini-3.5-flash-lite` | Multi-turn reasoning & tool execution loop. |
| **LLM Observation Synthesizer** | `gemini-3.5-flash-lite` | Structured tool observation generation when `observation_mode="llm"`. |
| **Trajectory Judge** | `gemini-3.6-flash` | 9-point quality rubric trajectory filtering. |

---

## CLI Command Usage

Generate agent trajectories without writing Python code using the `afterimage` CLI:

```bash
afterimage agent-trace \
  --app-name "banking_app" \
  --app-desc "Customer money transfer and account balance app." \
  -n 10 \
  -o "outputs/agent_trajectories.jsonl"
```
