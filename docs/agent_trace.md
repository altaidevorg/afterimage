# Environment-Free Agent Traces (`afterimage.agent_trace`)

`afterimage.agent_trace` provides an environment-free synthetic data generation pipeline for training API-calling AI agents. It combines the methodology from the **ESAT** research paper (*Environment-free Synthetic Data Generation for API-Calling Agents*, arXiv:2607.16900) and **Simula** paper (*Reasoning-Driven Synthetic Data Generation and Evaluation*, arXiv:2603.29791) with a sub-millisecond local **Declarative Tool Simulation Framework**.

Instead of setting up complex, executable backend applications or real databases, `afterimage.agent_trace` uses LLMs as offline schema architects and online teacher/judge agents, while delegating tool observation generation to a deterministic, local Python simulation engine.

---

## Key Benefits

- **Sub-Millisecond Execution:** Tool calls complete locally in `< 1 ms` (vs. 1,500 ms – 4,000 ms for LLM-simulated passes).
- **Zero Simulator Hallucinations:** Pydantic V2 response models guarantee 100% schema compliance.
- **Dynamic Virtual User Identities:** Synthesizes localized identity context profiles (`VirtualUserContextGenerator`) using Faker for grounded entity references across multi-turn trajectories.
- **Simula Integration:** Integrates `afterimage.simula` factor taxonomy trees and meta-prompt diversification for deep, challenging agent tasks (`task_synthesis_mode="simula"`).
- **Multi-Format Agent Tool Exporters:** Export trajectories natively into OpenAI Tool Calls, Anthropic Messages API, Hermes XML `<tool_call>`, and DPO preference pair formats.
- **Terminal Progress Indicators:** Terminal progress bar (`tqdm.asyncio`) and custom callback monitoring (`show_progress=True`).

---

## Architecture Overview

```
                                  ┌───────────────────────────┐
                                  │   BaseContextGenerator    │
                                  └─────────────┬─────────────┘
                                                │
          ┌───────────────────────┬─────────────┴─────────────┬─────────────────────────┐
          │                       │                           │                         │
┌─────────┴──────────────┐ ┌──────┴────────────────┐ ┌────────┴───────────────┐ ┌───────┴────────────────┐
│VirtualUserContextGen   │ │PersonaContextGenerator│ │CallableContextGen     │ │CompositeContextGen     │
│(Faker Identities/IDs)  │ │(Persona Integration)  │ │(User custom functions)│ │(Combines generators)   │
└─────────┬──────────────┘ └──────┬────────────────┘ └────────┬───────────────┘ └───────┬────────────────┘
          │                       │                           │                         │
          └───────────────────────┴─────────────┬─────────────┴─────────────────────────┘
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Task Synthesizer       │ (Grid or Simula Mode)
                                  └─────────────┬─────────────┘
                                                │ (Context Seed + Task Directive)
                                                ▼
                                  ┌───────────────────────────┐
                                  │   ReAct Teacher Loop      │ (< 1ms Declarative Simulation)
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Multi-Format Exporters    │ (OpenAI, Anthropic, Hermes, DPO)
                                  └───────────────────────────┘
```

---

## Getting Started Example

```python
import asyncio
import os
from afterimage.agent_trace import (
    AsyncAgentTraceGenerator,
    VirtualUserContextGenerator,
    ToolActionSpec,
    ToolParameterSpec,
)
from afterimage.exporters import export_dataset

async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    # 1. Initialize the Generator with Virtual User Context and Progress Indicator
    generator = AsyncAgentTraceGenerator(
        api_key=api_key,
        architect_model="gemini-3.6-flash",
        teacher_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.6-flash",
        context_generator=VirtualUserContextGenerator(seed=42),
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

    # 4. Generate Synthetic Agent Trajectories Concurrently with Progress Bar
    trajectories = await generator.generate(
        num_trajectories=10,
        max_turns=5,
        max_concurrency=4,
        show_progress=True,
    )

    print(f"Generated {len(trajectories)} valid synthetic agent trajectories.")

    # 5. Export Trajectories to Structured Tool Calling Format
    export_dataset(
        input_path="outputs/agent_trajectories.jsonl",
        format_name="openai_tools",
        output_path="outputs/agent_openai_tools.jsonl",
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Initial Context Architecture (`BaseContextGenerator`)

`afterimage.agent_trace` features a highly composable context generation system to seed virtual identities and initial database states into task synthesis prompts and declarative execution environments:

| Context Generator | Class Name | Description |
|---|---|---|
| **Virtual User** | `VirtualUserContextGenerator` | Generates realistic localized user identities, names, emails, addresses, user IDs, and account numbers using `Faker`. |
| **Persona Wrapper** | `PersonaContextGenerator` | Integrates `PersonaEntry` profiles (from `afterimage.persona_generator`) directly into initial state context. |
| **Custom Callable** | `CallableContextGenerator` | Wraps user-defined sync or async state seed functions. |
| **Composite** | `CompositeContextGenerator` | Merges state dictionaries produced by multiple context generators. |

---

## Export Formats for Agent Training

Use `export_dataset(input_path, format_name)` or the CLI to export trajectories into standard tool calling formats:

- **`openai_tools`**: Standard OpenAI Chat Completions tool-calling format (`tools` specification, `tool_calls` array, `role: "tool"`).
- **`anthropic_tools`**: Anthropic Messages API format with `tool_use` and `tool_result` content blocks.
- **`hermes_tools`**: Nous Hermes 2/3 and Qwen 2.5 XML `<tool_call>` format.
- **`agent_dpo`**: Trajectory preference pair format (`prompt`, `chosen`, `rejected`) for DPO/ORPO training.
- **`agent_sft`**: Sequential multi-turn message string format.

---

## Recommended Model Configuration

| Component | Default Model | Purpose |
|---|---|---|
| **Schema Architect** | `gemini-3.6-flash` | High-quality Pydantic response code generation with metadata tags. |
| **Task Synthesizer & Rewriter** | `gemini-3.5-flash-lite` | Combinatorial grid task synthesis & natural language rewriter. |
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
