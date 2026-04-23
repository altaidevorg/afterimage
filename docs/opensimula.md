# OpenSimula (`afterimage.simula`)

OpenSimula is an experimental, open implementation of mechanism-design ideas from **Simula** (Davidson et al., TMLR): factor taxonomies, weighted mix sampling, meta-prompt diversification, requirement critics with refinement, and a double-critic gate for multiple-choice items. It is **not** affiliated with Google and is **not** a reference port of internal systems.

For runnable scripts, see the [`examples/simula/`](https://github.com/alt-ai/afterimage/tree/main/examples/simula) directory in the repository.

## Imports

```python
from afterimage.simula import OpenSimula, SimulaInstructionGeneratorCallback
from afterimage.monitoring import GenerationMonitor
from afterimage.providers import LLMFactory, InMemoryDocumentProvider
```

## Generation monitoring

You can attach the same `GenerationMonitor` used elsewhere in AfterImage by passing `monitor=` into `OpenSimula` (see the **Simula / OpenSimula** page under API Reference).

When a monitor is set, structured LLM calls across the pipeline call `track_generation` with duration, success or failure, token counts when available, and **metadata** that includes:

- `component`: `"opensimula"`
- `operation`: a stable string such as `opensimula.taxonomy.propose_factors`, `opensimula.sampling.infer_strategies`, `opensimula.meta.generate_scenarios`, `opensimula.critics.requirement_critique`, `opensimula.double_critic.probe`, `opensimula.tasks.single_qa_json`, and similar paths for evaluation helpers.

Omit `monitor` (or pass `None`) to run without recording metrics.

Always call `GenerationMonitor.shutdown()` when the run finishes so background metric processing stops cleanly (see [Monitoring & observability](monitoring.md)).

### Sharing a monitor with `ConversationGenerator`

`SimulaInstructionGeneratorCallback` does not perform LLM calls by itself; it bridges Simula datapoints into conversation turns. If you use **both** `OpenSimula` and `ConversationGenerator` in one process and want one metrics stream, construct a single `GenerationMonitor`, pass it to `OpenSimula(..., monitor=monitor)`, and attach the same instance to the generator (or use the generator’s callback hooks only where they already record LLM usage).

## API reference

Module and class reference is generated on the **Simula / OpenSimula** API page (`api/simula` in the built site navigation).
