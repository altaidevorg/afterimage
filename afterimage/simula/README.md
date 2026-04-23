# OpenSimula (`afterimage.simula`)

Experimental, open implementation of the **Simula** mechanism-design ideas from Davidson et al. (TMLR): reasoning-driven taxonomies, weighted mix sampling, meta-prompt diversification, optional complexification, requirement critics with refinement, and a **double-critic** gate for multiple-choice items. This is **not** affiliated with Google and is **not** a reference port of internal systems.

## Quick import

```python
from afterimage.simula import OpenSimula, SimulaInstructionGeneratorCallback
from afterimage.providers import LLMFactory, InMemoryDocumentProvider
```

Use `OpenSimula` with any `LLMProvider` from `LLMFactory`. Persist taxonomies with `Checkpointer` (`bundle.save(cp)`, `spec.save(cp)`, `cp.write_run_config(OpenSimulaRunConfig(...))`, `cp.push_to_hub(...)`, `load_checkpoint`) or `save_checkpoint` / `push_checkpoint_to_hub`. Append accepted rows with `append_datapoints_jsonl`; generate batches with `OpenSimula.agenerate_single_qa_samples` / `aiter_single_qa_samples`. The **examples** under `examples/simula/` default to **`gemini-2.5-flash`**, call `configure_example_console()` to hide `httpx` / `google_genai` noise, and pass `show_progress=True` to `build_taxonomy()` for tqdm. See `examples/simula/README.md`.

## Monitoring (`GenerationMonitor`)

Pass an optional `GenerationMonitor` into `OpenSimula` so structured LLM work is mirrored into the same metrics pipeline as `ConversationGenerator` and other generators:

```python
from afterimage.monitoring import GenerationMonitor
from afterimage.simula import OpenSimula

monitor = GenerationMonitor(log_dir="./logs/opensimula")
sim = OpenSimula(llm, monitor=monitor)
try:
    bundle = await sim.build_taxonomy(instruction_y, show_progress=True)
    # ... infer_strategies, draw_meta_prompt, generate_* ...
finally:
    monitor.shutdown()
```

When `monitor` is not `None`, internal helpers call `track_generation` with latency, success or failure, token fields when the provider returns them, and metadata that always includes `component="opensimula"` plus an `operation` string (for example `opensimula.taxonomy.propose_factors`, `opensimula.sampling.infer_strategies`, `opensimula.meta.generate_scenarios`, `opensimula.critics.requirement_critique`, `opensimula.double_critic.probe`, `opensimula.tasks.single_qa_json`, and labels under `opensimula.eval.*` for taxonomy assignment and Elo batches).

`SimulaInstructionGeneratorCallback` does not call the LLM; it only replays precomputed scenario text into `ConversationGenerator`. To correlate Simula LLM metrics with conversation metrics in one process, share a single `GenerationMonitor` between `OpenSimula(..., monitor=m)` and `ConversationGenerator(..., monitor=m)` (and call `shutdown()` once at the end).

**Sphinx:** narrative doc is `docs/opensimula.md`; API autodoc lives on `docs/api/simula.rst` (built under *API Reference → Simula / OpenSimula*).
