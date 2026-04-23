# OpenSimula (`afterimage.simula`)

Experimental, open implementation of the **Simula** mechanism-design ideas from Davidson et al. (TMLR): reasoning-driven taxonomies, weighted mix sampling, meta-prompt diversification, optional complexification, requirement critics with refinement, and a **double-critic** gate for multiple-choice items. This is **not** affiliated with Google and is **not** a reference port of internal systems.

## Quick import

```python
from afterimage.simula import OpenSimula, SimulaInstructionGeneratorCallback
from afterimage.providers import LLMFactory, InMemoryDocumentProvider
```

Use `OpenSimula` with any `LLMProvider` from `LLMFactory`. The **examples** under `examples/simula/` default to **`gemini-2.5-flash`** (cheap, paper-aligned teacher family) and document each parameter against the Simula paper. See `examples/simula/README.md`.
