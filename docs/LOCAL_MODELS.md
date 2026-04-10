# Using AfterImage with Local Models

AfterImage supports any OpenAI-compatible local model server. No API key needed.

## Quick start

```bash
pip install afterimage
afterimage generate -c examples/configs/local.yaml
```

## vLLM

```bash
pip install vllm
vllm serve Qwen/Qwen3-1.7B --port 8000
```

Config:
```yaml
model:
  provider: local
  base_url: http://localhost:8000/v1
  model_name: Qwen/Qwen3-1.7B
```

## Ollama

```bash
ollama pull llama3.2
ollama serve
```

Config:
```yaml
model:
  provider: local
  base_url: http://localhost:11434/v1
  model_name: llama3.2
```

## llama.cpp

```bash
./llama-server -m model.gguf --port 8000
```

Config:
```yaml
model:
  provider: local
  base_url: http://localhost:8000/v1
  model_name: model
```

## Tips

- Lower `max_concurrency` for CPU inference (1-2 is usually best)
- `max_turns: 1` keeps generation fast for small models
- Quality gating (`auto_improve: true`) requires local embeddings: `pip install "afterimage[embeddings-local]"`
