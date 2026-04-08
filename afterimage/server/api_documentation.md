# AfterImage Server — API Documentation

**Version:** 0.11.4  
**Base URL:** `http://localhost:8000`  
**Interactive docs:** [`/docs`](http://localhost:8000/docs) (Swagger UI) · [`/redoc`](http://localhost:8000/redoc)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Configuration Reference](#configuration-reference)
4. [Data Models](#data-models)
5. [Endpoints](#endpoints)
   - [Health](#health)
   - [Generate — JSON body](#post-apiv1generate)
   - [Generate — File upload](#post-apiv1generateupload)
   - [Analyze Document](#post-apiv1analyze-document)
   - [List Jobs](#get-apiv1jobs)
   - [Get Job Status](#get-apiv1jobsjob_id)
   - [Stream Progress (SSE)](#get-apiv1jobsjob_idstream)
   - [Download Result](#get-apiv1jobsjob_idresult)
   - [Cancel Job](#delete-apiv1jobsjob_id)
6. [SSE Event Reference](#sse-event-reference)
7. [Error Reference](#error-reference)
8. [End-to-End Examples](#end-to-end-examples)

---

## Getting Started

### Install and run

```bash
# From the afterimage repo root
pip install -e ".[server]"

# Set your LLM API key (or add to .env)
export GEMINI_API_KEY=your_key_here

# Start the server
python -m afterimage.server
# or
afterimage-server --port 8000
```

### Verify it's running

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.11.4","active_jobs":0}
```

---

## Authentication

Authentication is **optional**. It is enabled only when `AFTERIMAGE_API_KEY` is set.

When enabled, every request must include:

```
Authorization: Bearer <your_api_key>
```

If the key is wrong or missing, the server returns `401 Unauthorized`.

When `AFTERIMAGE_API_KEY` is not set, all endpoints are open with no token required.

---

## Configuration Reference

All settings are read from environment variables with the `AFTERIMAGE_` prefix, or from a `.env` file in the working directory. LLM API keys are also accepted without the prefix as a convenience fallback.

| Environment Variable | Default | Description |
|---|---|---|
| `AFTERIMAGE_GEMINI_API_KEY` | — | Gemini API key (also read as `GEMINI_API_KEY`) |
| `AFTERIMAGE_DEEPSEEK_API_KEY` | — | DeepSeek API key (also read as `DEEPSEEK_API_KEY`) |
| `AFTERIMAGE_OPENAI_API_KEY` | — | OpenAI API key (also read as `OPENAI_API_KEY`) |
| `AFTERIMAGE_HOST` | `0.0.0.0` | Bind address |
| `AFTERIMAGE_PORT` | `8000` | Listen port |
| `AFTERIMAGE_WORKERS` | `1` | Uvicorn worker count |
| `AFTERIMAGE_DEFAULT_MODEL` | `gemini-2.0-flash` | Default LLM model |
| `AFTERIMAGE_DEFAULT_PROVIDER` | `gemini` | Default provider (`gemini`, `openai`, `deepseek`) |
| `AFTERIMAGE_MAX_CONCURRENT_JOBS` | `3` | Max parallel generation jobs |
| `AFTERIMAGE_MAX_DIALOGS_PER_REQUEST` | `1000` | Upper limit on `num_dialogs` per request |
| `AFTERIMAGE_RESULTS_DIR` | `./results` | Directory where result files are written |
| `AFTERIMAGE_JOB_DB_PATH` | `jobs.db` | SQLite file for job persistence |
| `AFTERIMAGE_API_KEY` | — | Server API key (enables auth when set) |
| `AFTERIMAGE_CORS_ORIGINS` | `["*"]` | CORS allowed origins |

**Example `.env`:**
```dotenv
GEMINI_API_KEY=AIzaSy...
DEEPSEEK_API_KEY=sk-...
AFTERIMAGE_MAX_CONCURRENT_JOBS=5
AFTERIMAGE_RESULTS_DIR=/data/results
```

---

## Data Models

### `GenerationRequest`

The request body for `POST /api/v1/generate`. At least one of `document_text` or `document_chunks` is required.

| Field | Type | Default | Description |
|---|---|---|---|
| `document_text` | `string \| null` | `null` | Full document as a plain string |
| `document_chunks` | `string[] \| null` | `null` | Pre-split document chunks |
| `chunk_size` | `integer` | `5000` | Characters per chunk when auto-splitting `document_text` |
| `num_dialogs` | `integer` | `10` | Number of QA / conversation pairs to generate (max: 1000) |
| `max_turns` | `integer` | `1` | Turns per dialog (1 = single Q&A, 2+ = multi-turn) |
| `max_concurrency` | `integer` | `4` | Parallel LLM workers (max: 32) |
| `respondent_prompt` | `string \| null` | `null` | System prompt for the assistant. Auto-generated when `null` and `auto_generate_prompts=true` |
| `correspondent_prompt` | `string \| null` | `null` | System prompt for the simulated user. Auto-generated when `null` |
| `auto_generate_prompts` | `boolean` | `true` | Use the LLM to analyze the document and craft context-appropriate prompts |
| `custom_instruction_prompt` | `string \| null` | `null` | Override the persona instruction template (supports `{persona}` and `{n_instructions}` placeholders) |
| `use_personas` | `boolean` | `true` | Generate diverse user personas from the document |
| `persona_iterations` | `integer` | `0` | Extra persona refinement passes (0 = single pass) |
| `model_name` | `string` | `"gemini-2.0-flash"` | LLM model identifier |
| `model_provider_name` | `string` | `"gemini"` | Provider: `"gemini"`, `"openai"`, or `"deepseek"` |
| `output_format` | `"jsonl" \| "json"` | `"jsonl"` | Result file format |
| `include_system_prompt_parts` | `boolean` | `true` | Include auto-generated prompt parts in the JSON result |

### `JobStatus`

```
"queued" | "running" | "completed" | "failed" | "cancelled"
```

### `GenerationPhase`

```
"analyzing_document" | "generating_personas" | "initializing"
| "generating" | "saving" | "complete"
```

### `JobProgress`

| Field | Type | Description |
|---|---|---|
| `completed` | `integer` | Dialogs generated so far |
| `total` | `integer` | Total dialogs requested |
| `percent` | `float` | Completion percentage (0–100) |
| `current_phase` | `GenerationPhase` | Current pipeline stage |
| `elapsed_seconds` | `float` | Wall-clock time since job started |
| `estimated_remaining_seconds` | `float \| null` | ETA (available once generation begins) |

### `JobResult`

| Field | Type | Description |
|---|---|---|
| `num_conversations` | `integer` | Number of conversations in the output |
| `download_url` | `string` | Relative path to `GET /api/v1/jobs/{job_id}/result` |
| `output_format` | `string` | `"jsonl"` or `"json"` |

---

## Endpoints

---

### Health

#### `GET /health`

Returns server health and the current number of active (queued + running) jobs.

**Response `200`:**
```json
{
  "status": "ok",
  "version": "0.11.4",
  "active_jobs": 2
}
```

#### `GET /ready`

Kubernetes-style readiness probe.

**Response `200`:**
```json
{ "ready": true }
```

---

### `POST /api/v1/generate`

Submit a new dataset generation job. Returns immediately with a job ID; generation runs in the background.

**Request body:** [`GenerationRequest`](#generationrequest)

**Response `202 Accepted`:**
```json
{
  "job_id": "f26ba273-ccce-4c49-be15-6c8e810836d7",
  "status": "queued",
  "created_at": "2026-03-09T10:43:02.728902",
  "estimated_duration_seconds": null,
  "links": {
    "self":   "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7",
    "status": "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7",
    "stream": "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7/stream",
    "result": "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7/result",
    "cancel": "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7"
  }
}
```

**Errors:**

| Status | Reason |
|---|---|
| `422` | No document supplied, `num_dialogs` exceeds server limit, or invalid field values |
| `401` | Missing / invalid API key (when auth is enabled) |

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Your document text here...",
    "num_dialogs": 50,
    "auto_generate_prompts": true,
    "use_personas": true,
    "model_name": "gemini-2.0-flash",
    "output_format": "jsonl"
  }'
```

---

### `POST /api/v1/generate/upload`

Same as `POST /api/v1/generate` but accepts a plain-text file via `multipart/form-data` instead of a JSON body. All generation parameters are passed as form fields.

**Form fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | **required** | UTF-8 encoded plain-text document |
| `num_dialogs` | integer | `10` | |
| `max_turns` | integer | `1` | |
| `max_concurrency` | integer | `4` | |
| `auto_generate_prompts` | boolean | `true` | |
| `use_personas` | boolean | `true` | |
| `model_name` | string | `"gemini-2.0-flash"` | |
| `model_provider_name` | string | `"gemini"` | |
| `output_format` | string | `"jsonl"` | |
| `chunk_size` | integer | `5000` | |

**Response `202`:** identical to `POST /api/v1/generate`

**Errors:**

| Status | Reason |
|---|---|
| `422` | File is not valid UTF-8, or invalid field values |

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/generate/upload \
  -F "file=@article.txt" \
  -F "num_dialogs=100" \
  -F "auto_generate_prompts=true"
```

---

### `POST /api/v1/analyze-document`

Analyzes a document excerpt using the LLM and returns context-appropriate system prompt roles. Use this to **preview** what `auto_generate_prompts` will produce before starting a full job.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `document_text` | `string` | **required** | Document to analyze |
| `excerpt_length` | `integer` | `4000` | Characters to send to the LLM (from the start of the document) |

**Response `200`:**
```json
{
  "respondent_role": "You are an expert in synthetic data generation and machine learning...",
  "correspondent_role": "You are a software engineer exploring dataset generation tools...",
  "instruction": "Answer questions clearly, providing code examples where relevant..."
}
```

**Errors:**

| Status | Reason |
|---|---|
| `422` | `document_text` missing |
| `503` | No LLM API key configured on the server |

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/analyze-document \
  -H "Content-Type: application/json" \
  -d '{"document_text": "AfterImage is a Python library...", "excerpt_length": 2000}'
```

---

### `GET /api/v1/jobs`

List all jobs, newest first. Supports pagination.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `page` | `integer` | `1` | Page number (≥ 1) |
| `per_page` | `integer` | `20` | Results per page (1–100) |

**Response `200`:**
```json
{
  "jobs": [
    {
      "job_id": "f26ba273-ccce-4c49-be15-6c8e810836d7",
      "status": "completed",
      "num_dialogs": 50,
      "model_name": "gemini-2.0-flash",
      "created_at": "2026-03-09T10:43:02.728902",
      "updated_at": "2026-03-09T10:47:15.391042"
    }
  ],
  "total": 14,
  "page": 1,
  "per_page": 20
}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/jobs?page=1&per_page=5"
```

---

### `GET /api/v1/jobs/{job_id}`

Get full status, progress, and result for a single job.

**Path parameter:** `job_id` — UUID returned by the generate endpoint.

**Response `200`:**
```json
{
  "job_id": "f26ba273-ccce-4c49-be15-6c8e810836d7",
  "status": "running",
  "progress": {
    "completed": 12,
    "total": 50,
    "percent": 24.0,
    "current_phase": "generating",
    "elapsed_seconds": 34.2,
    "estimated_remaining_seconds": 108.1
  },
  "result": null,
  "error": null,
  "created_at": "2026-03-09T10:43:02.728902",
  "updated_at": "2026-03-09T10:43:36.901234"
}
```

When `status` is `"completed"`, `result` is populated:
```json
"result": {
  "num_conversations": 50,
  "download_url": "/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7/result",
  "output_format": "jsonl"
}
```

When `status` is `"failed"`, `error` contains the failure message:
```json
"error": "No API key configured for provider 'gemini'."
```

**Errors:**

| Status | Reason |
|---|---|
| `404` | Job ID not found |

**Example:**
```bash
curl http://localhost:8000/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7
```

---

### `GET /api/v1/jobs/{job_id}/stream`

Subscribe to real-time progress via **Server-Sent Events (SSE)**. The connection stays open until the job reaches a terminal state (`completed`, `failed`, `cancelled`), then closes automatically.

**Headers required:**
```
Accept: text/event-stream
```

**Response:** `text/event-stream`

See [SSE Event Reference](#sse-event-reference) for the full event schema.

**Example (curl):**
```bash
curl -N -H "Accept: text/event-stream" \
  http://localhost:8000/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7/stream
```

**Example (Python with httpx):**
```python
import httpx, json

async with httpx.AsyncClient() as client:
    async with client.stream("GET", f"http://localhost:8000/api/v1/jobs/{job_id}/stream") as r:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                print(payload)
```

**Errors:**

| Status | Reason |
|---|---|
| `404` | Job ID not found |

---

### `GET /api/v1/jobs/{job_id}/result`

Download the generated dataset file. Only available when `status == "completed"`.

**Response:** file download

| `output_format` | Content-Type | File name |
|---|---|---|
| `jsonl` | `application/x-ndjson` | `dataset_{job_id}.jsonl` |
| `json` | `application/json` | `dataset_{job_id}.json` |

**JSONL format** — one conversation per line:
```jsonl
{"conversations": [{"role": "user", "content": "What is AfterImage?"}, {"role": "assistant", "content": "AfterImage is..."}], "instruction_context": "...", "response_context": "..."}
{"conversations": [...], "instruction_context": "...", "response_context": "..."}
```

**JSON format** — full object with optional system prompt parts:
```json
{
  "system_prompt_parts": [
    "You are an expert in synthetic data generation...",
    "Answer clearly and provide code examples where relevant."
  ],
  "conversations": [
    {
      "conversations": [
        {"role": "user", "content": "What is AfterImage?"},
        {"role": "assistant", "content": "AfterImage is..."}
      ],
      "instruction_context": "...",
      "response_context": "..."
    }
  ]
}
```

**Errors:**

| Status | Reason |
|---|---|
| `404` | Job ID not found, or result file missing on disk |
| `409` | Job is not yet completed (`status` is `queued`, `running`, etc.) |

**Example:**
```bash
curl -L http://localhost:8000/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7/result \
  -o dataset.jsonl
```

---

### `DELETE /api/v1/jobs/{job_id}`

Cancel a queued or running job. Has no effect on already-terminal jobs.

**Response `204 No Content`** — job cancelled successfully.

**Errors:**

| Status | Reason |
|---|---|
| `404` | Job ID not found |
| `409` | Job is already `completed`, `failed`, or `cancelled` |

**Example:**
```bash
curl -X DELETE http://localhost:8000/api/v1/jobs/f26ba273-ccce-4c49-be15-6c8e810836d7
```

---

## SSE Event Reference

All events follow the standard SSE format:

```
event: <event_name>
data: <json_payload>

```

### `progress`

Fired after every batch of conversations is saved to disk.

```
event: progress
data: {"job_id": "...", "completed": 12, "total": 50, "percent": 24.0, "current_phase": "generating", "elapsed_seconds": 34.2, "estimated_remaining_seconds": 108.1}
```

### `complete`

Fired once when the job finishes successfully.

```
event: complete
data: {"job_id": "...", "download_url": "/api/v1/jobs/.../result", "num_conversations": 50}
```

### `error`

Fired if the job fails.

```
event: error
data: {"job_id": "...", "error": "No API key configured for provider 'gemini'."}
```

### `cancelled`

Fired if the job is cancelled (via `DELETE` or internally).

```
event: cancelled
data: {"job_id": "..."}
```

### `heartbeat`

Sent every 15 seconds when there are no other events, to keep the connection alive through proxies.

```
event: heartbeat
data: {}
```

---

## Error Reference

All error responses follow FastAPI's standard format:

```json
{
  "detail": "Human-readable error message"
}
```

Validation errors (422) return an array:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "document_text"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

| Status | Meaning |
|---|---|
| `400` | Bad request |
| `401` | Invalid or missing API key |
| `404` | Resource not found |
| `409` | Conflict (e.g. cancelling a completed job, downloading an incomplete job) |
| `422` | Validation error — check `detail` for field-level errors |
| `503` | Server misconfigured (e.g. no LLM API key set) |

---

## End-to-End Examples

### Minimal — generate 10 QA pairs and download

```bash
API="http://localhost:8000"

# 1. Submit job
JOB=$(curl -s -X POST "$API/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Your document text here...",
    "num_dialogs": 10,
    "output_format": "jsonl"
  }')

JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB_ID"

# 2. Poll until done
while true; do
  STATUS=$(curl -s "$API/api/v1/jobs/$JOB_ID" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d['progress']['percent'])")
  echo "$STATUS"
  [[ "$STATUS" == completed* ]] && break
  [[ "$STATUS" == failed* ]] && exit 1
  sleep 3
done

# 3. Download
curl -L "$API/api/v1/jobs/$JOB_ID/result" -o dataset.jsonl
```

---

### Stream progress in real time

```bash
curl -N "$API/api/v1/jobs/$JOB_ID/stream"
```

Sample output:
```
event: progress
data: {"job_id": "...", "completed": 0, "total": 10, "percent": 0.0, "current_phase": "analyzing_document", ...}

event: progress
data: {"job_id": "...", "completed": 0, "total": 10, "percent": 0.0, "current_phase": "generating_personas", ...}

event: progress
data: {"job_id": "...", "completed": 4, "total": 10, "percent": 40.0, "current_phase": "generating", ...}

event: complete
data: {"job_id": "...", "download_url": "/api/v1/jobs/.../result", "num_conversations": 10}
```

---

### Python SDK pattern

```python
import asyncio, httpx, json

BASE = "http://localhost:8000"

async def generate_dataset(document: str, num_dialogs: int = 20) -> list[dict]:
    async with httpx.AsyncClient(base_url=BASE, timeout=600) as client:
        # Submit
        resp = await client.post("/api/v1/generate", json={
            "document_text": document,
            "num_dialogs": num_dialogs,
            "auto_generate_prompts": True,
            "output_format": "jsonl",
        })
        job_id = resp.json()["job_id"]
        print(f"Job submitted: {job_id}")

        # Stream progress
        async with client.stream("GET", f"/api/v1/jobs/{job_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event == "progress":
                        print(f"  {data['percent']:.0f}% — {data['current_phase']}")
                    elif event == "complete":
                        print(f"Done! {data['num_conversations']} conversations")
                        break
                    elif event == "error":
                        raise RuntimeError(data["error"])

        # Download
        result = await client.get(f"/api/v1/jobs/{job_id}/result")
        return [json.loads(line) for line in result.text.strip().splitlines()]

conversations = asyncio.run(generate_dataset("Your document text...", num_dialogs=20))
print(f"Downloaded {len(conversations)} conversations")
```

---

### Upload a file directly

```bash
curl -X POST "$API/api/v1/generate/upload" \
  -F "file=@/path/to/document.txt" \
  -F "num_dialogs=100" \
  -F "auto_generate_prompts=true" \
  -F "use_personas=true" \
  -F "output_format=jsonl"
```

---

### Preview auto-generated prompts before running

```bash
curl -X POST "$API/api/v1/analyze-document" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Your document text here...",
    "excerpt_length": 3000
  }'
```

```json
{
  "respondent_role": "You are an expert in synthetic data generation...",
  "correspondent_role": "You are a developer building training datasets...",
  "instruction": "Provide precise, practical answers with code examples."
}
```

Use the returned values as `respondent_prompt` / `correspondent_prompt` in your generation request if you want to tweak them before running the full job.
