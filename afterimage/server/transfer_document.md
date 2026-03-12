# AfterImage Server Transfer Document

Last updated: 2026-03-12  
Repository root: `/home/ubuntu/Projects/afterimage`  
Server package path: `afterimage/server`  
Current version: `0.11.4`

## 1. Purpose and Scope

The AfterImage server is a FastAPI service that turns a document (text/chunks/upload) into synthetic conversations using the core AfterImage library.

Primary capabilities:
- Submit async generation jobs.
- Track progress with polling or SSE.
- Download generated datasets (`jsonl` or wrapped `json`).
- Analyze a document to auto-build prompt roles before generation.

## 2. Runtime and Entrypoints

Main entrypoints:
- CLI module: `python -m afterimage.server` ([`afterimage/server/__main__.py`](./__main__.py))
- Script entrypoint: `afterimage-server` ([`pyproject.toml`](../../pyproject.toml))

App factory:
- [`afterimage/server/app.py`](./app.py): creates `FastAPI`, wires routers, initializes `JobStore`, `ResultStore`, and `JobManager`.

Dependencies for server mode:
- `fastapi`, `uvicorn`, `python-multipart`, `pydantic-settings`, `sse-starlette`, `aiosqlite`.

## 3. High-Level Architecture

### API layer (`routers/`)

- [`routers/health.py`](./routers/health.py)
  - `GET /health`
  - `GET /ready`
- [`routers/generation.py`](./routers/generation.py)
  - `POST /api/v1/generate`
  - `POST /api/v1/generate/upload`
- [`routers/documents.py`](./routers/documents.py)
  - `POST /api/v1/analyze-document`
- [`routers/jobs.py`](./routers/jobs.py)
  - `GET /api/v1/jobs`
  - `GET /api/v1/jobs/{job_id}`
  - `GET /api/v1/jobs/{job_id}/stream`
  - `GET /api/v1/jobs/{job_id}/result`
  - `DELETE /api/v1/jobs/{job_id}`

### Domain/services layer (`services/`)

- [`services/job_manager.py`](./services/job_manager.py)
  - Submits jobs, runs background tasks, controls server-level concurrency via `asyncio.Semaphore`.
  - Tracks SSE subscribers and broadcasts progress/terminal events.
- [`services/generation_service.py`](./services/generation_service.py)
  - Orchestrates full generation pipeline:
    1. API key resolve
    2. Document chunk prep
    3. Optional prompt analysis
    4. Optional persona generation
    5. Generator init
    6. Generation execution
    7. Result serialization
- [`services/prompt_analyzer.py`](./services/prompt_analyzer.py)
  - Single LLM call to infer `respondent_role`, `correspondent_role`, and instruction.

### Persistence layer (`storage/`)

- [`storage/job_store.py`](./storage/job_store.py): SQLite-backed job metadata persistence (`jobs.db` by default), plus in-memory cache.
- [`storage/result_store.py`](./storage/result_store.py): per-job result files under `results/{job_id}/`.

### Streaming layer (`ws/`)

- [`ws/progress.py`](./ws/progress.py): SSE generator with heartbeat every 15s.

## 4. Job Lifecycle

1. Client submits generation request.
2. `JobManager.submit()` creates `JobRecord(status="queued")`, persists to SQLite, starts background task.
3. Background task enters semaphore and marks job `running`.
4. `GenerationService.run()` emits phase-based progress:
   - `analyzing_document`
   - `generating_personas`
   - `initializing`
   - `generating`
   - `saving`
   - `complete`
5. On success:
   - result persisted (`conversations.jsonl` or `conversations.json`)
   - status set `completed`
   - SSE `complete` event broadcast
6. On failure:
   - status set `failed`
   - error string persisted
   - SSE `error` event broadcast
7. On cancellation:
   - cancel event set + background task cancelled
   - status `cancelled`
   - SSE `cancelled` event broadcast

## 5. Configuration and Environment

Config source: [`server/config.py`](./config.py), `AFTERIMAGE_` prefixed env vars + `.env` fallback for bare keys.

Important vars:
- `AFTERIMAGE_HOST` (default `0.0.0.0`)
- `AFTERIMAGE_PORT` (default `8000`)
- `AFTERIMAGE_WORKERS` (default `1`)
- `AFTERIMAGE_MAX_CONCURRENT_JOBS` (default `3`)
- `AFTERIMAGE_MAX_DIALOGS_PER_REQUEST` (default `1000`)
- `AFTERIMAGE_RESULTS_DIR` (default `./results`)
- `AFTERIMAGE_JOB_DB_PATH` (default `jobs.db`)
- `AFTERIMAGE_API_KEY` (enables auth when set)
- `AFTERIMAGE_CORS_ORIGINS` (default `["*"]`)
- API keys:
  - `AFTERIMAGE_GEMINI_API_KEY` or `GEMINI_API_KEY`
  - `AFTERIMAGE_OPENAI_API_KEY` or `OPENAI_API_KEY`
  - `AFTERIMAGE_DEEPSEEK_API_KEY` or `DEEPSEEK_API_KEY`

## 6. Security Model

- Auth is optional and global.
- If `AFTERIMAGE_API_KEY` is set, all protected routes require:
  - `Authorization: Bearer <key>`
- If unset, routes are open.
- CORS defaults to wildcard.

## 7. Data Layout

Default local artifacts:
- Job metadata DB: `jobs.db`
- Results dir: `results/`
- Per job:
  - `results/{job_id}/conversations.jsonl` (streamed generator output)
  - `results/{job_id}/conversations.json` (when `output_format="json"`)

SQLite table: `jobs`
- `job_id`, `status`, `request_json`, `progress_json`, `result_json`, `error`, timestamps.

## 8. Operations Runbook

Install and run:

```bash
uv pip install -e ".[server]"
uv run python -m afterimage.server --port 8000
```

Basic health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Quick end-to-end smoke test:

```bash
uv run python test_server.py --base-url http://localhost:8000 --quick
```

## 9. Known Issues / Technical Debt

1. `model_provider_name` is used for API-key selection but is not passed into `AsyncConversationGenerator`; generation defaults to provider `gemini`.
2. `PromptAnalyzer` is hardwired to `google.genai`; auto-prompt analysis can break for non-Gemini provider flows.
3. Jobs are persisted, but in-flight jobs are not resumed after process restart; stale `queued/running` records can remain.
4. `active_job_count()` only scans first 1000 jobs (`list_jobs(page=1, per_page=1000)`).
5. Cancelled jobs may leave partial result files in `results/{job_id}` (not cleaned automatically).
6. Config access is mixed (`get_config()` singleton vs app-state config dependency), which can cause inconsistency in custom embedding/testing setups.

### How to Solve

1. Provider mismatch in generation:
   Pass `model_provider_name=request.model_provider_name` when creating `AsyncConversationGenerator`. Also restrict `model_provider_name` in `GenerationRequest` to known providers (`gemini`, `openai`, `deepseek`) to prevent invalid values.
2. Gemini-only prompt analysis:
   Either (a) block `auto_generate_prompts=true` for non-Gemini providers with a clear `422` error, or (b) refactor `PromptAnalyzer` to use the same provider abstraction as generation so all supported providers work.
3. Stale in-flight jobs after restart:
   Add startup reconciliation in app lifespan. On boot, load jobs with status `queued`/`running` and mark them terminal (usually `failed` with an explicit restart reason), or explicitly requeue them if resumable behavior is desired.
4. Active job count truncation:
   Add a dedicated store method (`count_active_jobs`) that executes `SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')`, and use it in `JobManager.active_job_count()`.
5. Partial artifacts on cancellation:
   Define a cancellation retention policy. If cancelled outputs should be removed, call `ResultStore.delete_job_files(job_id)` in the cancellation path after generation stops to avoid races with ongoing writes.
6. Mixed config sources:
   Standardize on `request.app.state.config` everywhere (via `get_config_dep`) and inject config into services at construction time. Avoid reading `get_config()` inside request-time/service logic.

## 10. Key Files for Ownership Transfer

- [`afterimage/server/app.py`](./app.py)
- [`afterimage/server/config.py`](./config.py)
- [`afterimage/server/models.py`](./models.py)
- [`afterimage/server/routers/generation.py`](./routers/generation.py)
- [`afterimage/server/routers/jobs.py`](./routers/jobs.py)
- [`afterimage/server/services/job_manager.py`](./services/job_manager.py)
- [`afterimage/server/services/generation_service.py`](./services/generation_service.py)
- [`afterimage/server/storage/job_store.py`](./storage/job_store.py)
- [`afterimage/server/storage/result_store.py`](./storage/result_store.py)
- [`afterimage/server/api_documentation.md`](./api_documentation.md)
- [`test_server.py`](../../test_server.py)
