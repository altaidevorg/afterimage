#!/usr/bin/env python3
"""
AfterImage Server — Test Playground
Tests every endpoint: health, analyze-document, generate, job status,
SSE streaming, list jobs, result download, and job cancellation.

Usage:
    python test_server.py
    python test_server.py --base-url http://localhost:8000 --quick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENT = """
AfterImage is a Python library for generating synthetic conversation datasets.
It supports multiple document sources, persona-based question generation,
and integrates with LLMs such as Gemini and OpenAI.
The library provides AsyncConversationGenerator for multi-turn dialogs,
AsyncStructuredGenerator for single-turn structured outputs, and
PersonaGenerator to enrich documents with diverse user personas.
Storage backends include JSONLStorage and SQLStorage.
""".strip()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[str] = []

    def ok(self, name: str, detail: str = ""):
        tag = f"{GREEN}✓ PASS{RESET}"
        print(f"  {tag}  {name}" + (f"  {YELLOW}({detail}){RESET}" if detail else ""))
        self.passed.append(name)

    def fail(self, name: str, reason: str):
        tag = f"{RED}✗ FAIL{RESET}"
        print(f"  {tag}  {name}  {RED}{reason}{RESET}")
        self.failed.append((name, reason))

    def skip(self, name: str, reason: str):
        tag = f"{YELLOW}⊘ SKIP{RESET}"
        print(f"  {tag}  {name}  ({reason})")
        self.skipped.append(name)

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print()
        print(f"{BOLD}{'─' * 60}{RESET}")
        print(
            f"{BOLD}Results: {total} tests  "
            f"{GREEN}{len(self.passed)} passed{RESET}  "
            f"{RED}{len(self.failed)} failed{RESET}  "
            f"{YELLOW}{len(self.skipped)} skipped{RESET}{BOLD}{RESET}"
        )
        if self.failed:
            print(f"\n{RED}Failed tests:{RESET}")
            for name, reason in self.failed:
                print(f"  • {name}: {reason}")
        print(f"{BOLD}{'─' * 60}{RESET}")
        return len(self.failed) == 0


def section(title: str):
    print(f"\n{BOLD}{CYAN}▶ {title}{RESET}")


def _assert(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Test suites
# ──────────────────────────────────────────────────────────────────────────────


async def test_health(client: httpx.AsyncClient, r: TestResult):
    section("Health & Readiness")

    # GET /health
    try:
        resp = await client.get("/health")
        _assert(resp.status_code == 200, f"status {resp.status_code}")
        body = resp.json()
        _assert(body["status"] == "ok", f"status field: {body.get('status')}")
        _assert("version" in body, "missing version")
        _assert("active_jobs" in body, "missing active_jobs")
        r.ok(
            "GET /health",
            f"version={body['version']} active_jobs={body['active_jobs']}",
        )
    except Exception as e:
        r.fail("GET /health", str(e))

    # GET /ready
    try:
        resp = await client.get("/ready")
        _assert(resp.status_code == 200, f"status {resp.status_code}")
        _assert(resp.json().get("ready") is True, "ready != true")
        r.ok("GET /ready")
    except Exception as e:
        r.fail("GET /ready", str(e))

    # GET / → 404 (expected, not a defined route)
    try:
        resp = await client.get("/")
        _assert(resp.status_code == 404, f"expected 404, got {resp.status_code}")
        r.ok("GET / → 404 (expected)")
    except Exception as e:
        r.fail("GET / → 404", str(e))


async def test_docs(client: httpx.AsyncClient, r: TestResult):
    section("API Documentation")

    for path in ["/docs", "/redoc", "/openapi.json"]:
        try:
            resp = await client.get(path)
            _assert(resp.status_code == 200, f"status {resp.status_code}")
            r.ok(f"GET {path}")
        except Exception as e:
            r.fail(f"GET {path}", str(e))


async def test_analyze_document(
    client: httpx.AsyncClient, r: TestResult
) -> dict[str, str] | None:
    section("POST /api/v1/analyze-document")
    result = None

    # Valid request
    try:
        resp = await client.post(
            "/api/v1/analyze-document",
            json={"document_text": SAMPLE_DOCUMENT, "excerpt_length": 500},
            timeout=60,
        )
        _assert(
            resp.status_code == 200, f"status {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        for field in ("respondent_role", "correspondent_role", "instruction"):
            _assert(field in body and body[field], f"missing/empty field: {field}")
        result = body
        r.ok(
            "POST /api/v1/analyze-document",
            f"respondent_role={body['respondent_role'][:50]!r}",
        )
    except Exception as e:
        r.fail("POST /api/v1/analyze-document", str(e))

    # Missing document_text → 422
    try:
        resp = await client.post("/api/v1/analyze-document", json={})
        _assert(resp.status_code == 422, f"expected 422, got {resp.status_code}")
        r.ok("POST /api/v1/analyze-document (missing body → 422)")
    except Exception as e:
        r.fail("POST /api/v1/analyze-document (missing body → 422)", str(e))

    return result


async def test_generate_and_poll(
    client: httpx.AsyncClient,
    r: TestResult,
    num_dialogs: int = 2,
    output_format: str = "jsonl",
) -> str | None:
    """Submit a generation job and poll until completion. Returns job_id."""
    section(
        f"POST /api/v1/generate  (num_dialogs={num_dialogs}, format={output_format})"
    )
    job_id = None

    # Submit
    try:
        resp = await client.post(
            "/api/v1/generate",
            json={
                "document_text": SAMPLE_DOCUMENT,
                "num_dialogs": num_dialogs,
                "max_turns": 1,
                "max_concurrency": 2,
                "auto_generate_prompts": True,
                "use_personas": True,
                "persona_iterations": 0,
                "model_name": "gemini-2.0-flash",
                "model_provider_name": "gemini",
                "output_format": output_format,
                "include_system_prompt_parts": True,
            },
            timeout=30,
        )
        _assert(
            resp.status_code == 202, f"status {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        job_id = body.get("job_id")
        _assert(job_id, "missing job_id")
        _assert(
            body["status"] in ("queued", "running"),
            f"unexpected status: {body['status']}",
        )
        _assert("links" in body and "stream" in body["links"], "missing SSE link")
        r.ok("POST /api/v1/generate → 202", f"job_id={job_id}")
    except Exception as e:
        r.fail("POST /api/v1/generate → 202", str(e))
        return None

    # Validation errors
    try:
        resp = await client.post("/api/v1/generate", json={})
        _assert(resp.status_code == 422, f"expected 422, got {resp.status_code}")
        r.ok("POST /api/v1/generate (no document → 422)")
    except Exception as e:
        r.fail("POST /api/v1/generate (no document → 422)", str(e))

    try:
        resp = await client.post(
            "/api/v1/generate",
            json={"document_text": "x", "num_dialogs": 9999999},
        )
        _assert(resp.status_code in (422,), f"expected 422, got {resp.status_code}")
        r.ok("POST /api/v1/generate (num_dialogs too large → 422)")
    except Exception as e:
        r.fail("POST /api/v1/generate (num_dialogs too large → 422)", str(e))

    # Poll until done
    print(f"  {YELLOW}⏳ Polling job {job_id} …{RESET}", end="", flush=True)
    deadline = time.time() + 300
    final_status = None
    while time.time() < deadline:
        await asyncio.sleep(2)
        try:
            resp = await client.get(f"/api/v1/jobs/{job_id}", timeout=10)
            if resp.status_code == 200:
                body = resp.json()
                status = body["status"]
                pct = body.get("progress", {}).get("percent", 0)
                print(
                    f"\r  {YELLOW}⏳ {status} {pct:.0f}%{RESET}        ",
                    end="",
                    flush=True,
                )
                if status in ("completed", "failed", "cancelled"):
                    final_status = status
                    print()
                    break
        except Exception:
            pass
    else:
        print()
        r.fail("job completes within 300s", "timed out")
        return job_id

    if final_status == "completed":
        r.ok("job completes (status=completed)")
    else:
        body = await (await client.get(f"/api/v1/jobs/{job_id}")).aread()
        r.fail(
            "job completes",
            f"status={final_status} error={json.loads(body).get('error', '')}",
        )

    return job_id


async def test_job_status(client: httpx.AsyncClient, r: TestResult, job_id: str):
    section("GET /api/v1/jobs/{job_id}")

    # Valid job
    try:
        resp = await client.get(f"/api/v1/jobs/{job_id}")
        _assert(resp.status_code == 200, f"status {resp.status_code}")
        body = resp.json()
        for field in ("job_id", "status", "progress", "created_at", "updated_at"):
            _assert(field in body, f"missing field: {field}")
        r.ok("GET /api/v1/jobs/{job_id}", f"status={body['status']}")
    except Exception as e:
        r.fail("GET /api/v1/jobs/{job_id}", str(e))

    # Non-existent job → 404
    try:
        resp = await client.get("/api/v1/jobs/does-not-exist-000")
        _assert(resp.status_code == 404, f"expected 404, got {resp.status_code}")
        r.ok("GET /api/v1/jobs/nonexistent → 404")
    except Exception as e:
        r.fail("GET /api/v1/jobs/nonexistent → 404", str(e))

    # Completed job has result fields
    try:
        resp = await client.get(f"/api/v1/jobs/{job_id}")
        body = resp.json()
        if body["status"] == "completed":
            _assert(body.get("result") is not None, "missing result on completed job")
            _assert("download_url" in body["result"], "missing download_url")
            r.ok("completed job has result.download_url")
        else:
            r.skip(
                "completed job has result.download_url", f"job status={body['status']}"
            )
    except Exception as e:
        r.fail("completed job has result.download_url", str(e))


async def test_list_jobs(client: httpx.AsyncClient, r: TestResult):
    section("GET /api/v1/jobs")

    try:
        resp = await client.get("/api/v1/jobs")
        _assert(resp.status_code == 200, f"status {resp.status_code}")
        body = resp.json()
        for field in ("jobs", "total", "page", "per_page"):
            _assert(field in body, f"missing field: {field}")
        _assert(isinstance(body["jobs"], list), "jobs is not a list")
        r.ok("GET /api/v1/jobs", f"total={body['total']}")
    except Exception as e:
        r.fail("GET /api/v1/jobs", str(e))

    # Pagination
    try:
        resp = await client.get("/api/v1/jobs?page=1&per_page=1")
        _assert(resp.status_code == 200, f"status {resp.status_code}")
        body = resp.json()
        _assert(body["per_page"] == 1, f"per_page should be 1, got {body['per_page']}")
        _assert(len(body["jobs"]) <= 1, "returned more than 1 job")
        r.ok("GET /api/v1/jobs?page=1&per_page=1 (pagination)")
    except Exception as e:
        r.fail("GET /api/v1/jobs pagination", str(e))

    # Invalid pagination → 422
    try:
        resp = await client.get("/api/v1/jobs?page=0")
        _assert(resp.status_code == 422, f"expected 422, got {resp.status_code}")
        r.ok("GET /api/v1/jobs?page=0 → 422")
    except Exception as e:
        r.fail("GET /api/v1/jobs?page=0 → 422", str(e))


async def test_download_result(
    client: httpx.AsyncClient, r: TestResult, job_id: str, output_format: str
):
    section(f"GET /api/v1/jobs/{{job_id}}/result  (format={output_format})")

    try:
        resp = await client.get(f"/api/v1/jobs/{job_id}/result", timeout=30)
        _assert(
            resp.status_code == 200, f"status {resp.status_code}: {resp.text[:200]}"
        )

        content_type = resp.headers.get("content-type", "")
        if output_format == "json":
            data = resp.json()
            _assert("conversations" in data, "missing conversations key in JSON result")
            _assert(len(data["conversations"]) > 0, "conversations list is empty")
            r.ok(
                "GET /result → 200 (JSON)",
                f"content-type={content_type}  conversations={len(data['conversations'])}",
            )
        else:
            lines = [l for l in resp.text.strip().splitlines() if l.strip()]
            _assert(len(lines) > 0, "JSONL result is empty")
            first = json.loads(lines[0])
            _assert(
                "conversations" in first, "first JSONL line missing 'conversations'"
            )
            r.ok(
                "GET /result → 200 (JSONL)",
                f"content-type={content_type}  lines={len(lines)}",
            )
    except Exception as e:
        r.fail("GET /api/v1/jobs/{job_id}/result", str(e))

    # Non-existent job result → 404
    try:
        resp = await client.get("/api/v1/jobs/does-not-exist-000/result")
        _assert(resp.status_code == 404, f"expected 404, got {resp.status_code}")
        r.ok("GET /result for unknown job → 404")
    except Exception as e:
        r.fail("GET /result for unknown job → 404", str(e))


async def test_sse_stream(client: httpx.AsyncClient, r: TestResult):
    section("SSE  GET /api/v1/jobs/{job_id}/stream")

    # Submit a quick job to stream
    resp = await client.post(
        "/api/v1/generate",
        json={
            "document_text": SAMPLE_DOCUMENT,
            "num_dialogs": 2,
            "max_turns": 1,
            "max_concurrency": 2,
            "auto_generate_prompts": False,
            "respondent_prompt": "You are a helpful assistant.",
            "correspondent_prompt": "You are a curious user.",
            "use_personas": False,
            "model_name": "gemini-2.0-flash",
            "output_format": "jsonl",
        },
        timeout=30,
    )
    if resp.status_code != 202:
        r.skip("SSE stream events", f"job submission failed: {resp.status_code}")
        return

    job_id = resp.json()["job_id"]
    events: list[dict[str, str]] = []
    deadline = time.time() + 300

    try:
        async with client.stream(
            "GET", f"/api/v1/jobs/{job_id}/stream", timeout=300
        ) as stream:
            _assert(stream.status_code == 200, f"SSE status {stream.status_code}")
            async for line in stream.aiter_lines():
                if time.time() > deadline:
                    break
                line = line.strip()
                if line.startswith("event:"):
                    events.append({"event": line.split(":", 1)[1].strip()})
                elif line.startswith("data:") and events:
                    events[-1]["data"] = line.split(":", 1)[1].strip()
                    # Stop once we get a terminal event
                    if events[-1].get("event") in ("complete", "error", "cancelled"):
                        break

        event_names = [e.get("event") for e in events]
        _assert(len(events) > 0, "no SSE events received")
        r.ok("SSE stream returns events", f"events received: {event_names}")

        has_terminal = any(e in event_names for e in ("complete", "error", "cancelled"))
        if has_terminal:
            r.ok("SSE stream has terminal event", f"{event_names[-1]}")
        else:
            r.fail("SSE stream has terminal event", f"last events: {event_names[-3:]}")

        # Validate complete event payload
        complete_events = [e for e in events if e.get("event") == "complete"]
        if complete_events:
            payload = json.loads(complete_events[0].get("data", "{}"))
            _assert(
                "download_url" in payload,
                f"complete event missing download_url: {payload}",
            )
            r.ok("SSE complete event has download_url")

    except Exception as e:
        r.fail("SSE stream", str(e))


async def test_cancel_job(client: httpx.AsyncClient, r: TestResult):
    section("DELETE /api/v1/jobs/{job_id}  (cancel)")

    # Cancel a non-existent job → 404
    try:
        resp = await client.delete("/api/v1/jobs/does-not-exist-000")
        _assert(resp.status_code == 404, f"expected 404, got {resp.status_code}")
        r.ok("DELETE /jobs/nonexistent → 404")
    except Exception as e:
        r.fail("DELETE /jobs/nonexistent → 404", str(e))

    # Submit a large job and immediately cancel it
    try:
        resp = await client.post(
            "/api/v1/generate",
            json={
                "document_text": SAMPLE_DOCUMENT * 10,
                "num_dialogs": 50,
                "use_personas": False,
                "auto_generate_prompts": False,
                "respondent_prompt": "You are a helpful assistant.",
                "correspondent_prompt": "You are a curious user.",
                "model_name": "gemini-2.0-flash",
                "output_format": "jsonl",
            },
            timeout=10,
        )
        _assert(resp.status_code == 202, f"submit status {resp.status_code}")
        job_id = resp.json()["job_id"]

        await asyncio.sleep(1)  # Let it start

        resp = await client.delete(f"/api/v1/jobs/{job_id}")
        _assert(resp.status_code == 204, f"cancel status {resp.status_code}")
        r.ok("DELETE /jobs/{job_id} → 204")

        await asyncio.sleep(1)
        resp = await client.get(f"/api/v1/jobs/{job_id}")
        body = resp.json()
        _assert(
            body["status"] == "cancelled", f"expected cancelled, got {body['status']}"
        )
        r.ok("cancelled job has status=cancelled")

        # Try to cancel again → 409
        resp = await client.delete(f"/api/v1/jobs/{job_id}")
        _assert(resp.status_code == 409, f"expected 409, got {resp.status_code}")
        r.ok("DELETE already-cancelled job → 409")

    except Exception as e:
        r.fail("cancel job flow", str(e))


async def test_file_upload(client: httpx.AsyncClient, r: TestResult):
    section("POST /api/v1/generate/upload  (file upload)")

    try:
        content = SAMPLE_DOCUMENT.encode("utf-8")
        resp = await client.post(
            "/api/v1/generate/upload",
            files={"file": ("test_doc.txt", content, "text/plain")},
            data={
                "num_dialogs": "2",
                "use_personas": "false",
                "auto_generate_prompts": "false",
                "output_format": "jsonl",
            },
            timeout=30,
        )
        _assert(
            resp.status_code == 202, f"status {resp.status_code}: {resp.text[:300]}"
        )
        job_id = resp.json().get("job_id")
        _assert(job_id, "missing job_id")
        r.ok("POST /generate/upload → 202", f"job_id={job_id}")
    except Exception as e:
        r.fail("POST /generate/upload", str(e))

    # Non-UTF-8 file → 422
    try:
        resp = await client.post(
            "/api/v1/generate/upload",
            files={"file": ("bad.bin", b"\xff\xfe\x00", "application/octet-stream")},
            data={"num_dialogs": "2"},
            timeout=10,
        )
        _assert(resp.status_code == 422, f"expected 422, got {resp.status_code}")
        r.ok("POST /generate/upload (non-UTF8 → 422)")
    except Exception as e:
        r.fail("POST /generate/upload (non-UTF8 → 422)", str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────


async def run_tests(base_url: str, api_key: str | None, quick: bool):
    r = TestResult()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    print(f"\n{BOLD}AfterImage Server Test Playground{RESET}")
    print(f"Base URL : {CYAN}{base_url}{RESET}")
    print(f"API key  : {'set' if api_key else 'not set (auth disabled)'}")
    print(f"Mode     : {'quick (2 dialogs, skip SSE)' if quick else 'full'}")

    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, timeout=60
    ) as client:
        # Basic health — abort early if server unreachable
        await test_health(client, r)
        if r.failed and any("GET /health" in f[0] for f in r.failed):
            print(f"\n{RED}Server unreachable at {base_url}. Aborting.{RESET}")
            r.summary()
            return

        await test_docs(client, r)
        await test_analyze_document(client, r)

        # Full generation + poll (JSONL format)
        job_id_jsonl = await test_generate_and_poll(
            client, r, num_dialogs=2, output_format="jsonl"
        )
        if job_id_jsonl:
            await test_job_status(client, r, job_id_jsonl)
            await test_list_jobs(client, r)
            await test_download_result(client, r, job_id_jsonl, "jsonl")

        # Generation + poll (JSON format)
        job_id_json = await test_generate_and_poll(
            client, r, num_dialogs=2, output_format="json"
        )
        if job_id_json:
            await test_download_result(client, r, job_id_json, "json")

        # SSE streaming (skipped in quick mode to keep runtime down)
        if not quick:
            await test_sse_stream(client, r)
        else:
            r.skipped.append("SSE stream")
            section("SSE  GET /api/v1/jobs/{job_id}/stream")
            print(f"  {YELLOW}⊘ SKIP{RESET}  SSE stream  (--quick)")

        await test_cancel_job(client, r)
        await test_file_upload(client, r)

    success = r.summary()
    sys.exit(0 if success else 1)


def main():
    parser = argparse.ArgumentParser(description="AfterImage Server Test Playground")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key", default=None, help="Bearer token if server auth is enabled"
    )
    parser.add_argument("--quick", action="store_true", help="Skip SSE streaming test")
    args = parser.parse_args()

    asyncio.run(run_tests(args.base_url, args.api_key, args.quick))


if __name__ == "__main__":
    main()
