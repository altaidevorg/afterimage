"""Orchestrates the full generation pipeline, mapping generate_qa.py stages to services."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from afterimage import (
    AsyncConversationGenerator,
    InMemoryDocumentProvider,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    WithContextRespondentPromptModifier,
    JSONLStorage,
)

from ..models import (
    AnalyzeDocumentResponse,
    GenerationPhase,
    GenerationRequest,
    JobProgress,
)
from ..storage.result_store import ResultStore
from .prompt_analyzer import PromptAnalyzer

ProgressCallback = Callable[[JobProgress], Awaitable[None]]


@dataclass
class GenerationResult:
    num_conversations: int
    result_path: str
    output_format: str
    system_prompt_parts: list[str] = field(default_factory=list)


class GenerationService:
    """Runs the full AfterImage generation pipeline for a single job."""

    def __init__(self, result_store: ResultStore):
        self._result_store = result_store

    async def run(
        self,
        job_id: str,
        request: GenerationRequest,
        progress_callback: ProgressCallback | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> GenerationResult:
        start_time = time.monotonic()

        async def _report(
            phase: GenerationPhase, completed: int = 0, total: int = 0
        ) -> None:
            if progress_callback is None:
                return
            elapsed = time.monotonic() - start_time
            pct = (completed / total * 100.0) if total > 0 else 0.0
            remaining: float | None = None
            if completed > 0 and total > 0 and elapsed > 0:
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else None
            await progress_callback(
                JobProgress(
                    completed=completed,
                    total=total,
                    percent=round(pct, 1),
                    current_phase=phase,
                    elapsed_seconds=round(elapsed, 1),
                    estimated_remaining_seconds=round(remaining, 1)
                    if remaining is not None
                    else None,
                )
            )

        def _check_cancel() -> None:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("Job cancelled by user")

        # ------------------------------------------------------------------
        # Phase 1: Resolve API key
        # ------------------------------------------------------------------
        from ..config import get_config

        config = get_config()
        api_key = config.get_api_key(request.model_provider_name)
        if not api_key:
            raise ValueError(
                f"No API key configured for provider '{request.model_provider_name}'. "
                "Set the corresponding AFTERIMAGE_*_API_KEY environment variable."
            )

        # ------------------------------------------------------------------
        # Phase 2: Prepare document chunks
        # ------------------------------------------------------------------
        chunks = self._prepare_chunks(request)
        docs = InMemoryDocumentProvider(chunks)

        # ------------------------------------------------------------------
        # Phase 3: Optionally analyze document for auto prompts
        # ------------------------------------------------------------------
        prompt_parts: AnalyzeDocumentResponse | None = None
        system_prompt_parts_list: list[str] = []

        if request.auto_generate_prompts and not (
            request.respondent_prompt and request.correspondent_prompt
        ):
            await _report("analyzing_document", 0, request.num_dialogs)
            _check_cancel()
            analyzer = PromptAnalyzer(api_key=api_key, model_name=request.model_name)
            source_text = chunks[0] if chunks else ""
            prompt_parts = await analyzer.analyze(source_text)

        lang = request.force_language
        lang_rule = f"ALWAYS respond in {lang.upper()}." if lang else ""

        respondent_prompt = request.respondent_prompt or (
            f"{prompt_parts.respondent_role} {prompt_parts.instruction} {lang_rule}".strip()
            if prompt_parts
            else f"You are a helpful expert. {lang_rule}".strip()
        )
        # If the caller supplied their own respondent_prompt but force_language is set,
        # append the language rule so WithContextRespondentPromptModifier cannot override it.
        if lang and lang_rule not in respondent_prompt:
            respondent_prompt = f"{respondent_prompt} {lang_rule}"

        correspondent_prompt = request.correspondent_prompt or (
            prompt_parts.correspondent_role
            if prompt_parts
            else "You are a curious user."
        )

        if request.include_system_prompt_parts and prompt_parts:
            system_prompt_parts_list = [
                prompt_parts.respondent_role,
                prompt_parts.instruction,
            ]

        # ------------------------------------------------------------------
        # Phase 4: Generate personas (optional)
        # ------------------------------------------------------------------
        if request.use_personas:
            await _report("generating_personas", 0, request.num_dialogs)
            _check_cancel()
            persona_gen = PersonaGenerator(api_key=api_key)
            # Cap max_docs so we don't generate personas for every chunk when
            # num_dialogs is small. Each doc yields ~3-5 personas; num_dialogs
            # is a safe upper bound — we'll never need more docs than dialogs.
            max_docs = min(request.num_dialogs, len(chunks))
            await persona_gen.generate_from_documents(
                docs, n_iterations=request.persona_iterations, max_docs=max_docs
            )

        # ------------------------------------------------------------------
        # Phase 5: Build instruction callback and prompt modifier
        # ------------------------------------------------------------------
        await _report("initializing", 0, request.num_dialogs)
        _check_cancel()

        # Resolve the instruction prompt: explicit > auto-built language-enforcing > library default
        resolved_instruction_prompt: str | None = request.custom_instruction_prompt
        if not resolved_instruction_prompt and lang:
            resolved_instruction_prompt = _build_language_instruction_prompt(lang)

        instruction_callback: Any
        if request.use_personas:
            cb_kwargs: dict[str, Any] = dict(
                api_key=api_key,
                documents=docs,
                num_random_contexts=1,
                n_instructions=1,
            )
            if resolved_instruction_prompt:
                cb_kwargs["prompt"] = resolved_instruction_prompt
            instruction_callback = PersonaInstructionGeneratorCallback(**cb_kwargs)
        else:
            from afterimage import ContextualInstructionGeneratorCallback

            cb_kwargs2: dict[str, Any] = dict(
                api_key=api_key,
                documents=docs,
                num_random_contexts=1,
            )
            if resolved_instruction_prompt:
                cb_kwargs2["prompt"] = resolved_instruction_prompt
            instruction_callback = ContextualInstructionGeneratorCallback(**cb_kwargs2)

        prompt_modifier = WithContextRespondentPromptModifier()

        # ------------------------------------------------------------------
        # Phase 6: Set up storage
        # ------------------------------------------------------------------
        job_dir = self._result_store.job_dir(job_id)
        conversations_path = job_dir / "conversations.jsonl"
        storage = _ProgressStorage(
            conversations_path=str(conversations_path),
            total=request.num_dialogs,
            report_fn=lambda completed: _report(
                "generating", completed, request.num_dialogs
            ),
        )

        # ------------------------------------------------------------------
        # Phase 7: Create and run the generator
        # ------------------------------------------------------------------
        await _report("generating", 0, request.num_dialogs)
        _check_cancel()

        generator = AsyncConversationGenerator(
            respondent_prompt=respondent_prompt,
            correspondent_prompt=correspondent_prompt,
            api_key=api_key,
            model_name=request.model_name,
            instruction_generator_callback=instruction_callback,
            respondent_prompt_modifier=prompt_modifier,
            storage=storage,
        )

        await generator.generate(
            num_dialogs=request.num_dialogs,
            max_turns=request.max_turns,
            max_concurrency=request.max_concurrency,
        )

        _check_cancel()

        # ------------------------------------------------------------------
        # Phase 8: Save results
        # ------------------------------------------------------------------
        await _report("saving", request.num_dialogs, request.num_dialogs)
        conversations = generator.load_conversations()
        num_conversations = len(conversations)

        if request.output_format == "json":
            conversations_data = []
            for conv in conversations:
                if hasattr(conv, "model_dump"):
                    conversations_data.append(conv.model_dump())
                else:
                    conversations_data.append(
                        {
                            "persona": getattr(conv, "persona", None),
                            "conversations": [
                                {"role": t.role, "content": t.content}
                                for t in getattr(conv, "conversations", [])
                            ],
                        }
                    )
            result_path = str(
                self._result_store.save_conversations_json(
                    job_id,
                    conversations_data,
                    system_prompt_parts=system_prompt_parts_list or None,
                )
            )
        else:
            result_path = str(conversations_path)

        await _report("complete", num_conversations, num_conversations)

        return GenerationResult(
            num_conversations=num_conversations,
            result_path=result_path,
            output_format=request.output_format,
            system_prompt_parts=system_prompt_parts_list,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_chunks(request: GenerationRequest) -> list[str]:
        if request.document_chunks:
            return request.document_chunks
        if request.document_text:
            text = request.document_text
            size = request.chunk_size
            return [text[i : i + size] for i in range(0, len(text), size)]
        raise ValueError(
            "No document provided. Supply one of: document_text, document_chunks."
        )


def _build_language_instruction_prompt(language: str) -> str:
    """Return an instruction prompt that overrides AfterImage's default
    'same language as context' rule and enforces a specific output language."""
    lang_upper = language.upper()
    return (
        "You are an expert actor and roleplayer.\n"
        "You will be given a persona description and a context.\n"
        "Ask {n_instructions} questions that a person matching your persona would ask this expert.\n\n"
        "Persona:\n{persona}\n\n"
        "Rules:\n"
        f"1. EVERYTHING MUST BE IN {lang_upper}.\n"
        f"2. STRICTLY FORBIDDEN: DO NOT USE ANY LANGUAGE OTHER THAN {lang_upper}.\n"
        f"3. BE CONSISTENT WITH YOUR PERSONA BUT USE ONLY {lang_upper}.\n"
        "4. Ask questions relevant to the context, but do not directly quote or reference the context."
    )


class _ProgressStorage(JSONLStorage):
    """JSONLStorage that fires an async progress callback on every batch save.

    JSONLStorage.asave_conversations offloads writes to a ThreadPoolExecutor, which
    calls save_conversations from a worker thread with no event loop.  We capture
    the running loop at construction time and use run_coroutine_threadsafe so the
    callback is always scheduled on the correct loop regardless of calling thread.
    """

    def __init__(
        self,
        conversations_path: str,
        total: int,
        report_fn: Callable[[int], Any],
    ):
        super().__init__(conversations_path=conversations_path)
        self._total = total
        self._report_fn = report_fn  # callable(completed) -> coroutine
        self._saved = 0
        # Capture the running loop — __init__ is always called from async context
        self._loop = asyncio.get_running_loop()

    def _fire(self, completed: int) -> None:
        """Schedule the async report_fn on the event loop from any thread."""
        asyncio.run_coroutine_threadsafe(self._report_fn(completed), self._loop)

    def save_conversations(self, conversations):  # type: ignore[override]
        super().save_conversations(conversations)
        self._saved += len(conversations)
        self._fire(self._saved)

    async def asave_conversations(self, conversations):  # type: ignore[override]
        # The parent dispatches to run_in_executor → save_conversations (our override),
        # which handles _saved increment and _fire. No double-counting needed here.
        await super().asave_conversations(conversations)
