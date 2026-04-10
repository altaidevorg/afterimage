import asyncio
import logging
import math
import time
from typing import Literal, Optional, Union

from tqdm import tqdm

from .common import (
    default_model_name,
    default_safety_settings,
    resolve_generation_max_concurrency,
)
from .key_management import SmartKeyPool
from .monitoring import GenerationMonitor
from .prompts import (
    parse_personas,
    persona_to_persona_generation_prompt_tmpl,
    text_to_persona_generation_prompt_tmpl,
)
from .providers import DocumentProvider, InMemoryDocumentProvider, LLMFactory
from .storage import BaseStorage, JSONLStorage
from .types import Document, PersonaEntry


EXPECTED_PERSONA_COUNT = 5
MAX_PERSONA_GENERATION_ATTEMPTS = 3


class PersonaGenerationContractError(ValueError):
    """Raised when persona generation cannot satisfy the fixed-width contract."""


class PersonaGenerator:
    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        model_provider_name: Literal["gemini", "openai", "deepseek"] = "gemini",
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
        max_concurrency: int | None = None,
    ):
        self.key_pool = (
            api_key
            if isinstance(api_key, SmartKeyPool)
            else SmartKeyPool.from_single_key(api_key)
        )

        self.model_provider_name = model_provider_name
        self.model_name = model_name if model_name is not None else default_model_name
        self.safety_settings = (
            safety_settings if safety_settings is not None else default_safety_settings
        )

        self.storage = storage or JSONLStorage()
        self.monitor = monitor
        self.max_concurrency = self._resolve_max_concurrency(max_concurrency)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    def _resolve_max_concurrency(self, max_concurrency: int | None) -> int:
        return resolve_generation_max_concurrency(
            self.model_provider_name,
            max_concurrency,
        )

    def _normalize_persona_text(self, persona: str) -> str:
        return " ".join(persona.split()).strip()

    def _normalize_personas(self, personas: list[str]) -> list[str]:
        normalized_personas: list[str] = []
        seen: set[str] = set()

        for persona in personas:
            normalized = self._normalize_persona_text(persona)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_personas.append(normalized)

        return normalized_personas

    def _select_fixed_width_personas(self, personas: list[str]) -> list[str]:
        normalized_personas = self._normalize_personas(personas)
        if len(normalized_personas) < EXPECTED_PERSONA_COUNT:
            raise PersonaGenerationContractError(
                "Persona generation did not return enough unique personas"
            )
        return normalized_personas[:EXPECTED_PERSONA_COUNT]

    def _generate_personas_with_retry(
        self,
        generate_content,
        prompt: str,
    ):
        last_error: PersonaGenerationContractError | None = None

        for _ in range(MAX_PERSONA_GENERATION_ATTEMPTS):
            response = generate_content(prompt)
            try:
                personas = self._select_fixed_width_personas(
                    parse_personas(response.text)
                )
            except PersonaGenerationContractError as error:
                last_error = error
                continue
            return personas, response

        raise last_error or PersonaGenerationContractError(
            "Persona generation could not satisfy the fixed-width contract"
        )

    async def _agenerate_personas_with_retry(
        self,
        generate_content,
        prompt: str,
    ):
        last_error: PersonaGenerationContractError | None = None

        for _ in range(MAX_PERSONA_GENERATION_ATTEMPTS):
            response = await generate_content(prompt)
            try:
                personas = self._select_fixed_width_personas(
                    parse_personas(response.text)
                )
            except PersonaGenerationContractError as error:
                last_error = error
                continue
            return personas, response

        raise last_error or PersonaGenerationContractError(
            "Persona generation could not satisfy the fixed-width contract"
        )

    def _track_success(
        self,
        *,
        start_time: float,
        response,
        operation: str,
        text_length: int,
        generation: int | None = None,
    ) -> None:
        if self.monitor is None:
            return

        metadata = {
            "operation": operation,
            "text_length": text_length,
        }
        if generation is not None:
            metadata["generation"] = generation

        self.monitor.track_generation(
            duration=time.time() - start_time,
            success=True,
            prompt_token_count=response.prompt_token_count,
            completion_token_count=response.completion_token_count,
            total_token_count=response.total_token_count,
            model_name=response.model_name,
            metadata=metadata,
        )

    def _track_failure(
        self,
        *,
        start_time: float,
        error: Exception,
        operation: str,
        text_length: int,
        generation: int | None = None,
    ) -> None:
        if self.monitor is None:
            return

        metadata = {
            "operation": operation,
            "text_length": text_length,
        }
        if generation is not None:
            metadata["generation"] = generation

        self.monitor.track_generation(
            duration=time.time() - start_time,
            success=False,
            error=str(error),
            metadata=metadata,
        )

    def expected_persona_count(self, n_iterations: int) -> int:
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")

        total = 0
        layer_size = EXPECTED_PERSONA_COUNT
        for _ in range(n_iterations + 1):
            total += layer_size
            layer_size *= EXPECTED_PERSONA_COUNT
        return total

    def _resolve_active_doc_count(
        self,
        docs_to_process: list[Document],
        provider,
    ) -> int:
        if hasattr(provider, "_doc_sampling_weights"):
            active_doc_count = sum(
                1
                for doc in docs_to_process
                if provider._doc_sampling_weights.get(doc.id, 0.0) > 0
            )
        else:
            active_doc_count = len(docs_to_process)

        return max(active_doc_count, 1)

    def _resolve_target_per_document(
        self,
        provider,
        docs_to_process: list[Document],
        target_data_count: int | None,
        num_random_contexts: int,
    ) -> int | None:
        target_context_usage_count = None
        if hasattr(provider, "get_target_context_usage_count"):
            target_context_usage_count = provider.get_target_context_usage_count()
        else:
            target_context_usage_count = getattr(
                provider,
                "target_context_usage_count",
                None,
            )

        if (
            isinstance(target_context_usage_count, int)
            and target_context_usage_count > 0
        ):
            contexts_per_row = max(int(num_random_contexts), 1)
            return max(math.ceil(target_context_usage_count / contexts_per_row), 1)

        if target_data_count is None:
            return None

        requested = max(int(target_data_count), 1)
        active_doc_count = self._resolve_active_doc_count(docs_to_process, provider)
        return max(math.ceil(requested / active_doc_count), 1)

    def _resolve_auto_n_iterations(self, target_per_document: int | None) -> int:
        if target_per_document is None or target_per_document <= 0:
            return 0

        current_n = 0
        current_total = self.expected_persona_count(current_n)
        if target_per_document <= current_total:
            return current_n

        previous_n = current_n
        previous_total = current_total

        while current_total < target_per_document:
            previous_n = current_n
            previous_total = current_total
            current_n += 1
            current_total = self.expected_persona_count(current_n)

        if (target_per_document - previous_total) <= (
            current_total - target_per_document
        ):
            return previous_n
        return current_n

    def _resolve_generation_iterations(
        self,
        *,
        provider,
        docs_to_process: list[Document],
        n_iterations: int | None,
        target_data_count: int | None,
        num_random_contexts: int,
    ) -> int:
        if n_iterations is not None:
            if n_iterations < 0:
                raise ValueError("n_iterations must be >= 0")
            return n_iterations

        target_per_document = self._resolve_target_per_document(
            provider=provider,
            docs_to_process=docs_to_process,
            target_data_count=target_data_count,
            num_random_contexts=num_random_contexts,
        )
        resolved_iterations = self._resolve_auto_n_iterations(target_per_document)

        if self.monitor is not None:
            self.monitor.log_info(
                "Resolved persona generation iterations",
                n_iterations=resolved_iterations,
                target_per_document=target_per_document,
                active_doc_count=self._resolve_active_doc_count(
                    docs_to_process, provider
                ),
            )

        return resolved_iterations

    def generate_from_text(self, text: str) -> list[str]:
        api_key = self.key_pool.get_next_key()
        llm = LLMFactory.create(
            self.model_provider_name,
            self.model_name,
            api_key,
            safety_settings=self.safety_settings,
        )
        start_time = time.time()
        try:
            prompt = text_to_persona_generation_prompt_tmpl.format(text=text)
            personas, response = self._generate_personas_with_retry(
                llm.generate_content,
                prompt,
            )
            self._track_success(
                start_time=start_time,
                response=response,
                operation="text_to_persona_generation",
                text_length=len(text),
            )
            return personas
        except Exception as error:
            if not isinstance(error, PersonaGenerationContractError):
                self.key_pool.report_error(api_key)
            self._track_failure(
                start_time=start_time,
                error=error,
                operation="text_to_persona_generation",
                text_length=len(text),
            )
            raise

    async def agenerate_from_text(self, text: str) -> list[str]:
        async with self.semaphore:
            api_key = await self.key_pool.aget_next_key()
            llm = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key,
                safety_settings=self.safety_settings,
            )
            start_time = time.time()
            try:
                prompt = text_to_persona_generation_prompt_tmpl.format(text=text)
                personas, response = await self._agenerate_personas_with_retry(
                    llm.agenerate_content,
                    prompt,
                )
                self._track_success(
                    start_time=start_time,
                    response=response,
                    operation="text_to_persona_generation",
                    text_length=len(text),
                )
                return personas
            except Exception as error:
                if not isinstance(error, PersonaGenerationContractError):
                    await self.key_pool.areport_error(api_key)
                self._track_failure(
                    start_time=start_time,
                    error=error,
                    operation="text_to_persona_generation",
                    text_length=len(text),
                )
                raise

    def generate_from_persona(self, persona: str, generation: int = 1) -> list[str]:
        api_key = self.key_pool.get_next_key()
        llm = LLMFactory.create(
            self.model_provider_name,
            self.model_name,
            api_key,
            safety_settings=self.safety_settings,
        )
        start_time = time.time()
        try:
            prompt = persona_to_persona_generation_prompt_tmpl.format(personas=persona)
            personas, response = self._generate_personas_with_retry(
                llm.generate_content,
                prompt,
            )
            self._track_success(
                start_time=start_time,
                response=response,
                operation="persona_to_persona_generation",
                text_length=len(persona),
                generation=generation,
            )
            return personas
        except Exception as error:
            if not isinstance(error, PersonaGenerationContractError):
                self.key_pool.report_error(api_key)
            self._track_failure(
                start_time=start_time,
                error=error,
                operation="persona_to_persona_generation",
                text_length=len(persona),
                generation=generation,
            )
            raise

    async def agenerate_from_persona(
        self, persona: str, generation: int = 1
    ) -> list[str]:
        async with self.semaphore:
            api_key = await self.key_pool.aget_next_key()
            llm = LLMFactory.create(
                self.model_provider_name,
                self.model_name,
                api_key,
                safety_settings=self.safety_settings,
            )
            start_time = time.time()
            try:
                prompt = persona_to_persona_generation_prompt_tmpl.format(
                    personas=persona
                )
                personas, response = await self._agenerate_personas_with_retry(
                    llm.agenerate_content,
                    prompt,
                )
                self._track_success(
                    start_time=start_time,
                    response=response,
                    operation="persona_to_persona_generation",
                    text_length=len(persona),
                    generation=generation,
                )
                return personas
            except Exception as error:
                if not isinstance(error, PersonaGenerationContractError):
                    await self.key_pool.areport_error(api_key)
                self._track_failure(
                    start_time=start_time,
                    error=error,
                    operation="persona_to_persona_generation",
                    text_length=len(persona),
                    generation=generation,
                )
                raise

    async def _agenerate_persona_chains(
        self, base_personas: list[str], depth: int
    ) -> list[PersonaEntry]:
        all_entries = []
        current_personas = base_personas

        for i in range(depth):
            new_personas = []
            results = await asyncio.gather(
                *[
                    self.agenerate_from_persona(p, generation=i + 1)
                    for p in current_personas
                ],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    logging.warning(f"Persona generation failed: {result}")
                    continue
                new_personas.extend(result)

            entry = PersonaEntry(
                descriptions=new_personas,
                metadata={"generation_depth": i + 1},
            )
            all_entries.append(entry)
            current_personas = new_personas

            if not new_personas:
                break

        return all_entries

    async def generate_from_documents(
        self,
        documents: Union[DocumentProvider, list[str]],
        max_docs: int | None = None,
        n_iterations: int | None = None,
        target_data_count: int | None = None,
        num_random_contexts: int = 1,
    ):
        if isinstance(documents, list):
            documents = InMemoryDocumentProvider(documents)
        if max_docs is not None and max_docs < len(documents):
            docs_to_process = documents.get_documents(n=max_docs)
        else:
            docs_to_process = documents.get_all()

        resolved_iterations = self._resolve_generation_iterations(
            provider=documents,
            docs_to_process=docs_to_process,
            n_iterations=n_iterations,
            target_data_count=target_data_count,
            num_random_contexts=num_random_contexts,
        )

        pbar = tqdm(total=len(docs_to_process), desc="Generating Personas...")
        task_errors: list[Exception] = []
        completed_results: list[tuple[Document, Document]] = []

        async def worker_task(doc: Document):
            enriched_doc = doc.model_copy(deep=True)
            base_personas = await self.agenerate_from_text(doc.text)
            persona_entries = [
                PersonaEntry(
                    descriptions=base_personas, metadata={"generation_depth": 0}
                )
            ]
            if resolved_iterations > 0:
                deeper_personas = await self._agenerate_persona_chains(
                    base_personas, depth=resolved_iterations
                )
                persona_entries.extend(deeper_personas)

            pbar.update(1)
            enriched_doc.personas.extend(persona_entries)
            return doc, enriched_doc

        tasks = [asyncio.create_task(worker_task(doc)) for doc in docs_to_process]

        for future in asyncio.as_completed(tasks):
            try:
                original_doc, enriched_doc = await future
                completed_results.append((original_doc, enriched_doc))
            except Exception as e:
                logging.error(f"A task failed: {e}", exc_info=True)
                task_errors.append(e)

        pbar.close()

        if task_errors:
            raise task_errors[0]

        enriched_docs = [enriched_doc for _, enriched_doc in completed_results]
        if self.storage and enriched_docs:
            await self.storage.asave_documents(enriched_docs)

        for original_doc, enriched_doc in completed_results:
            original_doc.personas = enriched_doc.personas
