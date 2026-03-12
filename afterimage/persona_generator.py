import asyncio
import logging
import time
from typing import Optional, Union, Literal

from tqdm import tqdm

from .common import (
    default_model_name,
    default_safety_settings,
    resolve_generation_max_concurrency,
)
from .key_management import SmartKeyPool
from .providers import LLMFactory, DocumentProvider, InMemoryDocumentProvider
from .storage import BaseStorage, JSONLStorage
from .types import PersonaEntry, Document
from .monitoring import GenerationMonitor
from .prompts import (
    text_to_persona_generation_prompt_tmpl,
    parse_personas,
    persona_to_persona_generation_prompt_tmpl,
)


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
            response = llm.generate_content(prompt)
            personas = parse_personas(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={
                        "operation": "text_to_persona_generation",
                        "text_length": len(text),
                    },
                )
            return personas
        except Exception as e:
            self.key_pool.report_error(api_key)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "text_to_persona_generation",
                        "text_length": len(text),
                    },
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
                response = await llm.agenerate_content(prompt)
                personas = parse_personas(response.text)
                if self.monitor:
                    self.monitor.track_generation(
                        duration=time.time() - start_time,
                        success=True,
                        prompt_token_count=response.prompt_token_count,
                        completion_token_count=response.completion_token_count,
                        total_token_count=response.total_token_count,
                        model_name=response.model_name,
                        metadata={
                            "operation": "text_to_persona_generation",
                            "text_length": len(text),
                        },
                    )
                return personas
            except Exception as e:
                await self.key_pool.areport_error(api_key)
                if self.monitor:
                    self.monitor.track_generation(
                        duration=time.time() - start_time,
                        success=False,
                        error=str(e),
                        metadata={
                            "operation": "text_to_persona_generation",
                            "text_length": len(text),
                        },
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
            response = llm.generate_content(prompt)
            personas = parse_personas(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    prompt_token_count=response.prompt_token_count,
                    completion_token_count=response.completion_token_count,
                    total_token_count=response.total_token_count,
                    model_name=response.model_name,
                    metadata={
                        "operation": "persona_to_persona_generation",
                        "text_length": len(persona),
                    },
                )
            return personas
        except Exception as e:
            self.key_pool.report_error(api_key)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e),
                    metadata={
                        "operation": "persona_to_persona_generation",
                        "text_length": len(persona),
                        "generation": generation,
                    },
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
                response = await llm.agenerate_content(prompt)
                personas = parse_personas(response.text)
                if self.monitor:
                    self.monitor.track_generation(
                        duration=time.time() - start_time,
                        success=True,
                        prompt_token_count=response.prompt_token_count,
                        completion_token_count=response.completion_token_count,
                        total_token_count=response.total_token_count,
                        model_name=response.model_name,
                        metadata={
                            "operation": "persona_to_persona_generation",
                            "text_length": len(persona),
                            "generation": generation,
                        },
                    )
                return personas
            except Exception as e:
                await self.key_pool.areport_error(api_key)
                if self.monitor:
                    self.monitor.track_generation(
                        duration=time.time() - start_time,
                        success=False,
                        error=str(e),
                        metadata={
                            "operation": "persona_to_persona_generation",
                            "text_length": len(persona),
                            "generation": generation,
                        },
                    )
                raise

    async def _agenerate_persona_chains(
        self, base_personas: list[str], depth: int
    ) -> list[PersonaEntry]:
        all_entries = []
        current_personas = base_personas

        for i in range(depth):
            new_personas = []
            # run persona→persona generation concurrently for current layer
            results = await asyncio.gather(
                *[
                    self.agenerate_from_persona(p, generation=i + 1)
                    for p in current_personas
                ],
                return_exceptions=True,
            )

            for r in results:
                if isinstance(r, Exception):
                    logging.warning(f"Persona generation failed: {r}")
                    continue
                new_personas.extend(r)

            entry = PersonaEntry(
                descriptions=new_personas,
                metadata={"generation_depth": i + 1},
            )
            all_entries.append(entry)
            current_personas = new_personas  # feed forward

            if not new_personas:
                break  # stop early if model stopped producing

        return all_entries

    async def generate_from_documents(
        self,
        documents: Union[DocumentProvider, list[str]],
        max_docs: int | None = None,
        n_iterations: int = 0,
    ):
        if isinstance(documents, list):
            documents = InMemoryDocumentProvider(documents)
        if max_docs is not None and max_docs < len(documents):
            docs_to_process = documents.get_documents(n=max_docs)
        else:
            docs_to_process = documents.get_all()

        pbar = tqdm(total=len(docs_to_process), desc="Generating Personas...")

        async def worker_task(doc: Document):
            base_personas = await self.agenerate_from_text(doc.text)
            if not base_personas:
                pbar.update(1)
                return

            doc.personas.append(
                PersonaEntry(
                    descriptions=base_personas, metadata={"generation_depth": 0}
                )
            )
            if n_iterations > 0:
                deeper_personas = await self._agenerate_persona_chains(
                    base_personas, depth=n_iterations
                )
                doc.personas.extend(deeper_personas)

            if self.storage:
                await self.storage.asave_documents([doc])

            pbar.update(1)

        tasks = [asyncio.create_task(worker_task(doc)) for doc in docs_to_process]

        for future in asyncio.as_completed(tasks):
            try:
                await future
            except Exception as e:
                logging.error(f"A task failed: {e}", exc_info=True)

        pbar.close()
