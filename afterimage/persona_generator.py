import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Literal

from tqdm import tqdm

from .common import default_model_name, default_safety_settings
from .key_management import SmartKeyPool
from .providers import LLMFactory, DocumentProvider
from .storage import BaseStorage, JSONLStorage
from .types import PersonaEntry
from .monitoring import GenerationMonitor
from .prompts import persona_generation_prompt_tmpl, parse_personas
from datetime import datetime


class PersonaGenerator:
    def __init__(
        self,
        api_key: str | SmartKeyPool,
        model_name: str | None = None,
        safety_settings: List[Dict[str, str]] | None = None,
        model_provider_name: Literal["gemini", "openai"] = "gemini",
        storage: Optional[BaseStorage] = None,
        monitor: Optional[GenerationMonitor] = None,
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

    def generate(self, text: str) -> list[str]:
        api_key = self.key_pool.get_next_key()
        llm = LLMFactory.create(
            self.model_provider_name,
            self.model_name,
            api_key,
            safety_settings=self.safety_settings,
        )
        start_time = time.time()
        try:
            prompt = persona_generation_prompt_tmpl.format(text=text)
            response = llm.generate_content(prompt)
            personas = parse_personas(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    metadata={
                        "operation": "persona_generation",
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
                        "operation": "persona_generation",
                        "text_length": len(text),
                    },
                )
            raise

    async def generate_async(self, text: str) -> list[str]:
        api_key = await self.key_pool.aget_next_key()
        llm = LLMFactory.create(
            self.model_provider_name,
            self.model_name,
            api_key,
            safety_settings=self.safety_settings,
        )
        start_time = time.time()
        try:
            prompt = persona_generation_prompt_tmpl.format(text=text)
            response = await llm.agenerate_content(prompt)
            personas = parse_personas(response.text)
            if self.monitor:
                self.monitor.track_generation(
                    duration=time.time() - start_time,
                    success=True,
                    metadata={
                        "operation": "persona_generation",
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
                        "operation": "persona_generation",
                        "text_length": len(text),
                    },
                )
            raise

    async def generate_for_documents(
        self,
        documents: Union[DocumentProvider, list[str]],
        max_docs: int | None = None,
        max_concurrency: int = 4,
    ):
        if isinstance(documents, list):
            docs_to_process = documents
        else:
            docs_to_process = documents.get_documents(n=max_docs)

        semaphore = asyncio.Semaphore(max_concurrency)
        pbar = tqdm(total=len(docs_to_process), desc="Generating Personas...")

        async def worker_task(doc_text):
            async with semaphore:
                personas = await self.generate_async(doc_text)
                if personas:
                    entry = PersonaEntry(
                        source_document=doc_text,
                        personas=personas,
                        timestamp=datetime.now(),
                    )
                    if self.storage:
                        await self.storage.asave_personas([entry])
                pbar.update(1)

        tasks = [asyncio.create_task(worker_task(doc)) for doc in docs_to_process]

        for future in asyncio.as_completed(tasks):
            try:
                await future
            except Exception as e:
                logging.error(f"A task failed: {e}", exc_info=True)

        pbar.close()
