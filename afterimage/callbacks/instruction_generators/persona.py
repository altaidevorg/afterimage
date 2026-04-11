import math
import random
import threading
from collections import Counter
from typing import Any, Literal, Optional, Union

from ...key_management import SmartKeyPool
from ...prompts import default_persona_instruction_generation_prompt
from ...monitoring import GenerationMonitor
from ...providers import DocumentProvider
from ...types import Document
from ._utils import context_ids_from_documents, persona_fields_from_candidate
from .contextual import ContextualInstructionGeneratorCallback
from .persona_sampling import PersonaCandidate, PersonaSelectionState


class PersonaInstructionGeneratorCallback(ContextualInstructionGeneratorCallback):
    """Generates instructions based on randomly sampled contexts and personas.

    It works very similarly to `~ContextualInstructionGeneratorCallback` but it also samples a persona from the sampled documents.
        This usually results in more diverse yet still contextually relevant instructions.

    Args:
        api_key: API key for the generative AI service.
        documents: Either a list of texts or a DocumentProvider instance providing context to ground the instructions.
            For each round of generation `num_random_contexts` documents are sampled from this collection.
        prompt: Prompt that guides the instruction generation. If None, uses the default instruction generation prompt.
        model_name: Model name to use.
        model_provider_name: Model provider name to use.
        num_random_contexts: Number of contexts to sample for each round of generation.
        n_instructions: Number of instructions to generate in each round of generation.
        separator_text: Separator text for merging contexts if more than one context is sampled.
        safety_settings: Safety settings for the model. Mainly intended for Gemini models.
            Deprecated and may be removed in the future.
        monitor: GenerationMonitor instance to use for tracking.
            If `None`, Conversation and/or structured generators will set their own monitor"""

    def __init__(
        self,
        api_key: str | SmartKeyPool,
        documents: Union[list[str], DocumentProvider],
        prompt: str | None = None,
        model_name: str | None = None,
        model_provider_name: Literal[
            "gemini", "openai", "deepseek", "local"
        ] = "gemini",
        num_random_contexts: int = 1,
        n_instructions: int = 3,
        separator_text: str = "\n" + "-" * 80 + "\n\n",
        safety_settings: Optional[dict] = None,
        monitor: GenerationMonitor | None = None,
        llm_create_extras: dict[str, Any] | None = None,
    ):
        super().__init__(
            api_key=api_key,
            documents=documents,
            prompt=prompt
            if prompt is not None
            else default_persona_instruction_generation_prompt,
            model_name=model_name,
            model_provider_name=model_provider_name,
            num_random_contexts=num_random_contexts,
            n_instructions=n_instructions,
            separator_text=separator_text,
            safety_settings=safety_settings,
            monitor=monitor,
            llm_create_extras=llm_create_extras,
        )
        self._persona_target_per_document: int | None = None
        self._persona_selection_state: dict[str, PersonaSelectionState] = {}
        self._persona_selection_lock = threading.Lock()

    def _resolve_persona_target_from_context_usage(
        self,
        target_context_usage_count: int,
    ) -> int:
        contexts_per_row = max(int(self.num_random_contexts), 1)
        return max(math.ceil(target_context_usage_count / contexts_per_row), 1)

    def configure_persona_sampling(self, num_requested: int | None = None) -> None:
        with self._persona_selection_lock:
            self._persona_selection_state = {}

        target_context_usage_count = None
        if hasattr(self.provider, "get_target_context_usage_count"):
            target_context_usage_count = self.provider.get_target_context_usage_count()
        else:
            target_context_usage_count = getattr(
                self.provider,
                "target_context_usage_count",
                None,
            )

        if (
            isinstance(target_context_usage_count, int)
            and target_context_usage_count > 0
        ):
            self._persona_target_per_document = (
                self._resolve_persona_target_from_context_usage(
                    target_context_usage_count
                )
            )
            return

        if num_requested is None:
            self._persona_target_per_document = None
            return

        all_docs = self.provider.get_all()
        if hasattr(self.provider, "_doc_sampling_weights"):
            active_doc_count = sum(
                1
                for doc in all_docs
                if self.provider._doc_sampling_weights.get(doc.id, 0.0) > 0
            )
        else:
            active_doc_count = len(all_docs)

        active_doc_count = max(active_doc_count, 1)
        requested = max(int(num_requested), 1)
        inferred_target = math.ceil(requested / active_doc_count)
        self._persona_target_per_document = max(inferred_target, 1)

    def _normalize_generation_depth(self, raw_depth) -> int:
        try:
            depth = int(raw_depth)
        except (TypeError, ValueError):
            return 0
        return depth if depth >= 0 else 0

    def _flatten_document_personas(self, doc: Document) -> list[PersonaCandidate]:
        candidates: list[PersonaCandidate] = []

        for persona_entry in doc.personas:
            metadata = getattr(persona_entry, "metadata", {}) or {}
            generation_depth = self._normalize_generation_depth(
                metadata.get("generation_depth")
            )
            for description in persona_entry.descriptions:
                if not description:
                    continue
                candidates.append(
                    PersonaCandidate(
                        text=description,
                        generation_depth=generation_depth,
                    )
                )

        return sorted(candidates, key=lambda candidate: candidate.generation_depth)

    def _build_persona_selection_state(self, doc: Document) -> PersonaSelectionState:
        persona_candidates = self._flatten_document_personas(doc)
        if not persona_candidates:
            return PersonaSelectionState(mode="cycle")

        target = self._persona_target_per_document
        if target is None:
            return PersonaSelectionState(mode="cycle", active_pool=persona_candidates)

        total_personas = len(persona_candidates)
        target = max(target, 1)

        if target <= total_personas:
            active_pool = (
                persona_candidates
                if target == total_personas
                else persona_candidates[:target]
            )
            return PersonaSelectionState(mode="cycle", active_pool=active_pool)

        max_depth = max(candidate.generation_depth for candidate in persona_candidates)
        depth_counts = Counter(
            candidate.generation_depth for candidate in persona_candidates
        )
        weights = [
            float((max_depth - candidate.generation_depth) + 1)
            / depth_counts[candidate.generation_depth]
            for candidate in persona_candidates
        ]
        return PersonaSelectionState(
            mode="weighted",
            population=persona_candidates,
            weights=weights,
        )

    def _get_persona_selection_state(self, doc: Document) -> PersonaSelectionState:
        with self._persona_selection_lock:
            state = self._persona_selection_state.get(doc.id)
            if state is None:
                state = self._build_persona_selection_state(doc)
                self._persona_selection_state[doc.id] = state
            return state

    def _sample_persona_candidate(
        self,
        docs: list[Document],
    ) -> PersonaCandidate | None:
        docs_with_personas = [
            doc for doc in docs if self._flatten_document_personas(doc)
        ]
        if not docs_with_personas:
            return None

        selected_doc = random.choice(docs_with_personas)
        return self._get_persona_selection_state(selected_doc).next_candidate()

    def _sample(self) -> tuple[list[Document], PersonaCandidate | None]:
        docs = self.provider.get_documents(self.num_random_contexts)
        return docs, self._sample_persona_candidate(docs)

    def _run_persona_generation(self, original_prompt: str):
        random_contexts, persona_candidate = self._sample()
        persona, persona_generation_depth = persona_fields_from_candidate(
            persona_candidate
        )

        system_prompt = self.prompt
        if "{persona}" in system_prompt:
            system_prompt = system_prompt.format(persona=persona)

        model = self._create_model(system_instruction=system_prompt)
        full_context = self._merge_contexts([c.text for c in random_contexts])
        prompt = self._format_contextual_prompt(original_prompt, full_context)
        context_id, context_ids = context_ids_from_documents(random_contexts)
        return (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        )

    def generate(self, original_prompt):
        """Generates instructions based on the provided prompt, sampled context and persona.

        Args:
            original_prompt (str): The prompt guiding instruction generation.

        Returns:
            GeneratedInstructions: The instructions generated along with the context and persona used.
        """
        (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        ) = self._run_persona_generation(original_prompt)
        return self._execute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
            persona=persona,
            persona_generation_depth=persona_generation_depth,
        )

    async def agenerate(self, original_prompt):
        """Generates instructions based on the provided prompt, sampled context and persona asynchronously."""
        (
            model,
            prompt,
            full_context,
            context_id,
            context_ids,
            persona,
            persona_generation_depth,
        ) = self._run_persona_generation(original_prompt)
        return await self._aexecute_generation(
            model=model,
            prompt=prompt,
            full_context=full_context,
            context_id=context_id,
            context_ids=context_ids,
            persona=persona,
            persona_generation_depth=persona_generation_depth,
        )
