"""Sampling strategy for conversation and structured generation.

Coordinates persona selection, context/document selection, and instruction
dispatch. Extracts sampling-related logic from BaseGenerator so that the
orchestrator and generators do not need to understand sampling internals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .metadata_utils import extract_unique_context_ids

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Configures and tracks sampling state for document and persona selection.

    This class owns the logic for:
    - Inferring target context usage counts from stopping callbacks
    - Configuring document provider sampling weights
    - Configuring persona sampling targets on instruction callbacks
    - Recording context usage back to document providers after generation
    """

    def __init__(self, monitor: Optional[Any] = None):
        self._monitor = monitor

    @property
    def monitor(self) -> Optional[Any]:
        return self._monitor

    @monitor.setter
    def monitor(self, value: Any) -> None:
        self._monitor = value

    @staticmethod
    def iter_stopping_callbacks(callbacks) -> Any:
        """Yield callbacks, recursively flattening composite callback containers."""
        for callback in callbacks or []:
            yield callback
            nested_callbacks = getattr(callback, "_callbacks", None)
            if nested_callbacks:
                yield from SamplingStrategy.iter_stopping_callbacks(nested_callbacks)

    def infer_target_context_usage_count(
        self,
        provider,
        stopping_criteria,
    ) -> int | None:
        """Infer a context usage target from stopping callbacks bound to a provider."""
        inferred_targets: list[int] = []
        for callback in self.iter_stopping_callbacks(stopping_criteria):
            callback_provider = getattr(callback, "provider", None)
            target_visits = getattr(callback, "target_visits", None)
            if (
                callback_provider is provider
                and isinstance(target_visits, int)
                and target_visits > 0
            ):
                inferred_targets.append(target_visits)

        return max(inferred_targets) if inferred_targets else None

    def configure_context_sampling(
        self,
        instruction_generator_callback,
        stopping_criteria,
    ) -> None:
        """Configure provider sampling weights from stopping criteria when possible."""
        provider = getattr(instruction_generator_callback, "provider", None)
        if provider is None or not hasattr(provider, "set_target_context_usage_count"):
            return

        if getattr(provider, "_target_context_usage_count_explicit", False):
            return

        inferred_target = self.infer_target_context_usage_count(
            provider,
            stopping_criteria,
        )
        provider.set_target_context_usage_count(inferred_target)

        if self._monitor is not None:
            self._monitor.log_info(
                "Configured document sampling target",
                target_context_usage_count=inferred_target,
                provider_type=provider.__class__.__name__,
            )

    def configure_persona_sampling(
        self,
        instruction_generator_callback,
        num_requested: int | None,
        stopping_criteria=None,
    ) -> None:
        """Configure persona-aware callbacks with the effective request target."""
        configure_persona_sampling = getattr(
            instruction_generator_callback,
            "configure_persona_sampling",
            None,
        )
        if configure_persona_sampling is None:
            return

        inferred_num_requested = num_requested
        if inferred_num_requested is None:
            fixed_targets: list[int] = []
            for callback in self.iter_stopping_callbacks(stopping_criteria):
                callback_n = getattr(callback, "n", None)
                if (
                    callback.__class__.__name__ == "FixedNumberStoppingCallback"
                    and isinstance(callback_n, int)
                    and callback_n > 0
                ):
                    fixed_targets.append(callback_n)
            inferred_num_requested = max(fixed_targets) if fixed_targets else None

        configure_persona_sampling(num_requested=inferred_num_requested)

        if self._monitor is not None:
            self._monitor.log_info(
                "Configured persona sampling target",
                target_personas_per_document=getattr(
                    instruction_generator_callback,
                    "_persona_target_per_document",
                    None,
                ),
                callback_type=instruction_generator_callback.__class__.__name__,
            )

    @staticmethod
    def record_context_usage(
        instruction_generator_callback,
        item,
    ) -> None:
        """Report successful context usage back to the document provider."""
        provider = getattr(instruction_generator_callback, "provider", None)
        if provider is None or not hasattr(provider, "report_doc_usage"):
            return

        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, dict):
            return

        for context_id in extract_unique_context_ids(metadata):
            provider.report_doc_usage(context_id)
