from ..base import (
    BaseStoppingCallback,
)
from ..providers import DocumentProvider
from ..types import (
    GenerationState,
)


class FixedNumberStoppingCallback(BaseStoppingCallback):
    """Stops after generating a fixed number of samples."""

    def __init__(self, n: int):
        self.n = n

    async def should_stop(self, state: GenerationState) -> bool:
        return state.num_generated >= self.n


class ContextCoverageStoppingCallback(BaseStoppingCallback):
    """Stops after all (or a percentage of) contexts have been used N times."""

    def __init__(
        self,
        provider: DocumentProvider,
        target_visits: int = 1,
        coverage_threshold: float = 1.0,
    ):
        self.provider = provider
        self.target_visits = target_visits
        self.coverage_threshold = coverage_threshold

    async def should_stop(self, state: GenerationState) -> bool:
        all_docs = self.provider.get_all()
        if not all_docs:
            return True

        covered_count = 0
        for doc in all_docs:
            context_count = state.context_counts.get(doc.id, 0)
            if context_count >= self.target_visits:
                covered_count += 1

        actual_coverage = covered_count / len(all_docs)

        return actual_coverage >= self.coverage_threshold


class PersonaUsageStoppingCallback(BaseStoppingCallback):
    """Stops after N unique personas have been utilized."""

    def __init__(self, n_personas: int):
        self.n_personas = n_personas

    async def should_stop(self, state: GenerationState) -> bool:
        return len(state.unique_personas) >= self.n_personas


class BudgetStoppingCallback(BaseStoppingCallback):
    """Stops when token usage exceeds a threshold.

    Args:
        max_prompt_tokens: Maximum number of prompt tokens to use.
        max_completion_tokens: Maximum number of completion tokens to use.
        max_total_tokens: Maximum number of total tokens to use.
    """

    def __init__(
        self,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_total_tokens = max_total_tokens

    async def should_stop(self, state: GenerationState) -> bool:
        if state.monitor is None:
            return False

        if (
            self.max_prompt_tokens is None
            and self.max_completion_tokens is None
            and self.max_total_tokens is None
        ):
            return False

        # We rely on internal _metrics of monitor
        with state.monitor._lock:
            if self.max_total_tokens is not None:
                total_tokens = sum(
                    m["value"] for m in state.monitor._metrics["total_token_count"]
                )
                if total_tokens >= self.max_total_tokens:
                    return True

            if self.max_completion_tokens is not None:
                total_tokens = sum(
                    m["value"] for m in state.monitor._metrics["completion_token_count"]
                )
                if total_tokens >= self.max_completion_tokens:
                    return True

            if self.max_prompt_tokens is not None:
                total_tokens = sum(
                    m["value"] for m in state.monitor._metrics["prompt_token_count"]
                )
                if total_tokens >= self.max_prompt_tokens:
                    return True

        return False


class RateLimitStoppingCallback(BaseStoppingCallback):
    """Stops if error rate exceeds a threshold."""

    def __init__(self, max_error_rate: float = 0.5, min_samples: int = 10):
        self.max_error_rate = max_error_rate
        self.min_samples = min_samples

    async def should_stop(self, state: GenerationState) -> bool:
        if state.num_generated < self.min_samples:
            return False

        if state.monitor is None:
            return False

        recent_errors = state.monitor.get_metrics("error_rate")
        if not recent_errors:
            return False

        return recent_errors["mean"] >= self.max_error_rate
