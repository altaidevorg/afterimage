import random
from typing import Dict, List, Optional

from ..providers.llm_providers import LLMProvider
from .types import AppDomainSpec, GridTaskBucket


TASK_SYNTHESIZER_PROMPT = """You are an expert agentic task synthesizer.
Generate a realistic, grounded multi-turn agent task based on the following app APIs and structural constraints.

Target Apps: {app_names}
Available APIs:
{api_specs}

Under-utilized APIs to prioritize spotlighting in this task:
{least_invoked_apis}

Bucket Constraints:
- Difficulty: {difficulty}
- Action Type: {action_type}
- Task Focus: {task_focus}
- Number of Apps: {num_apps}

Instruction: Generate a single clear, realistic task requirement that a user would ask an AI agent to perform using these apps.
Output only the procedural task text.
"""


TASK_REWRITER_PROMPT = """You are an expert natural language task rewriter for AI user interfaces.
Your job is to rewrite a verbose, step-by-step synthetic procedural directive into a natural, direct user request.

Verbose Directive: "{verbose_task}"

Examples:
- Verbose: "Go to my expenses app, find non-group expenses, iterate over all comments on them, count unique commenter user IDs, and tell me the total count."
- Rewritten Intent: "How many unique people have commented on my non-group expenses and payments in total on ExpensesApp?"

- Verbose: "Call banking.get_transactions with status=unread, get reference_id 5001, then call banking.get_comments for 5001."
- Rewritten Intent: "Check my unread notifications for transactions with comments and tell me who commented."

Rewritten Natural User Request:
"""


class InverseFrequencySampler:
    """Tracks API usage frequency across generated tasks to prevent endpoint coverage collapse."""

    def __init__(self):
        self.invocation_counts: Dict[str, int] = {}  # "app.action" -> count

    def record_usage(self, app_name: str, action_name: str) -> None:
        key = f"{app_name}.{action_name}"
        self.invocation_counts[key] = self.invocation_counts.get(key, 0) + 1

    def get_least_invoked_actions(
        self, app_domains: Dict[str, AppDomainSpec], top_k: int = 10
    ) -> List[str]:
        """Returns top_k least invoked action endpoints across given app domains."""
        all_actions = []
        for app_name, spec in app_domains.items():
            for act in spec.actions:
                key = f"{app_name}.{act.action_name}"
                count = self.invocation_counts.get(key, 0)
                all_actions.append((count, key, act.description))

        all_actions.sort(key=lambda x: x[0])
        return [
            f"- {key}: {desc} (invoked {cnt} times)"
            for cnt, key, desc in all_actions[:top_k]
        ]


class GridTaskSynthesizer:
    """Combinatorial grid synthesizer and procedural rewriter for synthetic agent tasks."""

    DIFFICULTIES = ["easy", "medium", "hard"]
    ACTION_TYPES = ["read", "write", "mixed"]
    TASK_FOCI = ["constraint_satisfaction", "derivation", "iteration", "open"]

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.5-flash-lite",
    ):
        self.llm_provider = llm_provider
        self.model_name = getattr(llm_provider, "model_name", model_name)
        self.sampler = InverseFrequencySampler()

    def sample_grid_bucket(self, num_apps_available: int) -> GridTaskBucket:
        """Samples a combinatorial bucket from the task grid."""
        max_apps = min(num_apps_available, 3)
        return GridTaskBucket(
            difficulty=random.choice(self.DIFFICULTIES),
            action_type=random.choice(self.ACTION_TYPES),
            task_focus=random.choice(self.TASK_FOCI),
            num_apps=random.choice(list(range(1, max_apps + 1))),
        )

    async def synthesize_task(
        self,
        app_domains: Dict[str, AppDomainSpec],
        bucket: Optional[GridTaskBucket] = None,
    ) -> tuple[str, List[str], GridTaskBucket]:
        """Synthesizes a natural agent task grounded in registered app domains."""
        if not app_domains:
            raise ValueError("No app domains registered for task synthesis.")

        effective_bucket = bucket or self.sample_grid_bucket(len(app_domains))
        selected_app_names = random.sample(
            list(app_domains.keys()), min(effective_bucket.num_apps, len(app_domains))
        )
        selected_domains = {k: app_domains[k] for k in selected_app_names}

        api_specs_lines = []
        for app_name, spec in selected_domains.items():
            for act in spec.actions:
                api_specs_lines.append(
                    f"- {app_name}.{act.action_name}: {act.description}"
                )

        least_invoked = self.sampler.get_least_invoked_actions(selected_domains)

        prompt = TASK_SYNTHESIZER_PROMPT.format(
            app_names=", ".join(selected_app_names),
            api_specs="\n".join(api_specs_lines),
            least_invoked_apis="\n".join(least_invoked) if least_invoked else "None",
            difficulty=effective_bucket.difficulty,
            action_type=effective_bucket.action_type,
            task_focus=effective_bucket.task_focus,
            num_apps=effective_bucket.num_apps,
        )

        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.7,
        )
        verbose_task = response.text.strip().strip('"')

        # Procedural Task Rewriting step to compress step-by-step input into natural user intent
        rewritten_task = await self.rewrite_task_intent(verbose_task)

        # Record API usages for sampled domains
        for app_name in selected_app_names:
            for act in app_domains[app_name].actions:
                self.sampler.record_usage(app_name, act.action_name)

        return rewritten_task, selected_app_names, effective_bucket

    async def rewrite_task_intent(self, verbose_task: str) -> str:
        """Compresses verbose procedural directives into natural intent-level user queries."""
        prompt = TASK_REWRITER_PROMPT.format(verbose_task=verbose_task)
        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.3,
        )
        cleaned = response.text.strip().strip('"')
        return cleaned or verbose_task
