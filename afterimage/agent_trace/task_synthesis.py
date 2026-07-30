import json
import random
from typing import Any, Dict, List, Optional

from ..providers.llm_providers import LLMProvider
from .context import BaseContextGenerator, VirtualUserContextGenerator
from .types import AppDomainSpec, GridTaskBucket


TASK_SYNTHESIZER_PROMPT = """You are an expert agentic task and environment state synthesizer.
Generate a realistic, grounded multi-turn agent task AND a matching initial environment seed state based on the following app APIs, structural constraints, and virtual user identity profile.

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

Virtual User Identity & Initial Seed Context Blueprint:
```json
{context_seed_snippet}
```

Instruction:
1. Generate a single clear, realistic task requirement that the virtual user above would ask an AI agent to perform using these apps.
2. Ensure specific entity identifiers (e.g., user_id, account_id, entity records) in the task text or initial state match or build upon the Virtual User Blueprint provided above.
3. Provide a JSON block at the end with the complete initial seed context data (e.g. user_id, account_ids, expense/item records, comments) matching the synthesized task scenario.

Format Output:
PROMPT: <procedural task text>
INITIAL_CONTEXT:
```json
{{
  ...
}}
```
"""


TASK_REWRITER_PROMPT = """You are an expert natural language task rewriter for AI user interfaces.
Your job is to rewrite a verbose, step-by-step synthetic procedural directive into a natural, direct user request.

CRITICAL CONSTRAINT: Preserve all specific account numbers, IDs, merchant names, category filters, and monetary amounts mentioned in the verbose directive.

Verbose Directive: "{verbose_task}"

Examples:
- Verbose: "Go to my expenses app, find non-group expenses for user 101, iterate over all comments on them, count unique commenter user IDs, and tell me the total count."
- Rewritten Intent: "How many unique people have commented on my non-group expenses and payments in total on ExpensesApp for user 101?"

- Verbose: "Call banking.get_account_balance with account_id=1001, check if balance >= 500, then call banking.transfer_money from 1001 to 1002 for 250."
- Rewritten Intent: "Check if I have at least $500 in my checking account (1001), and if so, transfer $250 to my savings account (1002)."

Rewritten Natural User Request:
"""


class InverseFrequencySampler:
    """Tracks API usage frequency across generated tasks to prevent endpoint coverage collapse.

    Attributes:
        invocation_counts: Mapping from ``"app.action"`` endpoint string to invocation frequency.
    """

    def __init__(self):
        self.invocation_counts: Dict[str, int] = {}

    def record_usage(self, app_name: str, action_name: str) -> None:
        """Records an API endpoint invocation.

        Args:
            app_name: Name of the application domain.
            action_name: Name of the action endpoint.
        """
        key = f"{app_name}.{action_name}"
        self.invocation_counts[key] = self.invocation_counts.get(key, 0) + 1

    def get_least_invoked_actions(
        self, app_domains: Dict[str, AppDomainSpec], top_k: int = 10
    ) -> List[str]:
        """Returns top_k least invoked action endpoints across given app domains.

        Args:
            app_domains: Registered application domain specifications.
            top_k: Number of least invoked actions to return. Defaults to 10.

        Returns:
            List[str]: Formatted descriptions of least invoked API endpoints.
        """
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
    """Combinatorial grid synthesizer and procedural rewriter for synthetic agent tasks.

    Uses a 360-bucket grid across difficulties, action types, task foci, and app counts
    to synthesize grounded user tasks and matching initial context states.

    Args:
        llm_provider: LLM provider instance for task generation and rewriting.
        model_name: Default LLM model name. Defaults to ``"gemini-3.5-flash-lite"``.
        context_generator: Extensible context generator instance. Defaults to
            :class:`VirtualUserContextGenerator`.

    Example:
        >>> synthesizer = GridTaskSynthesizer(llm_provider)
        >>> task, context, apps, bucket = await synthesizer.synthesize_task(app_domains)
    """

    DIFFICULTIES = ["easy", "medium", "hard"]
    ACTION_TYPES = ["read", "write", "mixed"]
    TASK_FOCI = ["constraint_satisfaction", "derivation", "iteration", "open"]

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.5-flash-lite",
        context_generator: Optional[BaseContextGenerator] = None,
    ):
        self.llm_provider = llm_provider
        self.model_name = getattr(llm_provider, "model_name", model_name)
        self.sampler = InverseFrequencySampler()
        self.context_generator = (
            context_generator if context_generator is not None else VirtualUserContextGenerator()
        )

    def sample_grid_bucket(self, num_apps_available: int) -> GridTaskBucket:
        """Samples a combinatorial bucket from the task complexity grid.

        Args:
            num_apps_available: Number of registered app domains available.

        Returns:
            GridTaskBucket: Sampled grid bucket.
        """
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
    ) -> tuple[str, Dict[str, Any], List[str], GridTaskBucket]:
        """Synthesizes a natural agent task grounded in registered app domains.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional pre-sampled grid bucket.

        Returns:
            tuple[str, Dict[str, Any], List[str], GridTaskBucket]: A tuple containing:
                - Natural language user task directive.
                - Initial context dictionary.
                - List of target app domain names.
                - Effective grid task bucket.

        Raises:
            ValueError: If app_domains map is empty.
        """
        if not app_domains:
            raise ValueError("No app domains registered for task synthesis.")

        effective_bucket = bucket or self.sample_grid_bucket(len(app_domains))
        selected_app_names = random.sample(
            list(app_domains.keys()), min(effective_bucket.num_apps, len(app_domains))
        )
        selected_domains = {k: app_domains[k] for k in selected_app_names}

        # Synthesize initial seed context payload dynamically
        seed_context = await self.context_generator.generate_context(
            app_domains=selected_domains, bucket=effective_bucket
        )
        context_snippet = self.context_generator.render_prompt_snippet(seed_context)

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
            context_seed_snippet=context_snippet,
        )

        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.7,
        )
        raw_text = response.text.strip()
        verbose_task, parsed_context = self._parse_synthesizer_output(raw_text)

        # Merge generated seed context with LLM output context
        final_context = dict(seed_context)
        if parsed_context:
            final_context.update(parsed_context)

        # Procedural Task Rewriting step to compress step-by-step directive into natural user intent
        rewritten_task = await self.rewrite_task_intent(verbose_task)

        # Record API usages for sampled domains
        for app_name in selected_app_names:
            for act in app_domains[app_name].actions:
                self.sampler.record_usage(app_name, act.action_name)

        return rewritten_task, final_context, selected_app_names, effective_bucket

    def _parse_synthesizer_output(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Parses prompt text and initial context JSON from synthesizer output.

        Args:
            text: Raw output string from LLM task synthesizer.

        Returns:
            tuple[str, Dict[str, Any]]: Verbose task text and parsed context dictionary.
        """
        
        verbose_task = text
        initial_context: Dict[str, Any] = {}

        if "PROMPT:" in text:
            parts = text.split("PROMPT:", 1)[1]
            if "INITIAL_CONTEXT:" in parts:
                prompt_part, json_part = parts.split("INITIAL_CONTEXT:", 1)
                verbose_task = prompt_part.strip()
                if "```json" in json_part:
                    json_str = (
                        json_part.split("```json", 1)[1].split("```", 1)[0].strip()
                    )
                elif "```" in json_part:
                    json_str = json_part.split("```", 1)[1].split("```", 1)[0].strip()
                else:
                    json_str = json_part.strip()

                try:
                    initial_context = json.loads(json_str)
                except Exception:
                    initial_context = {}
            else:
                verbose_task = parts.strip()

        return verbose_task.strip().strip('"'), initial_context

    async def rewrite_task_intent(self, verbose_task: str) -> str:
        """Compresses verbose procedural directives into natural intent-level user queries.

        Args:
            verbose_task: Detailed step-by-step procedural instruction.

        Returns:
            str: Compressed natural language user query.
        """
        prompt = TASK_REWRITER_PROMPT.format(verbose_task=verbose_task)
        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.3,
        )
        cleaned = response.text.strip().strip('"')
        return cleaned or verbose_task

