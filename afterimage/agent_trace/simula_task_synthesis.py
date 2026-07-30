"""Simula-driven task synthesis for environment-free synthetic agent traces.

Integrates Google's Simula reasoning-driven synthetic data methodology
(:mod:`afterimage.simula`) with :mod:`afterimage.agent_trace` execution environments.
Uses tree-structured factor taxonomies and strategy meta-prompts to generate
deep multi-domain agent tasks.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from ..monitoring import GenerationMonitor
from ..providers.llm_providers import LLMProvider
from ..simula.pipeline import OpenSimula
from .context import BaseContextGenerator, VirtualUserContextGenerator
from .task_synthesis import TASK_REWRITER_PROMPT
from .types import AppDomainSpec, GridTaskBucket


SIMULA_SYNTHESIS_PROMPT = """You are an expert AI agent task synthesizer.
Generate a concrete, grounded multi-turn agent task directive and matching initial context based on the following app APIs, virtual user identity, and Simula taxonomy requirement focus.

Target Apps: {app_names}
Available APIs:
{api_specs}

Simula Taxonomy Focus: {taxonomy_focus}

Virtual User Identity & Context Blueprint:
```json
{context_seed_snippet}
```

Instruction:
1. Generate a single clear, realistic task requirement that the virtual user above would ask an AI agent to perform using these apps to satisfy the Simula taxonomy focus.
2. Ensure entity identifiers (e.g. user_id, account_id, expense_id) match or build upon the Virtual User Blueprint provided above.
3. Provide a JSON block at the end with the complete initial seed context data matching the synthesized task scenario.

Format Output:
PROMPT: <procedural task text>
INITIAL_CONTEXT:
```json
{{
  ...
}}
```
"""


class SimulaTaskSynthesizer:
    """Advanced task synthesizer using Simula taxonomy expansion and strategy sampling.

    Combines factor taxonomy trees (domain constraints, task depth, user intent diversity)
    with virtual user identity contexts to synthesize high-complexity agent directives.

    Args:
        llm_provider: LLM provider instance for Simula pipeline.
        context_generator: Extensible context generator instance. Defaults to
            :class:`VirtualUserContextGenerator`.
        monitor: Optional generation monitor.

    Example:
        >>> synthesizer = SimulaTaskSynthesizer(llm_provider)
        >>> task, context, apps, bucket = await synthesizer.synthesize_task(app_domains)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_generator: Optional[BaseContextGenerator] = None,
        monitor: Optional[GenerationMonitor] = None,
    ):
        self.llm_provider = llm_provider
        self.monitor = monitor
        self.simula = OpenSimula(llm=llm_provider, temperature=0.5, monitor=monitor)
        self.context_generator = (
            context_generator if context_generator is not None else VirtualUserContextGenerator()
        )

    async def synthesize_task(
        self,
        app_domains: Dict[str, AppDomainSpec],
        bucket: Optional[GridTaskBucket] = None,
    ) -> tuple[str, Dict[str, Any], List[str], GridTaskBucket]:
        """Synthesizes an agent task directive using Simula strategy meta-prompts.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional pre-sampled grid bucket.

        Returns:
            tuple[str, Dict[str, Any], List[str], GridTaskBucket]: A tuple containing:
                - Synthesized agent task directive.
                - Initial context dictionary.
                - List of target app domain names.
                - Effective grid task bucket.

        Raises:
            ValueError: If app_domains map is empty.
        """
        if not app_domains:
            raise ValueError("No app domains registered for task synthesis.")

        effective_bucket = bucket or GridTaskBucket(
            difficulty="hard",
            action_type="mixed",
            task_focus="constraint_satisfaction",
            num_apps=min(len(app_domains), 3),
        )

        selected_app_names = random.sample(
            list(app_domains.keys()), min(effective_bucket.num_apps, len(app_domains))
        )
        selected_domains = {k: app_domains[k] for k in selected_app_names}

        # Generate virtual user seed context
        seed_context = await self.context_generator.generate_context(
            app_domains=selected_domains, bucket=effective_bucket
        )
        context_snippet = self.context_generator.render_prompt_snippet(seed_context)

        # Construct Simula instruction prompt from app domain actions
        api_lines = []
        for app_name, spec in selected_domains.items():
            for act in spec.actions:
                api_lines.append(f"- {app_name}.{act.action_name}: {act.description}")

        instruction_y = (
            f"Generate a complex multi-turn user directive for an AI agent interacting with "
            f"apps [{', '.join(selected_app_names)}]. APIs:\n" + "\n".join(api_lines)
        )

        # Build factor taxonomy bundle via Simula
        bundle = await self.simula.build_taxonomy(
            instruction_y=instruction_y,
            target_depth_D=2,
            proposal_N=2,
            max_factors=3,
        )

        # Extract factor focus string from factor taxonomy
        taxonomy_focus = "General multi-turn API task"
        if bundle and bundle.taxonomies:
            tax = random.choice(bundle.taxonomies)
            root_node = tax.nodes.get(tax.root_id)
            child_nodes = [node for node in tax.nodes.values() if node.parent_id == tax.root_id]
            if child_nodes:
                target_node = random.choice(child_nodes)
                taxonomy_focus = f"{target_node.label}"
            elif root_node:
                taxonomy_focus = f"{root_node.label}"

        prompt = SIMULA_SYNTHESIS_PROMPT.format(
            app_names=", ".join(selected_app_names),
            api_specs="\n".join(api_lines),
            taxonomy_focus=taxonomy_focus,
            context_seed_snippet=context_snippet,
        )

        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.7,
        )
        raw_text = response.text.strip()
        verbose_task, parsed_context = self._parse_synthesizer_output(raw_text)

        final_context = dict(seed_context)
        if parsed_context:
            final_context.update(parsed_context)

        # Rewrite verbose procedural output into natural user intent
        rewritten_task = await self.rewrite_task_intent(verbose_task)

        return rewritten_task, final_context, selected_app_names, effective_bucket

    def _parse_synthesizer_output(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Parses output text into task prompt string and context dictionary."""
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
        """Compresses procedural directive into natural user intent."""
        prompt = TASK_REWRITER_PROMPT.format(verbose_task=verbose_task)
        response = await self.llm_provider.agenerate_content(
            prompt=prompt,
            temperature=0.3,
        )
        cleaned = response.text.strip().strip('"')
        return cleaned or verbose_task
