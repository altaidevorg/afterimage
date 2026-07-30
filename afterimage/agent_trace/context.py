"""Context generator framework for environment-free agent trace task synthesis.

Provides extensible, composable abstractions for synthesizing initial environment
context states (virtual user identities, persona attributes, seed database records)
to ground synthetic AI agent task directives and tool interactions.
"""

from __future__ import annotations

import abc
import inspect
import json
import random
from typing import Any, Callable, Dict, List, Optional, Union

from faker import Faker

from .types import AppDomainSpec, GridTaskBucket


class BaseContextGenerator(abc.ABC):
    """Abstract base class for all initial context generators.

    Initial context generators synthesize structured state payloads (e.g., virtual
    user profiles, database seed records, account IDs) that anchor synthetic agent
    tasks into realistic, deterministic execution environments.
    """

    @abc.abstractmethod
    async def generate_context(
        self,
        app_domains: Optional[Dict[str, AppDomainSpec]] = None,
        bucket: Optional[GridTaskBucket] = None,
    ) -> Dict[str, Any]:
        """Synthesizes an initial context dictionary for task generation.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional grid task bucket specifying complexity constraints.

        Returns:
            Dict[str, Any]: Key-value pair payload representing initial context state.
        """
        pass

    def render_prompt_snippet(self, context: Dict[str, Any]) -> str:
        """Formats context dictionary into a clean markdown JSON snippet for LLM prompts.

        Args:
            context: Context state payload.

        Returns:
            str: Markdown-formatted JSON block for prompt insertion.
        """
        if not context:
            return "{}"
        try:
            return json.dumps(context, indent=2, ensure_ascii=False)
        except Exception:
            return str(context)


class VirtualUserContextGenerator(BaseContextGenerator):
    """Generates realistic virtual user identity profiles using Faker.

    Synthesizes localized virtual user identities including personal details,
    account identifiers, contact info, physical location, device specs, and
    domain-specific seed parameters.

    Args:
        locale: Locale string for Faker identity generation (e.g. ``"en_US"``).
        seed: Optional integer seed for reproducible generation.
        extra_fields_generator: Optional callable producing domain-specific extra fields.

    Example:
        >>> generator = VirtualUserContextGenerator(locale="en_US", seed=42)
        >>> context = await generator.generate_context()
        >>> print(context["user_name"])
        'Alice Smith'
    """

    def __init__(
        self,
        locale: str = "en_US",
        seed: Optional[int] = None,
        extra_fields_generator: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.locale = locale
        self.seed = seed
        self.faker = Faker(locale)
        if seed is not None:
            self.faker.seed_instance(seed)
        self.extra_fields_generator = extra_fields_generator

    async def generate_context(
        self,
        app_domains: Optional[Dict[str, AppDomainSpec]] = None,
        bucket: Optional[GridTaskBucket] = None,
    ) -> Dict[str, Any]:
        """Synthesizes a virtual user identity profile with realistic entity identifiers.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional grid task bucket.

        Returns:
            Dict[str, Any]: Dictionary containing virtual user attributes and IDs.
        """
        user_id = self.faker.random_int(min=101, max=999)
        account_id = self.faker.random_int(min=1001, max=9999)
        savings_account_id = account_id + 1

        context: Dict[str, Any] = {
            "user_id": user_id,
            "user_name": self.faker.name(),
            "user_email": self.faker.email(),
            "user_phone": self.faker.phone_number(),
            "account_id": account_id,
            "savings_account_id": savings_account_id,
            "checking_balance": round(random.uniform(500.0, 5500.0), 2),
            "savings_balance": round(random.uniform(1000.0, 25000.0), 2),
            "city": self.faker.city(),
            "street_address": self.faker.street_address(),
            "membership_tier": random.choice(["Standard", "Gold", "Platinum", "VIP"]),
        }

        if self.extra_fields_generator:
            extra = self.extra_fields_generator()
            if isinstance(extra, dict):
                context.update(extra)

        return context


class PersonaContextGenerator(BaseContextGenerator):
    """Integrates persona profiles into initial context payloads.

    Wraps persona attributes (e.g., from ``afterimage.persona_generator.PersonaEntry``)
    or persona dictionaries to inject user background, expertise level, and communication
    preferences into task synthesis context.

    Args:
        persona: Persona entry object or dictionary containing persona details.

    Example:
        >>> persona_data = {"persona_name": "Tech Enthusiast", "expertise": "expert"}
        >>> gen = PersonaContextGenerator(persona_data)
        >>> ctx = await gen.generate_context()
    """

    def __init__(self, persona: Union[Dict[str, Any], Any]):
        if hasattr(persona, "model_dump"):
            self.persona_data = persona.model_dump(mode="json")
        elif isinstance(persona, dict):
            self.persona_data = dict(persona)
        else:
            self.persona_data = {"persona": str(persona)}

    async def generate_context(
        self,
        app_domains: Optional[Dict[str, AppDomainSpec]] = None,
        bucket: Optional[GridTaskBucket] = None,
    ) -> Dict[str, Any]:
        """Injects persona attributes into context state.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional grid task bucket.

        Returns:
            Dict[str, Any]: Persona context payload.
        """
        return {"persona_context": self.persona_data}


class CallableContextGenerator(BaseContextGenerator):
    """Wraps user-defined callables to produce initial context state.

    Supports both synchronous and asynchronous functions returning dictionary state payloads.

    Args:
        func: Sync or async callable returning a dictionary.

    Example:
        >>> gen = CallableContextGenerator(lambda: {"custom_key": 42})
        >>> ctx = await gen.generate_context()
    """

    def __init__(self, func: Callable[..., Any]):
        self.func = func

    async def generate_context(
        self,
        app_domains: Optional[Dict[str, AppDomainSpec]] = None,
        bucket: Optional[GridTaskBucket] = None,
    ) -> Dict[str, Any]:
        """Invokes the wrapped callable to produce context state.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional grid task bucket.

        Returns:
            Dict[str, Any]: Result of callable execution.
        """
        if inspect.iscoroutinefunction(self.func):
            res = await self.func()
        else:
            res = self.func()

        if isinstance(res, dict):
            return res
        return {"data": res}


class CompositeContextGenerator(BaseContextGenerator):
    """Composes multiple initial context generators into a unified context provider.

    Executes all child context generators in sequence, merging their produced state
    dictionaries into a single context payload.

    Args:
        generators: List of child :class:`BaseContextGenerator` instances.

    Example:
        >>> gen = CompositeContextGenerator([
        ...     VirtualUserContextGenerator(),
        ...     CallableContextGenerator(lambda: {"order_id": 999})
        ... ])
        >>> ctx = await gen.generate_context()
    """

    def __init__(self, generators: List[BaseContextGenerator]):
        self.generators = generators

    async def generate_context(
        self,
        app_domains: Optional[Dict[str, AppDomainSpec]] = None,
        bucket: Optional[GridTaskBucket] = None,
    ) -> Dict[str, Any]:
        """Generates context from all child generators and merges their dictionaries.

        Args:
            app_domains: Map of registered application domain specifications.
            bucket: Optional grid task bucket.

        Returns:
            Dict[str, Any]: Merged context dictionary.
        """
        merged: Dict[str, Any] = {}
        for gen in self.generators:
            res = await gen.generate_context(app_domains=app_domains, bucket=bucket)
            if isinstance(res, dict):
                merged.update(res)
        return merged
