"""Stateful simulation context and sub-millisecond declarative execution engine.

This module provides local simulation utilities that resolve field values for generated
Pydantic V2 response models using a 4-tier fallback generator hierarchy.
"""

import datetime
import random
import uuid
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, EmailStr

try:
    from faker import Faker

    fake = Faker()
except ImportError:
    fake = None

T = TypeVar("T", bound=BaseModel)


class SimulationContext:
    """Stores generated stateful entities and foreign key lookup pools across turns.

    Attributes:
        store (Dict[str, List[Any]]): Internal mapping from entity keys to list of generated IDs.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initializes a new SimulationContext instance.

        Args:
            seed (Optional[int]): Optional random seed for reproducible entity synthesis.
        """
        self._seed = seed
        self._random = random.Random(seed)
        if seed is not None and fake:
            Faker.seed(seed)
        self.store: Dict[str, List[Any]] = {}

    def record_entity(self, key: str, value: Any) -> None:
        """Stores a primary identifier or generated entity attribute into state.

        Args:
            key (str): Entity key identifier (e.g. ``"user.user_id"``).
            value (Any): Primitive value or identifier to record.
        """
        if key not in self.store:
            self.store[key] = []
        if value not in self.store[key]:
            self.store[key].append(value)

    def sample_fk(self, key: str) -> Any:
        """Samples an existing foreign key identifier from state, or synthesizes a fresh record.

        Args:
            key (str): Foreign key lookup string (e.g. ``"user.user_id"`` or ``"fk:user.user_id"``).

        Returns:
            Any: A sampled existing identifier or a freshly registered integer identifier.
        """
        candidates = [key]
        if "." in key:
            entity, field = key.split(".", 1)
            candidates.extend([f"{entity}_{field}", field])

        for c in candidates:
            if c in self.store and self.store[c]:
                return self._random.choice(self.store[c])

        fallback_id = self._random.randint(1000, 9999)
        self.record_entity(key, fallback_id)
        return fallback_id

    def mutate_state(self, mutation_tag: str, entity_value: Any) -> None:
        """Applies state mutation directive during stateful API calls (e.g. POST/PATCH).

        Args:
            mutation_tag (str): Directive tag such as ``"append:user_orders"``.
            entity_value (Any): Entity payload to record into context.
        """
        if mutation_tag.startswith("append:"):
            target_key = mutation_tag.split("append:", 1)[1]
            self.record_entity(target_key, entity_value)

    def clear(self) -> None:
        """Clears all stored context state."""
        self.store.clear()


class DeclarativeEngine:
    """Core sub-millisecond local declarative simulation generator engine.

    Uses a 4-tier fallback resolution hierarchy:
        1. Explicit Generator Annotations (``id``, ``fk``, ``money``, ``faker``, ``enum``).
        2. Specialized Pydantic / standard library types (``EmailStr``, ``UUID``, ``datetime``).
        3. Field constraints (``ge``, ``le``, ``min_length``, ``max_length``).
        4. Primitive fallbacks (``int``, ``str``, ``float``, ``bool``).

    Attributes:
        ctx (SimulationContext): Stateful entity lookup context.
    """

    def __init__(self, context: Optional[SimulationContext] = None) -> None:
        """Initializes DeclarativeEngine.

        Args:
            context (Optional[SimulationContext]): Context instance or None to create fresh context.
        """
        self.ctx = context or SimulationContext()

    def _extract_constraint(
        self, field_info: Any, constraint_name: str
    ) -> Optional[Any]:
        val = getattr(field_info, constraint_name, None)
        if val is not None:
            return val
        metadata = getattr(field_info, "metadata", [])
        for meta in metadata:
            if hasattr(meta, constraint_name):
                return getattr(meta, constraint_name)
        return None

    def _coerce_type(self, val: Any, annotation: Any) -> Any:
        """Coerces a synthesized value to match the target Pydantic field annotation."""
        if val is None or annotation is None:
            return val

        if annotation is str:
            return str(val) if not isinstance(val, str) else val

        if annotation is int:
            if isinstance(val, int):
                return val
            try:
                return int(float(str(val)))
            except Exception:
                return val

        if annotation is float:
            if isinstance(val, (int, float)):
                return float(val)
            try:
                return float(str(val))
            except Exception:
                return val

        if annotation is bool:
            return bool(val)

        return val

    def _synthesize_semantic_fallback(
        self, field_name: str, annotation: Any, field_info: Any
    ) -> Any:
        """Synthesizes realistic domain data based on field naming semantics when explicit tags are absent."""
        lower_name = field_name.lower()

        if annotation is str:
            if lower_name in ("status", "state"):
                return self.ctx._random.choice(
                    ["completed", "pending", "active", "success", "failed"]
                )
            elif lower_name in (
                "message",
                "comment",
                "description",
                "note",
                "summary",
                "body",
                "text",
                "reason",
            ):
                return (
                    fake.sentence()
                    if fake
                    else f"Sample {field_name.replace('_', ' ')}."
                )
            elif lower_name in (
                "name",
                "full_name",
                "author_name",
                "customer_name",
                "user_name",
            ):
                return (
                    fake.name()
                    if fake
                    else f"User_{self.ctx._random.randint(100, 999)}"
                )
            elif lower_name in ("company", "merchant", "vendor", "organization"):
                return fake.company() if fake else "Acme Corp"
            elif lower_name in ("category", "type", "genre"):
                return self.ctx._random.choice(
                    [
                        "general",
                        "billing",
                        "support",
                        "travel",
                        "groceries",
                        "dining",
                        "entertainment",
                    ]
                )
            elif "email" in lower_name:
                return (
                    fake.email()
                    if fake
                    else f"user_{self.ctx._random.randint(100, 999)}@example.com"
                )
            elif any(k in lower_name for k in ("url", "link", "uri")):
                return fake.url() if fake else "https://example.com"
            elif "city" in lower_name:
                return fake.city() if fake else "New York"
            elif "country" in lower_name:
                return fake.country() if fake else "United States"
            elif "address" in lower_name:
                return fake.address() if fake else "123 Main St"
            elif "phone" in lower_name:
                return fake.phone_number() if fake else "+1-555-0199"
            elif lower_name in ("currency", "currency_code"):
                return self.ctx._random.choice(["USD", "EUR", "GBP"])
            else:
                return (
                    fake.sentence(nb_words=3).rstrip(".")
                    if fake
                    else f"sample_{field_name}"
                )

        if annotation is float:
            if any(
                k in lower_name
                for k in ("amount", "balance", "price", "cost", "total", "subtotal", "fee")
            ):
                ge_val = self._extract_constraint(field_info, "ge")
                le_val = self._extract_constraint(field_info, "le")
                ge = 5.0 if ge_val is None else float(ge_val)
                le = 500.0 if le_val is None else float(le_val)
                return round(self.ctx._random.uniform(ge, le), 2)

        return None

    def synthesize_field(self, field_name: str, field_info: Any) -> Any:
        """Synthesizes a field value using 4-tier fallback hierarchy with type coercion and semantic heuristics.

        Args:
            field_name (str): Name of the target model field.
            field_info (Any): Pydantic FieldInfo object.

        Returns:
            Any: Generated schema-compliant field value.
        """
        extra = getattr(field_info, "json_schema_extra", None) or {}
        gen_type = extra.get("generator")
        annotation = getattr(field_info, "annotation", None)

        # Tier 1: Explicit Generator Annotations (with automatic type coercion)
        if gen_type:
            if gen_type == "id":
                val = self.ctx._random.randint(10000, 99999)
                self.ctx.record_entity(field_name, val)
                return self._coerce_type(val, annotation)
            elif isinstance(gen_type, str) and gen_type.startswith("fk:"):
                target_key = gen_type.split("fk:", 1)[1]
                val = self.ctx.sample_fk(target_key)
                return self._coerce_type(val, annotation)
            elif gen_type == "money":
                ge_val = self._extract_constraint(field_info, "ge")
                le_val = self._extract_constraint(field_info, "le")
                ge = 0.01 if ge_val is None else float(ge_val)
                le = 500.0 if le_val is None else float(le_val)
                val = round(self.ctx._random.uniform(ge, le), 2)
                return self._coerce_type(val, annotation)
            elif gen_type == "enum":
                values = extra.get("values", ["default_1", "default_2"])
                val = self.ctx._random.choice(values)
                return self._coerce_type(val, annotation)
            elif isinstance(gen_type, str) and gen_type.startswith("faker:"):
                provider = gen_type.split("faker:", 1)[1]
                if fake and hasattr(fake, provider):
                    val = getattr(fake, provider)()
                    return self._coerce_type(val, annotation)

        # Tier 2: Specialized Types
        if annotation is EmailStr:
            return (
                fake.email()
                if fake
                else f"user_{self.ctx._random.randint(100, 999)}@example.com"
            )
        elif annotation is uuid.UUID:
            val = uuid.uuid4()
            self.ctx.record_entity(field_name, str(val))
            return val
        elif annotation is datetime.datetime:
            return datetime.datetime.now(datetime.timezone.utc)
        elif annotation is datetime.date:
            return datetime.date.today()

        # Tier 3 & Tier 4: Field Constraints, Semantic Heuristics & Primitive Fallbacks
        semantic_val = self._synthesize_semantic_fallback(
            field_name, annotation, field_info
        )
        if semantic_val is not None:
            return self._coerce_type(semantic_val, annotation)

        if annotation is int:
            ge_val = self._extract_constraint(field_info, "ge")
            le_val = self._extract_constraint(field_info, "le")
            ge = 1 if ge_val is None else int(ge_val)
            le = 100 if le_val is None else int(le_val)
            return self.ctx._random.randint(ge, le)
        elif annotation is float:
            ge_val = self._extract_constraint(field_info, "ge")
            le_val = self._extract_constraint(field_info, "le")
            ge = 1.0 if ge_val is None else float(ge_val)
            le = 100.0 if le_val is None else float(le_val)
            return round(self.ctx._random.uniform(ge, le), 2)
        elif annotation is str:
            return (
                fake.sentence(nb_words=3).rstrip(".")
                if fake
                else f"sample_{field_name}"
            )
        elif annotation is bool:
            return True
        elif annotation is list or getattr(annotation, "__origin__", None) is list:
            item_type = (
                getattr(annotation, "__args__", (str,))[0]
                if hasattr(annotation, "__args__")
                else str
            )
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                return [
                    self.generate_response(item_type)
                    for _ in range(self.ctx._random.randint(1, 3))
                ]
            elif item_type is int:
                return [
                    self.ctx._random.randint(1, 50)
                    for _ in range(self.ctx._random.randint(1, 3))
                ]
            else:
                return [
                    fake.sentence(nb_words=2).rstrip(".") if fake else "item"
                    for _ in range(self.ctx._random.randint(1, 3))
                ]

        return None

    def generate_response(self, model_cls: Type[T]) -> T:
        """Recursively populates a Pydantic V2 BaseModel with type-safe validation recovery.

        Args:
            model_cls (Type[T]): Target Pydantic V2 BaseModel class.

        Returns:
            T: Instantiated schema-compliant response model instance.
        """
        payload: Dict[str, Any] = {}
        fields = getattr(model_cls, "model_fields", {})
        for field_name, field_info in fields.items():
            payload[field_name] = self.synthesize_field(field_name, field_info)

        try:
            return model_cls(**payload)
        except Exception:
            # Fallback type coercion across payload fields if validation error occurs
            coerced_payload: Dict[str, Any] = {}
            for field_name, field_info in fields.items():
                raw_val = payload.get(field_name)
                annotation = getattr(field_info, "annotation", None)
                coerced_payload[field_name] = self._coerce_type(raw_val, annotation)
            return model_cls(**coerced_payload)
