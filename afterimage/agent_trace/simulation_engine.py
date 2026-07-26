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
        if seed is not None:
            random.seed(seed)
            if fake:
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
                return random.choice(self.store[c])

        fallback_id = random.randint(1000, 9999)
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

    def _extract_constraint(self, field_info: Any, constraint_name: str) -> Optional[Any]:
        val = getattr(field_info, constraint_name, None)
        if val is not None:
            return val
        metadata = getattr(field_info, "metadata", [])
        for meta in metadata:
            if hasattr(meta, constraint_name):
                return getattr(meta, constraint_name)
        return None

    def synthesize_field(self, field_name: str, field_info: Any) -> Any:
        """Synthesizes a field value using 4-tier fallback hierarchy.

        Args:
            field_name (str): Name of the target model field.
            field_info (Any): Pydantic FieldInfo object.

        Returns:
            Any: Generated schema-compliant field value.
        """
        extra = getattr(field_info, "json_schema_extra", None) or {}
        gen_type = extra.get("generator")

        # Tier 1: Explicit Generator Annotations
        if gen_type:
            if gen_type == "id":
                val = random.randint(10000, 99999)
                self.ctx.record_entity(field_name, val)
                return val
            elif isinstance(gen_type, str) and gen_type.startswith("fk:"):
                target_key = gen_type.split("fk:", 1)[1]
                return self.ctx.sample_fk(target_key)
            elif gen_type == "money":
                ge_val = self._extract_constraint(field_info, "ge")
                le_val = self._extract_constraint(field_info, "le")
                ge = 0.01 if ge_val is None else float(ge_val)
                le = 500.0 if le_val is None else float(le_val)
                return round(random.uniform(ge, le), 2)
            elif gen_type == "enum":
                values = extra.get("values", ["default_1", "default_2"])
                return random.choice(values)
            elif isinstance(gen_type, str) and gen_type.startswith("faker:"):
                provider = gen_type.split("faker:", 1)[1]
                if fake and hasattr(fake, provider):
                    return getattr(fake, provider)()

        # Tier 2: Specialized Types
        annotation = getattr(field_info, "annotation", None)
        if annotation == EmailStr:
            return fake.email() if fake else f"user_{random.randint(100,999)}@example.com"
        elif annotation == uuid.UUID:
            val = uuid.uuid4()
            self.ctx.record_entity(field_name, str(val))
            return val
        elif annotation == datetime.datetime:
            return datetime.datetime.now(datetime.timezone.utc)
        elif annotation == datetime.date:
            return datetime.date.today()

        # Tier 3 & Tier 4: Field Constraints & Primitive Fallbacks
        if annotation == int:
            ge_val = self._extract_constraint(field_info, "ge")
            le_val = self._extract_constraint(field_info, "le")
            ge = 1 if ge_val is None else int(ge_val)
            le = 100 if le_val is None else int(le_val)
            return random.randint(ge, le)
        elif annotation == float:
            ge_val = self._extract_constraint(field_info, "ge")
            le_val = self._extract_constraint(field_info, "le")
            ge = 1.0 if ge_val is None else float(ge_val)
            le = 100.0 if le_val is None else float(le_val)
            return round(random.uniform(ge, le), 2)
        elif annotation == str:
            return fake.word() if fake else f"sample_{field_name}"
        elif annotation == bool:
            return True
        elif annotation == list or getattr(annotation, "__origin__", None) is list:
            item_type = getattr(annotation, "__args__", (str,))[0] if hasattr(annotation, "__args__") else str
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                return [self.generate_response(item_type) for _ in range(random.randint(1, 3))]
            elif item_type == int:
                return [random.randint(1, 50) for _ in range(random.randint(1, 3))]
            else:
                return [fake.word() if fake else "item" for _ in range(random.randint(1, 3))]

        return None

    def generate_response(self, model_cls: Type[T]) -> T:
        """Recursively populates a Pydantic V2 BaseModel in sub-millisecond execution time.

        Args:
            model_cls (Type[T]): Target Pydantic V2 BaseModel class.

        Returns:
            T: Instantiated schema-compliant response model instance.
        """
        payload: Dict[str, Any] = {}
        fields = getattr(model_cls, "model_fields", {})
        for field_name, field_info in fields.items():
            payload[field_name] = self.synthesize_field(field_name, field_info)
        return model_cls(**payload)
