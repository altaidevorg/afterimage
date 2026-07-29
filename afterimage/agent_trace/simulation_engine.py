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
    """Stores generated stateful entities, account balances, and foreign key lookup pools across turns."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initializes a new SimulationContext instance."""
        self._seed = seed
        self._random = random.Random(seed)
        if seed is not None and fake:
            Faker.seed(seed)
        self.store: Dict[str, List[Any]] = {}
        self.account_balances: Dict[int, Dict[str, float]] = {}

    def record_entity(self, key: str, value: Any) -> None:
        """Stores a primary identifier or generated entity attribute into state."""
        if key not in self.store:
            self.store[key] = []
        if value not in self.store[key]:
            self.store[key].append(value)

    def sample_fk(self, key: str) -> Any:
        """Samples an existing foreign key identifier from state, or synthesizes a fresh record."""
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

    def get_or_create_account_balance(self, account_id: Any) -> Dict[str, float]:
        """Retrieves persistent stateful account balance or initializes a realistic balance record."""
        try:
            account_id_int = int(account_id)
        except Exception:
            account_id_int = self._random.randint(10000, 99999)

        if account_id_int not in self.account_balances:
            total = round(self._random.uniform(500.0, 5000.0), 2)
            avail = round(total * self._random.uniform(0.7, 0.95), 2)
            self.account_balances[account_id_int] = {
                "total_balance": total,
                "available_balance": avail,
            }
        return self.account_balances[account_id_int]

    def apply_transfer(self, sender_id: Any, receiver_id: Any, amount: Any) -> None:
        """Applies a fund transfer mutation, deducting from sender and crediting receiver."""
        try:
            s_id = int(sender_id)
            r_id = int(receiver_id)
            amt = float(amount)
        except Exception:
            return

        s_bal = self.get_or_create_account_balance(s_id)
        r_bal = self.get_or_create_account_balance(r_id)

        s_bal["available_balance"] = round(s_bal["available_balance"] - amt, 2)
        s_bal["total_balance"] = round(s_bal["total_balance"] - amt, 2)
        r_bal["available_balance"] = round(r_bal["available_balance"] + amt, 2)
        r_bal["total_balance"] = round(r_bal["total_balance"] + amt, 2)

    def mutate_state(self, mutation_tag: str, entity_value: Any) -> None:
        """Applies state mutation directive during stateful API calls."""
        if mutation_tag.startswith("append:"):
            target_key = mutation_tag.split("append:", 1)[1]
            self.record_entity(target_key, entity_value)

    def clear(self) -> None:
        """Clears all stored context state."""
        self.store.clear()
        self.account_balances.clear()


class DeclarativeEngine:
    """Core sub-millisecond local declarative simulation generator engine."""

    def __init__(self, context: Optional[SimulationContext] = None) -> None:
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

    def _match_parameter_echo(
        self, field_name: str, annotation: Any, extra: Dict[str, Any], parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """Checks for explicit parameter generator annotations, direct name matches, or semantic synonym echoes."""
        gen_type = extra.get("generator")

        # 1. Explicit generator annotation: "param:<param_name>" or "echo"
        if isinstance(gen_type, str):
            if gen_type == "echo" and field_name in parameters:
                return self._coerce_type(parameters[field_name], annotation)
            elif gen_type.startswith("param:"):
                target_p = gen_type.split("param:", 1)[1]
                if target_p in parameters and parameters[target_p] is not None:
                    return self._coerce_type(parameters[target_p], annotation)

        # 2. Direct parameter name match
        if field_name in parameters and parameters[field_name] is not None:
            return self._coerce_type(parameters[field_name], annotation)

        # 3. Semantic Synonym Parameter Matching
        lower_field = field_name.lower()
        if lower_field == "account_id" and "account_id" in parameters:
            return self._coerce_type(parameters["account_id"], annotation)
        elif lower_field in ("sender_account_id", "sender_id") and "sender_id" in parameters:
            return self._coerce_type(parameters["sender_id"], annotation)
        elif lower_field in ("recipient_account_id", "receiver_account_id", "receiver_id", "recipient_id"):
            val = parameters.get("receiver_id") if "receiver_id" in parameters else parameters.get("recipient_id")
            if val is not None:
                return self._coerce_type(val, annotation)
        elif lower_field == "amount" and "amount" in parameters:
            return self._coerce_type(parameters["amount"], annotation)
        elif lower_field == "user_id" and "user_id" in parameters:
            return self._coerce_type(parameters["user_id"], annotation)
        elif lower_field == "expense_id" and "expense_id" in parameters:
            return self._coerce_type(parameters["expense_id"], annotation)
        elif lower_field == "category" and "category" in parameters:
            return self._coerce_type(parameters["category"], annotation)

        return None

    def synthesize_field(
        self, field_name: str, field_info: Any, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Synthesizes a field value using parameter echoing and 4-tier fallback hierarchy."""
        extra = getattr(field_info, "json_schema_extra", None) or {}
        gen_type = extra.get("generator")
        annotation = getattr(field_info, "annotation", None)
        params = parameters or {}

        # Tier 0: Parameter Echoing & Synonym Matching
        echoed_val = self._match_parameter_echo(field_name, annotation, extra, params)
        if echoed_val is not None:
            return echoed_val

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
                    self.generate_response(item_type, parameters=params)
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

    def generate_response(
        self, model_cls: Type[T], parameters: Optional[Dict[str, Any]] = None
    ) -> T:
        """Recursively populates a Pydantic V2 BaseModel with parameter echoing and stateful context lookup."""
        params = parameters or {}
        payload: Dict[str, Any] = {}
        fields = getattr(model_cls, "model_fields", {})

        for field_name, field_info in fields.items():
            payload[field_name] = self.synthesize_field(
                field_name, field_info, parameters=params
            )

        # Stateful Account Balance Lookup
        if "account_id" in params and any(
            f in fields for f in ("available_balance", "total_balance")
        ):
            acc_id = params["account_id"]
            bal = self.ctx.get_or_create_account_balance(acc_id)
            if "available_balance" in fields:
                payload["available_balance"] = bal["available_balance"]
            if "total_balance" in fields:
                payload["total_balance"] = bal["total_balance"]

        # Record synthesized entity IDs into context entity stores
        for field_name, val in payload.items():
            if val is not None and (
                "_id" in field_name or field_name in ("id", "user_id", "account_id", "expense_id")
            ):
                self.ctx.record_entity(field_name, val)
                if "account" in field_name:
                    self.ctx.record_entity("account_id", val)
                if "user" in field_name:
                    self.ctx.record_entity("user_id", val)

        try:
            return model_cls(**payload)
        except Exception:
            coerced_payload: Dict[str, Any] = {}
            for field_name, field_info in fields.items():
                raw_val = payload.get(field_name)
                annotation = getattr(field_info, "annotation", None)
                coerced_payload[field_name] = self._coerce_type(raw_val, annotation)
            return model_cls(**coerced_payload)
