from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel

from ..providers.llm_providers import LLMProvider
from .types import AppDomainSpec, ToolActionSpec
from .verifier import SchemaVerifier, VerificationReport


SCHEMA_ARCHITECT_PROMPT_TEMPLATE = """You are an expert LLM Schema Architect for synthetic tool environments.
Your task is to generate valid Python source code containing Pydantic V2 response models for an application domain.

App Name: {app_name}
App Description: {app_description}
Target Endpoints: {endpoints_desc}

### ANNOTATION PROTOCOL CONSTRAINTS:
Annotate fields using Pydantic's `Field(json_schema_extra={{...}})` parameter following these exact guidelines:
1. `json_schema_extra={{"generator": "id"}}` for primary keys (e.g. `user_id: int`, `expense_id: int`). Ensure primary and foreign key IDs use `int` types.
2. `json_schema_extra={{"generator": "fk:<entity>.<field>"}}` for foreign keys (e.g. `user_id: int = Field(..., json_schema_extra={{"generator": "fk:user.user_id"}})`).
3. `json_schema_extra={{"generator": "money"}}` for monetary float amounts (e.g. `amount: float`).
4. `json_schema_extra={{"generator": "faker:<method>"}}` for fake string data (e.g. `faker:name`, `faker:email`, `faker:company`, `faker:sentence`, `faker:date_time_iso8601`).
5. `json_schema_extra={{"generator": "enum", "values": ["active", "pending", "completed"]}}` for categorical status strings.

### NESTED COLLECTION REQUIREMENTS:
- For collection endpoints (e.g. `list_expenses`, `get_transactions`, `list_comments`), construct dedicated item models (e.g. `ExpenseRecord`, `CommentRecord`) and include a list field in the main response model (e.g. `expenses: List[ExpenseRecord]`).
- Do NOT flatten collection endpoints into simple single-object `id/status/message` fallbacks.

### EXAMPLE SCHEMAS:
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ExpenseRecord(BaseModel):
    expense_id: int = Field(json_schema_extra={{"generator": "id"}})
    user_id: int = Field(json_schema_extra={{"generator": "fk:user.user_id"}})
    amount: float = Field(json_schema_extra={{"generator": "money"}})
    category: str = Field(json_schema_extra={{"generator": "enum", "values": ["dining", "travel", "groceries", "utilities"]}})
    merchant: str = Field(json_schema_extra={{"generator": "faker:company"}})
    description: str = Field(json_schema_extra={{"generator": "faker:sentence"}})
    status: str = Field(default="completed", json_schema_extra={{"generator": "enum", "values": ["pending", "completed", "flagged"]}})

class ExpensesListResponse(BaseModel):
    expenses: List[ExpenseRecord] = Field(default_factory=list)
    total_count: int = Field(default=1)
```

### CODE FORMATTING REQUIREMENTS:
- Output ONLY valid, executable Python code inside a ```python ``` code block.
- Import `from pydantic import BaseModel, Field, EmailStr` and `from typing import List, Optional`.
- Inherit all response models from `BaseModel`.

{feedback_section}
"""


class SchemaArchitect:
    """LLM Schema Architect that generates Pydantic response model specifications with static verification feedback loop."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        model_name: str = "gemini-3.6-flash",
        max_retries: int = 3,
    ):
        self.llm_provider = llm_provider
        self.model_name = getattr(llm_provider, "model_name", model_name)
        self.max_retries = max_retries
        self.verifier = SchemaVerifier()

    async def generate_app_domain_schema(
        self,
        app_name: str,
        app_description: str,
        actions: List[ToolActionSpec],
        existing_primary_ids: Optional[set[str]] = None,
    ) -> Tuple[AppDomainSpec, Dict[str, Type[BaseModel]]]:
        """Generates Pydantic response models code for an app domain with static verification retries."""
        endpoints_desc_lines = []
        for a in actions:
            desc = f"- {a.action_name}: {a.description} -> ResponseModel: {a.response_model_name}"
            if a.response_schema_hint:
                desc += f" (Schema Hint: {a.response_schema_hint})"
            endpoints_desc_lines.append(desc)
        endpoints_desc = "\n".join(endpoints_desc_lines)

        feedback_section = ""
        last_code = ""
        report = VerificationReport(is_valid=False)

        for attempt in range(1, self.max_retries + 1):
            prompt = SCHEMA_ARCHITECT_PROMPT_TEMPLATE.format(
                app_name=app_name,
                app_description=app_description,
                endpoints_desc=endpoints_desc,
                feedback_section=feedback_section,
            )

            response = await self.llm_provider.agenerate_content(
                prompt=prompt,
                temperature=0.2,
            )
            raw_text = response.text
            code_str = self._extract_python_code(raw_text)

            report = self.verifier.verify_code(
                code_str, existing_declared_ids=existing_primary_ids
            )
            if report.is_valid:
                last_code = code_str
                break

            feedback_section = f"\n### PREVIOUS ATTEMPT VERIFICATION ERRORS (ATTEMPT {attempt}/{self.max_retries}):\n{report.get_prompt_feedback()}\nPlease fix all listed errors and output the corrected Python code."

        if not report.is_valid and not last_code:
            # Fallback minimal schema if LLM retries failed
            last_code = self._generate_fallback_code(actions)

        # Exec verified code into runtime model classes dictionary
        model_classes = self._compile_model_classes(last_code)

        app_spec = AppDomainSpec(
            app_name=app_name,
            description=app_description,
            actions=actions,
            response_models_code=last_code,
        )
        return app_spec, model_classes

    def _extract_python_code(self, text: str) -> str:
        """Extracts python code block from LLM response."""
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        return text.strip()

    def _compile_model_classes(self, code_str: str) -> Dict[str, Type[BaseModel]]:
        """Compiles Python source code string into live Pydantic BaseModel classes."""
        import typing
        from pydantic import EmailStr, Field
        from .verifier import SAFE_BUILTINS

        local_scope: Dict[str, Any] = {}
        exec_globals = {
            "__builtins__": SAFE_BUILTINS,
            "BaseModel": BaseModel,
            "Field": Field,
            "EmailStr": EmailStr,
            "List": typing.List,
            "Optional": typing.Optional,
            "Dict": typing.Dict,
            "Any": typing.Any,
            "Set": typing.Set,
            "Tuple": typing.Tuple,
        }
        try:
            exec(code_str, exec_globals, local_scope)
        except Exception:
            return {}

        model_classes: Dict[str, Type[BaseModel]] = {}
        for k, v in local_scope.items():
            if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel:
                model_classes[k] = v
        return model_classes

    def _generate_fallback_code(self, actions: List[ToolActionSpec]) -> str:
        """Generates fallback basic Pydantic code if LLM generation exhausts retries."""
        lines = [
            "from pydantic import BaseModel, Field",
            "from typing import List, Optional\n",
        ]
        for a in actions:
            model_name = a.response_model_name
            lines.extend(
                [
                    f"class {model_name}(BaseModel):",
                    "    id: int = Field(json_schema_extra={'generator': 'id'})",
                    "    status: str = Field(default='success')",
                    "    message: str = Field(default='Action completed')\n",
                ]
            )
        return "\n".join(lines)
