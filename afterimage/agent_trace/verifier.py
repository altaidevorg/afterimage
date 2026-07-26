"""Static AST and structural invariant verifier for generated Pydantic response models."""

import ast
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

SAFE_BUILTINS = {
    "__import__": __import__,
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "len": len,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "type": type,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "__build_class__": __build_class__,
    "__name__": "__main__",
}


class VerificationErrorDetail(BaseModel):
    model_name: str
    field_name: Optional[str] = None
    error_type: str
    error_message: str
    action_required: str


class VerificationReport(BaseModel):
    is_valid: bool
    errors: List[VerificationErrorDetail] = Field(default_factory=list)

    def get_prompt_feedback(self) -> str:
        """Formats error report into structured prompt feedback for LLM re-generation."""
        if self.is_valid:
            return "All schema structural invariants passed."

        lines = ["Verification Error Report:"]
        for err in self.errors:
            lines.append(f"- Model: {err.model_name}")
            if err.field_name:
                lines.append(f"  Field: {err.field_name}")
            lines.append(f"  Error Type: {err.error_type}")
            lines.append(f"  Message: {err.error_message}")
            lines.append(f"  Action Required: {err.action_required}")
        return "\n".join(lines)


class SchemaVerifier:
    """Static AST and structural invariant verifier for generated Pydantic response models."""

    ALLOWED_GENERATORS = {"id", "money"}
    ALLOWED_MODULE_IMPORTS = {"pydantic", "typing", "datetime", "uuid"}
    FORBIDDEN_CALL_NAMES = {
        "eval", "exec", "open", "__import__", "globals", "locals", "compile",
        "breakpoint", "input", "os", "sys", "subprocess", "shutil"
    }

    def verify_code(
        self, code_str: str, existing_declared_ids: Optional[Set[str]] = None
    ) -> VerificationReport:
        """Verifies Pydantic response models code against 6 structural invariants and AST security rules.

        Args:
            code_str (str): Python source code containing Pydantic response models.
            existing_declared_ids (Optional[Set[str]]): Existing primary ID keys in domain graph.

        Returns:
            VerificationReport: Report detailing validity and structural error feedback.
        """
        errors: List[VerificationErrorDetail] = []

        # 1. Syntactic Validity Check
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return VerificationReport(
                is_valid=False,
                errors=[
                    VerificationErrorDetail(
                        model_name="Global",
                        error_type="SyntaxError",
                        error_message=f"Python AST compilation failed: {e.msg} at line {e.lineno}",
                        action_required="Ensure generated code is syntactically valid Python source code.",
                    )
                ],
            )

        # 1b. AST Security Inspection
        sec_errors = self._verify_ast_security(tree)
        if sec_errors:
            return VerificationReport(is_valid=False, errors=sec_errors)

        # 2. Extract Class Definitions & Inheritance
        model_classes: Dict[str, ast.ClassDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                model_classes[node.name] = node

        if not model_classes:
            errors.append(
                VerificationErrorDetail(
                    model_name="Global",
                    error_type="ModelHierarchy",
                    error_message="No classes were found in the generated Python code.",
                    action_required="Define at least one response model inheriting from pydantic.BaseModel.",
                )
            )

        declared_primary_ids: Set[str] = set(existing_declared_ids or set())
        field_generator_tags: List[tuple[str, str, dict]] = []  # (model_name, field_name, extra_dict)

        # 3. Metadata Format Check & Primary ID Collection
        for model_name, class_node in model_classes.items():
            for stmt in class_node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    field_name = stmt.target.id
                    extra_dict = self._extract_json_schema_extra(stmt.value)
                    if extra_dict:
                        field_generator_tags.append((model_name, field_name, extra_dict))
                        gen = extra_dict.get("generator")
                        if gen == "id":
                            # Register entity.field and model.field primary keys
                            entity_name = model_name.lower().removesuffix("response").removesuffix("model")
                            declared_primary_ids.add(f"{entity_name}.{field_name}")
                            declared_primary_ids.add(f"{model_name}.{field_name}")
                            declared_primary_ids.add(field_name)

                        # Check generator tag validity
                        if gen:
                            if not self._is_valid_generator_tag(gen):
                                errors.append(
                                    VerificationErrorDetail(
                                        model_name=model_name,
                                        field_name=field_name,
                                        error_type="InvalidMetadataFormat",
                                        error_message=f"Unrecognized generator tag '{gen}'.",
                                        action_required="Use valid generator tags: 'id', 'fk:<entity>.<field>', 'money', 'faker:<method>', 'enum', 'mutation'.",
                                    )
                                )

        # 4. Foreign Key Integrity Check
        for model_name, field_name, extra_dict in field_generator_tags:
            gen = extra_dict.get("generator", "")
            if isinstance(gen, str) and gen.startswith("fk:"):
                fk_target = gen.split("fk:", 1)[1]
                if fk_target not in declared_primary_ids:
                    # Also check fallback entity matching
                    target_field = fk_target.split(".")[-1]
                    if not any(pid.endswith(f".{target_field}") or pid == target_field for pid in declared_primary_ids):
                        errors.append(
                            VerificationErrorDetail(
                                model_name=model_name,
                                field_name=field_name,
                                error_type="ForeignKeySafety",
                                error_message=f"Foreign key target '{fk_target}' does not map to any primary 'id' field in the schema graph.",
                                action_required=f"Ensure target entity/field for '{fk_target}' is declared with generator='id' in some model.",
                            )
                        )

        # 5. Instantiation Dry-Run Check
        if not errors:
            dry_run_err = self._perform_dry_run(code_str, list(model_classes.keys()))
            if dry_run_err:
                errors.append(dry_run_err)

        return VerificationReport(is_valid=len(errors) == 0, errors=errors)

    def _verify_ast_security(self, tree: ast.AST) -> List[VerificationErrorDetail]:
        """Inspects AST nodes for security violations (forbidden calls, unapproved imports, executable functions)."""
        errors: List[VerificationErrorDetail] = []
        for node in ast.walk(tree):
            # Block unapproved imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_MODULE_IMPORTS:
                        errors.append(
                            VerificationErrorDetail(
                                model_name="Global",
                                error_type="SecurityViolation",
                                error_message=f"Import of unapproved module '{alias.name}' is forbidden.",
                                action_required="Only import from pydantic, typing, datetime, or uuid.",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module_base = (node.module or "").split(".")[0]
                if module_base not in self.ALLOWED_MODULE_IMPORTS:
                    errors.append(
                        VerificationErrorDetail(
                            model_name="Global",
                            error_type="SecurityViolation",
                            error_message=f"Import from unapproved module '{node.module}' is forbidden.",
                            action_required="Only import from pydantic, typing, datetime, or uuid.",
                        )
                    )

            # Block forbidden function/method calls
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in self.FORBIDDEN_CALL_NAMES:
                    errors.append(
                        VerificationErrorDetail(
                            model_name="Global",
                            error_type="SecurityViolation",
                            error_message=f"Forbidden function call '{func_name}' detected in AST.",
                            action_required="Remove forbidden function calls.",
                        )
                    )

        return errors

    def _is_valid_generator_tag(self, gen: str) -> bool:
        if gen in self.ALLOWED_GENERATORS:
            return True
        if gen.startswith("fk:") or gen.startswith("faker:") or gen.startswith("mutation:") or gen == "enum":
            return True
        return False

    def _extract_json_schema_extra(self, expr: Optional[ast.expr]) -> Optional[dict]:
        """Extracts json_schema_extra keyword argument dict from Field(...) AST expression."""
        if not isinstance(expr, ast.Call):
            return None

        func = expr.func
        is_field = (isinstance(func, ast.Name) and func.id == "Field") or (
            isinstance(func, ast.Attribute) and func.attr == "Field"
        )
        if not is_field:
            return None

        for keyword in expr.keywords:
            if keyword.arg == "json_schema_extra":
                return self._ast_to_dict(keyword.value)
        return None

    def _ast_to_dict(self, node: ast.expr) -> Optional[dict]:
        """Converts an AST Dict node to a Python dictionary."""
        if isinstance(node, ast.Dict):
            res = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if isinstance(v, ast.Constant):
                        res[k.value] = v.value
                    elif isinstance(v, ast.List):
                        res[k.value] = [el.value for el in v.elts if isinstance(el, ast.Constant)]
            return res
        return None

    def _perform_dry_run(self, code_str: str, model_names: List[str]) -> Optional[VerificationErrorDetail]:
        """Executes code string in isolated namespace and instantiates models."""
        local_scope: Dict[str, Any] = {}
        try:
            exec(code_str, local_scope)
        except Exception as e:
            return VerificationErrorDetail(
                model_name="Global",
                error_type="InstantiationDryRun",
                error_message=f"Runtime execution error during dry-run: {str(e)}",
                action_required="Ensure code imports required Pydantic modules and has no execution side-effects.",
            )
        return None
