import pytest
from afterimage.agent_trace.verifier import SchemaVerifier, VerificationReport


def test_schema_verifier_syntactic_validity():
    verifier = SchemaVerifier()
    invalid_code = "class BadModel(BaseModel:\n    id: int"
    report = verifier.verify_code(invalid_code)
    assert not report.is_valid
    assert any(err.error_type == "SyntaxError" for err in report.errors)


def test_schema_verifier_model_hierarchy():
    verifier = SchemaVerifier()
    code_no_models = "x = 42\ny = 100"
    report = verifier.verify_code(code_no_models)
    assert not report.is_valid
    assert any(err.error_type == "ModelHierarchy" for err in report.errors)


def test_schema_verifier_foreign_key_safety():
    verifier = SchemaVerifier()
    code_missing_fk_target = """
from pydantic import BaseModel, Field

class OrderResponse(BaseModel):
    order_id: int = Field(json_schema_extra={"generator": "id"})
    user_id: int = Field(json_schema_extra={"generator": "fk:user.id"})
"""
    report = verifier.verify_code(code_missing_fk_target)
    assert not report.is_valid
    assert any(err.error_type == "ForeignKeySafety" for err in report.errors)


def test_schema_verifier_valid_graph():
    verifier = SchemaVerifier()
    valid_code = """
from pydantic import BaseModel, Field

class UserResponse(BaseModel):
    user_id: int = Field(json_schema_extra={"generator": "id"})
    name: str = Field(json_schema_extra={"generator": "faker:name"})

class OrderResponse(BaseModel):
    order_id: int = Field(json_schema_extra={"generator": "id"})
    user_id: int = Field(json_schema_extra={"generator": "fk:user.user_id"})
    total: float = Field(json_schema_extra={"generator": "money"})
"""
    report = verifier.verify_code(valid_code)
    assert report.is_valid
    assert len(report.errors) == 0
