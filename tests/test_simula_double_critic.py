"""Double-critic acceptance logic."""

import pytest

from afterimage.simula.double_critic import accept_double_critique, double_critique_mcq
from afterimage.simula.types import DoubleCritiqueVerdict, MCQRow


def test_accept_double_critique_rule():
    assert accept_double_critique(
        DoubleCritiqueVerdict(
            claims_correct=True,
            claims_incorrect=False,
        )
    )
    assert not accept_double_critique(
        DoubleCritiqueVerdict(
            claims_correct=True,
            claims_incorrect=True,
        )
    )
    assert not accept_double_critique(
        DoubleCritiqueVerdict(
            claims_correct=False,
            claims_incorrect=False,
        )
    )


@pytest.mark.asyncio
async def test_double_critique_mcq_mock():
    from afterimage.providers.llm_providers import StructuredLLMResponse
    from afterimage.simula.schemas_llm import DoubleProbeCorrect, DoubleProbeIncorrect

    class Fake:
        calls = 0

        async def agenerate_structured(self, prompt, schema, temperature=0.7, **kwargs):
            Fake.calls += 1
            if schema is DoubleProbeCorrect:
                parsed = DoubleProbeCorrect(is_correct=True, rationale="ok")
            else:
                parsed = DoubleProbeIncorrect(is_incorrect=False, rationale="ok")
            return StructuredLLMResponse(
                text="",
                prompt_token_count=0,
                completion_token_count=0,
                total_token_count=0,
                finish_reason="stop",
                model_name="fake",
                raw_response=None,
                parsed=parsed,
            )

    row = MCQRow(
        question="2+2?",
        choices=["3", "4", "5", "6"],
        correct_index=1,
    )
    v = await double_critique_mcq(Fake(), row=row)
    assert accept_double_critique(v)
    assert Fake.calls == 2
