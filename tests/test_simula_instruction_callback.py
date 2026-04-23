"""Simula instruction callback for ConversationGenerator."""

import pytest

from afterimage.simula.tasks.multiturn_bridge import SimulaInstructionGeneratorCallback


@pytest.mark.asyncio
async def test_simula_callback_round_robin():
    cb = SimulaInstructionGeneratorCallback(
        [
            ("First user message", {"mix": "m1"}),
            ("Second dialog opener", {"mix": "m2"}),
        ]
    )
    g1 = await cb.agenerate("corr")
    assert g1.instructions == ["First user message"]
    g2 = await cb.agenerate("corr")
    assert g2.instructions == ["Second dialog opener"]
    with pytest.raises(IndexError):
        await cb.agenerate("corr")
