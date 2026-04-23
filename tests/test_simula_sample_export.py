"""JSONL export for DataPointRecord."""

from __future__ import annotations

import json
from pathlib import Path

from afterimage.simula.sample_export import append_datapoints_jsonl
from afterimage.simula.types import DataPointLineage, DataPointRecord


def _rec(q: str = "Q?", a: str = "A.") -> DataPointRecord:
    return DataPointRecord(
        task="single_qa",
        payload={"question": q, "answer": a},
        lineage=DataPointLineage(
            instruction_y="y",
            mix_id="m1",
            meta_prompt_id="meta1",
        ),
    )


def test_append_datapoints_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data" / "rows.jsonl"
    n = append_datapoints_jsonl(path, [_rec("1"), _rec("2")])
    assert n == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    r0 = DataPointRecord.model_validate_json(lines[0])
    assert r0.payload["question"] == "1"
    append_datapoints_jsonl(path, [_rec("3")])
    lines2 = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines2) == 3
    assert json.loads(lines2[2])["payload"]["question"] == "3"
