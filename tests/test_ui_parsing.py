import os
import json
import pandas as pd
from examples.demo_ui.utils import create_document_provider_from_file


def test_parsing():
    # 1. Test TXT
    txt_path = "test_context.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Line 1\nLine 2\n\nLine 3")

    provider = create_document_provider_from_file(txt_path)
    docs = provider.get_all()
    print(f"TXT: Found {len(docs)} docs. Content: {[d.text for d in docs]}")
    assert len(docs) == 3
    os.remove(txt_path)

    # 2. Test CSV
    csv_path = "test_context.csv"
    df = pd.DataFrame({"text": ["Row 1", "Row 2"], "other": ["a", "b"]})
    df.to_csv(csv_path, index=False)

    provider = create_document_provider_from_file(csv_path, key="text")
    docs = provider.get_all()
    print(f"CSV: Found {len(docs)} docs. Content: {[d.text for d in docs]}")
    assert len(docs) == 2
    os.remove(csv_path)

    # 3. Test JSONL
    jsonl_path = "test_context.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"content": "JSONL 1"}) + "\n")
        f.write(json.dumps({"content": "JSONL 2"}) + "\n")

    provider = create_document_provider_from_file(jsonl_path, key="content")
    docs = provider.get_all()
    print(f"JSONL: Found {len(docs)} docs. Content: {[d.text for d in docs]}")
    assert len(docs) == 2
    os.remove(jsonl_path)

    print("All parsing tests passed!")


if __name__ == "__main__":
    test_parsing()
