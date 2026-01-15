import json
import torch
import gc
from typing import Dict, Any, List
from datasets import load_dataset


def clean_memory():
    """Clean RAM and GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def load_tools_schema(tools_file: str) -> List[Dict[str, Any]]:
    """Load tools schema file"""
    with open(tools_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_dataset(dataset_file: str):
    """Load JSONL file"""
    return load_dataset("json", data_files=dataset_file, split="train")


def format_chat(sample: Dict[str, Any], tokenizer, tools_schema: List[Dict[str, Any]]) -> Dict[str, str]:
    """Convert to chat format"""
    output_data = sample['output']
    
    # Convert string to JSON if needed
    if isinstance(output_data, str):
        try:
            output_data = json.loads(output_data)
        except json.JSONDecodeError:
            output_data = {"response": output_data}
    
    # Create messages
    messages = [
        {"role": "user", "content": sample['instruction']},
        {"role": "model", "content": output_data.get('response', "")}
    ]
    
    # Tool calls varsa ekle
    if output_data.get('tool_calls'):
        messages[1]["tool_calls"] = output_data['tool_calls']
    
    # Tokenizer ile format
    text = tokenizer.apply_chat_template(
        messages, 
        tools=tools_schema, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": text}


def save_results(results: Dict[str, Any], output_file: str):
    """Save results as JSON"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved: {output_file}")
