#!/usr/bin/env python3
"""
Model Evaluation Script
=======================
Evaluate fine-tuned model with detailed metrics.
"""

import os
import sys
import json
import ast
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add parent directory to path to import from core package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.parsing_utils import try_parse_args, extract_balanced_braces

import training_config as config
from utils import load_tools_schema


def load_model(model_path):
    """Load fine-tuned model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    return tokenizer, model


def load_test_data():
    """Load test dataset"""
    test_file = "data/test_dataset.jsonl"
    
    if not os.path.exists(test_file):
        print(f"[ERROR] Test dataset not found: {test_file}")
        print("[INFO] Please train the model first.\n")
        sys.exit(1)
    
    dataset = load_dataset("json", data_files=test_file, split="train")
    test_data = list(dataset)
    
    return test_data





def generate_tool_calls(model, tokenizer, messages, tools):
    """Model inference - generate tool calls"""
    # Convert to Function Gemma format
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse tool calls from response
    try:
        tool_calls = []
        
        # Try Function Gemma format first: <function=name>args</function>
        if "<function=" in response:
            pattern = r'<function=(\w+)>'
            for match in re.finditer(pattern, response):
                tool_name = match.group(1)
                content_start = match.end()
                
                # Find the closing </function> tag
                end_tag = "</function>"
                end_idx = response.find(end_tag, content_start)
                if end_idx == -1:
                    continue
                
                args_str = response[content_start:end_idx].strip()
                arguments = try_parse_args(args_str)
                
                if arguments is not None:
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    })
                else:
                    print(f"  [WARNING] Failed to parse args for {tool_name}: {args_str[:100]}")
        
        # Try alternative format: <start_function_call>call:name{...}<end_function_call>
        elif "<start_function_call>" in response:
            pattern = r'<start_function_call>call:(\w+)'
            for match in re.finditer(pattern, response):
                tool_name = match.group(1)
                brace_start = match.end()
                
                # Find opening brace
                while brace_start < len(response) and response[brace_start] != '{':
                    brace_start += 1
                
                # Extract balanced braces content
                args_str = extract_balanced_braces(response, brace_start)
                
                if args_str:
                    arguments = try_parse_args(args_str)
                    
                    if arguments is not None:
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments)
                            }
                        })
                    else:
                        print(f"  [WARNING] Failed to parse args for {tool_name}: {args_str[:100]}")
        
        # Try plain JSON array format (some models output this)
        elif response.strip().startswith('['):
            try:
                parsed = json.loads(response.strip())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "function" in item:
                            tool_calls.append(item)
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    except Exception as e:
        print(f"  [WARNING] Parse error: {e}")
        return []


def compare_tool_calls(expected, predicted):
    """Compare and categorize tool calls"""
    exp_count = len(expected)
    pred_count = len(predicted)
    
    # Exact match
    if exp_count == pred_count == 0:
        return "no_tools_needed", {}
    
    if exp_count == 0 and pred_count > 0:
        return "false_positive", {"expected": 0, "predicted": pred_count}
    
    if exp_count > 0 and pred_count == 0:
        return "missed_all", {"expected": exp_count, "predicted": 0}
    
    # Check tool call count
    if exp_count != pred_count:
        if pred_count < exp_count:
            return "missing_tools", {"expected": exp_count, "predicted": pred_count}
        else:
            return "extra_tools", {"expected": exp_count, "predicted": pred_count}
    
    # Same count, check content
    correct_tools = 0
    correct_args = 0
    
    for i in range(exp_count):
        exp_func = expected[i].get("function", {})
        pred_func = predicted[i].get("function", {})
        
        exp_name = exp_func.get("name", "")
        pred_name = pred_func.get("name", "")
        
        if exp_name == pred_name:
            correct_tools += 1
            
            # Check arguments
            try:
                exp_args_raw = exp_func.get("arguments", "{}")
                pred_args_raw = pred_func.get("arguments", "{}")
                
                # Parse if string, otherwise use as-is
                if isinstance(exp_args_raw, str):
                    exp_args = json.loads(exp_args_raw)
                else:
                    exp_args = exp_args_raw
                
                if isinstance(pred_args_raw, str):
                    pred_args = json.loads(pred_args_raw)
                else:
                    pred_args = pred_args_raw
                
                if exp_args == pred_args:
                    correct_args += 1
            except Exception as e:
                pass
    
    if correct_tools == exp_count and correct_args == exp_count:
        return "exact_match", {}
    elif correct_tools == exp_count:
        return "correct_tools_wrong_args", {"correct_tools": correct_tools, "correct_args": correct_args}
    else:
        return "wrong_tool_names", {"correct_tools": correct_tools, "total": exp_count}


def evaluate_model(model_path=None):
    """Main model evaluation function"""
    if model_path is None:
        model_path = config.OUTPUT_DIR
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    
    # Load model
    tokenizer, model = load_model(model_path)
    
    # Load test data
    test_data = load_test_data()
    
    # Load tools schema (optional)
    tools = load_tools_schema(config.TOOLS_FILE) if os.path.exists(config.TOOLS_FILE) else None
    
    # Evaluation
    print("\n[INFO] Starting evaluation on test dataset...")
    
    results = {
        "exact_match": [],
        "correct_tools_wrong_args": [],
        "missing_tools": [],
        "extra_tools": [],
        "wrong_tool_names": [],
        "missed_all": [],
        "false_positive": [],
        "no_tools_needed": [],
    }
    
    for idx, item in enumerate(test_data, 1):
        # Parse instruction and output
        user_message = item.get("instruction", "")
        output_data = item.get("output", {})
        
        if isinstance(output_data, str):
            try:
                output_data = json.loads(output_data)
            except json.JSONDecodeError:
                output_data = {}
        
        # Expected tool calls
        expected_tool_calls = output_data.get("tool_calls", [])
        expected_response = output_data.get("response", "")
        
        # Messages format for model
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        print(f"\n[{idx}/{len(test_data)}]")
        print(f"User: {user_message}")
        
        # Generate prediction
        predicted_tool_calls = generate_tool_calls(
            model, tokenizer, messages, tools
        )
        
        # Compare
        category, details = compare_tool_calls(expected_tool_calls, predicted_tool_calls)
        
        # Format expected
        print(f"\nExpected:")
        if expected_tool_calls:
            for tc in expected_tool_calls:
                func = tc.get("function", {})
                print(f"  - {func.get('name')}: {func.get('arguments')}")
        else:
            print(f"  (no tool calls)")
        
        # Format predicted
        print(f"\nPredicted:")
        if predicted_tool_calls:
            for tc in predicted_tool_calls:
                func = tc.get("function", {})
                print(f"  - {func.get('name')}: {func.get('arguments')}")
        else:
            print(f"  (no tool calls)")
        
        print(f"\nResult: {category}")
        if details:
            print(f"Details: {details}")
        print("-" * 80)
        
        results[category].append({
            "index": idx,
            "user_message": user_message,
            "expected": expected_tool_calls,
            "predicted": predicted_tool_calls,
            "details": details
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    total = len(test_data)
    
    print(f"\nTotal Tests: {total}\n")
    
    # Category descriptions
    category_names = {
        "exact_match": "[OK] Exact Match",
        "correct_tools_wrong_args": "[PARTIAL] Correct Tool, Wrong Args",
        "missing_tools": "[ERROR] Missing Tools",
        "extra_tools": "[ERROR] Extra Tools",
        "wrong_tool_names": "[ERROR] Wrong Tool Names",
        "missed_all": "[ERROR] No Tools Generated",
        "false_positive": "[ERROR] Unnecessary Tools",
        "no_tools_needed": "[OK] No Tools Needed",
    }
    
    for category, items in results.items():
        count = len(items)
        percentage = (count / total * 100) if total > 0 else 0
        if count > 0:
            display_name = category_names.get(category, category)
            print(f"{display_name:.<40} {count:>3} ({percentage:>5.1f}%)")
    
    print("=" * 80)
    
    # Save detailed results (silently)
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == "__main__":
    evaluate_model()
