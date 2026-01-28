"""
Chat handler for the trained model.
"""
import os
import re
import json
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import get_training_dir


def extract_balanced_braces(text: str, start_pos: int) -> str:
    """Extract content within balanced braces starting at start_pos."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return ""
    
    depth = 0
    end_pos = start_pos
    
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    
    return text[start_pos + 1:end_pos]


def parse_tool_args(args_str: str) -> dict:
    """Parse tool arguments from string, handling various formats."""
    args_str = args_str.strip()
    if not args_str:
        return {}
    
    # Clean up escape tags
    args_str = re.sub(r'<escape>([^<]*)</escape>', r'"\1"', args_str)
    
    # Convert Python None/True/False to JSON
    args_str = re.sub(r'\bNone\b', 'null', args_str)
    args_str = re.sub(r'\bTrue\b', 'true', args_str)
    args_str = re.sub(r'\bFalse\b', 'false', args_str)
    
    # Quote unquoted keys (only after { or ,)
    args_str = re.sub(r'([\{,])\s*(\w+)\s*:', r'\1"\2":', args_str)
    
    # Try JSON parse
    try:
        return json.loads("{" + args_str + "}")
    except json.JSONDecodeError:
        pass
    
    # Fallback: manual parsing
    result = {}
    pattern = r'"?(\w+)"?\s*:\s*(?:"([^"]*)"|(\w+))'
    for match in re.finditer(pattern, args_str):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        if value and value.lower() not in ('null', 'none'):
            result[key] = value
    
    return result


def format_tool_calls(raw_response: str) -> str:
    """
    Parse raw model response and format it nicely.
    
    Input: "OK, I'll lock the doors.<start_function_call>call:lock_door{...}<end_function_call>"
    Output: "OK, I'll lock the doors.\n\n**Called Functions:**\n- lock_door(door='garage')"
    """
    # Extract text before first function call
    text_part = raw_response
    if "<start_function_call>" in raw_response:
        text_part = raw_response.split("<start_function_call>")[0].strip()
    
    # Find all function calls using balanced brace extraction
    formatted_calls = []
    pattern = r'<start_function_call>call:(\w+)\{'
    
    for match in re.finditer(pattern, raw_response):
        func_name = match.group(1)
        brace_start = match.end() - 1  # Position of '{'
        args_str = extract_balanced_braces(raw_response, brace_start)
        
        # Parse arguments
        args = parse_tool_args(args_str)
        
        # Format non-None arguments
        formatted_args = []
        for key, value in args.items():
            if value is not None:
                if isinstance(value, str):
                    formatted_args.append(f"{key}='{value}'")
                else:
                    formatted_args.append(f"{key}={value}")
        
        args_display = ", ".join(formatted_args) if formatted_args else ""
        formatted_calls.append(f"- `{func_name}({args_display})`")
    
    if not formatted_calls:
        return raw_response
    
    # Build formatted response
    result = text_part
    if formatted_calls:
        result += "\n\n**Called Functions:**\n" + "\n".join(formatted_calls)
    
    return result


def chat_with_trained_model(message: str, history: list):
    """Chat with the trained model."""
    if not message or not message.strip():
        return "", history
    
    try:
        # Load model from training directory
        training_dir = get_training_dir()
        model_path = os.path.join(training_dir, "final_model_stable")
        
        if not os.path.exists(model_path):
            gr.Warning("Model not found. Please train a model first.")
            return "", history
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Use tokenizer's chat template which matches training
        messages = [{"role": "user", "content": message}]
        
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)
        
        # Create attention mask to avoid warning
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        # Generate
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode - extract new tokens only
        raw_response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Format the response nicely
        response = format_tool_calls(raw_response)
        
        # Update history (Gradio 6.0 expects dictionary format)
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        return "", new_history
        
    except Exception as e:
        gr.Warning(f"Error during generation: {str(e)}")
        # Return history unchanged on error
        return "", history
