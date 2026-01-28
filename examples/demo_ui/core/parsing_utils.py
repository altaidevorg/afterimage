"""
Shared parsing utilities for the demo UI.
"""
import json
import ast
import re

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


def try_parse_args(args_str: str) -> dict:
    """
    Try multiple methods to parse tool arguments.
    Consolidated from chat.py and evaluate.py.
    """
    args_str = args_str.strip()
    if not args_str:
        return {}

    # Method 1: Direct JSON parse (with added braces if missing)
    try:
        if not args_str.startswith('{'):
            return json.loads("{" + args_str + "}")
        return json.loads(args_str)
    except json.JSONDecodeError:
        pass

    # Method 2: Handle <escape> tags
    # Supports both <escape>val</escape> and <escape>val<escape>
    cleaned = re.sub(r'<escape>([^<]*)</?(?:escape)?>', r'"\1"', args_str)
    
    # Method 3: Convert Python values to JSON and add quotes to keys
    def json_repair(s):
        # Add quotes around unquoted keys (only after { or ,)
        s = re.sub(r'([\{,])\s*(\w+)\s*:', r'\1"\2":', s)
        # Replace Python None/True/False with JSON null/true/false using word boundaries
        s = re.sub(r'\bNone\b', 'null', s)
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        return s

    repaired = json_repair(cleaned)
    try:
        if not repaired.startswith('{'):
            return json.loads("{" + repaired + "}")
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Method 4: Try ast.literal_eval (handles Python dict syntax)
    try:
        # Wrap in braces if it looks like key-value pairs without them
        eval_str = cleaned if cleaned.startswith('{') else "{" + cleaned + "}"
        return ast.literal_eval(eval_str)
    except (ValueError, SyntaxError, TypeError):
        pass

    # Method 5: Fallback manual parsing for very broken strings
    result = {}
    pattern = r'"?(\w+)"?\s*:\s*(?:"([^"]*)"|(\w+))'
    for match in re.finditer(pattern, cleaned):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        if value and value.lower() not in ('null', 'none'):
            result[key] = value
            
    return result
