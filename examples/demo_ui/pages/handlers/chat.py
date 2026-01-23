"""
Chat handler for the trained model.
"""
import os
import json
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import get_training_dir

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
        
        # Generate
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Decode
        # Extract the new tokens only
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # User requested RAW response showing all tags
        # Wrap in code block to prevent browser from hiding <tags>
        response = f"```text\n{response}\n```"
        
        # Update history (Gradio 6.0 expects dictionary format)
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
        return "", new_history
        
    except Exception as e:
        gr.Warning(f"Error during generation: {str(e)}")
        # Return history unchanged on error, effectively ignoring the latest message but showing the warning
        return "", history
