#!/usr/bin/env python3
"""
Function Gemma Training Script
================================
Fine-tune Gemma model for function calling.
"""

import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

import training_config as config
from utils import clean_memory, load_tools_schema
from utils_custom_split import prepare_dataset_with_tool_call_test


def parse_args():
    """Parse command line arguments for developer mode"""
    parser = argparse.ArgumentParser(description="Train Function Gemma model")
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=config.PER_DEVICE_BATCH_SIZE, help="Batch size per device")
    parser.add_argument("--grad_accumulation", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--test_size", type=float, default=config.TEST_SIZE, help="Test split ratio")
    parser.add_argument("--logging_steps", type=int, default=config.LOGGING_STEPS, help="Logging frequency")
    return parser.parse_args()


def validate_token():
    """Validate HuggingFace token"""
    if not config.HF_TOKEN or config.HF_TOKEN.startswith("hf_xx"):
        print("[ERROR] Invalid HF_TOKEN!")
        print("[INFO] Add to .env file: HF_TOKEN=hf_xxxxxxxxxxxx")
        sys.exit(1)
    print("[OK] HuggingFace token validated")


def check_files():
    """Check required files"""
    if not os.path.exists(config.DATASET_FILE):
        print(f"[ERROR] Dataset file not found: {config.DATASET_FILE}")
        sys.exit(1)
    
    print(f"[OK] Dataset: {config.DATASET_FILE}")
    
    if os.path.exists(config.TOOLS_FILE):
        print(f"[OK] Tools: {config.TOOLS_FILE}")
    else:
        print(f"[INFO] Tools schema not found (optional)")


def load_model():
    """Load model and tokenizer"""
    print(f"\n[INFO] Loading model: {config.MODEL_ID}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, 
        token=config.HF_TOKEN
    )
    tokenizer.padding_side = 'right'
    
    # Model (Safe mode - FP32)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32,  # Safe mode
        attn_implementation="eager",
        token=config.HF_TOKEN
    )
    
    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print("[OK] Model loaded")
    return tokenizer, model


def create_trainer(model, tokenizer, train_dataset, eval_dataset, hyperparams):
    """Create trainer with custom hyperparameters"""
    args = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=hyperparams['num_epochs'],
        
        per_device_train_batch_size=hyperparams['batch_size'],
        gradient_accumulation_steps=hyperparams['grad_accumulation'],
        gradient_checkpointing=True,
        
        fp16=config.USE_FP16,
        bf16=config.USE_BF16,
        optim=config.OPTIMIZER,
        
        logging_steps=hyperparams['logging_steps'],
        
        # Evaluation & Saving
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=config.LOAD_BEST_MODEL,
        save_only_model=True,  # Save model only (no optimizer state)
        
        learning_rate=hyperparams['learning_rate'],
        packing=False,
        report_to="none",
        dataset_text_field="text"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer


def main():
    """Main training function"""
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("FUNCTION GEMMA TRAINING")
    print("=" * 60)
    
    # Print hyperparameters
    print("\n[INFO] Hyperparameters:")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Gradient Accumulation: {args.grad_accumulation}")
    print(f"  - Test Split: {args.test_size}")
    print(f"  - Logging Steps: {args.logging_steps}")
    print()
    
    # 1. Preparation
    validate_token()
    check_files()
    clean_memory()
    
    # 2. Load model
    tokenizer, model = load_model()
    
    # 3. Prepare dataset (custom split: only tool call samples for test)
    tools_schema = load_tools_schema(config.TOOLS_FILE) if os.path.exists(config.TOOLS_FILE) else None
    split_dataset = prepare_dataset_with_tool_call_test(
        config.DATASET_FILE,
        tokenizer,
        tools_schema,
        test_size=args.test_size,
        seed=config.RANDOM_SEED
    )
    
    # 4. Create trainer with custom hyperparameters
    print("\n[INFO] Creating trainer...")
    hyperparams = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'grad_accumulation': args.grad_accumulation,
        'logging_steps': args.logging_steps,
    }
    
    trainer = create_trainer(
        model,
        tokenizer,
        split_dataset["train"],
        split_dataset["test"],
        hyperparams
    )
    
    # 5. Start training
    print("\n[INFO] TRAINING STARTED...")
    print("=" * 60)
    trainer.train()
    
    print("\n" + "=" * 60)
    print("[OK] TRAINING COMPLETED!")
    print("=" * 60)
    
    # 6. Save model (silently)
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # Save test dataset (raw format)
    test_output_file = "data/test_dataset.jsonl"
    
    with open(test_output_file, "w", encoding="utf-8") as f:
        for item in split_dataset["test_raw"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("[OK] Model saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
