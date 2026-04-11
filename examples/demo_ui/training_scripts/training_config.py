import os
from dotenv import find_dotenv, load_dotenv

# Resolve .env from cwd or a parent directory (e.g. repo root when cwd is training_scripts).
load_dotenv(find_dotenv(usecwd=True))

# ==========================================
# TOKEN AND MODEL SETTINGS
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "google/functiongemma-270m-it"

# ==========================================
# DATASET SETTINGS
# ==========================================
DATASET_FILE = "data/toolcalldataset.jsonl"  # All data
TOOLS_FILE = "data/tools.json"
TEST_SIZE = 0.10  # Test split ratio (10%)
RANDOM_SEED = 42  # Seed for reproducibility

# ==========================================
# TRAINING SETTINGS
# ==========================================
OUTPUT_DIR = "./final_model_stable"
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
LOGGING_STEPS = 5

# Use FP16/BF16 (safe mode)
USE_FP16 = False
USE_BF16 = False

# Optimizer
OPTIMIZER = "adamw_torch"

# Evaluation
EVAL_STRATEGY = "epoch"  # Evaluate at end of each epoch
SAVE_STRATEGY = "no"  # Checkpoint saving (only final model)
SAVE_TOTAL_LIMIT = 0  # Checkpoint limit (0 = no saving)
LOAD_BEST_MODEL = False  # False because no checkpoints

# ==========================================
# EVALUATION SETTINGS
# ==========================================
EVAL_OUTPUT_DIR = "./evaluation_results"
EVAL_BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
