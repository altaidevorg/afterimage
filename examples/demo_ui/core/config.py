"""
Configuration and environment management for the demo UI.
"""
import os


# --- API Configuration ---

def get_api_key() -> str:
    """
    Get the Gemini API key from environment.
    
    Raises:
        ValueError: If the API key is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return api_key


# --- Path Configuration ---

def get_training_dir() -> str:
    """Get the path to the training directory."""
    return os.path.join(os.path.dirname(__file__), "..", "training_scripts")


def get_data_dir() -> str:
    """Get the path to the training data directory."""
    return os.path.join(get_training_dir(), "data")


# --- Environment Setup ---

def setup_environment():
    """Set up environment variables for the application."""
    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


# --- Display Configuration ---

SPINNERS = ["◐", "◓", "◑", "◒"]
PROGRESS_BAR_LENGTH = 40
ANIMATION_FPS = 0.15  # seconds per frame
