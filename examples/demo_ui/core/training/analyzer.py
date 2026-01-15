"""
Dataset analysis functionality for training.
"""
import sys
from typing import Tuple, List, Dict, Any

from ..config import get_training_dir, get_data_dir
from ..file_utils import copy_dataset_file


def run_analysis(training_file) -> Tuple[Dict[str, Any], List[str], List[int]]:
    """
    Analyze uploaded dataset and return statistics.
    
    Args:
        training_file: File object from Gradio upload
        
    Returns:
        Tuple of (stats_dict, tool_names, tool_counts) for Gradio outputs
        - stats_dict: Dictionary with dataset statistics
        - tool_names: List of tool names
        - tool_counts: List of counts corresponding to each tool
    """
    if not training_file:
        return {}, [], []
    
    data_dir = get_data_dir()
    
    try:
        # Copy dataset to training directory
        dataset_path = copy_dataset_file(training_file, data_dir)
        
        # Import and run analysis directly
        training_dir = get_training_dir()
        sys.path.insert(0, training_dir)
        try:
            from analyze_dataset import analyze_dataset
            stats = analyze_dataset(dataset_path)
        finally:
            sys.path.remove(training_dir)
        
        # Prepare data for plots
        tool_names = list(stats.get('tool_distribution', {}).keys())
        tool_counts = list(stats.get('tool_distribution', {}).values())
        
        return stats, tool_names, tool_counts
            
    except Exception as e:
        return {"error": str(e)}, [], []
