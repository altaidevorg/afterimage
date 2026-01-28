"""
Event handlers for the Train Model page.
"""
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime
from glob import glob
from typing import Any, List, Tuple
import zipfile

import gradio as gr
import pandas as pd


from core.config import get_training_dir, get_datasets_dir


# =============================================================================
# Utility Functions
# =============================================================================

def get_meta_path(jsonl_path: str) -> str:
    """Convert .jsonl path to .meta.json path."""
    return jsonl_path.replace(".jsonl", ".meta.json")


def extract_filename_from_label(label: str) -> str:
    """Extract filename from choice label like 'filename.jsonl (123 KB)'."""
    return label.split(" (")[0] if " (" in label else label


def get_dataset_path(filename: str) -> str:
    """Get full path to a dataset file."""
    return os.path.join(get_datasets_dir(), filename)


# =============================================================================
# Model Download Functions  
# =============================================================================

def get_model_download_label(model_dir: str) -> str:
    """Calculate model directory size and return formatted label."""
    try:
        total_size_mb = 0
        if os.path.exists(model_dir):
            for dirpath, _, filenames in os.walk(model_dir):
                for filename in filenames:
                    try:
                        filepath = os.path.join(dirpath, filename)
                        total_size_mb += os.path.getsize(filepath)
                    except OSError:
                        pass # Ignore temporary file errors
            
            total_size_mb /= (1024 * 1024)
        
        return f"Download Model (~{total_size_mb:.1f} MB)"
    except Exception:
        return "Download Model"



def _list_datasets():
    """Return dataset metadata for UI lists."""
    datasets_dir = get_datasets_dir()
    jsonl_files = glob(os.path.join(datasets_dir, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)

    choices = []
    table_rows = []
    for path in jsonl_files:
        basename = os.path.basename(path)
        size_kb = os.path.getsize(path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        choices.append(f"{basename} ({size_kb:.1f} KB)")
        table_rows.append(
            {"Name": basename, "Size KB": f"{size_kb:.1f}", "Modified": mtime}
        )

    return choices, table_rows


def _get_dataset_category(path: str) -> str:
    """Get category from dataset metadata file."""
    meta_path = get_meta_path(path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("category", "Uncategorized")
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return "Uncategorized"


def get_datasets_by_category() -> dict[str, List[str]]:
    """
    Get all datasets grouped by category.
    
    Returns:
        Dictionary mapping category names to lists of dataset choice labels
    """
    datasets_dir = get_datasets_dir()
    jsonl_files = glob(os.path.join(datasets_dir, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)
    
    grouped: dict[str, List[str]] = {}
    
    for path in jsonl_files:
        basename = os.path.basename(path)
        size_kb = os.path.getsize(path) / 1024
        choice_label = f"{basename} ({size_kb:.1f} KB)"
        
        category = _get_dataset_category(path)
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(choice_label)
    
    # Sort categories with "Uncategorized" at the end
    sorted_grouped = {}
    for cat in sorted(grouped.keys()):
        if cat != "Uncategorized":
            sorted_grouped[cat] = grouped[cat]
    if "Uncategorized" in grouped:
        sorted_grouped["Uncategorized"] = grouped["Uncategorized"]
    
    return sorted_grouped


def get_dataset_categories() -> List[str]:
    """Get list of all unique dataset categories."""
    datasets_dir = get_datasets_dir()
    jsonl_files = glob(os.path.join(datasets_dir, "*.jsonl"))
    
    categories = set()
    for path in jsonl_files:
        categories.add(_get_dataset_category(path))
    
    # Sort with "Uncategorized" at the end
    result = sorted(c for c in categories if c != "Uncategorized")
    if "Uncategorized" in categories:
        result.append("Uncategorized")
    
    return result


def update_dataset_category(path: str, category: str) -> bool:
    """
    Update the category of a dataset in its metadata file.
    
    Args:
        path: Path to the .jsonl file
        category: New category name
        
    Returns:
        True if successful, False otherwise
    """
    meta_path = get_meta_path(path)
    
    try:
        # Load existing metadata or create new
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        
        # Update category
        meta["category"] = category or "Uncategorized"
        
        # Save back
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Failed to update dataset category: {e}")
        return False


def _choice_to_path(choice: str) -> str:
    """Convert a dataset choice label to a full file path."""
    filename = extract_filename_from_label(choice)
    return get_dataset_path(filename)


def _parse_samples_from_meta(meta: dict) -> int:
    """Safely parse 'total_samples' from metadata."""
    samples = meta.get("total_samples", 0)
    if isinstance(samples, str) and samples.isdigit():
        return int(samples)
    return samples if isinstance(samples, int) else 0


def choice_to_path(choice: str) -> str:
    """Public wrapper for converting choice labels to paths."""
    return _choice_to_path(choice)


def load_datasets_list():
    """Load list of saved datasets for the UI."""
    choices, _ = _list_datasets()
    return gr.update(choices=choices, value=[])


def on_dataset_select(selected_names: List[str] | None):
    """When datasets are selected, return paths, overview, distribution dict, config dict, and button updates."""
    empty_overview = """
    <div style="text-align: center; padding: 30px; color: #94a3b8;">
        Select datasets to see statistics
    </div>
    """
    if not selected_names:
        return (
            None, 
            empty_overview,  # overview HTML
            {},  # tool_dist_state
            {},  # filter_config_state
            gr.update(interactive=False),  # train_btn
            gr.update(interactive=False),  # train_dev_btn
            0,  # total_samples
        )
    
    datasets_dir = get_datasets_dir()
    full_paths = []
    
    # Aggregated stats
    total_samples = 0
    total_size_kb = 0
    tools_counter = Counter()
    has_meta = False
    
    for name in selected_names:
        filename = extract_filename_from_label(name)
        full_path = os.path.join(datasets_dir, filename)
        
        if os.path.exists(full_path):
            full_paths.append(full_path)
            total_size_kb += os.path.getsize(full_path) / 1024
            
            # Try to load metadata
            meta_path = get_meta_path(full_path)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    has_meta = True
                    # Parse samples
                    total_samples += _parse_samples_from_meta(meta)
                    
                    # Parse tool distribution
                    tool_dist = meta.get("tool_distribution", {})
                    if tool_dist:
                        tools_counter.update(tool_dist)
                    else:
                        tools_used = meta.get("tools_used", [])
                        for t in tools_used:
                            tools_counter[t] += 1
                            
                except Exception:
                    pass
    
    if not full_paths:
        return (
            None, 
            '<div style="text-align: center; padding: 30px; color: #ef4444;">Dataset files not found</div>',
            {},
            {},
            gr.update(interactive=False),
            gr.update(interactive=False),
            0,  # total_samples
        )

    # Build tool distribution dict
    dist_dict = dict(tools_counter) if tools_counter else {}
    
    # Build overview HTML dashboard - 3 Stat Cards (total_samples from metadata)
    overview = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Datasets</div>
            <div style="font-size: 24px; font-weight: 700; color: #334155;">{len(full_paths)}</div>
        </div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Size</div>
            <div style="font-size: 24px; font-weight: 700; color: #334155;">{total_size_kb:.1f} KB</div>
        </div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Samples</div>
            <div style="font-size: 24px; font-weight: 700; color: #334155;">{total_samples if has_meta else "?"}</div>
        </div>
    </div>
    """
    
    # Add warning if metadata is missing
    if not has_meta:
        overview += """
        <div style="padding: 12px; border-radius: 6px; background: #fff7ed; border: 1px solid #ffedd5; color: #9a3412; font-size: 13px; margin-top: 12px;">
            Complete metadata is not available for one or more selected datasets. Filtering may not work as expected.
        </div>
        """
    
    return (
        full_paths, 
        overview,  # overview HTML content
        dist_dict,  # tool_dist_state
        dist_dict,  # filter_config_state (initially same as max)
        gr.update(interactive=True),
        gr.update(interactive=True),
        total_samples,  # total samples from metadata
    )


def update_action_buttons(selected: List[str] | None):
    """Enable/disable Edit/Delete/Merge buttons based on selection."""
    count = len(selected) if selected else 0
    return (
        gr.update(interactive=(count == 1)),
        gr.update(interactive=(count == 1)),
        gr.update(interactive=(count >= 2)),
    )


def update_action_buttons_inline(selected: List[str] | None):
    """Enable/disable buttons and show inline rename for single selection."""
    count = len(selected) if selected else 0
    
    if count == 1:
        # Single selection: show rename input with current name
        path = _choice_to_path(selected[0])
        current_name = os.path.basename(path)
        return (
            gr.update(interactive=True),   # delete_btn
            gr.update(interactive=False),  # merge_btn (need 2+)
            gr.update(visible=True),       # rename_row
            gr.update(value=current_name), # rename_input
            path,                          # rename_target_path
        )
    elif count >= 2:
        # Multiple selection: hide rename, enable merge
        return (
            gr.update(interactive=False),  # delete_btn (need 1)
            gr.update(interactive=True),   # merge_btn
            gr.update(visible=False),      # rename_row
            gr.update(value=""),           # rename_input
            None,                          # rename_target_path
        )
    else:
        # No selection
        return (
            gr.update(interactive=False),  # delete_btn
            gr.update(interactive=False),  # merge_btn
            gr.update(visible=False),      # rename_row
            gr.update(value=""),           # rename_input
            None,                          # rename_target_path
        )


def open_delete_confirm(selected: List[str] | None):
    """Open delete confirmation for a single selected dataset."""
    if not selected or len(selected) != 1:
        raise gr.Error("Select exactly one dataset to delete.")
    path = _choice_to_path(selected[0])
    return gr.update(visible=True), path


def cancel_delete():
    """Close delete confirmation dialog."""
    return gr.update(visible=False), None


def confirm_delete(path: str):
    """Delete dataset file and related metadata."""
    if not path or not os.path.exists(path):
        raise gr.Error("Dataset file not found.")

    try:
        os.remove(path)
        meta_path = get_meta_path(path)
        lock_path = path + ".lock"
        if os.path.exists(meta_path):
            os.remove(meta_path)
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError as e:
        raise gr.Error(f"Failed to delete dataset files: {e}")

    return gr.update(visible=False), None


def open_rename_dialog(selected: List[str] | None):
    """Open rename dialog for a single selected dataset."""
    if not selected or len(selected) != 1:
        raise gr.Error("Select exactly one dataset to rename.")
    path = _choice_to_path(selected[0])
    current_name = os.path.basename(path)
    return gr.update(visible=True), path, current_name


def cancel_rename():
    """Close rename dialog."""
    return gr.update(visible=False), None


def confirm_rename(old_path: str, new_name: str):
    """Rename dataset file and related metadata."""
    if not old_path or not os.path.exists(old_path):
        raise gr.Error("Dataset file not found.")
    if not new_name or not new_name.strip():
        raise gr.Error("Please enter a new name.")
    
    # Sanitize the new name to prevent path traversal
    new_name = os.path.basename(new_name.strip())
    if not new_name:
        raise gr.Error("Please enter a valid file name.")
    if not new_name.endswith(".jsonl"):
        new_name = f"{new_name}.jsonl"
    
    datasets_dir = get_datasets_dir()
    new_path = os.path.join(datasets_dir, new_name)
    
    if new_path == old_path:
        return gr.update(visible=False), None
    
    if os.path.exists(new_path):
        raise gr.Error("A dataset with this name already exists.")
    
    # Rename main file and metadata atomically with rollback
    try:
        os.rename(old_path, new_path)
        
        # Rename metadata if exists
        old_meta = get_meta_path(old_path)
        if os.path.exists(old_meta):
            new_meta = get_meta_path(new_path)
            try:
                os.rename(old_meta, new_meta)
            except Exception as e:
                # Roll back the primary file rename for consistency
                os.rename(new_path, old_path)
                raise gr.Error(f"Metadata rename failed, operation rolled back: {e}")
    except gr.Error:
        raise  # Re-raise Gradio errors as-is
    except Exception as e:
        raise gr.Error(f"Failed to rename dataset: {e}")
    
    gr.Info(f"Renamed to {new_name}")
    return gr.update(visible=False), None


def inline_rename(old_path: str, new_name: str):
    """Inline rename - simpler version without dialog state management."""
    if not old_path or not os.path.exists(old_path):
        raise gr.Error("Dataset file not found.")
    if not new_name or not new_name.strip():
        raise gr.Error("Please enter a new name.")
    
    # Sanitize the new name to prevent path traversal
    new_name = os.path.basename(new_name.strip())
    if not new_name:
        raise gr.Error("Please enter a valid file name.")
    if not new_name.endswith(".jsonl"):
        new_name = f"{new_name}.jsonl"
    
    datasets_dir = get_datasets_dir()
    new_path = os.path.join(datasets_dir, new_name)
    
    # No change needed
    if new_path == old_path:
        return
    
    if os.path.exists(new_path):
        raise gr.Error("A dataset with this name already exists.")
    
    # Rename main file and metadata atomically with rollback
    try:
        os.rename(old_path, new_path)
        
        # Rename metadata if exists
        old_meta = get_meta_path(old_path)
        if os.path.exists(old_meta):
            new_meta = get_meta_path(new_path)
            try:
                os.rename(old_meta, new_meta)
            except Exception as e:
                # Roll back the primary file rename for consistency
                os.rename(new_path, old_path)
                raise gr.Error(f"Metadata rename failed, operation rolled back: {e}")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Failed to rename dataset: {e}")
    
    gr.Info(f"Renamed to {new_name}")


def merge_datasets(selected: List[str] | None, new_name: str):
    """Merge selected datasets into a new JSONL file with combined metadata."""
    if not selected or len(selected) < 2:
        raise gr.Error("Select at least two datasets to merge.")
    if not new_name or not new_name.strip():
        raise gr.Error("Provide a name for the merged dataset.")

    # Sanitize filename to prevent path traversal
    filename = os.path.basename(new_name.strip())
    if not filename:
        raise gr.Error("Provide a valid name for the merged dataset.")
    if not filename.endswith(".jsonl"):
        filename = f"{filename}.jsonl"

    output_path = os.path.join(get_datasets_dir(), filename)
    if os.path.exists(output_path):
        raise gr.Error("A dataset with this name already exists.")

    # Aggregate metadata from source datasets
    total_samples = 0
    merged_tool_dist = Counter()
    merged_tools_used = set()
    source_files = []

    # First pass: collect metadata and validate files
    for choice in selected:
        path = _choice_to_path(choice)
        if not os.path.exists(path):
            raise gr.Error(f"Missing dataset: {os.path.basename(path)}")
        source_files.append(path)

        # Read source metadata if available
        meta_path = get_meta_path(path)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                # Aggregate samples
                total_samples += _parse_samples_from_meta(meta)
                
                # Aggregate tool distribution
                tool_dist = meta.get("tool_distribution", {})
                if tool_dist:
                    merged_tool_dist.update(tool_dist)
                
                # Aggregate tools used
                tools_used = meta.get("tools_used", [])
                merged_tools_used.update(tools_used)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                gr.Warning(f"Could not read metadata for {os.path.basename(path)}: {e}")

    # Second pass: merge JSONL content and count lines if no metadata
    lines_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for path in source_files:
            with open(path, "r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)
                    lines_written += 1

    # Use line count if metadata was incomplete
    if total_samples == 0:
        total_samples = lines_written

    # Create merged metadata file
    merged_meta = {
        "total_samples": total_samples,
        "category": "Uncategorized",  # Merged datasets default to Uncategorized
        "tool_distribution": dict(merged_tool_dist),
        "tools_used": sorted(merged_tools_used),
        "source_datasets": [os.path.basename(p) for p in source_files],
        "merged_at": datetime.now().isoformat(),
    }

    meta_output_path = get_meta_path(output_path)
    with open(meta_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)

    return gr.Info(f"Merged {len(selected)} datasets into {filename} ({total_samples} samples)")


def get_dataset_tool_distribution(dataset_path: str) -> dict:
    """
    Get tool distribution for a dataset from its metadata.
    
    Returns: dict mapping tool_name -> count
    """
    meta_path = get_meta_path(dataset_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("tool_distribution", {})
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _extract_tool_from_row(row_data: dict) -> str | None:
    """Extract the primary tool name from a dataset row."""
    tool_calls = None
    
    # Format 1: output.tool_calls (our storage format)
    output = row_data.get("output", {})
    if isinstance(output, dict):
        tool_calls = output.get("tool_calls", [])
    
    # Format 2: messages array with assistant tool_calls (OpenAI format)
    if not tool_calls:
        messages = row_data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    break
    
    # Format 3: direct tool_calls field
    if not tool_calls:
        tool_calls = row_data.get("tool_calls", [])
    
    # Extract tool name from first tool_call
    if tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        if isinstance(tc, dict):
            func = tc.get("function", {})
            if isinstance(func, dict):
                return func.get("name")
    
    return None


def split_dataset(
    dataset_path: str,
    tool_groups: dict[str, list[str]],
    base_name: str | None = None
) -> List[str]:
    """
    Split a dataset by tool groups into multiple datasets.
    
    Args:
        dataset_path: Path to source dataset
        tool_groups: Dict mapping group_name -> list of tool names
                    e.g. {"smart_home": ["light_control", "thermostat"], "other": ["calendar"]}
        base_name: Base name for output files (default: source name)
    
    Returns:
        List of created dataset filenames
    """
    if not os.path.exists(dataset_path):
        raise gr.Error(f"Dataset not found: {dataset_path}")
    
    if not tool_groups or len(tool_groups) < 2:
        raise gr.Error("Need at least 2 groups to split")
    
    # Validate no empty groups
    for group_name, tools in tool_groups.items():
        if not tools:
            raise gr.Error(f"Group '{group_name}' has no tools assigned")
    
    # Prepare output files
    source_name = os.path.basename(dataset_path).replace(".jsonl", "")
    # Sanitize base_name to prevent path traversal
    base = os.path.basename(base_name.strip()) if base_name else source_name
    
    datasets_dir = get_datasets_dir()
    output_files = {}
    output_handles = {}
    group_counts = {name: 0 for name in tool_groups}
    group_tool_dist = {name: Counter() for name in tool_groups}
    
    # Create tool -> group mapping
    tool_to_group = {}
    for group_name, tools in tool_groups.items():
        for tool in tools:
            tool_to_group[tool] = group_name
    
    try:
        # Open output files
        for group_name in tool_groups:
            filename = f"{base}_{group_name}.jsonl"
            filepath = os.path.join(datasets_dir, filename)
            
            # Auto-increment if file exists
            counter = 1
            while os.path.exists(filepath):
                filename = f"{base}_{group_name}_{counter}.jsonl"
                filepath = os.path.join(datasets_dir, filename)
                counter += 1
            
            output_files[group_name] = filepath
            output_handles[group_name] = open(filepath, "w", encoding="utf-8")
        
        # Read and split
        with open(dataset_path, "r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row = json.loads(line)
                    tool_name = _extract_tool_from_row(row)
                    
                    if tool_name and tool_name in tool_to_group:
                        group = tool_to_group[tool_name]
                        output_handles[group].write(line + "\n")
                        group_counts[group] += 1
                        group_tool_dist[group][tool_name] += 1
                    # Unmatched rows are skipped
                except json.JSONDecodeError:
                    continue
        
    finally:
        # Close all handles
        for handle in output_handles.values():
            handle.close()
    
    # Create metadata for each output file
    created_files = []
    source_meta_path = get_meta_path(dataset_path)
    source_category = DEFAULT_CATEGORY
    
    if os.path.exists(source_meta_path):
        try:
            with open(source_meta_path, "r", encoding="utf-8") as f:
                source_meta = json.load(f)
            source_category = source_meta.get("category", DEFAULT_CATEGORY)
        except (json.JSONDecodeError, IOError):
            pass
    
    for group_name, filepath in output_files.items():
        count = group_counts[group_name]
        
        # Remove empty files
        if count == 0:
            os.remove(filepath)
            continue
        
        # Create metadata
        meta = {
            "name": os.path.basename(filepath).replace(".jsonl", ""),
            "category": source_category,
            "created_at": datetime.now().isoformat(),
            "total_samples": count,
            "tools_used": list(tool_groups[group_name]),
            "tool_distribution": dict(group_tool_dist[group_name]),
            "split_from": os.path.basename(dataset_path),
        }
        
        meta_path = get_meta_path(filepath)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        created_files.append(os.path.basename(filepath))
    
    if not created_files:
        raise gr.Error("No data matched any tool group")
    
    return created_files


# Import for split_dataset
DEFAULT_CATEGORY = "Uncategorized"


def create_analyze_handler(analyze_fn):
    """
    Create the dataset analysis handler.
    
    Args:
        analyze_fn: Function to analyze the dataset
        
    Returns:
        Handler function for dataset analysis
    """
    def on_analyze(training_file):
        """Analyze dataset and show statistics + charts."""
        if not training_file:
            return (
                "Error: Please upload a training dataset first",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),  # train_btn
                gr.update(interactive=False),  # train_dev_btn
            )
        
        try:
            stats, tool_names, tool_counts = analyze_fn(training_file)
        except Exception as e:
            print(f"Analysis error: {e}")
            return (
                f"Error: Analysis failed - {str(e)}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        
        if not stats or 'error' in stats:
            return (
                "Error: Could not analyze dataset",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        
        # Format statistics markdown
        test_pct_of_total = stats['test_samples']/stats['total_samples']*100 if stats['total_samples'] > 0 else 0
        test_pct_of_tool = stats['test_samples']/stats['tool_call_samples']*100 if stats['tool_call_samples'] > 0 else 0
        
        total_samples = stats['total_samples'] if stats['total_samples'] > 0 else 1  # Prevent division by zero
        
        stats_text = f"""
### Dataset Statistics

**Total Samples:** {stats['total_samples']}

**Distribution:**
- With Tool Calls: {stats['tool_call_samples']} ({stats['tool_call_samples']/total_samples*100:.1f}%)
- Without Tool Calls: {stats['no_tool_call_samples']} ({stats['no_tool_call_samples']/total_samples*100:.1f}%)

**Data Split:**
- Training: {stats['train_samples']} samples ({stats['train_samples']/total_samples*100:.1f}% of total)
- Test: {stats['test_samples']} samples ({test_pct_of_tool:.1f}% of tool calls, {test_pct_of_total:.1f}% of total)

**Available Tools:** {len(stats['tool_distribution'])} different tools

**Sample Instructions:**
"""
        for i, instr in enumerate(stats.get('sample_instructions', []), 1):
            stats_text += f"\n{i}. {instr}"
        
        # Prepare chart data as markdown (BarPlot causes freeze issues)
        tool_md = "### Tool Distribution\n\n"
        if tool_names:
            for name, count in zip(tool_names, tool_counts):
                bar = "█" * min(count * 2, 40)  # Simple text bar
                tool_md += f"**{name}**: {bar} ({count})\n\n"
        else:
            tool_md += "_No tool data available_"
        
        split_md = "### Train/Test Split\n\n"
        split_md += f"**Training**: {'█' * min(stats['train_samples'] // 2, 40)} ({stats['train_samples']})\n\n"
        split_md += f"**Test**: {'█' * min(stats['test_samples'] // 2, 40)} ({stats['test_samples']})\n\n"
        
        return (
            stats_text,
            gr.update(value=tool_md, visible=True if tool_names else False),
            gr.update(value=split_md, visible=True),
            gr.update(interactive=True),  # Enable train button (normal)
            gr.update(interactive=True),  # Enable train button (dev)
        )
    
    return on_analyze


def on_train_complete(status, logs):
    """Enable evaluation button and trigger download preparation when training completes."""
    if not status:
        return (
            gr.update(), gr.update(),  # eval_btn, eval_status
            gr.update(), gr.update(),  # download_status, download_model_btn
            gr.update(), gr.update(),  # download_status_dev, download_model_dev_btn
        )
    
    if "Complete" in status:
        # Enable evaluation button
        eval_btn_update = gr.update(interactive=True)
        eval_status_update = gr.update(value="Status: Ready to evaluate")
        
        # Calculate size for label
        # Calculate size for label
        try:
            training_dir = get_training_dir()
            model_dir = os.path.join(training_dir, "final_model_stable")
            label = get_model_download_label(model_dir)
        except Exception:
            label = "Download Model"

        # Show download status messages
        download_status_update = gr.update(value="Training complete. Click to download.", visible=True)
        download_status_dev_update = gr.update(value="Training complete. Click to download.", visible=True)
        
        # Make buttons visible and interactive with size label
        download_btn_update = gr.update(visible=True, value=None, label=label, interactive=True)
        download_dev_btn_update = gr.update(visible=True, value=None, label=label, interactive=True)
        
        return (
            eval_btn_update, eval_status_update,
            download_status_update, download_btn_update,
            download_status_dev_update, download_dev_btn_update,
        )
    
    return (
        gr.update(), gr.update(),
        gr.update(), gr.update(),
        gr.update(), gr.update(),
    )

def set_download_loading():
    """Disable download button and show loading text."""
    return gr.update(interactive=False, label="Compressing... (Check Progress Bar)")


def generate_model_zip(progress=gr.Progress()):
    """Compress and return model zip path for automatic download with standard progress bar."""
    # Get model directory
    training_dir = get_training_dir()
    model_dir = os.path.join(training_dir, "final_model_stable")
    
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        gr.Warning("Model directory not found!")
        return gr.update(interactive=True, label="Download Model (Not Found)")
    
    try:
        # 1. Calculate total size, list files, and find latest mtime
        files_to_zip = []
        last_model_mtime = 0
        
        for dirpath, _, filenames in os.walk(model_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                files_to_zip.append(filepath)
                try:
                    f_mtime = os.path.getmtime(filepath)
                    if f_mtime > last_model_mtime:
                        last_model_mtime = f_mtime
                except OSError:
                    # File might have been removed during the walk, ignore it.
                    pass
        
        if not files_to_zip:
            gr.Warning("Model directory is empty!")
            return gr.update(interactive=True, label="Download Model (Empty)")
            
        # 2. Check if valid zip already exists
        zip_path = os.path.join(tempfile.gettempdir(), "trained_model.zip")
            
        # If zip exists and is newer than or equal to model, skip compression
        if os.path.exists(zip_path):
            zip_mtime = os.path.getmtime(zip_path)
            if zip_mtime >= last_model_mtime:
                final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                gr.Info(f"Using cached zip. Downloading... ({final_size_mb:.1f} MB)")
                return gr.update(
                    value=zip_path,
                    interactive=True,
                    label=f"Download Model ({final_size_mb:.1f} MB)"
                )

        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        # Sort files by size (smallest first) to show early progress
        files_to_zip.sort(key=os.path.getsize)
            
        gr.Info("Starting compression...")
        progress(0, desc="Starting compression...")
        
        # 3. Compress with progress
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, filepath in enumerate(files_to_zip):
                # Archive name relative to training_dir so it unzips as "final_model_stable/..."
                arcname = os.path.relpath(filepath, training_dir)
                zf.write(filepath, arcname)
                
                # Update progress based on file count
                pct = (i + 1) / len(files_to_zip)
                progress(pct, desc=f"Compressing file {i+1}/{len(files_to_zip)}")
        
        # Get final size
        final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        
        gr.Info(f"Compression complete. Downloading... ({final_size_mb:.1f} MB)")
        return gr.update(
            value=zip_path,
            interactive=True,
            label=f"Download Model ({final_size_mb:.1f} MB)"
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to create model zip: {e}")
        gr.Error(f"Failed to create zip: {e}")
        return gr.update(interactive=True, label="Download Model (Error)")

