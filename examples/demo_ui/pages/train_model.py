"""
Train Model page - 3-step wizard for dataset selection, training, and evaluation.
"""
import os
import json
from typing import List, Any, Tuple

import gradio as gr

from core.config import get_datasets_dir, MAX_CATEGORIES, MAX_TOOL_SLIDERS
from .handlers.training import (
    on_train_complete, generate_model_zip, on_dataset_select, set_download_loading,
    get_datasets_by_category, get_dataset_categories, update_dataset_category,
    inline_rename, merge_datasets, confirm_delete, split_dataset,
    get_meta_path, extract_filename_from_label, get_dataset_path,
    get_dataset_tool_distribution,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CATEGORY = "Uncategorized"
NO_SELECTION_TEXT = "*No datasets selected*"
MAX_SPLIT_TOOLS = 15  # Maximum number of tools to show in split UI


# =============================================================================
# Helper Functions
# =============================================================================

def collect_selected_items(*checkbox_values) -> List[str]:
    """Collect all selected items from multiple checkbox groups."""
    selected = []
    for value in checkbox_values:
        selected.extend(value or [])
    return selected


def create_category_toggle_handler(cat_index: int):
    """
    Create a handler for category checkbox toggle.
    
    When checked: selects all datasets in category.
    When unchecked: clears selection only if all were selected.
    """
    def toggle(checked: bool, current_selection: List[str], all_cat_datasets: List[List[str]]):
        category_datasets = all_cat_datasets[cat_index] if cat_index < len(all_cat_datasets) else []
        current_selection = current_selection or []
        
        if checked:
            return category_datasets
        else:
            # Only clear if all were selected (explicit deselect)
            if len(current_selection) == len(category_datasets) and len(category_datasets) > 0:
                return []
            return current_selection
    return toggle


def create_category_sync_handler(cat_index: int):
    """
    Create a handler that syncs category checkbox with item selection.
    
    Returns True if all items in category are selected.
    """
    def sync(selected: List[str], all_cat_datasets: List[List[str]]) -> bool:
        cat_datasets = all_cat_datasets[cat_index] if cat_index < len(all_cat_datasets) else []
        return len(selected) == len(cat_datasets) and len(cat_datasets) > 0
    return sync


# =============================================================================
# Page Builder
# =============================================================================

def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn, chat_fn=None):
    """Create the Train Model wizard page with 4 steps."""
    
    # Pre-load categories at startup
    initial_grouped = get_datasets_by_category()
    cat_names = list(initial_grouped.keys())
    
    def get_cat_data(idx: int) -> Tuple[str, List[str]]:
        """Get category name and datasets for a given slot index."""
        if idx < len(cat_names):
            name = cat_names[idx]
            return name, initial_grouped[name]
        return None, []
    
    # Pre-compute category data for all slots
    cat_data = [get_cat_data(i) for i in range(MAX_CATEGORIES)]
    
    with gr.Blocks() as page:
        gr.Markdown("# Model Training Wizard")
        
        step_state = gr.State(0)
        selected_dataset_path = gr.State(None)
        tool_dist_state = gr.State({})
        filter_config_state = gr.State({})
        tool_names_state = gr.State([])  # List of tool names in order
        total_samples_state = gr.State(0)  # Total samples from metadata

        # STEP 1
        with gr.Group(visible=True) as step_1_group:
            gr.Markdown("## Step 1/3: Select Datasets")
            
            with gr.Row():
                with gr.Column(scale=1, elem_id="dataset-library-col"):
                    with gr.Row():
                        gr.Markdown("### Dataset Library")
                        refresh_datasets_btn = gr.Button("↻", size="sm", scale=0, min_width=40)
                    
                    # Create category slots dynamically
                    all_cat_selects = []
                    all_cbs = []
                    initial_cat_datasets = [cat_data[i][1] for i in range(MAX_CATEGORIES)]
                    
                    # State to track current datasets per category (updated on refresh)
                    cat_datasets_state = gr.State(initial_cat_datasets)
                    
                    with gr.Column(elem_id="dataset-list-scroll"):
                        for i in range(MAX_CATEGORIES):
                            cat_name, cat_ds = cat_data[i]
                            
                            if cat_name:
                                cat_select = gr.Checkbox(label=f"{cat_name} ({len(cat_ds)})", value=False)
                                cb = gr.CheckboxGroup(choices=cat_ds, value=[], label=None, show_label=False)
                            else:
                                cat_select = gr.Checkbox(visible=False, value=False)
                                cb = gr.CheckboxGroup(choices=[], value=[], visible=False)
                            
                            all_cat_selects.append(cat_select)
                            all_cbs.append(cb)
                    
                    gr.Markdown("---")
                    selection_summary = gr.Markdown(NO_SELECTION_TEXT)
                    
                    # Actions for selected datasets
                    with gr.Row(visible=False) as actions_row:
                        edit_btn = gr.Button("Edit", size="sm", variant="secondary")
                        split_btn = gr.Button("Split", size="sm", variant="secondary")
                        merge_btn = gr.Button("Merge", size="sm", variant="secondary")
                        delete_btn = gr.Button("Delete", size="sm", variant="stop")
            
                with gr.Column(scale=2):
                    # Empty state
                    with gr.Group(visible=True) as empty_panel:
                        gr.HTML('<div style="text-align:center;padding:60px;color:#94a3b8;"><div style="font-size:16px;">Select datasets to see training overview</div></div>')
                    
                    # Overview panel
                    with gr.Group(visible=False) as overview_panel:
                        gr.Markdown("### Training Overview")
                        dataset_overview = gr.HTML()
                        gr.Markdown("#### Filter by Tool (samples to include)")
                        samples_summary = gr.Markdown("**0 / 0 samples**")
                        
                        # Static slider slots with containers for proper hiding
                        tool_sliders = []
                        tool_containers = []
                        for i in range(MAX_TOOL_SLIDERS):
                            with gr.Group(visible=False) as container:
                                s = gr.Slider(
                                    minimum=0, maximum=1, value=0, step=1,
                                    label=f"Tool {i}",
                                    elem_id=f"tool-slider-{i}"
                                )
                            tool_sliders.append(s)
                            tool_containers.append(container)
                    
                    # Edit panel (rename + category)
                    with gr.Group(visible=False) as edit_panel:
                        gr.Markdown("### Edit Dataset")
                        edit_name_input = gr.Textbox(label="Dataset Name", placeholder="dataset_name.jsonl")
                        edit_category_dropdown = gr.Dropdown(
                            label="Category",
                            choices=cat_names + (["Uncategorized"] if "Uncategorized" not in cat_names else []),
                            value="Uncategorized",
                            allow_custom_value=True,
                            info="Select existing or type new category"
                        )
                        edit_target = gr.State(None)
                        with gr.Row():
                            edit_save_btn = gr.Button("Save", variant="primary")
                            edit_cancel_btn = gr.Button("Cancel", variant="secondary")
                    
                    # Merge panel
                    with gr.Group(visible=False) as merge_panel:
                        gr.Markdown("### Merge Datasets")
                        merge_info = gr.Markdown("")
                        merge_name_input = gr.Textbox(label="Merged Dataset Name", placeholder="merged_dataset.jsonl")
                        with gr.Row():
                            merge_save_btn = gr.Button("Merge", variant="primary")
                            merge_cancel_btn = gr.Button("Cancel", variant="secondary")
                    
                    # Split panel
                    with gr.Group(visible=False) as split_panel:
                        gr.Markdown("### Split Dataset by Tools")
                        split_info = gr.Markdown("")
                        split_target = gr.State(None)  # Source dataset path
                        split_tools_state = gr.State([])  # List of tool names
                        
                        with gr.Row():
                            split_num_groups = gr.Slider(
                                minimum=2, maximum=MAX_SPLIT_TOOLS, value=2, step=1,
                                label="Number of Groups"
                            )
                        
                        gr.Markdown("**Assign tools to groups:**")
                        
                        # Tool assignment dropdowns (dynamically shown)
                        split_tool_rows = []
                        split_tool_dropdowns = []
                        for i in range(15):  # Max 15 tools
                            with gr.Row(visible=False, equal_height=True) as tool_row:
                                tool_dropdown = gr.Dropdown(
                                    choices=["Group 1", "Group 2"],
                                    value="Group 1",
                                    label=f"Tool {i+1}",
                                    scale=1
                                )
                            split_tool_rows.append(tool_row)
                            split_tool_dropdowns.append(tool_dropdown)
                        
                        split_base_name = gr.Textbox(
                            label="Base Name for Split Files",
                            placeholder="dataset_name (will create dataset_name_group1.jsonl, etc.)"
                        )
                        
                        with gr.Row():
                            split_save_btn = gr.Button("Split", variant="primary")
                            split_cancel_btn = gr.Button("Cancel", variant="secondary")
                    
                    # Delete confirmation
                    with gr.Group(visible=False) as delete_panel:
                        gr.Markdown("### Confirm Delete")
                        delete_info = gr.Markdown("")
                        delete_target = gr.State(None)
                        with gr.Row():
                            delete_confirm_btn = gr.Button("Delete", variant="stop")
                            delete_cancel_btn = gr.Button("Cancel", variant="secondary")
            
            with gr.Row():
                gr.Column(scale=2)
                with gr.Column(scale=1):
                    next_btn_1 = gr.Button("Next: Configure Training", variant="primary", interactive=False, size="lg")

        # STEP 2
        with gr.Group(visible=False) as step_2_group:
            gr.Markdown("## Step 2/3: Train Model")
            with gr.Tabs():
                with gr.Tab("Normal Mode"):
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")
                    status_output = gr.Markdown("Status: Ready")
                    logs_output = gr.Code(language="shell", lines=15)
                    download_status = gr.Markdown("", visible=False)
                    download_model_btn = gr.DownloadButton(label="Download Model", visible=False)
                with gr.Tab("Developer Mode"):
                    with gr.Row():
                        num_epochs = gr.Slider(label="Epochs", minimum=1, maximum=10, value=3)
                        batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, value=1)
                    train_dev_btn = gr.Button("Start Training (Dev)", variant="primary", size="lg")
                    dev_status_output = gr.Markdown("Status: Ready")
                    dev_logs_output = gr.Code(language="shell", lines=15)
                    download_status_dev = gr.Markdown("", visible=False)
                    download_model_dev_btn = gr.DownloadButton(label="Download Model", visible=False)
            with gr.Row():
                back_btn_2 = gr.Button("Back", variant="secondary")
                next_btn_2 = gr.Button("Next: Evaluate", variant="primary", interactive=False)

        # STEP 3
        with gr.Group(visible=False) as step_3_group:
            gr.Markdown("## Step 3/3: Evaluate Model")
            eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")
            eval_status = gr.Markdown("Status: Ready")
            with gr.Column(elem_id="eval-output-scroll"):
                eval_output = gr.Code(language="shell", lines=25)
            with gr.Row():
                back_btn_3 = gr.Button("Back", variant="secondary")
                next_btn_3 = gr.Button("Next: Chat", variant="primary", interactive=False)

        # STEP 4
        with gr.Group(visible=False) as step_4_group:
            gr.Markdown("## Chat with Model")
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Message...", scale=4)
                send_btn = gr.Button("Send", variant="primary")
            with gr.Row():
                back_btn_4 = gr.Button("Back", variant="secondary")
                restart_btn = gr.Button("Restart", variant="secondary")

        # Navigation between wizard steps
        def navigate(current_step: int, direction: int) -> Tuple[int, Any, Any, Any, Any]:
            """Navigate between wizard steps. Direction: +1 forward, -1 backward."""
            new_step = max(0, min(3, current_step + direction))
            return (
                new_step,
                gr.update(visible=new_step == 0),
                gr.update(visible=new_step == 1),
                gr.update(visible=new_step == 2),
                gr.update(visible=new_step == 3),
            )
        
        nav_outputs = [step_state, step_1_group, step_2_group, step_3_group, step_4_group]
        next_btn_1.click(lambda step: navigate(step, 1), inputs=[step_state], outputs=nav_outputs)
        back_btn_2.click(lambda step: navigate(step, -1), inputs=[step_state], outputs=nav_outputs)
        next_btn_2.click(lambda step: navigate(step, 1), inputs=[step_state], outputs=nav_outputs)
        back_btn_3.click(lambda step: navigate(step, -1), inputs=[step_state], outputs=nav_outputs)
        next_btn_3.click(lambda step: navigate(step, 1), inputs=[step_state], outputs=nav_outputs)
        back_btn_4.click(lambda step: navigate(step, -1), inputs=[step_state], outputs=nav_outputs)
        restart_btn.click(lambda step: navigate(step, -step), inputs=[step_state], outputs=nav_outputs)

        # ============================================================
        # Selection Handlers
        # ============================================================
        
        def on_selection_change(*checkbox_values):
            """
            Handle dataset selection changes.
            
            Updates: summary text, overview panel, tool sliders, and action buttons.
            """
            all_selected = collect_selected_items(*checkbox_values)
            
            # Base outputs for no selection
            if not all_selected:
                slider_updates = [gr.update(minimum=0, maximum=1, value=0) for _ in range(MAX_TOOL_SLIDERS)]
                container_updates = [gr.update(visible=False) for _ in range(MAX_TOOL_SLIDERS)]
                return [
                    NO_SELECTION_TEXT,
                    None,
                    gr.update(visible=True),   # empty_panel
                    gr.update(visible=False),  # overview_panel
                    "",
                    {},  # tool_dist_state
                    {},  # filter_config_state
                    [],  # tool_names_state
                    0,   # total_samples_state
                    gr.update(interactive=False),
                    gr.update(visible=False),  # actions_row
                    "**0 / 0 samples**",  # samples_summary
                ] + slider_updates + container_updates
            
            paths, overview, dist, config, _, _, total_samples = on_dataset_select(all_selected)
            can_next = paths is not None and len(paths) > 0
            summary = f"**{len(all_selected)} dataset(s) selected**"
            
            # Build slider updates
            sorted_tools = sorted(dist.items(), key=lambda x: x[1], reverse=True) if dist else []
            tool_names = [t[0] for t in sorted_tools]
            
            slider_updates = []
            container_updates = []
            slider_sum = 0
            for i in range(MAX_TOOL_SLIDERS):
                if i < len(sorted_tools):
                    tool_name, tool_count = sorted_tools[i]
                    slider_sum += tool_count
                    slider_updates.append(gr.update(
                        label=f"{tool_name} ({tool_count} samples)",
                        minimum=0,
                        maximum=max(1, tool_count),
                        value=tool_count
                    ))
                    container_updates.append(gr.update(visible=True))
                else:
                    slider_updates.append(gr.update(minimum=0, maximum=1, value=0))
                    container_updates.append(gr.update(visible=False))
            
            # Seçilen toplam sample'dan fazla olamaz
            selected = min(slider_sum, total_samples)
            samples_text = f"**{selected} / {total_samples} samples**"
            
            return [
                summary,
                paths,
                gr.update(visible=False),  # empty_panel
                gr.update(visible=True),   # overview_panel
                overview,
                dist,
                config,
                tool_names,
                total_samples,  # total_samples_state
                gr.update(interactive=can_next),
                gr.update(visible=True),   # actions_row
                samples_text,  # samples_summary
            ] + slider_updates + container_updates
        
        change_outputs = [
            selection_summary, selected_dataset_path, empty_panel, overview_panel,
            dataset_overview, tool_dist_state, filter_config_state, tool_names_state,
            total_samples_state, next_btn_1, actions_row, samples_summary
        ] + tool_sliders + tool_containers
        
        # Wire all checkbox changes
        for cb in all_cbs:
            cb.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        
        # Category checkbox -> select/deselect all in that category
        for i, (cat_select, cb) in enumerate(zip(all_cat_selects, all_cbs)):
            cat_select.change(
                create_category_toggle_handler(i), 
                inputs=[cat_select, cb, cat_datasets_state], 
                outputs=[cb]
            )
        
        # Sync category checkbox when individual items change
        for i, (cb, cat_select) in enumerate(zip(all_cbs, all_cat_selects)):
            cb.change(
                create_category_sync_handler(i), 
                inputs=[cb, cat_datasets_state], 
                outputs=[cat_select]
            )
        
        # ============================================================
        # Slider change handlers - update filter_config_state and samples_summary
        # ============================================================
        
        def update_config_and_summary(*args) -> Tuple[dict, str]:
            """
            Update filter config when sliders change.
            
            Args format: [slider0..sliderN, config, names, total_samples]
            Returns: (updated_config, summary_text)
            """
            slider_values = args[:MAX_TOOL_SLIDERS]
            config = args[MAX_TOOL_SLIDERS]
            names = args[MAX_TOOL_SLIDERS + 1]
            total_samples = args[MAX_TOOL_SLIDERS + 2]
            
            new_config = dict(config) if config else {}
            
            # Update config with current slider values
            selected_sum = 0
            for i, val in enumerate(slider_values):
                if names and i < len(names) and val is not None:
                    new_config[names[i]] = int(val)
                    selected_sum += int(val)
            
            # Selected cannot exceed total samples
            selected = min(selected_sum, total_samples) if total_samples else selected_sum
            summary_text = f"**{selected} / {total_samples} samples**"
            
            return new_config, summary_text
        
        slider_inputs = tool_sliders + [filter_config_state, tool_names_state, total_samples_state]
        
        for slider in tool_sliders:
            slider.change(
                update_config_and_summary,
                inputs=slider_inputs,
                outputs=[filter_config_state, samples_summary]
            )
        
        # ============================================================
        # Edit Handlers (rename + category)
        # ============================================================
        
        def show_edit(*checkbox_values):
            """Show edit panel for a single selected dataset."""
            all_selected = collect_selected_items(*checkbox_values)
            if len(all_selected) != 1:
                gr.Warning("Select exactly one dataset to edit")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None
            
            label = all_selected[0]
            filename = extract_filename_from_label(label)
            path = get_dataset_path(filename)
            
            # Get current category from metadata
            current_category = DEFAULT_CATEGORY
            meta_path = get_meta_path(path)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    current_category = meta.get("category", DEFAULT_CATEGORY)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            
            # Get all available categories for dropdown
            all_categories = get_dataset_categories()
            if current_category not in all_categories:
                all_categories.append(current_category)
            if DEFAULT_CATEGORY not in all_categories:
                all_categories.append(DEFAULT_CATEGORY)
            
            return (
                gr.update(visible=False),  # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=True),   # edit_panel
                gr.update(visible=False),  # merge_panel
                gr.update(visible=False),  # split_panel
                gr.update(value=filename), # edit_name_input
                gr.update(choices=all_categories, value=current_category),  # edit_category_dropdown
                path  # edit_target
            )
        
        def do_edit(old_path: str, new_name: str, new_category: str):
            """Save dataset edits (rename and/or category change)."""
            if not old_path or not os.path.exists(old_path):
                raise gr.Error("Dataset not found")
            
            old_name = os.path.basename(old_path)
            new_name = new_name.strip() if new_name else old_name
            if not new_name.endswith(".jsonl"):
                new_name += ".jsonl"
            
            # Rename if name changed
            if new_name != old_name:
                inline_rename(old_path, new_name)
                old_path = get_dataset_path(new_name)
            
            # Update category
            new_category = new_category.strip() if new_category else DEFAULT_CATEGORY
            update_dataset_category(old_path, new_category)
            
            gr.Info(f"Saved: {new_name} [{new_category}]")
        
        def cancel_edit():
            """Cancel edit and return to overview."""
            return gr.update(visible=False), gr.update(visible=True)
        
        def refresh_after_action():
            """Refresh dataset list after edit/merge/delete operations."""
            new_grouped = get_datasets_by_category()
            new_cats = list(new_grouped.keys())
            
            # Build new cat_datasets list for state
            new_cat_datasets = []
            
            results = []
            for i in range(MAX_CATEGORIES):
                if i < len(new_cats):
                    cat = new_cats[i]
                    datasets = new_grouped[cat]
                    new_cat_datasets.append(datasets)
                    results.append(gr.update(label=f"{cat} ({len(datasets)})", visible=True, value=False))
                    results.append(gr.update(choices=datasets, value=[]))
                else:
                    new_cat_datasets.append([])
                    results.append(gr.update(visible=False, value=False))
                    results.append(gr.update(choices=[], value=[]))
            
            results.extend([
                gr.update(visible=True),   # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=False),  # edit_panel
                gr.update(visible=False),  # merge_panel
                gr.update(visible=False),  # split_panel
                gr.update(visible=False),  # delete_panel
                gr.update(visible=False),  # actions_row
                NO_SELECTION_TEXT,         # selection_summary
                new_cat_datasets,          # cat_datasets_state
            ])
            
            return results
        
        # Build refresh outputs list dynamically
        refresh_outputs = []
        for cat_select, cb in zip(all_cat_selects, all_cbs):
            refresh_outputs.append(cat_select)
            refresh_outputs.append(cb)
        refresh_outputs.extend([empty_panel, overview_panel, edit_panel, merge_panel, split_panel, delete_panel, actions_row, selection_summary, cat_datasets_state])
        
        edit_btn.click(
            show_edit, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, edit_panel, merge_panel, split_panel, edit_name_input, edit_category_dropdown, edit_target]
        )
        
        edit_save_btn.click(do_edit, inputs=[edit_target, edit_name_input, edit_category_dropdown]).then(
            refresh_after_action,
            outputs=refresh_outputs
        )
        
        edit_cancel_btn.click(cancel_edit, outputs=[edit_panel, overview_panel])
        
        # ============================================================
        # Merge Handlers
        # ============================================================
        
        def show_merge(*checkbox_values):
            """Show merge panel for 2+ selected datasets."""
            all_selected = collect_selected_items(*checkbox_values)
            if len(all_selected) < 2:
                gr.Warning("Select at least 2 datasets to merge")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            info = f"Merging **{len(all_selected)}** datasets:\n" + "\n".join(f"- {s}" for s in all_selected)
            
            return (
                gr.update(visible=False),  # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=False),  # edit_panel
                gr.update(visible=True),   # merge_panel
                gr.update(visible=False),  # split_panel
                info
            )
        
        def do_merge(*args):
            """Execute dataset merge. Last arg is new_name, rest are checkbox values."""
            new_name = args[-1]
            all_selected = collect_selected_items(*args[:-1])
            merge_datasets(all_selected, new_name)
            gr.Info(f"Created merged dataset: {new_name}")
        
        def cancel_merge():
            """Cancel merge and return to overview."""
            return gr.update(visible=False), gr.update(visible=True)
        
        merge_btn.click(
            show_merge, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, edit_panel, merge_panel, split_panel, merge_info]
        )
        
        merge_save_btn.click(do_merge, inputs=[*all_cbs, merge_name_input]).then(
            refresh_after_action,
            outputs=refresh_outputs
        )
        
        merge_cancel_btn.click(cancel_merge, outputs=[merge_panel, overview_panel])
        
        # ============================================================
        # Split Handlers
        # ============================================================
        
        def show_split(*checkbox_values):
            """Show split panel for a single dataset with multiple tools."""
            all_selected = collect_selected_items(*checkbox_values)
            if len(all_selected) != 1:
                gr.Warning("Select exactly one dataset to split")
                # Return updates for all outputs (5 panels + info + target + tools + base_name + num_groups)
                base_outputs = [gr.update()] * 5 + [gr.update(), None, [], gr.update(), gr.update()]
                row_outputs = [gr.update(visible=False)] * len(split_tool_rows)
                dropdown_outputs = [gr.update()] * len(split_tool_rows)
                return base_outputs + row_outputs + dropdown_outputs
            
            label = all_selected[0]
            filename = extract_filename_from_label(label)
            path = get_dataset_path(filename)
            
            # Get tool distribution
            tool_dist = get_dataset_tool_distribution(path)
            if len(tool_dist) < 2:
                gr.Warning("Dataset must have at least 2 different tools to split")
                base_outputs = [gr.update()] * 5 + [gr.update(), None, [], gr.update(), gr.update()]
                row_outputs = [gr.update(visible=False)] * len(split_tool_rows)
                dropdown_outputs = [gr.update()] * len(split_tool_rows)
                return base_outputs + row_outputs + dropdown_outputs
            
            # Sort tools by count
            sorted_tools = sorted(tool_dist.items(), key=lambda x: x[1], reverse=True)
            tool_names = [t[0] for t in sorted_tools]
            
            info = f"**Splitting:** {filename}\n\n**Tools ({len(tool_names)}):**\n"
            for tool, count in sorted_tools:
                info += f"- {tool}: {count} samples\n"
            
            base_name = filename.replace(".jsonl", "")
            num_tools = len(tool_names)
            
            # Build group choices based on number of tools
            group_choices = [f"Group {i+1}" for i in range(num_tools)]
            
            # Build outputs
            base_outputs = [
                gr.update(visible=False),  # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=False),  # edit_panel
                gr.update(visible=False),  # merge_panel
                gr.update(visible=True),   # split_panel
                info,                      # split_info
                path,                      # split_target
                tool_names,                # split_tools_state
                gr.update(value=base_name), # split_base_name
                gr.update(minimum=2, maximum=num_tools, value=2),  # split_num_groups
            ]
            
            # Tool row visibility and dropdown setup
            row_outputs = []
            dropdown_outputs = []
            for i in range(len(split_tool_rows)):
                if i < len(tool_names):
                    tool_name = tool_names[i]
                    count = tool_dist.get(tool_name, 0)
                    row_outputs.append(gr.update(visible=True))
                    dropdown_outputs.append(gr.update(
                        label=f"{tool_name} ({count} samples)",
                        choices=["Group 1", "Group 2"],  # Initial 2 groups
                        value="Group 1"
                    ))
                else:
                    row_outputs.append(gr.update(visible=False))
                    dropdown_outputs.append(gr.update())
            
            return base_outputs + row_outputs + dropdown_outputs
        
        def update_group_choices(num_groups: int, tools_list: list):
            """Update dropdown choices when number of groups changes.
            
            If num_groups equals tool count, auto-assign each tool to its own group.
            """
            num_groups = int(num_groups)
            choices = [f"Group {i+1}" for i in range(num_groups)]
            
            # Auto-assign if group count matches tool count
            if tools_list and num_groups == len(tools_list):
                return [
                    gr.update(choices=choices, value=f"Group {i+1}") 
                    for i in range(len(split_tool_dropdowns))
                ]
            
            # Otherwise default all to Group 1
            return [gr.update(choices=choices, value=choices[0]) for _ in split_tool_dropdowns]
        
        def do_split(target_path, tools_list, base_name, num_groups, *dropdown_values):
            """Execute the split operation."""
            if not target_path or not tools_list:
                raise gr.Error("No dataset selected for split")
            
            # Build tool groups from dropdown selections
            num_groups = int(num_groups)
            tool_groups = {f"group{i+1}": [] for i in range(num_groups)}
            
            for i, tool_name in enumerate(tools_list):
                if i < len(dropdown_values):
                    group_selection = dropdown_values[i]
                    # Extract group number from "Group X"
                    group_num = int(group_selection.split()[-1])
                    group_key = f"group{group_num}"
                    tool_groups[group_key].append(tool_name)
            
            # Remove empty groups
            tool_groups = {k: v for k, v in tool_groups.items() if v}
            
            if len(tool_groups) < 2:
                raise gr.Error("Assign tools to at least 2 different groups")
            
            created = split_dataset(target_path, tool_groups, base_name)
            gr.Info(f"Created {len(created)} datasets: {', '.join(created)}")
        
        def cancel_split():
            """Cancel split and return to overview."""
            return gr.update(visible=False), gr.update(visible=True)
        
        # Split button outputs
        split_show_outputs = [
            empty_panel, overview_panel, edit_panel, merge_panel, split_panel,
            split_info, split_target, split_tools_state, split_base_name, split_num_groups
        ] + split_tool_rows + split_tool_dropdowns
        
        split_btn.click(show_split, inputs=all_cbs, outputs=split_show_outputs)
        
        # Update dropdowns when group count changes
        split_num_groups.change(
            update_group_choices,
            inputs=[split_num_groups, split_tools_state],
            outputs=split_tool_dropdowns
        )
        
        # Execute split
        split_save_btn.click(
            do_split,
            inputs=[split_target, split_tools_state, split_base_name, split_num_groups] + split_tool_dropdowns
        ).then(refresh_after_action, outputs=refresh_outputs)
        
        split_cancel_btn.click(cancel_split, outputs=[split_panel, overview_panel])
        
        # ============================================================
        # Delete Handlers
        # ============================================================
        
        def show_delete(*checkbox_values):
            """Show delete confirmation panel."""
            all_selected = collect_selected_items(*checkbox_values)
            if not all_selected:
                gr.Warning("Select at least one dataset to delete")
                return gr.update(), gr.update(), gr.update(), None
            
            info = f"**Delete {len(all_selected)} dataset(s)?**\n\nThis cannot be undone.\n\n" + "\n".join(f"- {s}" for s in all_selected)
            
            paths = [get_dataset_path(extract_filename_from_label(label)) for label in all_selected]
            
            return gr.update(visible=False), gr.update(visible=True), info, paths
        
        def do_delete(paths: List[str]):
            """Delete selected datasets."""
            if not paths:
                return
            for path in paths:
                if os.path.exists(path):
                    confirm_delete(path)
            gr.Info(f"Deleted {len(paths)} dataset(s)")
        
        def cancel_delete_fn():
            """Cancel delete and return to overview."""
            return gr.update(visible=False), gr.update(visible=True)
        
        delete_btn.click(
            show_delete, inputs=all_cbs,
            outputs=[overview_panel, delete_panel, delete_info, delete_target]
        )
        
        delete_confirm_btn.click(do_delete, inputs=[delete_target]).then(
            refresh_after_action,
            outputs=refresh_outputs
        )
        
        delete_cancel_btn.click(cancel_delete_fn, outputs=[delete_panel, overview_panel])
        
        # Refresh datasets button
        refresh_datasets_btn.click(
            refresh_after_action,
            outputs=refresh_outputs
        )

        # ============================================================
        # Training
        # ============================================================
        train_btn.click(
            train_fn, inputs=[selected_dataset_path, filter_config_state], outputs=[status_output, logs_output]
        ).then(
            on_train_complete, inputs=[status_output, logs_output],
            outputs=[eval_btn, eval_status, download_status, download_model_btn, download_status_dev, download_model_dev_btn]
        ).then(lambda: gr.update(interactive=True), outputs=[next_btn_2])
        
        download_model_btn.click(set_download_loading, outputs=[download_model_btn]).then(generate_model_zip, outputs=[download_model_btn])
        download_model_dev_btn.click(set_download_loading, outputs=[download_model_dev_btn]).then(generate_model_zip, outputs=[download_model_dev_btn])
        eval_btn.click(eval_fn, outputs=[eval_status, eval_output]).then(lambda: gr.update(interactive=True), outputs=[next_btn_3])
        
        if chat_fn:
            send_btn.click(chat_fn, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(chat_fn, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

    return page
