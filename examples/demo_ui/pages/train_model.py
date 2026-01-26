import os
import json
import gradio as gr

from .handlers.training import (
    on_train_complete, generate_model_zip, on_dataset_select, set_download_loading,
    get_datasets_by_category, get_dataset_categories, update_dataset_category,
    inline_rename, merge_datasets, confirm_delete,
)

MAX_TOOL_SLIDERS = 15
MAX_CATEGORIES = 10


def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn, chat_fn=None):
    
    # Pre-load categories at startup
    initial_grouped = get_datasets_by_category()
    cat_names = list(initial_grouped.keys())
    
    def get_cat_data(idx):
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
                    all_cat_datasets = []
                    
                    with gr.Column(elem_id="dataset-list-scroll"):
                        for i in range(MAX_CATEGORIES):
                            cat_name, cat_ds = cat_data[i]
                            all_cat_datasets.append(cat_ds)
                            
                            if cat_name:
                                cat_select = gr.Checkbox(label=f"{cat_name} ({len(cat_ds)})", value=False)
                                cb = gr.CheckboxGroup(choices=cat_ds, value=[], label=None, show_label=False)
                            else:
                                cat_select = gr.Checkbox(visible=False, value=False)
                                cb = gr.CheckboxGroup(choices=[], value=[], visible=False)
                            
                            all_cat_selects.append(cat_select)
                            all_cbs.append(cb)
                    
                    gr.Markdown("---")
                    selection_summary = gr.Markdown("*No datasets selected*")
                    
                    # Actions for selected datasets
                    with gr.Row(visible=False) as actions_row:
                        edit_btn = gr.Button("Edit", size="sm", variant="secondary")
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

        # Navigation
        def nav(s, d): 
            n = max(0, min(3, s+d))
            return n, gr.update(visible=n==0), gr.update(visible=n==1), gr.update(visible=n==2), gr.update(visible=n==3)
        
        nav_outs = [step_state, step_1_group, step_2_group, step_3_group, step_4_group]
        next_btn_1.click(lambda s: nav(s,1), inputs=[step_state], outputs=nav_outs)
        back_btn_2.click(lambda s: nav(s,-1), inputs=[step_state], outputs=nav_outs)
        next_btn_2.click(lambda s: nav(s,1), inputs=[step_state], outputs=nav_outs)
        back_btn_3.click(lambda s: nav(s,-1), inputs=[step_state], outputs=nav_outs)
        next_btn_3.click(lambda s: nav(s,1), inputs=[step_state], outputs=nav_outs)
        back_btn_4.click(lambda s: nav(s,-1), inputs=[step_state], outputs=nav_outs)
        restart_btn.click(lambda s: nav(s,-s), inputs=[step_state], outputs=nav_outs)

        # ============================================================
        # Selection Handlers
        # ============================================================
        
        def on_selection_change(*args):
            """When any checkbox changes, update summary, overview, and sliders."""
            all_selected = []
            for v in args:
                all_selected.extend(v or [])
            
            # Base outputs for no selection
            if not all_selected:
                slider_updates = [gr.update(minimum=0, maximum=1, value=0) for _ in range(MAX_TOOL_SLIDERS)]
                container_updates = [gr.update(visible=False) for _ in range(MAX_TOOL_SLIDERS)]
                return [
                    "*No datasets selected*",
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
        # Smart toggle: only deselect all if ALL were selected (prevents sync loop)
        def make_category_toggle(category_datasets):
            """Create a toggle handler for a category checkbox."""
            def toggle(checked, current_selection):
                current_selection = current_selection or []
                if checked:
                    return category_datasets  # Select all
                else:
                    # Only clear if all were selected (user explicitly deselected)
                    # If partial selection, this is from sync - keep current
                    if len(current_selection) == len(category_datasets):
                        return []
                    return current_selection
            return toggle
        
        # Wire category toggles using helper function
        for i, (cat_select, cb, cat_ds) in enumerate(zip(all_cat_selects, all_cbs, all_cat_datasets)):
            cat_select.change(make_category_toggle(cat_ds), inputs=[cat_select, cb], outputs=[cb])
        
        # Sync category checkbox when individual items change
        def make_sync_handler(cat_ds):
            """Create a sync handler that checks if all items are selected."""
            def sync(selected):
                return len(selected) == len(cat_ds) and len(cat_ds) > 0
            return sync
        
        for cb, cat_select, cat_ds in zip(all_cbs, all_cat_selects, all_cat_datasets):
            cb.change(make_sync_handler(cat_ds), inputs=[cb], outputs=[cat_select])
        
        # ============================================================
        # Slider change handlers - update filter_config_state and samples_summary
        # ============================================================
        
        def update_config_and_summary(*args):
            """Update config and show selected/total samples."""
            # args = [slider0, slider1, ..., sliderN, config, names, total_samples]
            slider_values = args[:MAX_TOOL_SLIDERS]
            config = args[MAX_TOOL_SLIDERS]
            names = args[MAX_TOOL_SLIDERS + 1]
            total_samples = args[MAX_TOOL_SLIDERS + 2]  # from metadata (line count)
            
            new_config = dict(config) if config else {}
            
            # Update config with current slider values
            selected_sum = 0
            for i, val in enumerate(slider_values):
                if names and i < len(names) and val is not None:
                    new_config[names[i]] = int(val)
                    selected_sum += int(val)
            
            # Seçilen toplam sample'dan fazla olamaz
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
        
        def show_edit(*args):
            all_selected = []
            for v in args:
                all_selected.extend(v or [])
            if len(all_selected) != 1:
                gr.Warning("Select exactly one dataset to edit")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None
            
            label = all_selected[0]
            filename = label.split(" (")[0] if " (" in label else label
            
            from core.config import get_datasets_dir
            path = os.path.join(get_datasets_dir(), filename)
            
            # Get current category from metadata
            current_category = "Uncategorized"
            meta_path = path.replace(".jsonl", ".meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    current_category = meta.get("category", "Uncategorized")
                except:
                    pass
            
            # Get all available categories for dropdown
            all_categories = get_dataset_categories()
            if current_category not in all_categories:
                all_categories.append(current_category)
            if "Uncategorized" not in all_categories:
                all_categories.append("Uncategorized")
            
            return (
                gr.update(visible=False),  # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=True),   # edit_panel
                gr.update(visible=False),  # merge_panel
                gr.update(value=filename), # edit_name_input
                gr.update(choices=all_categories, value=current_category),  # edit_category_dropdown
                path  # edit_target
            )
        
        def do_edit(old_path, new_name, new_category):
            if not old_path or not os.path.exists(old_path):
                raise gr.Error("Dataset not found")
            
            old_name = os.path.basename(old_path)
            new_name = new_name.strip() if new_name else old_name
            if not new_name.endswith(".jsonl"):
                new_name += ".jsonl"
            
            # Rename if name changed
            if new_name != old_name:
                inline_rename(old_path, new_name)
                # Update path to new location
                from core.config import get_datasets_dir
                old_path = os.path.join(get_datasets_dir(), new_name)
            
            # Update category
            new_category = new_category.strip() if new_category else "Uncategorized"
            update_dataset_category(old_path, new_category)
            
            gr.Info(f"Saved: {new_name} [{new_category}]")
        
        def cancel_edit():
            return gr.update(visible=False), gr.update(visible=True)
        
        def refresh_after_action():
            new_grouped = get_datasets_by_category()
            new_cats = list(new_grouped.keys())
            
            results = []
            for i in range(MAX_CATEGORIES):
                if i < len(new_cats):
                    cat = new_cats[i]
                    ds = new_grouped[cat]
                    results.append(gr.update(label=f"{cat} ({len(ds)})", visible=True))
                    results.append(gr.update(choices=ds, value=[]))
                else:
                    results.append(gr.update(visible=False))
                    results.append(gr.update(choices=[], value=[]))
            
            results.extend([
                gr.update(visible=True),   # empty_panel
                gr.update(visible=False),  # overview_panel
                gr.update(visible=False),  # edit_panel
                gr.update(visible=False),  # merge_panel
                gr.update(visible=False),  # delete_panel
                gr.update(visible=False),  # actions_row
                "*No datasets selected*",  # selection_summary
            ])
            
            return results
        
        # Build refresh outputs list dynamically
        refresh_outputs = []
        for cat_select, cb in zip(all_cat_selects, all_cbs):
            refresh_outputs.append(cat_select)
            refresh_outputs.append(cb)
        refresh_outputs.extend([empty_panel, overview_panel, edit_panel, merge_panel, delete_panel, actions_row, selection_summary])
        
        edit_btn.click(
            show_edit, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, edit_panel, merge_panel, edit_name_input, edit_category_dropdown, edit_target]
        )
        
        edit_save_btn.click(do_edit, inputs=[edit_target, edit_name_input, edit_category_dropdown]).then(
            refresh_after_action,
            outputs=refresh_outputs
        )
        
        edit_cancel_btn.click(cancel_edit, outputs=[edit_panel, overview_panel])
        
        # ============================================================
        # Merge Handlers
        # ============================================================
        
        def show_merge(*args):
            all_selected = []
            for v in args:
                all_selected.extend(v or [])
            if len(all_selected) < 2:
                gr.Warning("Select at least 2 datasets to merge")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            info = f"Merging **{len(all_selected)}** datasets:\n" + "\n".join(f"- {s}" for s in all_selected)
            
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                info
            )
        
        def do_merge(*args):
            # Last arg is new_name, rest are checkbox values
            new_name = args[-1]
            all_selected = []
            for v in args[:-1]:
                all_selected.extend(v or [])
            merge_datasets(all_selected, new_name)
            gr.Info(f"Created merged dataset: {new_name}")
        
        def cancel_merge():
            return gr.update(visible=False), gr.update(visible=True)
        
        merge_btn.click(
            show_merge, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, edit_panel, merge_panel, merge_info]
        )
        
        merge_save_btn.click(do_merge, inputs=[*all_cbs, merge_name_input]).then(
            refresh_after_action,
            outputs=refresh_outputs
        )
        
        merge_cancel_btn.click(cancel_merge, outputs=[merge_panel, overview_panel])
        
        # ============================================================
        # Delete Handlers
        # ============================================================
        
        def show_delete(*args):
            all_selected = []
            for v in args:
                all_selected.extend(v or [])
            if not all_selected:
                gr.Warning("Select at least one dataset to delete")
                return gr.update(), gr.update(), gr.update(), None
            
            info = f"**Delete {len(all_selected)} dataset(s)?**\n\nThis cannot be undone.\n\n" + "\n".join(f"- {s}" for s in all_selected)
            
            from core.config import get_datasets_dir
            paths = []
            for label in all_selected:
                filename = label.split(" (")[0] if " (" in label else label
                paths.append(os.path.join(get_datasets_dir(), filename))
            
            return gr.update(visible=False), gr.update(visible=True), info, paths
        
        def do_delete(paths):
            if not paths:
                return
            for path in paths:
                if os.path.exists(path):
                    confirm_delete(path)
            gr.Info(f"Deleted {len(paths)} dataset(s)")
        
        def cancel_delete_fn():
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
