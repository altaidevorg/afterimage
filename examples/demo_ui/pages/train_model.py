import os
import json
import gradio as gr

from .handlers.training import (
    on_train_complete, generate_model_zip, on_dataset_select, set_download_loading,
    get_datasets_by_category, inline_rename, merge_datasets, confirm_delete,
)

MAX_TOOL_SLIDERS = 10


def create_train_model_page(analyze_fn, train_fn, train_dev_fn, eval_fn, chat_fn=None):
    
    # Pre-load categories at startup
    initial_grouped = get_datasets_by_category()
    cat_names = list(initial_grouped.keys())
    
    def get_cat_data(idx):
        if idx < len(cat_names):
            name = cat_names[idx]
            return name, initial_grouped[name]
        return None, []
    
    cat0_name, cat0_ds = get_cat_data(0)
    cat1_name, cat1_ds = get_cat_data(1)
    cat2_name, cat2_ds = get_cat_data(2)
    cat3_name, cat3_ds = get_cat_data(3)
    cat4_name, cat4_ds = get_cat_data(4)
    
    with gr.Blocks() as page:
        gr.Markdown("# Model Training Wizard")
        
        step_state = gr.State(0)
        selected_dataset_path = gr.State(None)
        tool_dist_state = gr.State({})
        filter_config_state = gr.State({})
        tool_names_state = gr.State([])  # List of tool names in order

        # STEP 1
        with gr.Group(visible=True) as step_1_group:
            gr.Markdown("## Step 1/3: Select Datasets")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Library")
                    
                    # Category 0
                    if cat0_name:
                        cat0_select = gr.Checkbox(label=f"{cat0_name} ({len(cat0_ds)})", value=False)
                        cb0 = gr.CheckboxGroup(choices=cat0_ds, value=[], label=None, show_label=False)
                    else:
                        cat0_select = gr.Checkbox(visible=False, value=False)
                        cb0 = gr.CheckboxGroup(choices=[], value=[], visible=False)
                    
                    # Category 1
                    if cat1_name:
                        cat1_select = gr.Checkbox(label=f"{cat1_name} ({len(cat1_ds)})", value=False)
                        cb1 = gr.CheckboxGroup(choices=cat1_ds, value=[], label=None, show_label=False)
                    else:
                        cat1_select = gr.Checkbox(visible=False, value=False)
                        cb1 = gr.CheckboxGroup(choices=[], value=[], visible=False)
                    
                    # Category 2
                    if cat2_name:
                        cat2_select = gr.Checkbox(label=f"{cat2_name} ({len(cat2_ds)})", value=False)
                        cb2 = gr.CheckboxGroup(choices=cat2_ds, value=[], label=None, show_label=False)
                    else:
                        cat2_select = gr.Checkbox(visible=False, value=False)
                        cb2 = gr.CheckboxGroup(choices=[], value=[], visible=False)
                    
                    # Category 3
                    if cat3_name:
                        cat3_select = gr.Checkbox(label=f"{cat3_name} ({len(cat3_ds)})", value=False)
                        cb3 = gr.CheckboxGroup(choices=cat3_ds, value=[], label=None, show_label=False)
                    else:
                        cat3_select = gr.Checkbox(visible=False, value=False)
                        cb3 = gr.CheckboxGroup(choices=[], value=[], visible=False)
                    
                    # Category 4
                    if cat4_name:
                        cat4_select = gr.Checkbox(label=f"{cat4_name} ({len(cat4_ds)})", value=False)
                        cb4 = gr.CheckboxGroup(choices=cat4_ds, value=[], label=None, show_label=False)
                    else:
                        cat4_select = gr.Checkbox(visible=False, value=False)
                        cb4 = gr.CheckboxGroup(choices=[], value=[], visible=False)
                    
                    all_cbs = [cb0, cb1, cb2, cb3, cb4]
                    all_cat_selects = [cat0_select, cat1_select, cat2_select, cat3_select, cat4_select]
                    
                    gr.Markdown("---")
                    selection_summary = gr.Markdown("*No datasets selected*")
                    
                    # Actions for selected datasets
                    with gr.Row(visible=False) as actions_row:
                        rename_btn = gr.Button("Rename", size="sm", variant="secondary")
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
                        samples_summary = gr.Markdown("**0 / 0 samples selected**")
                        
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
                    
                    # Rename panel
                    with gr.Group(visible=False) as rename_panel:
                        gr.Markdown("### Rename Dataset")
                        rename_input = gr.Textbox(label="New Name", placeholder="new_name.jsonl")
                        rename_target = gr.State(None)
                        with gr.Row():
                            rename_save_btn = gr.Button("Save", variant="primary")
                            rename_cancel_btn = gr.Button("Cancel", variant="secondary")
                    
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
            eval_output = gr.Code(language="shell", lines=20)
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
        
        def on_selection_change(v0, v1, v2, v3, v4):
            """When any checkbox changes, update summary, overview, and sliders."""
            all_selected = (v0 or []) + (v1 or []) + (v2 or []) + (v3 or []) + (v4 or [])
            
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
                    gr.update(interactive=False),
                    gr.update(visible=False),  # actions_row
                    "**0 / 0 samples selected**",  # samples_summary
                ] + slider_updates + container_updates
            
            paths, overview, dist, config, _, _, total_samples = on_dataset_select(all_selected)
            can_next = paths is not None and len(paths) > 0
            summary = f"**{len(all_selected)} dataset(s) selected**"
            
            # Build slider updates
            sorted_tools = sorted(dist.items(), key=lambda x: x[1], reverse=True) if dist else []
            tool_names = [t[0] for t in sorted_tools]
            
            # Use distribution sum for both values (consistent)
            total_from_dist = sum(dist.values()) if dist else 0
            
            slider_updates = []
            container_updates = []
            for i in range(MAX_TOOL_SLIDERS):
                if i < len(sorted_tools):
                    tool_name, tool_count = sorted_tools[i]
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
            
            samples_text = f"**{total_from_dist} / {total_from_dist} samples selected**"
            
            return [
                summary,
                paths,
                gr.update(visible=False),  # empty_panel
                gr.update(visible=True),   # overview_panel
                overview,
                dist,
                config,
                tool_names,
                gr.update(interactive=can_next),
                gr.update(visible=True),   # actions_row
                samples_text,  # samples_summary
            ] + slider_updates + container_updates
        
        change_outputs = [
            selection_summary, selected_dataset_path, empty_panel, overview_panel,
            dataset_overview, tool_dist_state, filter_config_state, tool_names_state,
            next_btn_1, actions_row, samples_summary
        ] + tool_sliders + tool_containers
        
        # Wire all checkbox changes
        cb0.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        cb1.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        cb2.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        cb3.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        cb4.change(on_selection_change, inputs=all_cbs, outputs=change_outputs)
        
        # Category checkbox -> select/deselect all in that category
        # Smart toggle: only deselect all if ALL were selected (prevents sync loop)
        def toggle_cat0(checked, current):
            current = current or []
            if checked:
                return cat0_ds  # User wants to select all
            else:
                # Only clear if all were selected (user clicked to deselect)
                # If partial selection, this is from sync - keep current
                if len(current) == len(cat0_ds):
                    return []
                return current
        
        def toggle_cat1(checked, current):
            current = current or []
            if checked:
                return cat1_ds
            else:
                if len(current) == len(cat1_ds):
                    return []
                return current
        
        def toggle_cat2(checked, current):
            current = current or []
            if checked:
                return cat2_ds
            else:
                if len(current) == len(cat2_ds):
                    return []
                return current
        
        def toggle_cat3(checked, current):
            current = current or []
            if checked:
                return cat3_ds
            else:
                if len(current) == len(cat3_ds):
                    return []
                return current
        
        def toggle_cat4(checked, current):
            current = current or []
            if checked:
                return cat4_ds
            else:
                if len(current) == len(cat4_ds):
                    return []
                return current
        
        cat0_select.change(toggle_cat0, inputs=[cat0_select, cb0], outputs=[cb0])
        cat1_select.change(toggle_cat1, inputs=[cat1_select, cb1], outputs=[cb1])
        cat2_select.change(toggle_cat2, inputs=[cat2_select, cb2], outputs=[cb2])
        cat3_select.change(toggle_cat3, inputs=[cat3_select, cb3], outputs=[cb3])
        cat4_select.change(toggle_cat4, inputs=[cat4_select, cb4], outputs=[cb4])
        
        # Sync category checkbox when individual items change
        cb0.change(lambda s: len(s) == len(cat0_ds) and len(cat0_ds) > 0, inputs=[cb0], outputs=[cat0_select])
        cb1.change(lambda s: len(s) == len(cat1_ds) and len(cat1_ds) > 0, inputs=[cb1], outputs=[cat1_select])
        cb2.change(lambda s: len(s) == len(cat2_ds) and len(cat2_ds) > 0, inputs=[cb2], outputs=[cat2_select])
        cb3.change(lambda s: len(s) == len(cat3_ds) and len(cat3_ds) > 0, inputs=[cb3], outputs=[cat3_select])
        cb4.change(lambda s: len(s) == len(cat4_ds) and len(cat4_ds) > 0, inputs=[cb4], outputs=[cat4_select])
        
        # ============================================================
        # Slider change handlers - update filter_config_state and samples_summary
        # ============================================================
        
        def update_config_and_summary(*args):
            """Update config and recalculate total samples."""
            # args = [slider0, slider1, ..., sliderN, config, names, dist]
            slider_values = args[:MAX_TOOL_SLIDERS]
            config = args[MAX_TOOL_SLIDERS]
            names = args[MAX_TOOL_SLIDERS + 1]
            dist = args[MAX_TOOL_SLIDERS + 2]
            
            new_config = dict(config) if config else {}
            
            # Update config with current slider values
            for i, val in enumerate(slider_values):
                if names and i < len(names) and val is not None:
                    new_config[names[i]] = int(val)
            
            # Calculate totals from distribution (consistent source)
            total_max = sum(dist.values()) if dist else 0
            total_selected = 0
            for i, val in enumerate(slider_values):
                if names and i < len(names) and val is not None:
                    total_selected += int(val)
            
            summary_text = f"**{total_selected} / {total_max} samples selected**"
            
            return new_config, summary_text
        
        slider_inputs = tool_sliders + [filter_config_state, tool_names_state, tool_dist_state]
        
        for slider in tool_sliders:
            slider.change(
                update_config_and_summary,
                inputs=slider_inputs,
                outputs=[filter_config_state, samples_summary]
            )
        
        # ============================================================
        # Rename Handlers
        # ============================================================
        
        def show_rename(v0, v1, v2, v3, v4):
            all_selected = (v0 or []) + (v1 or []) + (v2 or []) + (v3 or []) + (v4 or [])
            if len(all_selected) != 1:
                gr.Warning("Select exactly one dataset to rename")
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None
            
            label = all_selected[0]
            filename = label.split(" (")[0] if " (" in label else label
            
            from core.config import get_datasets_dir
            path = os.path.join(get_datasets_dir(), filename)
            
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=filename),
                path
            )
        
        def do_rename(old_path, new_name):
            inline_rename(old_path, new_name)
            gr.Info(f"Renamed to {new_name}")
        
        def cancel_rename():
            return gr.update(visible=False), gr.update(visible=True)
        
        def refresh_after_action():
            new_grouped = get_datasets_by_category()
            new_cats = list(new_grouped.keys())
            
            results = []
            for i in range(5):
                if i < len(new_cats):
                    cat = new_cats[i]
                    ds = new_grouped[cat]
                    results.append(gr.update(label=f"{cat} ({len(ds)})", visible=True))
                    results.append(gr.update(choices=ds, value=[]))
                else:
                    results.append(gr.update(visible=False))
                    results.append(gr.update(choices=[], value=[]))
            
            results.extend([
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "*No datasets selected*",
            ])
            
            return results
        
        rename_btn.click(
            show_rename, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, rename_panel, merge_panel, rename_input, rename_target]
        )
        
        rename_save_btn.click(do_rename, inputs=[rename_target, rename_input]).then(
            refresh_after_action,
            outputs=[
                cat0_select, cb0, cat1_select, cb1, cat2_select, cb2, cat3_select, cb3, cat4_select, cb4,
                empty_panel, overview_panel, rename_panel, merge_panel, delete_panel, actions_row, selection_summary
            ]
        )
        
        rename_cancel_btn.click(cancel_rename, outputs=[rename_panel, overview_panel])
        
        # ============================================================
        # Merge Handlers
        # ============================================================
        
        def show_merge(v0, v1, v2, v3, v4):
            all_selected = (v0 or []) + (v1 or []) + (v2 or []) + (v3 or []) + (v4 or [])
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
        
        def do_merge(v0, v1, v2, v3, v4, new_name):
            all_selected = (v0 or []) + (v1 or []) + (v2 or []) + (v3 or []) + (v4 or [])
            merge_datasets(all_selected, new_name)
            gr.Info(f"Created merged dataset: {new_name}")
        
        def cancel_merge():
            return gr.update(visible=False), gr.update(visible=True)
        
        merge_btn.click(
            show_merge, inputs=all_cbs,
            outputs=[empty_panel, overview_panel, rename_panel, merge_panel, merge_info]
        )
        
        merge_save_btn.click(do_merge, inputs=[*all_cbs, merge_name_input]).then(
            refresh_after_action,
            outputs=[
                cat0_select, cb0, cat1_select, cb1, cat2_select, cb2, cat3_select, cb3, cat4_select, cb4,
                empty_panel, overview_panel, rename_panel, merge_panel, delete_panel, actions_row, selection_summary
            ]
        )
        
        merge_cancel_btn.click(cancel_merge, outputs=[merge_panel, overview_panel])
        
        # ============================================================
        # Delete Handlers
        # ============================================================
        
        def show_delete(v0, v1, v2, v3, v4):
            all_selected = (v0 or []) + (v1 or []) + (v2 or []) + (v3 or []) + (v4 or [])
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
            outputs=[
                cat0_select, cb0, cat1_select, cb1, cat2_select, cb2, cat3_select, cb3, cat4_select, cb4,
                empty_panel, overview_panel, rename_panel, merge_panel, delete_panel, actions_row, selection_summary
            ]
        )
        
        delete_cancel_btn.click(cancel_delete_fn, outputs=[delete_panel, overview_panel])

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
