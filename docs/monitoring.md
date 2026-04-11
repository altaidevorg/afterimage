# Monitoring & Observability

When generating thousands of conversations, you need visibility into the process. Is it working? How fast is it? Are errors occurring? Afterimage provides a robust, thread-safe **Monitoring System** to track these metrics in real-time, visualize them, and export them for analysis.

## `GenerationMonitor`

The central component is the `GenerationMonitor`. It collects metrics from the generator relative to performance, health, and quality, and routes them to various handlers (files, logs, or custom dashboards).

### Initialization

You can attach a monitor to any generator (`ConversationGenerator`, `PersonaGenerator`, etc.). The monitor uses background threads to process metrics without blocking the main generation loop.

```python
from afterimage import ConversationGenerator, GenerationMonitor

# 1. Initialize Monitor
# This writes metrics to metrics.jsonl and logs to afterimage.log under log_dir.
# If log_dir is omitted, a timestamped directory is created under ./.afterimage-monitoring/
monitor = GenerationMonitor(
    log_dir="./logs",
    metrics_interval=60,  # stored on the instance; not used by workers in the current implementation
)

# 2. Attach to Generator
generator = ConversationGenerator(
    ...,
    monitor=monitor
)
```

### Metrics Tracked

The monitor automatically tracks a wide range of metrics:

*   **Performance**:
    *   `generation_time`: Time taken to generate one conversation.
    *   `prompt_token_count`: Input tokens used.
    *   `completion_token_count`: Output tokens generated.
    *   `total_token_count`: Total token usage.
    *   `conversation_length`: Number of turns in the generated conversation.
*   **Health**:
    *   `success_rate`: Binary tracking of successful generations (1.0) vs failures (0.0).
    *   `error_rate`: Binary tracking of errors.
    *   `api_errors`: Specific API failures.
*   **Quality** (if Evaluation is running):
    *   `evaluation_score_<type>`: Scores from evaluators (e.g., `evaluation_score_coherence`).
    *   `evaluation_time`: Time taken for evaluation steps.

### Exporting Data

You can export your collected metrics to various formats for external analysis (e.g., in Jupyter Notebooks or Excel).

```python
# Export to JSON
monitor.export_metrics("metrics_export.json", format="json")

# Export to CSV (creates separate files for each metric type)
monitor.export_metrics("metrics_export.csv", format="csv")

# Export to Excel (creates a multi-sheet workbook)
monitor.export_metrics("metrics_report.xlsx", format="excel")

# Export to Parquet (efficient binary format)
monitor.export_metrics("metrics.parquet", format="parquet")
```

You can also filter exports by a time window:

```python
from datetime import timedelta
# Export only the last hour of data
monitor.export_metrics("last_hour.csv", format="csv", window=timedelta(hours=1))
```

## Visualization

The `GenerationMonitor` has built-in plotting capabilities using `matplotlib` and `seaborn`. It can generate a suite of plots to help you understand your generation run.

```python
# Generate and save all standard plots to the log directory
monitor.visualize_metrics()

# Or specify a custom directory
monitor.visualize_metrics(save_dir="./plots")
```

The standard visualizations include:
1.  **Success/Error Rate Over Time**: Rolling averages of success and failure rates.
2.  **Generation Time Distribution**: Histogram of latencies.
3.  **Token Usage Over Time**: Trends for prompt, completion, and total tokens.
4.  **Evaluation Scores Over Time**: Trends for quality metrics.
5.  **Evaluation Time Distribution**: Histogram of evaluation latencies.

## Alerts

`GenerationMonitor` exposes `alert_handlers` plus threshold kwargs (`alert_min_success_rate`, `alert_max_generation_time_seconds`, `alert_max_error_rate`, token means, `alert_max_conversation_length_mean`). The helper `_check_alerts` encodes the built-in rules using rolling **five-minute means** from `get_metrics` (for example: low success rate below 0.8 by default, mean generation time above 30s, mean error rate above 0.2, mean prompt/completion/total token counts above 4096/4096/8192, mean `conversation_length` **above** 2 turns for the `long_conversations` alert).

**Important:** in the current codebase those rules are **not** invoked on a timer—`metrics_interval` is stored but worker threads only dequeue metrics/logs and forward them to handlers. Nothing calls `_check_alerts` automatically. To use the built-in thresholds today, poll from your own loop (using `get_metrics`) or invoke your own checks alongside generation.

### Custom alert handlers

Register callables that accept an `Alert` dataclass (`name`, `message`, `level`, `timestamp`, `data`). They only run if something in your integration dispatches alerts (for example your own scheduler calling into the same logic as `_check_alerts`, or a custom metric handler that emits alerts).

```python
def stop_on_critical_error(alert):
    if alert.level == "error":
        print(f"CRITICAL ALERT: {alert.name} - {alert.message}")


monitor = GenerationMonitor(
    log_dir="./logs",
    alert_handlers=[stop_on_critical_error],
)
```

## Internals & Extensibility (For Developers)

### Threading Model
The `GenerationMonitor` uses a producer-consumer architecture to ensure monitoring does not impact generation performance.
*   **Producers**: `record_metric`, `log_info`, etc., simply put items into a thread-safe `queue.Queue`.
*   **Consumers**: Background worker threads (`_metric_worker`, `_log_worker`) pull items from the queues and process them (writing to files, forwarding to handlers).

### Custom Handlers
By default, the monitor uses `FileMetricHandler` and `FileLogHandler`. You can implement your own handlers (e.g., to send metrics to Datadog, Prometheus, or WandB) by implementing the `MetricHandler` or `LogHandler` protocols.

```python
from typing import Dict, Any

class WandBMetricHandler:
    def handle_metric(self, metric_name: str, value: float, metadata: Dict[str, Any]) -> None:
        import wandb
        wandb.log({metric_name: value, **metadata})

monitor = GenerationMonitor(
    metric_handlers=[WandBMetricHandler()]
)
```
