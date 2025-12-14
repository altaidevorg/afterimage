# Monitoring & Observability

When generating thousands of conversations, you need visibility into the process. Is it working? How fast is it? Are errors occurring? Afterimage provides a **Monitoring System** to track these metrics in real-time.

## `GenerationMonitor`

The central component is the `GenerationMonitor`. It collects metrics from the generator and routes them to various handlers (files, logs, or custom dashboards).

### Initialization

You can attach a monitor to any generator (`AsyncConversationGenerator`, `PersonaGenerator`, etc.).

```python
from afterimage import AsyncConversationGenerator, GenerationMonitor

# 1. Initialize Monitor
# This will save metrics to 'metrics.jsonl' and logs to 'generation_metrics.log' in the current dir
monitor = GenerationMonitor(log_dir=".")

# 2. Attach to Generator
generator = AsyncConversationGenerator(
    ...,
    monitor=monitor
)
```

### Metrics Tracked

The monitor automatically tracks:

*   **Performance**:
    *   `generation_time`: Time taken to generate one conversation.
    *   `tokens_total`, `tokens_per_second`: Throughput usage.
*   **Health**:
    *   `error_rate`: Percentage of failed generations.
    *   `api_errors`: Specific API failures.
*   **Quality** (if Evaluation is running):
    *   `coherence_score`, `grounding_score`, etc.

## Visualization

The `GenerationMonitor` has built-in plotting capabilities using `matplotlib` and `seaborn`. This is useful for analyzing your generation run after it completes.

```python
# Plot generation time over the last hour
fig = monitor.plot_metric(
    metric_name="generation_time",
    window=timedelta(hours=1),
    rolling_window=10
)
fig.savefig("latency_chart.png")

# Get raw data
metrics = monitor.get_metrics("error_rate")
print(f"Current Error Rate: {metrics['average']}")
```

## Alerts

You can set up alerts to notify you (or stop the process) if something goes wrong.

```python
def stop_on_high_error(alert):
    if alert.name == "high_error_rate":
        print("CRITICAL: Stopping generation due to errors!")
        # Logic to stop generation...

monitor = GenerationMonitor(
    alert_handlers=[stop_on_high_error]
)
```

## Handlers

By default, the monitor uses:
*   **`FileMetricHandler`**: Saves structured metrics to a JSONL file.
*   **`FileLogHandler`**: Saves human-readable logs to a text file.

You can implement your own handlers (e.g., to send metrics to Datadog or Prometheus) by implementing the `MetricHandler` protocol.

---
[Previous: Structured Generation](structured_generation.md) | [Next: Advanced Configuration](advanced_usage.md)
