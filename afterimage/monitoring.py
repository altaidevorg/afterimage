import asyncio
import json
import logging
import queue
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Literal, Protocol

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LogLevel = Literal["info", "warning", "error"]


class MetricHandler(Protocol):
    """Protocol for custom metric handling."""

    def handle_metric(
        self, metric_name: str, value: float, metadata: dict[str, Any]
    ) -> None:
        """Handle a metric event."""
        pass


class LogHandler(Protocol):
    """Protocol for custom log handling."""

    def handle_log(self, message: dict[str, Any]) -> None:
        """Handle a log message."""
        pass


class FileMetricHandler(MetricHandler):
    """Default file-based metric handler."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = log_dir / "metrics.jsonl"

    def handle_metric(
        self, metric_name: str, value: float, metadata: dict[str, Any]
    ) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            # Use timestamp from metadata if available
            entry = {
                "metric": metric_name,
                "value": value,
                **metadata,  # This includes the timestamp from the caller
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()  # Ensure immediate write


class FileLogHandler(LogHandler):
    """Default file-based log handler."""

    def __init__(self, log_dir: Path):
        self.logger = logging.getLogger(f"afterimage.monitoring.{uuid.uuid4().hex}")
        self.logger.propagate = False
        handler = logging.FileHandler(log_dir / "afterimage.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

    def handle_log(self, message: dict[str, Any]) -> None:
        self.logger.error(json.dumps(message))


@dataclass
class Alert:
    """Represents a monitoring alert."""

    name: str
    message: str
    level: str
    timestamp: datetime
    data: dict[str, Any]


@dataclass(frozen=True)
class ModelTokenUsage:
    """Token usage for a single model."""

    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TokenUsageReport:
    """Token usage report: per-model breakdown plus totals for cost calculation."""

    by_model: list[ModelTokenUsage]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class GenerationMonitor:
    """Monitors and tracks conversation generation metrics."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        metric_handlers: list[MetricHandler] | None = None,
        log_handlers: list[LogHandler] | None = None,
        alert_handlers: list[Callable[[Alert], None]] | None = None,
        metrics_interval: int = 60,  # seconds
        shutdown_timeout: int = 5,
        *,
        alert_min_success_rate: float | None = None,
        alert_max_generation_time_seconds: float | None = None,
        alert_max_error_rate: float | None = None,
        alert_max_prompt_token_mean: float | None = None,
        alert_max_completion_token_mean: float | None = None,
        alert_max_total_token_mean: float | None = None,
        alert_max_conversation_length_mean: float | None = None,
        token_usage_callback: Callable[[TokenUsageReport], None] | None = None,
        token_usage_callback_interval_seconds: float = 60.0,
    ):
        """Initialize generation monitor.

        Args:
            log_dir: Directory to save metrics logs
            metric_handlers: List of custom metric handlers
            log_handlers: List of custom log handlers
            alert_handlers: List of callables to handle alerts
            metrics_interval: How often to calculate metrics (seconds)
            shutdown_timeout: Timeout for graceful shutdown (seconds)
            alert_min_success_rate: Alert if success_rate mean below this (default 0.8).
            alert_max_generation_time_seconds: Alert if generation_time mean above this in seconds (default 30).
            alert_max_error_rate: Alert if error_rate mean above this (default 0.2).
            alert_max_prompt_token_mean: Alert if prompt_token_count mean above this (default 4096).
            alert_max_completion_token_mean: Alert if completion_token_count mean above this (default 4096).
            alert_max_total_token_mean: Alert if total_token_count mean above this (default 8192).
            alert_max_conversation_length_mean: Alert if conversation_length mean above this (default 2).
            token_usage_callback: If set, called periodically with current TokenUsageReport for easy tracking.
            token_usage_callback_interval_seconds: How often to invoke token_usage_callback (default 60). Used only when token_usage_callback is set.
        """
        self.log_dir = (
            Path(log_dir)
            if log_dir
            else Path(".afterimage-monitoring")
            / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Initialize handlers
        self.metric_handlers = metric_handlers or [FileMetricHandler(self.log_dir)]
        self.log_handlers = log_handlers or [FileLogHandler(self.log_dir)]
        self.alert_handlers = alert_handlers or []

        self.metrics_interval = metrics_interval
        self.shutdown_timeout = shutdown_timeout

        # Alert thresholds (None = use default)
        self._alert_min_success_rate = (
            0.8 if alert_min_success_rate is None else alert_min_success_rate
        )
        self._alert_max_generation_time_seconds = (
            30.0
            if alert_max_generation_time_seconds is None
            else alert_max_generation_time_seconds
        )
        self._alert_max_error_rate = (
            0.2 if alert_max_error_rate is None else alert_max_error_rate
        )
        self._alert_max_prompt_token_mean = (
            4096.0
            if alert_max_prompt_token_mean is None
            else alert_max_prompt_token_mean
        )
        self._alert_max_completion_token_mean = (
            4096.0
            if alert_max_completion_token_mean is None
            else alert_max_completion_token_mean
        )
        self._alert_max_total_token_mean = (
            8192.0 if alert_max_total_token_mean is None else alert_max_total_token_mean
        )
        self._alert_max_conversation_length_mean = (
            2.0
            if alert_max_conversation_length_mean is None
            else alert_max_conversation_length_mean
        )

        self._token_usage_callback = token_usage_callback
        self._token_usage_callback_interval = token_usage_callback_interval_seconds

        # Initialize metrics storage
        self._metrics = defaultdict(list)
        self._lock = Lock()
        self._async_lock = asyncio.Lock()

        # Initialize queues
        self.metric_queue = queue.Queue()
        self.log_queue = queue.Queue()

        # Start worker threads
        self._shutdown = Event()
        self._workers = [
            Thread(target=self._metric_worker, daemon=True),
            Thread(target=self._log_worker, daemon=True),
        ]
        if self._token_usage_callback is not None:
            self._workers.append(Thread(target=self._token_usage_worker, daemon=True))
        for worker in self._workers:
            worker.start()

    def _metric_worker(self):
        """Process metrics from queue."""
        while not self._shutdown.is_set():
            try:
                metric = self.metric_queue.get(timeout=1)

                # Store internally with datetime
                with self._lock:
                    self._store_metric(**metric)

                # Convert datetime to ISO format for handlers
                timestamp = metric["metadata"].pop("timestamp")
                handler_metric = {
                    "metric_name": metric["metric_name"],
                    "value": metric["value"],
                    "metadata": {
                        **metric["metadata"],
                        "timestamp": timestamp.isoformat()
                        if isinstance(timestamp, datetime)
                        else timestamp,
                    },
                }

                # Send to handlers
                for handler in self.metric_handlers:
                    try:
                        handler.handle_metric(**handler_metric)
                    except Exception as e:
                        self._enqueue_log(
                            {
                                "level": "ERROR",
                                "message": f"Metric handler failed: {str(e)}",
                                "error": str(e),
                            }
                        )

            except queue.Empty:
                continue
            except Exception as e:
                self._enqueue_log(
                    {
                        "level": "ERROR",
                        "message": f"Metric worker failed: {str(e)}",
                        "error": str(e),
                    }
                )

    def _token_usage_worker(self):
        """Periodically invoke token_usage_callback with current total token usage."""
        while not self._shutdown.is_set():
            if self._shutdown.wait(timeout=self._token_usage_callback_interval):
                break
            if self._token_usage_callback is None:
                continue
            try:
                report = self.get_total_token_usage()
                self._token_usage_callback(report)
            except Exception as e:
                self._enqueue_log(
                    {
                        "level": "ERROR",
                        "message": f"Token usage callback failed: {str(e)}",
                        "error": str(e),
                    },
                )

    def _log_worker(self):
        """Process logs from queue."""
        while not self._shutdown.is_set():
            try:
                log = self.log_queue.get(timeout=1)
                for handler in self.log_handlers:
                    try:
                        handler.handle_log(log)
                    except Exception as e:
                        print(f"Log handler failed: {str(e)}")  # Last resort logging
            except queue.Empty:
                continue

    def _store_metric(self, metric_name: str, value: float, metadata: dict[str, Any]):
        """Store metric in internal storage."""
        timestamp = datetime.now()
        self._metrics[metric_name].append(
            {"timestamp": timestamp, "value": value, **(metadata or {})}
        )

    def _enqueue_log(self, message: dict[str, Any], level: LogLevel = "info"):
        """Add log message to queue."""
        message["level"] = level
        self.log_queue.put(message)

    def log_info(self, message: str, **data):
        """Log an info message."""
        self._enqueue_log({"message": message, **data}, "info")

    def log_warning(self, message: str, **data):
        """Log a warning message."""
        self._enqueue_log({"message": message, **data}, "warning")

    def log_error(self, message: str, error: Exception = None, **data):
        """Log an error message."""
        error_data = {}
        if error is not None:
            error_data["error"] = str(error)
            error_data["error_type"] = error.__class__.__name__
        self._enqueue_log({"message": message, **error_data, **data}, "error")

    def record_metric(
        self, metric_name: str, value: float, metadata: dict[str, Any] | None = None
    ):
        """Record metric using queue."""
        timestamp = (metadata.get("timestamp") if metadata else None) or datetime.now()
        meta = dict(metadata) if metadata else {}
        meta.setdefault("timestamp", timestamp)

        self.metric_queue.put(
            {
                "metric_name": metric_name,
                "value": value,
                "metadata": meta,
            }
        )

    def track_generation(self, duration: float, success: bool, **kwargs):
        """Track generation metrics using queue.

        Args:
            duration: Time taken for generation.
            success: Whether generation completed successfully.
            **kwargs: Additional metrics or metadata for logging.
                Some common metrics include:
                - prompt_token_count: Number of tokens in the prompt (input tokens).
                - completion_token_count: Number of tokens in the completion (output tokens).
                - total_token_count: Total number of tokens used (input + output tokens).
                - model_name: Name of the model used.
                - finish_reason: Reason for generation completion.
                - error: Error message if generation failed.
                - turns: Number of turns in the conversation.
                These metrics are automatically converted to individual metrics and logged with the timestamp when they are passed as kwargs.
        """
        timestamp = datetime.now()  # Create timestamp once

        # Create JSON-serializable metrics for logging
        metrics = {
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "success": success,
            **kwargs,
        }

        # Record individual metrics with datetime object

        # success and error metrics
        self.record_metric("generation_time", duration, {"timestamp": timestamp})
        self.record_metric(
            "success_rate", 1.0 if success else 0.0, {"timestamp": timestamp}
        )
        if "error" in kwargs:
            self.record_metric(
                "error_rate", 1.0 if kwargs["error"] else 0.0, {"timestamp": timestamp}
            )
        if "model_name" in kwargs:
            self.record_metric(
                f"model_usage:{kwargs['model_name']}", 1.0, {"timestamp": timestamp}
            )
        if "finish_reason" in kwargs:
            self.record_metric(
                f"finish_reason:{kwargs['finish_reason']}",
                1.0,
                {"timestamp": timestamp},
            )

        # metrics to record individually (include model_name in metadata for token metrics so get_total_token_usage can group by model)
        metrics_to_record = [
            "prompt_token_count",
            "completion_token_count",
            "total_token_count",
            "conversation_length",
        ]
        token_meta: dict[str, Any] = {"timestamp": timestamp}
        if "model_name" in kwargs:
            token_meta["model_name"] = kwargs["model_name"]
        for metric in metrics_to_record:
            if metric not in kwargs:
                continue
            raw = kwargs[metric]
            if raw is None:
                continue
            self.record_metric(
                metric,
                raw,
                token_meta
                if metric != "conversation_length"
                else {"timestamp": timestamp},
            )

        # Log complete metrics
        self._enqueue_log({"message": "Generation metrics", "data": metrics})

    def track_evaluation(
        self,
        duration: float,
        success: bool,
        evaluator_type: str,
        scores: dict[str, float],
        **kwargs,
    ) -> None:
        """Track evaluation metrics.

        Args:
            duration: Time taken for evaluation
            success: Whether evaluation completed successfully
            evaluator_type: Type of evaluator (e.g., 'coherence', 'factuality')
            scores: Dictionary of evaluation scores
            **kwargs: Additional metadata
        """
        timestamp = datetime.now()

        # Create JSON-serializable metrics
        metrics = {
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "success": success,
            "evaluator_type": evaluator_type,
            "scores": scores,
            **kwargs,
        }

        # Record individual metrics
        self.record_metric(
            "evaluation_time",
            duration,
            metadata={
                "timestamp": timestamp,
                "evaluator_type": evaluator_type,
                "success": success,
            },
        )

        # Record scores as separate metrics
        for score_name, score_value in scores.items():
            feedback = (
                score_value.get("feedback", None)
                if isinstance(score_value, dict)
                else None
            )
            value = (
                score_value.get("score", 0)
                if isinstance(score_value, dict)
                else score_value
            )
            self.record_metric(
                f"evaluation_score_{score_name}",
                value,
                metadata={
                    "timestamp": timestamp,
                    "evaluator_type": evaluator_type,
                    "feedback": feedback,
                },
            )

        if "error" in kwargs:
            self.record_metric(
                "evaluation_error_rate",
                1 if kwargs["error"] else 0,
                metadata={
                    "timestamp": timestamp,
                    "evaluator_type": evaluator_type,
                },
            )

        # Record token usage for evaluation LLM calls (for cost calculation)
        eval_token_meta: dict[str, Any] = {"timestamp": timestamp}
        if "model_name" in kwargs:
            eval_token_meta["model_name"] = kwargs["model_name"]
        for token_metric in (
            "prompt_token_count",
            "completion_token_count",
            "total_token_count",
        ):
            if token_metric not in kwargs:
                continue
            raw = kwargs[token_metric]
            if raw is None:
                continue
            self.record_metric(
                token_metric,
                raw,
                eval_token_meta,
            )

        # Log complete metrics
        self._enqueue_log({"message": "Evaluation metrics", "data": metrics})

    def shutdown(self):
        """Gracefully shutdown monitoring."""
        self._shutdown.set()
        for worker in self._workers:
            worker.join(timeout=self.shutdown_timeout)

    def get_metrics(
        self,
        metric_name: str,
        window: timedelta = timedelta(minutes=5),
    ) -> dict[str, float]:
        """Get aggregated metrics for a time window.

        Args:
            metric_name: Name of metric to retrieve
            window: Time window for aggregation

        Returns:
            Dict containing metric aggregates
        """
        with self._lock:
            if metric_name not in self._metrics:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

            now = datetime.now()
            window_start = now - window

            # Filter metrics within window
            values = [
                m["value"]
                for m in self._metrics[metric_name]
                if m["timestamp"] >= window_start
            ]

            if not values:
                return {}

            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    def get_total_token_usage(
        self, window: timedelta | None = None
    ) -> TokenUsageReport:
        """Get total token usage summed across all events, optionally within a time window.
        Grouped by model name for cost calculation (one entry per model).

        Args:
            window: If set, only include events with timestamp >= (now - window).
                    If None, include all events.

        Returns:
            TokenUsageReport with by_model (one entry per model) and total_* fields.
        """
        with self._lock:
            now = datetime.now()
            window_start = (now - window) if window else None

            def in_window(m: dict[str, Any]) -> bool:
                if window_start is None:
                    return True
                ts = m.get("timestamp")
                return ts is not None and ts >= window_start

            by_model: dict[str, dict[str, int]] = {}

            for metric_key, key in [
                ("prompt_token_count", "prompt_tokens"),
                ("completion_token_count", "completion_tokens"),
                ("total_token_count", "total_tokens"),
            ]:
                if metric_key not in self._metrics:
                    continue
                for entry in self._metrics[metric_key]:
                    if not in_window(entry):
                        continue
                    model_name = entry.get("model_name") or "unknown"
                    if model_name not in by_model:
                        by_model[model_name] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }
                    raw_val = entry.get("value")
                    delta = int(raw_val) if raw_val is not None else 0
                    by_model[model_name][key] = by_model[model_name][key] + delta

            total_prompt = total_completion = total_all = 0
            model_usages: list[ModelTokenUsage] = []
            for model_name, counts in sorted(by_model.items()):
                model_usages.append(
                    ModelTokenUsage(
                        model_name=model_name,
                        prompt_tokens=counts["prompt_tokens"],
                        completion_tokens=counts["completion_tokens"],
                        total_tokens=counts["total_tokens"],
                    )
                )
                total_prompt += counts["prompt_tokens"]
                total_completion += counts["completion_tokens"]
                total_all += counts["total_tokens"]

            return TokenUsageReport(
                by_model=model_usages,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_tokens=total_all,
            )

    def _check_alerts(self, metrics: dict[str, Any]):
        """Check metrics against configurable alert thresholds."""
        # Check success rate
        recent_success = self.get_metrics("success_rate", timedelta(minutes=5))
        if recent_success and recent_success["mean"] < self._alert_min_success_rate:
            self._send_alert(
                Alert(
                    name="low_success_rate",
                    message=f"Success rate dropped to {recent_success['mean']:.1%}",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_success,
                )
            )

        # Check generation time
        recent_time = self.get_metrics("generation_time", timedelta(minutes=5))
        if (
            recent_time
            and recent_time["mean"] > self._alert_max_generation_time_seconds
        ):
            self._send_alert(
                Alert(
                    name="high_generation_time",
                    message=f"Average generation time: {recent_time['mean']:.1f}s",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_time,
                )
            )

        # Check error rate
        recent_errors = self.get_metrics("error_rate", timedelta(minutes=5))
        if recent_errors and recent_errors["mean"] > self._alert_max_error_rate:
            self._send_alert(
                Alert(
                    name="high_error_rate",
                    message=f"Error rate increased to {recent_errors['mean']:.1%}",
                    level="error",
                    timestamp=datetime.now(),
                    data=recent_errors,
                )
            )

        # Check token usage spikes
        recent_tokens = self.get_metrics("prompt_token_count", timedelta(minutes=5))
        if recent_tokens and recent_tokens["mean"] > self._alert_max_prompt_token_mean:
            self._send_alert(
                Alert(
                    name="high_token_usage:prompt",
                    message=f"Average token usage: {recent_tokens['mean']:.0f}",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_tokens,
                )
            )
        recent_tokens = self.get_metrics("completion_token_count", timedelta(minutes=5))
        if (
            recent_tokens
            and recent_tokens["mean"] > self._alert_max_completion_token_mean
        ):
            self._send_alert(
                Alert(
                    name="high_token_usage:completion",
                    message=f"Average token usage: {recent_tokens['mean']:.0f}",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_tokens,
                )
            )
        recent_tokens = self.get_metrics("total_token_count", timedelta(minutes=5))
        if recent_tokens and recent_tokens["mean"] > self._alert_max_total_token_mean:
            self._send_alert(
                Alert(
                    name="high_token_usage:total",
                    message=f"Average token usage: {recent_tokens['mean']:.0f}",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_tokens,
                )
            )

        # Check for long conversations
        recent_turns = self.get_metrics("conversation_length", timedelta(minutes=5))
        if (
            recent_turns
            and recent_turns["mean"] > self._alert_max_conversation_length_mean
        ):
            self._send_alert(
                Alert(
                    name="long_conversations",
                    message=f"Average conversation length: {recent_turns['mean']:.1f} turns",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_turns,
                )
            )

    def _send_alert(self, alert: Alert):
        """Send alert to all handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                warnings.warn(f"Alert handler failed: {e}")

    def save_metrics(self) -> Path:
        """Save current metrics to disk.

        Returns:
            Path to saved metrics file
        """
        metrics_file = self.log_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"

        with self._lock:
            with open(metrics_file, "w") as f:
                json.dump(
                    self._metrics, f, default=str
                )  # Handle datetime serialization

        return metrics_file

    def export_metrics(
        self,
        output_path: str | Path,
        format: str = "json",
        window: timedelta | None = None,
    ) -> None:
        """Export metrics data to various formats.

        Args:
            output_path: Path to save the exported data
            format: Export format ('json', 'csv', 'excel', 'parquet')
            window: Optional time window to filter metrics
        """
        output_path = Path(output_path)

        with self._lock:
            # Filter by time window if specified
            if window:
                now = datetime.now()
                window_start = now - window
                filtered_metrics = {
                    name: [m for m in values if m["timestamp"] >= window_start]
                    for name, values in self._metrics.items()
                }
            else:
                filtered_metrics = self._metrics

            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(filtered_metrics, f, default=str)

            elif format in ["csv", "excel"]:
                # Create a multi-sheet workbook
                dfs = {}

                for metric_name, values in filtered_metrics.items():
                    # Convert to DataFrame with metadata columns
                    rows = []
                    for entry in values:
                        row = {"timestamp": entry["timestamp"], "value": entry["value"]}
                        if "metadata" in entry:
                            for k, v in entry["metadata"].items():
                                row[f"metadata_{k}"] = v
                        rows.append(row)

                    dfs[metric_name] = pd.DataFrame(rows)

                if format == "csv":
                    # Save each metric to a separate CSV file
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    for metric_name, df in dfs.items():
                        metric_file = (
                            output_path.parent / f"{output_path.stem}_{metric_name}.csv"
                        )
                        df.to_csv(metric_file, index=False)

                else:  # excel
                    # Save all metrics as sheets in one Excel file
                    with pd.ExcelWriter(output_path) as writer:
                        for metric_name, df in dfs.items():
                            df.to_excel(writer, sheet_name=metric_name, index=False)

            elif format == "parquet":
                # Convert to a single DataFrame with metric_name column
                rows = []
                for metric_name, values in filtered_metrics.items():
                    for entry in values:
                        row = {
                            "metric_name": metric_name,
                            "timestamp": entry["timestamp"],
                            "value": entry["value"],
                        }
                        if "metadata" in entry:
                            for k, v in entry["metadata"].items():
                                row[f"metadata_{k}"] = v
                        rows.append(row)

                df = pd.DataFrame(rows)
                df.to_parquet(output_path, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

    def visualize_metrics(
        self,
        save_dir: str | Path | None = None,
        figsize: tuple = (12, 6),
        return_figures: bool = False,
    ) -> dict[str, plt.Figure] | None:
        """Generate visualizations for metrics.

        Args:
            save_dir: Optional directory to save plots.
            figsize: Figure size for plots.
            return_figures: Whether to return the figures.

        Returns:
            Dict of matplotlib figures if return_figures is True, otherwise None.
        """
        if save_dir is None:
            save_dir = self.log_dir / "plots"

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        figures = {}

        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            plt.style.use("default")
            warnings.warn(
                "Could not load seaborn style, using default matplotlib style"
            )

        # Convert metrics to DataFrames for plotting
        dfs = {}
        with self._lock:
            for metric_name, values in self._metrics.items():
                if values:
                    df = pd.DataFrame(values)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        dfs[metric_name] = df

        if not dfs:
            warnings.warn("No metrics data available for visualization")
            return figures

        # 1. Success/Error Rate Over Time
        if "success_rate" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["success_rate"]
            df["rolling_success"] = df["value"].rolling(window=10, min_periods=1).mean()

            if "error_rate" in dfs:
                df_error = dfs["error_rate"]
                df_error["rolling_error"] = (
                    df_error["value"].rolling(window=10, min_periods=1).mean()
                )
                ax.plot(
                    df_error["timestamp"],
                    df_error["rolling_error"],
                    color="red",
                    label="Error Rate (rolling avg)",
                )

            ax.plot(
                df["timestamp"],
                df["rolling_success"],
                color="green",
                label="Success Rate (rolling avg)",
            )
            ax.set_title("Success/Error Rate Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Rate")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["success_error_rate"] = fig

        # 2. Generation Time Distribution
        if "generation_time" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["generation_time"]
            sns.histplot(data=df["value"], ax=ax)
            ax.set_title("Generation Time Distribution")
            ax.set_xlabel("Time (seconds)")
            plt.tight_layout()
            figures["generation_time"] = fig

        # 3. Total Token Usage Over Time
        if "total_token_count" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["total_token_count"]
            df["rolling_avg"] = df["value"].rolling(window=10, min_periods=1).mean()
            ax.plot(df["timestamp"], df["rolling_avg"], color="blue")
            ax.set_title("Total Token Usage Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Tokens")
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["total_token_count"] = fig

        # 4. Prompt Token Usage Over Time
        if "prompt_token_count" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["prompt_token_count"]
            df["rolling_avg"] = df["value"].rolling(window=10, min_periods=1).mean()
            ax.plot(df["timestamp"], df["rolling_avg"], color="blue")
            ax.set_title("Prompt Token Usage Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Tokens")
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["prompt_token_count"] = fig

        # 5. Completion Token Usage Over Time
        if "completion_token_count" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["completion_token_count"]
            df["rolling_avg"] = df["value"].rolling(window=10, min_periods=1).mean()
            ax.plot(df["timestamp"], df["rolling_avg"], color="blue")
            ax.set_title("Completion Token Usage Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Tokens")
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["completion_token_count"] = fig

        # 6. Evaluation Scores Over Time
        evaluation_metrics = [
            metric for metric in dfs.keys() if metric.startswith("evaluation_score_")
        ]
        if evaluation_metrics:
            fig, ax = plt.subplots(figsize=figsize)
            for metric in evaluation_metrics:
                df = dfs[metric]
                metric_name = metric.replace("evaluation_score_", "")
                # Convert scores to numeric values
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                # Calculate rolling average only for valid numeric values
                df["rolling_avg"] = df["value"].rolling(window=5, min_periods=1).mean()
                ax.plot(
                    df["timestamp"],
                    df["rolling_avg"],
                    label=f"{metric_name.title()} Score",
                )
            ax.set_title("Evaluation Scores Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Score")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Move legend outside
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["evaluation_scores"] = fig

        # 7. Evaluation Time Distribution
        if "evaluation_time" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["evaluation_time"]
            sns.histplot(data=df["value"], ax=ax)
            ax.set_title("Evaluation Time Distribution")
            ax.set_xlabel("Time (seconds)")
            plt.tight_layout()
            figures["evaluation_time"] = fig

        # Save if directory provided
        for name, fig in figures.items():
            fig.savefig(save_dir / f"{name}.png")
        if return_figures:
            return figures

    def plot_metric(
        self,
        metric_name: str,
        window: timedelta = timedelta(hours=1),
        rolling_window: int = 10,
        figsize: tuple = (12, 8),
    ) -> plt.Figure:
        """Plot a specific metric over time.

        Args:
            metric_name: Name of metric to plot
            window: Time window for visualization
            rolling_window: Window size for rolling average
            figsize: Figure size for plot

        Returns:
            matplotlib figure
        """
        with self._lock:
            if metric_name not in self._metrics:
                raise ValueError(f"Metric {metric_name} not found")
            now = datetime.now()
            window_start = now - window
            values = [
                {"timestamp": m["timestamp"], "value": m["value"]}
                for m in self._metrics[metric_name]
                if m["timestamp"] >= window_start
            ]

        if not values:
            raise ValueError(f"No data for metric {metric_name} in specified window")

        df = pd.DataFrame(values)
        df["rolling_avg"] = (
            df["value"].rolling(window=rolling_window, min_periods=1).mean()
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            df["timestamp"], df["rolling_avg"], label=f"{metric_name} (rolling avg)"
        )
        ax.scatter(df["timestamp"], df["value"], alpha=0.2, label="Raw values")

        ax.set_title(f"{metric_name} Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(metric_name)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig
