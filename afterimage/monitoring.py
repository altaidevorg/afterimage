from typing import Dict, Any, List, Optional, Callable, Protocol
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
from threading import Lock, Event, Thread
import queue
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict


class MetricHandler(Protocol):
    """Protocol for custom metric handling."""

    def handle_metric(
        self, metric_name: str, value: float, metadata: Dict[str, Any]
    ) -> None:
        """Handle a metric event."""
        pass


class LogHandler(Protocol):
    """Protocol for custom log handling."""

    def handle_log(self, message: Dict[str, Any]) -> None:
        """Handle a log message."""
        pass


class FileMetricHandler(MetricHandler):
    """Default file-based metric handler."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = log_dir / "metrics.jsonl"

    def handle_metric(
        self, metric_name: str, value: float, metadata: Dict[str, Any]
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
        self.logger = logging.getLogger("afterimage.monitoring")
        handler = logging.FileHandler(log_dir / "generation_metrics.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

    def handle_log(self, message: Dict[str, Any]) -> None:
        self.logger.error(json.dumps(message))


@dataclass
class Alert:
    """Represents a monitoring alert."""

    name: str
    message: str
    level: str
    timestamp: datetime
    data: Dict[str, Any]


class GenerationMonitor:
    """Monitors and tracks conversation generation metrics."""

    def __init__(
        self,
        log_dir: Optional[str | Path] = None,
        metric_handlers: Optional[List[MetricHandler]] = None,
        log_handlers: Optional[List[LogHandler]] = None,
        alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
        metrics_interval: int = 60,  # seconds
        shutdown_timeout: int = 5,
    ):
        """Initialize generation monitor.

        Args:
            log_dir: Directory to save metrics logs
            metric_handlers: List of custom metric handlers
            log_handlers: List of custom log handlers
            alert_handlers: List of callables to handle alerts
            metrics_interval: How often to calculate metrics (seconds)
            shutdown_timeout: Timeout for graceful shutdown (seconds)
        """
        self.log_dir = Path(log_dir) if log_dir else Path("monitoring")
        self.log_dir.mkdir(exist_ok=True)

        # Initialize handlers
        self.metric_handlers = metric_handlers or [FileMetricHandler(self.log_dir)]
        self.log_handlers = log_handlers or [FileLogHandler(self.log_dir)]
        self.alert_handlers = alert_handlers or []

        self.metrics_interval = metrics_interval
        self.shutdown_timeout = shutdown_timeout

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

    def _store_metric(self, metric_name: str, value: float, metadata: Dict[str, Any]):
        """Store metric in internal storage."""
        timestamp = datetime.now()
        self._metrics[metric_name].append(
            {"timestamp": timestamp, "value": value, **(metadata or {})}
        )

    def _enqueue_log(self, message: Dict[str, Any]):
        """Add log message to queue."""
        self.log_queue.put(message)

    def record_metric(
        self, metric_name: str, value: float, metadata: Optional[Dict] = None
    ):
        """Record metric using queue."""
        timestamp = (
            metadata.pop("timestamp", None) if metadata else None
        ) or datetime.now()

        self.metric_queue.put(
            {
                "metric_name": metric_name,
                "value": value,
                "metadata": {
                    "timestamp": timestamp,
                    **(metadata or {}),
                },
            }
        )

    def track_generation(self, duration: float, success: bool, **kwargs):
        """Track generation metrics using queue."""
        timestamp = datetime.now()  # Create timestamp once

        # Create JSON-serializable metrics for logging
        metrics = {
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "success": success,
            **kwargs,
        }

        # Record individual metrics with datetime object
        self.record_metric("generation_time", duration, {"timestamp": timestamp})
        self.record_metric(
            "success_rate", 1 if success else 0, {"timestamp": timestamp}
        )

        if "error" in kwargs:
            self.record_metric(
                "error_rate", 1 if kwargs["error"] else 0, {"timestamp": timestamp}
            )

        if "tokens" in kwargs:
            self.record_metric(
                "token_usage", kwargs["tokens"], {"timestamp": timestamp}
            )

        if "turns" in kwargs:
            self.record_metric(
                "conversation_length", kwargs["turns"], {"timestamp": timestamp}
            )

        # Log complete metrics
        self._enqueue_log({"message": "Generation metrics", "data": metrics})

    def track_evaluation(
        self,
        duration: float,
        success: bool,
        evaluator_type: str,
        scores: Dict[str, float],
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
                if isinstance(score_value, Dict)
                else None
            )
            value = (
                score_value.get("score", 0)
                if isinstance(score_value, Dict)
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
    ) -> Dict[str, float]:
        """Get aggregated metrics for a time window.

        Args:
            metric_name: Name of metric to retrieve
            window: Time window for aggregation

        Returns:
            Dict containing metric aggregates
        """
        with self._lock:
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

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert conditions."""
        # Check success rate
        recent_success = self.get_metrics("success_rate", timedelta(minutes=5))
        if recent_success and recent_success["mean"] < 0.8:  # Below 80% success
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
        if recent_time and recent_time["mean"] > 30:  # Over 30s average
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
        if recent_errors and recent_errors["mean"] > 0.2:  # Over 20% errors
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
        recent_tokens = self.get_metrics("token_usage", timedelta(minutes=5))
        if recent_tokens and recent_tokens["mean"] > 5000:  # High token usage
            self._send_alert(
                Alert(
                    name="high_token_usage",
                    message=f"Average token usage: {recent_tokens['mean']:.0f}",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_tokens,
                )
            )

        # Check for short conversations
        recent_turns = self.get_metrics("conversation_length", timedelta(minutes=5))
        if recent_turns and recent_turns["mean"] < 2:  # Less than 2 turns on average
            self._send_alert(
                Alert(
                    name="short_conversations",
                    message=f"Average conversation length: {recent_turns['mean']:.1f} turns",
                    level="warning",
                    timestamp=datetime.now(),
                    data=recent_turns,
                )
            )

    def _send_alert(self, alert: Alert):
        """Send alert to all handlers."""
        print(f"Alert: {alert.name} - {alert.message}")

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
        window: Optional[timedelta] = None,
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
        self, save_dir: Optional[str | Path] = None, figsize: tuple = (12, 6)
    ) -> Dict[str, plt.Figure]:
        """Generate visualizations for metrics.

        Args:
            save_dir: Optional directory to save plots
            figsize: Figure size for plots

        Returns:
            Dict of matplotlib figures
        """
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

        # 3. Token Usage Over Time
        if "token_usage" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["token_usage"]
            df["rolling_avg"] = df["value"].rolling(window=10, min_periods=1).mean()
            ax.plot(df["timestamp"], df["rolling_avg"], color="blue")
            ax.set_title("Token Usage Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Tokens")
            plt.xticks(rotation=45)
            plt.tight_layout()
            figures["token_usage"] = fig

        # 4. Evaluation Scores Over Time
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

        # 5. Evaluation Time Distribution
        if "evaluation_time" in dfs:
            fig, ax = plt.subplots(figsize=figsize)
            df = dfs["evaluation_time"]
            sns.histplot(data=df["value"], ax=ax)
            ax.set_title("Evaluation Time Distribution")
            ax.set_xlabel("Time (seconds)")
            plt.tight_layout()
            figures["evaluation_time"] = fig

        # Save if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(save_dir / f"{name}.png")

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
        now = datetime.now()
        window_start = now - window

        # Filter metrics by time window
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
