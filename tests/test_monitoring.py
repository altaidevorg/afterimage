"""Tests for :class:`~afterimage.monitoring.GenerationMonitor`."""

import time
from pathlib import Path

import pytest

from afterimage.monitoring import Alert, GenerationMonitor


def test_check_alerts_manual_low_success_rate(tmp_path: Path) -> None:
    monitor = GenerationMonitor(
        log_dir=tmp_path / "m",
        metrics_interval=0,
        alert_handlers=[],
    )
    names: list[str] = []

    def capture(alert: Alert) -> None:
        names.append(alert.name)

    monitor.alert_handlers = [capture]
    monitor.track_generation(1.0, success=False)
    time.sleep(0.2)
    monitor.check_alerts()
    monitor.shutdown()
    assert "low_success_rate" in names


def test_periodic_alert_worker_invokes_handlers(tmp_path: Path) -> None:
    received: list[str] = []

    def handler(alert: Alert) -> None:
        received.append(alert.name)

    monitor = GenerationMonitor(
        log_dir=tmp_path / "mon",
        metrics_interval=1,
        alert_handlers=[handler],
    )
    try:
        monitor.track_generation(1.0, success=False)
        time.sleep(2.5)
    finally:
        monitor.shutdown()

    assert "low_success_rate" in received


def test_metrics_interval_zero_skips_alert_thread(tmp_path: Path) -> None:
    monitor = GenerationMonitor(log_dir=tmp_path / "z", metrics_interval=0)
    try:
        assert len(monitor._workers) == 2
    finally:
        monitor.shutdown()
