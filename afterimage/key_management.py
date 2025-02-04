from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional
import warnings
from contextlib import contextmanager
import google.generativeai as genai


@dataclass
class KeyStats:
    """Statistics and state for a single API key"""

    key: str
    hourly_calls: int = 0
    daily_calls: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_count: int = 0
    is_active: bool = True
    last_hourly_reset: datetime = datetime.now()
    last_daily_reset: datetime = datetime.now()


class SmartKeyPool:
    """Manages a pool of API keys with usage tracking and rate limiting."""

    def __init__(
        self,
        api_keys: List[str],
        hourly_limit: Optional[int] = None,
        daily_limit: Optional[int] = None,
        error_threshold: int = 10,
        cooldown_period: int = 600,
    ):
        """Initialize the key pool with configuration.

        Args:
            api_keys: List of API keys to manage
            hourly_limit: Maximum calls per hour per key. None means unlimited.
            daily_limit: Maximum calls per day per key. None means unlimited.
            error_threshold: Number of errors before cooling down a key
            cooldown_period: Seconds to wait after error threshold reached
        """
        self._keys: Dict[str, KeyStats] = {key: KeyStats(key=key) for key in api_keys}
        self._lock = Lock()
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.error_threshold = error_threshold
        self.cooldown_period = cooldown_period

    @classmethod
    def from_single_key(cls, api_key: str) -> "SmartKeyPool":
        """Create a KeyPool instance from a single API key."""
        return cls(api_keys=[api_key])

    def _reset_counters(self, stats: KeyStats) -> None:
        """Reset usage counters if time period has elapsed."""
        now = datetime.now()

        if now - stats.last_hourly_reset > timedelta(hours=1):
            stats.hourly_calls = 0
            stats.last_hourly_reset = now

        if now - stats.last_daily_reset > timedelta(days=1):
            stats.daily_calls = 0
            stats.last_daily_reset = now

    def _is_key_available(self, stats: KeyStats) -> bool:
        """Check if a key is available for use."""
        if not stats.is_active:
            if datetime.now() - stats.last_error > timedelta(
                seconds=self.cooldown_period
            ):
                stats.is_active = True
                stats.error_count = 0
            else:
                return False

        self._reset_counters(stats)
        return (
            self.hourly_limit is None or stats.hourly_calls < self.hourly_limit
        ) and (self.daily_limit is None or stats.daily_calls < self.daily_limit)

    def get_next_key(self) -> str:
        """Get the next available API key using a smart selection strategy.

        Returns:
            str: The selected API key

        Raises:
            RuntimeError: If no keys are available
        """
        with self._lock:
            # First try to find a key that hasn't been used recently
            now = datetime.now()
            available_keys = [
                stats for stats in self._keys.values() if self._is_key_available(stats)
            ]

            if not available_keys:
                raise RuntimeError(
                    "No API keys available. All keys are at limit or cooling down."
                )

            # Sort by last used time and usage counts to distribute load
            selected = min(
                available_keys,
                key=lambda x: (
                    x.last_used or datetime.min,
                    x.hourly_calls,
                    x.daily_calls,
                ),
            )

            selected.hourly_calls += 1
            selected.daily_calls += 1
            selected.last_used = now

            return selected.key

    def report_error(self, key: str) -> None:
        """Report an error for a key, potentially triggering cooldown."""
        with self._lock:
            if key not in self._keys:
                return

            stats = self._keys[key]
            stats.error_count += 1
            stats.last_error = datetime.now()

            if stats.error_count >= self.error_threshold:
                stats.is_active = False
                warnings.warn(
                    f"API key {key[:8]}... has been temporarily disabled due to errors"
                )

    def get_stats(self) -> Dict[str, Dict]:
        """Get current statistics for all keys."""
        with self._lock:
            return {
                key: {
                    "hourly_calls": stats.hourly_calls,
                    "daily_calls": stats.daily_calls,
                    "is_active": stats.is_active,
                    "error_count": stats.error_count,
                    "last_used": stats.last_used,
                    "last_error": stats.last_error,
                }
                for key, stats in self._keys.items()
            }

    @contextmanager
    def configure_api(self, key: str):
        """Thread-safe context manager for API configuration."""
        try:
            genai.configure(api_key=key)
            yield
        finally:
            pass  # Reset if needed
