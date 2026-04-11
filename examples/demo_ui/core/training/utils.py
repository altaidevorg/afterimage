"""
Utility functions for training UI display.
"""

from ..config import SPINNERS, PROGRESS_BAR_LENGTH


def format_time(time_str: str) -> str:
    """
    Convert MM:SS to readable format.

    Args:
        time_str: Time string in MM:SS format

    Returns:
        Formatted time string (e.g., "5m 30s" or "45s")
    """
    if ":" not in time_str:
        return time_str

    parts = time_str.split(":")
    if len(parts) == 2:
        mins, secs = int(parts[0]), int(parts[1])
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"
    return time_str


def make_progress_display(
    percent: int, remaining: str = "", spinner_idx: int = 0
) -> str:
    """
    Create an animated progress display.

    Args:
        percent: Progress percentage (0-100)
        remaining: Remaining time string
        spinner_idx: Spinner animation index

    Returns:
        Formatted progress display string
    """
    filled = int(PROGRESS_BAR_LENGTH * percent / 100)
    empty = PROGRESS_BAR_LENGTH - filled

    bar = "█" * filled + "░" * empty
    spinner = SPINNERS[spinner_idx % len(SPINNERS)]

    lines = [
        f"{spinner} Training in progress...\n\n",
        f"[{bar}] {percent}%\n\n",
    ]

    if remaining and remaining != "00:00":
        lines.append(f"Time remaining: ~{format_time(remaining)}")
    elif percent >= 100:
        lines.append("Finalizing...")
    else:
        lines.append("Calculating time...")

    return "".join(lines)
