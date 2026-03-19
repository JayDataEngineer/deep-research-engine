"""
Extension Metrics

Collects telemetry and usage statistics for extensions.
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime


class ExtensionMetrics:
    """
    Telemetry and metrics collector for extensions.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        # Extension-level metrics
        self._load_times: Dict[str, float] = {}
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._usage_counts: Dict[str, int] = defaultdict(int)
        self._last_used: Dict[str, float] = {}
        self._reload_counts: Dict[str, int] = defaultdict(int)

        # Install/uninstall tracking
        self._installed: Dict[str, float] = {}
        self._uninstalled: Dict[str, float] = {}

        # Session start
        self._session_start = time.time()

    def record_load(self, name: str, load_time: float) -> None:
        """Record extension load time."""
        self._load_times[name] = load_time

    def record_error(self, name: str) -> None:
        """Record an error for an extension."""
        self._error_counts[name] += 1

    def record_usage(self, name: str) -> None:
        """Record that an extension was used."""
        self._usage_counts[name] += 1
        self._last_used[name] = time.time()

    def record_reload(self, name: str) -> None:
        """Record an extension reload."""
        self._reload_counts[name] += 1

    def record_install(self, name: str) -> None:
        """Record an extension installation."""
        self._installed[name] = time.time()

    def record_uninstall(self, name: str) -> None:
        """Record an extension uninstallation."""
        self._uninstalled[name] = time.time()

    def get_extension_metrics(self, name: str) -> Dict[str, Any]:
        """
        Get metrics for a specific extension.

        Args:
            name: Extension name

        Returns:
            Metrics dict
        """
        return {
            "name": name,
            "load_time": self._load_times.get(name),
            "error_count": self._error_counts.get(name, 0),
            "usage_count": self._usage_counts.get(name, 0),
            "reload_count": self._reload_counts.get(name, 0),
            "last_used": self._last_used.get(name),
            "last_used_timestamp": datetime.fromtimestamp(self._last_used[name]).isoformat() if name in self._last_used else None,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.

        Returns:
            Complete metrics dict
        """
        all_names = set()
        all_names.update(self._load_times.keys())
        all_names.update(self._error_counts.keys())
        all_names.update(self._usage_counts.keys())

        return {
            "session_start": datetime.fromtimestamp(self._session_start).isoformat(),
            "uptime_seconds": time.time() - self._session_start,
            "extensions": {
                name: self.get_extension_metrics(name)
                for name in all_names
            },
            "summary": self._get_summary(),
        }

    def _get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_errors = sum(self._error_counts.values())
        total_usage = sum(self._usage_counts.values())
        total_reloads = sum(self._reload_counts.values())

        return {
            "total_extensions": len(self._load_times),
            "total_errors": total_errors,
            "total_usage": total_usage,
            "total_reloads": total_reloads,
            "extensions_with_errors": len([k for k, v in self._error_counts.items() if v > 0]),
            "most_used": self._get_most_used(),
        }

    def _get_most_used(self) -> list:
        """Get most used extensions."""
        sorted_usage = sorted(
            self._usage_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"name": name, "count": count}
            for name, count in sorted_usage[:10]
        ]

    def reset_extension(self, name: str) -> None:
        """Reset metrics for an extension."""
        self._error_counts.pop(name, None)
        self._usage_counts.pop(name, None)
        self._reload_counts.pop(name, None)
        self._last_used.pop(name, None)

    def reset_all(self) -> None:
        """Reset all metrics."""
        self._load_times.clear()
        self._error_counts.clear()
        self._usage_counts.clear()
        self._last_used.clear()
        self._reload_counts.clear()
