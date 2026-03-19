"""Stats Manager

Tracks usage statistics for the backend.
Similar to SillyTavern's stats endpoint.
Refactored to use BaseManager infrastructure.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extensions.libraries.base.manager_base import BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


class StatsManager(BaseManager):
    """Manages usage statistics tracking using BaseManager infrastructure.

    Uses a dual-file structure:
    - usage_stats.json - overall statistics
    - daily_stats.json - daily breakdown
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the stats manager.

        Args:
            base_dir: Base directory for stats storage
        """
        super().__init__(base_dir=base_dir, feature_name="stats", storage_file="usage_stats.json")

        # Separate file for daily stats
        self.daily_file = self.base_dir / "daily_stats.json"
        self._daily_stats: Dict[str, Dict[str, Any]] = {}
        self._load_daily_stats()

    def _get_item_class(self):
        """StatsManager doesn't use item_class."""
        return None

    def _load(self):
        """Load overall stats from storage."""
        data = StorageManager.load_json_file(self.storage_file)
        if not data:
            data = self._get_default_stats()
        self._items = data if isinstance(data, dict) else data
        logger.info(f"Stats Manager initialized")

    def _load_daily_stats(self):
        """Load daily stats from storage."""
        if self.daily_file.exists():
            try:
                self._daily_stats = StorageManager.load_json_file(self.daily_file)
            except Exception as e:
                logger.warning(f"Failed to load daily stats: {e}")
                self._daily_stats = {}
        else:
            self._daily_stats = {}

    def _get_default_stats(self) -> Dict[str, Any]:
        """Get default stats structure."""
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_characters": 0,
            "start_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "models": {},
            "extensions": {},
            "features": {},
            "characters": {},
        }

    def _save(self):
        """Save both overall and daily stats to storage."""
        # Convert defaultdicts to dicts if any
        stats_to_save = {
            **self._items,
            "models": dict(self._items.get("models", {})),
            "extensions": dict(self._items.get("extensions", {})),
            "features": dict(self._items.get("features", {})),
            "characters": dict(self._items.get("characters", {})),
        }

        StorageManager.save_json_file(self.storage_file, stats_to_save)
        StorageManager.save_json_file(self.daily_file, self._daily_stats)

    def _get_today_key(self) -> str:
        """Get today's date key."""
        return datetime.now().strftime("%Y-%m-%d")

    def _ensure_stats_structure(self):
        """Ensure stats has default structure."""
        if "total_requests" not in self._items:
            self._items.update(self._get_default_stats())

    def _ensure_daily_structure(self, today: str):
        """Ensure daily stats has default structure."""
        if today not in self._daily_stats:
            self._daily_stats[today] = {
                "requests": 0,
                "tokens": 0,
                "models": {},
                "features": {},
            }

    # ========================================================================
    # Recording Stats
    # ========================================================================

    def record_request(
        self,
        model: str,
        tokens: int = 0,
        feature: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """Record a request.

        Args:
            model: Model name used
            tokens: Number of tokens processed
            feature: Feature used (optional)
            extension: Extension used (optional)
        """
        today = self._get_today_key()

        self._ensure_stats_structure()
        self._ensure_daily_structure(today)

        # Update overall stats
        self._items["total_requests"] += 1
        self._items["total_tokens"] += tokens
        self._items["last_updated"] = datetime.now().isoformat()

        if model:
            if "models" not in self._items:
                self._items["models"] = {}
            self._items["models"][model] = self._items["models"].get(model, 0) + 1

        if feature:
            if "features" not in self._items:
                self._items["features"] = {}
            self._items["features"][feature] = self._items["features"].get(feature, 0) + 1

        if extension:
            if "extensions" not in self._items:
                self._items["extensions"] = {}
            self._items["extensions"][extension] = self._items["extensions"].get(extension, 0) + 1

        # Update daily stats
        self._daily_stats[today]["requests"] += 1
        self._daily_stats[today]["tokens"] += tokens

        if model:
            self._daily_stats[today]["models"][model] = \
                self._daily_stats[today]["models"].get(model, 0) + 1

        if feature:
            self._daily_stats[today]["features"][feature] = \
                self._daily_stats[today]["features"].get(feature, 0) + 1

        self._save()

    def record_tokens(self, count: int, model: str = ""):
        """Record token usage.

        Args:
            count: Number of tokens
            model: Model name (optional)
        """
        today = self._get_today_key()

        self._ensure_stats_structure()
        self._ensure_daily_structure(today)

        self._items["total_tokens"] += count
        self._items["last_updated"] = datetime.now().isoformat()

        self._daily_stats[today]["tokens"] += count

        self._save()

    def record_character_usage(self, character_name: str):
        """Record character interaction.

        Args:
            character_name: Name of character
        """
        self._ensure_stats_structure()

        if "characters" not in self._items:
            self._items["characters"] = {}

        self._items["characters"][character_name] = \
            self._items["characters"].get(character_name, 0) + 1

        self._items["total_characters"] = \
            self._items.get("total_characters", 0) + 1

        self._save()

    # ========================================================================
    # Retrieving Stats
    # ========================================================================

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall usage statistics."""
        return {
            **self._items,
            "models": dict(self._items.get("models", {})),
            "extensions": dict(self._items.get("extensions", {})),
            "features": dict(self._items.get("features", {})),
            "characters": dict(self._items.get("characters", {})),
        }

    def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get stats for recent days.

        Args:
            days: Number of days to include

        Returns:
            List of daily stats
        """
        result = []
        cutoff = datetime.now() - timedelta(days=days)

        for date_key in sorted(self._daily_stats.keys(), reverse=True):
            try:
                date_obj = datetime.strptime(date_key, "%Y-%m-%d")
                if date_obj >= cutoff:
                    result.append({
                        "date": date_key,
                        **self._daily_stats[date_key],
                    })
            except ValueError:
                continue

        return result

    def get_model_stats(self) -> Dict[str, int]:
        """Get model usage statistics."""
        return dict(self._items.get("models", {}))

    def get_feature_stats(self) -> Dict[str, int]:
        """Get feature usage statistics."""
        return dict(self._items.get("features", {}))

    def get_character_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most used characters.

        Args:
            limit: Max number of characters

        Returns:
            List of character stats
        """
        characters = self._items.get("characters", {})
        sorted_chars = sorted(characters.items(), key=lambda x: x[1], reverse=True)

        return [
            {"name": name, "count": count}
            for name, count in sorted_chars[:limit]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key statistics.

        Returns:
            Summary dict
        """
        uptime = datetime.now() - datetime.fromisoformat(
            self._items.get("start_date", datetime.now().isoformat())
        )

        return {
            "total_requests": self._items.get("total_requests", 0),
            "total_tokens": self._items.get("total_tokens", 0),
            "total_characters": self._items.get("total_characters", 0),
            "uptime_days": uptime.days,
            "top_models": self.get_model_stats(),
            "top_features": self.get_feature_stats(),
            "recent_daily": self.get_daily_stats(days=7),
        }

    # ========================================================================
    # Management
    # ========================================================================

    def reset_stats(self):
        """Reset all statistics."""
        self._items = self._get_default_stats()
        self._daily_stats = {}
        self._save()

    def export_stats(self, output_path: str):
        """Export stats to a file.

        Args:
            output_path: Path to export file
        """
        data = {
            "overall": self.get_overall_stats(),
            "daily": self.get_daily_stats(days=365),
            "exported_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Stats exported to {output_path}")
