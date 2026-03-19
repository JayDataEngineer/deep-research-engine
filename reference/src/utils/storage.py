"""Storage Utilities

Common storage operations for all features.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class StorageManager:
    """Common storage operations across all features."""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if needed.

        Args:
            path: Directory path

        Returns:
            The path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def load_json_file(
        file_path: Path,
        item_class: Optional[Type] = None,
        default: Any = None,
    ) -> Any:
        """Load JSON file with error handling.

        Args:
            file_path: Path to JSON file
            item_class: Optional class to deserialize items
            default: Default value if file doesn't exist

        Returns:
            Loaded data or default
        """
        if not file_path.exists():
            return default if default is not None else []

        try:
            with open(file_path) as f:
                data = json.load(f)

            if item_class and data:
                if isinstance(data, list):
                    return [item_class.from_dict(item) for item in data]
                elif isinstance(data, dict):
                    return {k: item_class.from_dict(v) for k, v in data.items()}

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {e}")
            return default if default is not None else []
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return default if default is not None else []

    @staticmethod
    def save_json_file(file_path: Path, data: Any, indent: int = 2) -> bool:
        """Save data to JSON file.

        Args:
            file_path: Path to save to
            data: Data to save (list or dict)
            indent: JSON indentation

        Returns:
            True if successful
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert items with to_dict() method
            if isinstance(data, list):
                data = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in data
                ]
            elif isinstance(data, dict):
                data = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in data.items()
                }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=indent)

            return True

        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            return False

    @staticmethod
    def append_to_json_file(file_path: Path, item: Any) -> bool:
        """Append an item to a JSON file (list-based).

        Args:
            file_path: Path to JSON file
            item: Item to append

        Returns:
            True if successful
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing data
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
            else:
                data = []

            # Append item
            item_data = item.to_dict() if hasattr(item, 'to_dict') else item
            data.append(item_data)

            # Save
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to append to {file_path}: {e}")
            return False

    @staticmethod
    def generate_id_from_name(name: str) -> str:
        """Generate a safe ID from a name.

        Args:
            name: Name to convert

        Returns:
            Safe ID string (lowercase, spaces and slashes replaced with dashes)
        """
        return name.lower().replace(" ", "-").replace("/", "-").strip("-")

    @staticmethod
    def get_project_subdir(*path_parts: str) -> Path:
        """Get a path under the project data directory.

        Args:
            *path_parts: Path components

        Returns:
            Path under project data directory
        """
        from src.constants import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data" / "inputs"
        return data_dir.joinpath(*path_parts)

    @staticmethod
    def clean_old_files(
        directory: Path,
        pattern: str = "*.json",
        days: int = 30,
    ) -> int:
        """Remove files older than specified days.

        Args:
            directory: Directory to clean
            pattern: File pattern to match
            days: Age threshold in days

        Returns:
            Number of files removed
        """
        from datetime import datetime, timedelta

        if not directory.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        file_path.unlink()
                        removed += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")

        return removed


class AsyncStorageManager:
    """Async version of StorageManager for async operations."""

    @staticmethod
    async def load_json_file_async(
        file_path: Path,
        item_class: Optional[Type] = None,
        default: Any = None,
    ) -> Any:
        """Load JSON file asynchronously.

        Args:
            file_path: Path to JSON file
            item_class: Optional class to deserialize items
            default: Default value if file doesn't exist

        Returns:
            Loaded data or default
        """
        import aiofiles

        if not file_path.exists():
            return default if default is not None else []

        try:
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                data = json.loads(content)

            if item_class and data:
                if isinstance(data, list):
                    return [item_class.from_dict(item) for item in data]
                elif isinstance(data, dict):
                    return {k: item_class.from_dict(v) for k, v in data.items()}

            return data

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return default if default is not None else []

    @staticmethod
    async def save_json_file_async(
        file_path: Path,
        data: Any,
        indent: int = 2,
    ) -> bool:
        """Save data to JSON file asynchronously.

        Args:
            file_path: Path to save to
            data: Data to save
            indent: JSON indentation

        Returns:
            True if successful
        """
        import aiofiles

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert items with to_dict() method
            if isinstance(data, list):
                data = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in data
                ]
            elif isinstance(data, dict):
                data = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in data.items()
                }

            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=indent))

            return True

        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")
            return False
