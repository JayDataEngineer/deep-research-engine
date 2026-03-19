"""
Extension Discoverer

Scans the filesystem for extension directories and loads their configs.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import yaml

from ..base import BaseExtension
from ..config import validate_config, get_missing_fields


def load_yaml_config(config_path: Path, required: bool = True) -> Optional[Dict[str, Any]]:
    """Load YAML config file with optional requirement.

    Args:
        config_path: Path to YAML file
        required: If True, return None for missing files (default: True)

    Returns:
        Parsed config dict or None if error/not found
    """
    if not config_path.exists():
        if required:
            return None
        return {}

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def is_extension_enabled(config: Dict[str, Any]) -> bool:
    """Check if extension is enabled based on its config.

    Args:
        config: Extension config dictionary

    Returns:
        True if extension is enabled, False otherwise
    """
    extension_config = config.get("extension", {})
    return extension_config.get("enabled", True)


class ExtensionDiscoverer:
    """
    Discovers extensions in the filesystem.
    """

    def __init__(self, extensions_dir: Path):
        """
        Initialize the discoverer.

        Args:
            extensions_dir: Root directory containing extension categories
        """
        self.extensions_dir = extensions_dir

    async def scan(self) -> List[Tuple[Dict[str, Any], Path]]:
        """
        Scan for all extensions.

        Returns:
            List of (config, directory_path) tuples for discovered extensions
        """
        discovered = []

        for category_dir in self.extensions_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'core':
                continue

            # Skip non-category directories
            if category_dir.name.startswith('_'):
                continue

            # Find all extension directories in this category
            for ext_dir in self._find_extension_dirs(category_dir):
                config_path = ext_dir / "config.yaml"
                if not config_path.exists():
                    continue

                config = self._load_config(config_path)
                if not config:
                    continue

                # Must have 'extension' key
                if 'extension' not in config:
                    continue

                # Skip disabled extensions - use shared utility
                if not is_extension_enabled(config):
                    ext_name = config['extension']['name']
                    print(f"[Discoverer] ⏭️  Skipping disabled: {ext_name}")
                    continue

                ext_name = config['extension']['name']
                category = config['extension'].get('category', 'unknown')
                print(f"[Discoverer] Found: {ext_name} ({category})")

                discovered.append((config, ext_dir))

        return discovered

    def _find_extension_dirs(self, base_dir: Path) -> List[Path]:
        """
        Recursively find all directories containing config.yaml.

        Args:
            base_dir: Base directory to search

        Returns:
            List of directories containing config.yaml
        """
        extension_dirs = []
        for item in base_dir.rglob("*"):
            if item.is_dir():
                config_path = item / "config.yaml"
                if config_path.exists():
                    extension_dirs.append(item)
        return extension_dirs

    def _load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse YAML config file with validation.

        Args:
            config_path: Path to config.yaml

        Returns:
            Parsed config dict or None if error
        """
        try:
            # Use shared config utility instead of manual yaml.safe_load
            config = load_yaml_config(config_path, required=False)

            if not config:
                print(f"[Discoverer] ⚠️  Empty config: {config_path}")
                return None

            # Validate schema
            is_valid, validated_schema, errors = validate_config(config_path)

            if not is_valid:
                ext_name = config.get('extension', {}).get('name', config_path.parent.name)
                print(f"[Discoverer] ❌ Config validation failed for '{ext_name}':")
                for error in errors:
                    print(f"[Discoverer]    - {error}")
                # Still return config if it has basic required fields
                if 'extension' in config and 'name' in config.get('extension', {}):
                    return config
                return None

            # Check for missing recommended fields
            missing = get_missing_fields(config_path)
            if missing:
                ext_name = config.get('extension', {}).get('name', config_path.parent.name)
                print(f"[Discoverer] ⚠️  '{ext_name}' missing recommended fields: {', '.join(missing)}")

            return config

        except Exception as e:
            print(f"[Discoverer] Error loading {config_path}: {e}")
            return None

    def get_extension_path(self, name: str) -> Optional[Path]:
        """
        Get the directory path for an extension by name.

        Args:
            name: Extension name

        Returns:
            Path to extension directory or None
        """
        # Search for extension with matching name in config
        for category_dir in self.extensions_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'core':
                continue

            for ext_dir in self._find_extension_dirs(category_dir):
                config_path = ext_dir / "config.yaml"
                if config_path.exists():
                    config = self._load_config(config_path)
                    if config and config.get('extension', {}).get('name') == name:
                        return ext_dir

        return None

    def list_all_names(self) -> List[str]:
        """
        List all extension names (including disabled).

        Returns:
            List of extension names
        """
        names = []
        for category_dir in self.extensions_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'core':
                continue

            for ext_dir in self._find_extension_dirs(category_dir):
                config_path = ext_dir / "config.yaml"
                if config_path.exists():
                    config = self._load_config(config_path)
                    if config and 'extension' in config:
                        names.append(config['extension']['name'])

        return names
