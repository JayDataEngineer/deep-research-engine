"""
ComfyUI Extension Factory - DRY solution for ComfyUI workflow extensions.

This module provides a generic factory that reads extension config.yaml files
and auto-generates extension classes, eliminating code duplication across ComfyUI extensions.

All functionality is inherited from BaseComfyUIExtension - this factory just
creates minimal wrapper classes.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.extensions.libraries.comfyui.base import BaseComfyUIExtension

logger = logging.getLogger(__name__)


class ComfyUIExtensionFactory:
    """
    Factory for creating ComfyUI extension classes from config.yaml.

    All extensions created by this factory:
    - Extend BaseComfyUIExtension (which extends ServiceExtension)
    - Connect to the external ComfyUI server
    - Use the shared ComfyUIEngine for workflow execution
    - Support both "text_to_image" and "image_edit" workflow types

    The factory creates minimal wrapper classes - all functionality is
    provided by BaseComfyUIExtension.
    """

    @staticmethod
    def create_from_directory(ext_dir: Path) -> BaseComfyUIExtension:
        """
        Create a ComfyUI extension instance from its config.yaml.

        This reads the extension's config.yaml file and creates an extension
        class that extends BaseComfyUIExtension. All functionality is inherited
        from the base class - no additional code is needed.

        Usage in extension directory:
            # extension.py
            from src.extensions.libraries.comfyui.factory import create_extension_from_config
            from pathlib import Path

            def register_extension():
                return create_extension_from_config(Path(__file__).parent)

        Args:
            ext_dir: Path to the extension directory

        Returns:
            An instantiated BaseComfyUIExtension subclass

        Raises:
            FileNotFoundError: If config.yaml is not found
            ValueError: If required configuration is missing
        """
        config_file = ext_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.yaml found in {ext_dir}")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Validate extension configuration
        ext_config = config.get("extension", {})
        name = ext_config.get("name")
        if not name:
            raise ValueError(f"No extension name in config.yaml")

        # Create a minimal subclass of BaseComfyUIExtension
        # The subclass exists only for naming/type identification
        parts = name.replace("_", " ").title().split()
        cls_name = f"{''.join(parts)}Extension"

        # Create the class with minimal overhead
        # All functionality is inherited from BaseComfyUIExtension
        extension_class = type(cls_name, (BaseComfyUIExtension,), {
            "__module__": f"comfyui.{name}",
        })

        # Instantiate and return
        return extension_class(config, ext_dir)


def create_extension_from_config(ext_dir: Path) -> BaseComfyUIExtension:
    """
    Convenience function to create an extension from its config.yaml.

    Usage in extension directory:
        # extension.py
        from src.extensions.libraries.comfyui.factory import create_extension_from_config
        from pathlib import Path

        def register_extension():
            return create_extension_from_config(Path(__file__).parent)

    Args:
        ext_dir: Path to the extension directory

    Returns:
        An instantiated BaseComfyUIExtension subclass
    """
    return ComfyUIExtensionFactory.create_from_directory(ext_dir)
