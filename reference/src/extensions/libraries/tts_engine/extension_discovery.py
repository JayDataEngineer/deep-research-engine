"""
Extension discovery system for TTS extensions.

⚠️ DEPRECATED: This module is for ComfyUI-style discovery (extension.py with
register_extension). New extensions should use config.yaml with the core
extension system in src.extensions.core.

This module is kept for backward compatibility only. The core ExtensionManager
in src.extensions.core.manager provides the same functionality with better
integration and support.

Migration guide:
- Instead of extension.py with register_extension(), use config.yaml
- Use ExtensionManager from src.extensions.core.manager instead
- See src.extensions.core.factory for extension creation utilities
"""

import importlib
import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

from src.extensions.core.base import BaseExtension, TTSExtensionInterface

logger = logging.getLogger(__name__)


def _deprecated_warning():
    """Issue deprecation warning."""
    warnings.warn(
        "TTS extension_discovery is deprecated. "
        "Use ExtensionManager from src.extensions.core.manager instead. "
        "See src.extensions.core.factory for extension creation utilities.",
        DeprecationWarning,
        stacklevel=3
    )


async def discover_extensions_from_path(extensions_dir: str) -> dict[str, type[BaseExtension]]:
    """
    Discover extensions from a given directory path.

    ⚠️ DEPRECATED: Use ExtensionManager from src.extensions.core.manager instead.

    Args:
        extensions_dir: Path to directory containing extension folders

    Returns:
        Dictionary mapping extension names to extension classes
    """
    _deprecated_warning()
    discovery = ExtensionDiscovery()
    return await discovery.discover_extensions(extensions_dir)


class ExtensionDiscovery:
    """System for discovering and loading ComfyUI-style TTS extensions."""

    def __init__(self):
        """Initialize the extension discovery system."""
        self.discovered_extensions: dict[str, type[BaseExtension]] = {}
        self.loaded_extensions: dict[str, BaseExtension] = {}
        self.failed_extensions: dict[str, str] = {}  # extension_name -> error_message

    async def discover_extensions(self, extensions_dir: str) -> dict[str, type[BaseExtension]]:
        """Discover and load all extensions from a directory.

        Args:
            extensions_dir: Path to extensions directory

        Returns:
            Dictionary of extension_name -> extension_class
        """
        extensions = {}
        extensions_path = Path(extensions_dir)

        if not extensions_path.exists():
            logger.warning(f"Extensions directory does not exist: {extensions_path}")
            return extensions

        logger.info(f"Discovering extensions in: {extensions_path}")

        for item in extensions_path.iterdir():
            if (
                not item.is_dir()
                or item.name.startswith(".")
                or item.name.startswith("__")
            ):
                continue

            try:
                extension = await self._load_extension_from_folder(item)
                if extension:
                    extensions[extension.get_name()] = extension
                    logger.info(f"Discovered extension: {extension.get_name()}")
            except Exception as e:
                logger.error(f"Failed to load extension from {item.name}: {str(e)}")

        logger.info(f"Discovered {len(extensions)} extensions from {extensions_path}")
        return extensions

    async def _load_extension_from_folder(self, folder_path: Path) -> BaseExtension | None:
        """Load extension from a folder following ComfyUI pattern.

        Args:
            folder_path: Path to extension folder

        Returns:
            Extension instance or None if failed
        """
        folder_name = folder_path.name

        # Add the extension's directory to sys.path temporarily
        original_path = sys.path[:]
        try:
            # Insert the extension's directory at the beginning of sys.path
            ext_dir = str(folder_path)
            if ext_dir not in sys.path:
                sys.path.insert(0, ext_dir)

            # Try to load __init__.py first
            init_path = folder_path / "__init__.py"
            if init_path.exists():
                module_name = f"extension_{folder_name}"
                spec = importlib.util.spec_from_file_location(module_name, init_path)
                module = importlib.util.module_from_spec(spec)

                # Add the module to sys.modules before exec_module
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Look for register_extension function
                if hasattr(module, "register_extension"):
                    register_func = module.register_extension
                    extension_instance = register_func()
                    return extension_instance

            # Try to load extension.py
            extension_path = folder_path / "extension.py"
            if extension_path.exists():
                module_name = f"extension_{folder_name}"
                spec = importlib.util.spec_from_file_location(
                    module_name, extension_path
                )
                module = importlib.util.module_from_spec(spec)

                # Add the module to sys.modules before exec_module
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Look for register_extension function
                if hasattr(module, "register_extension"):
                    register_func = module.register_extension
                    extension_instance = register_func()
                    return extension_instance


            logger.warning(f"No valid extension entry point found in {folder_name}")
            return None

        except Exception as e:
            logger.error(f"Error loading extension from {folder_name}: {str(e)}")
            return None
        finally:
            # Restore the original sys.path
            sys.path[:] = original_path
            # Clean up temporary modules
            module_name = f"extension_{folder_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

    async def discover_from_multiple_paths(
        self, extension_paths: list[str]
    ) -> dict[str, type[BaseExtension]]:
        """Discover extensions from multiple paths.

        Args:
            extension_paths: List of extension directory paths

        Returns:
            Dictionary of extension_name -> extension_class
        """
        all_extensions = {}

        for path in extension_paths:
            extensions = await self.discover_extensions(path)
            all_extensions.update(extensions)

        return all_extensions

    async def load_extensions(
        self, extension_paths: list[str], auto_load: bool = False
    ) -> dict[str, BaseExtension]:
        """Discover and optionally load extensions.

        Args:
            extension_paths: List of extension directory paths
            auto_load: Whether to automatically load discovered extensions

        Returns:
            Dictionary of extension_name -> extension_instance
        """
        # Discover extensions
        discovered = await self.discover_from_multiple_paths(extension_paths)
        self.discovered_extensions = discovered

        # Load extensions if requested
        if auto_load:
            for name, extension_class in discovered.items():
                try:
                    extension = extension_class()
                    if extension.load():
                        self.loaded_extensions[name] = extension
                        logger.info(f"Loaded extension: {name}")
                    else:
                        self.failed_extensions[name] = "Failed to load extension"
                        logger.warning(f"Failed to load extension: {name}")
                except Exception as e:
                    self.failed_extensions[name] = str(e)
                    logger.error(f"Error loading extension {name}: {str(e)}")

        return self.loaded_extensions

    def get_extension(self, name: str) -> BaseExtension | None:
        """Get a loaded extension by name.

        Args:
            name: Extension name

        Returns:
            Extension instance or None if not found
        """
        return self.loaded_extensions.get(name)

    def get_all_extensions(self) -> dict[str, BaseExtension]:
        """Get all loaded extensions.

        Returns:
            Dictionary of extension_name -> extension_instance
        """
        return self.loaded_extensions

    def get_discovered_extensions(self) -> dict[str, type[BaseExtension]]:
        """Get all discovered extensions (including failed ones).

        Returns:
            Dictionary of extension_name -> extension_class
        """
        return self.discovered_extensions

    def get_failed_extensions(self) -> dict[str, str]:
        """Get failed extensions and their error messages.

        Returns:
            Dictionary of extension_name -> error_message
        """
        return self.failed_extensions

    def reload_extension(self, name: str) -> bool:
        """Reload a specific extension.

        Args:
            name: Extension name to reload

        Returns:
            True if successful
        """
        if name in self.loaded_extensions:
            try:
                # Unload first
                extension = self.loaded_extensions[name]
                extension.unload()
                del self.loaded_extensions[name]

                # Remove from failed if it was there
                if name in self.failed_extensions:
                    del self.failed_extensions[name]

                # Reload
                if name in self.discovered_extensions:
                    extension_class = self.discovered_extensions[name]
                    new_extension = extension_class()
                    if new_extension.load():
                        self.loaded_extensions[name] = new_extension
                        logger.info(f"Reloaded extension: {name}")
                        return True
                    else:
                        self.failed_extensions[name] = "Failed to reload extension"
                        return False

            except Exception as e:
                self.failed_extensions[name] = str(e)
                logger.error(f"Error reloading extension {name}: {str(e)}")
                return False

        return False

    def unload_extension(self, name: str) -> bool:
        """Unload a specific extension.

        Args:
            name: Extension name to unload

        Returns:
            True if successful
        """
        if name in self.loaded_extensions:
            try:
                extension = self.loaded_extensions[name]
                extension.unload()
                del self.loaded_extensions[name]
                logger.info(f"Unloaded extension: {name}")
                return True
            except Exception as e:
                logger.error(f"Error unloading extension {name}: {str(e)}")
                return False

        return False

    def get_extension_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all extensions.

        Returns:
            Dictionary with extension information
        """
        info = {"loaded": {}, "discovered": {}, "failed": self.failed_extensions}

        # Loaded extensions info
        for name, extension in self.loaded_extensions.items():
            info["loaded"][name] = extension.get_info()

        # Discovered extensions info
        for name in self.discovered_extensions:
            if name not in self.loaded_extensions:
                info["discovered"][name] = {
                    "name": name,
                    "status": "discovered but not loaded",
                }

        return info


# Global extension discovery instance
extension_discovery = ExtensionDiscovery()
