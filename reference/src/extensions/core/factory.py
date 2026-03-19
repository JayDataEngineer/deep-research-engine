"""Extension Factory

Centralized factory for creating extension instances.
Eliminates duplicate register_extension() boilerplate across 35+ extensions.
"""

import logging
from pathlib import Path
from typing import Type, TypeVar, Callable, Optional

from src.extensions.core.manager.discover import load_yaml_config as get_extension_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseExtension")


def register_extension_factory(
    extension_class: Type[T],
    config_name: str = "config.yaml",
    validate: bool = True,
) -> Callable[[], T]:
    """
    Factory function that creates a register_extension function for an extension.

    This eliminates the 6-line boilerplate that was duplicated in 35+ extensions:

    Before (in every extension.py):
    ```python
    def register_extension():
        import yaml
        from pathlib import Path
        root_path = Path(__file__).parent
        config_path = root_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return MyExtension(config, root_path)
    ```

    After (in every extension.py):
    ```python
    from src.extensions.core.factory import register_extension_factory
    register_extension = register_extension_factory(MyExtension)
    ```

    Args:
        extension_class: The extension class to instantiate
        config_name: Name of the config file (default: "config.yaml")
        validate: Whether to validate the config structure

    Returns:
        A register_extension function that can be called by the extension loader
    """
    def register_extension() -> T:
        # Get the root path from the extension class's module
        # This assumes the extension class is defined in extension.py
        # and the config is in the same directory
        import inspect
        module_file = Path(inspect.getfile(extension_class))
        root_path = module_file.parent

        # Load config using centralized utility
        config = get_extension_config(root_path / config_name, required=True)

        # Create and return extension instance
        return extension_class(config, root_path)

    return register_extension


def create_extension(
    extension_class: Type[T],
    root_path: Path,
    config: Optional[dict] = None,
    config_name: str = "config.yaml",
) -> T:
    """
    Direct factory function to create an extension instance.

    Use this when you need to create an extension programmatically
    rather than through the register_extension pattern.

    Args:
        extension_class: The extension class to instantiate
        root_path: Path to the extension directory
        config: Optional pre-loaded config (if None, loads from config_name)
        config_name: Name of config file to load if config is not provided

    Returns:
        Extension instance
    """
    if config is None:
        config = get_extension_config(root_path / config_name, required=True)

    return extension_class(config, root_path)


__all__ = [
    "register_extension_factory",
    "create_extension",
]
