"""Centralized Python Path Setup Utilities.

Provides unified sys.path management for the backend and extensions.
Consolidates duplicate sys.path.insert patterns from 40+ files.

Usage:
    from src.shared.utils.path_utils import setup_python_path, setup_extension_path

    # For backend paths
    setup_python_path("/path/to/module")

    # For extension paths (ComfyUI-style)
    setup_extension_path("/path/to/extension")
"""

import sys
from pathlib import Path
from typing import Optional


def setup_python_path(
    path: str | Path,
    index: int = 0,
    quiet: bool = False,
) -> bool:
    """
    Add a path to sys.path if not already present.

    This is the centralized utility for all sys.path manipulations
    across the codebase, replacing 40+ duplicate patterns.

    Args:
        path: Path to add to sys.path
        index: Position to insert at (default: 0 for highest priority)
        quiet: If True, don't log the addition

    Returns:
        True if path was added, False if already present

    Examples:
        # Add parent directory to import from sibling modules
        setup_python_path(str(Path(__file__).parent.parent))

        # Add extension directory for ComfyUI-style loading
        setup_python_path(extension_dir)
    """
    path_str = str(path)

    if path_str in sys.path:
        return False

    sys.path.insert(index, path_str)

    if not quiet:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Added {path_str} to sys.path at position {index}")

    return True


def setup_extension_path(
    extension_dir: str | Path,
    add_parent: bool = True,
    quiet: bool = False,
) -> tuple[bool, bool]:
    """
    Set up sys.path for ComfyUI-style extension loading.

    Adds both the extension directory and its parent to sys.path,
    following the ComfyUI extension loading pattern.

    Args:
        extension_dir: Path to the extension directory
        add_parent: Whether to also add the parent directory (default: True)
        quiet: If True, don't log the additions

    Returns:
        Tuple of (parent_added, extension_added) booleans

    Examples:
        # In extension loading code
        setup_extension_path(extension_path)
    """
    extension_path = Path(extension_dir)

    if add_parent:
        extensions_parent = str(extension_path.parent)
        parent_added = setup_python_path(extensions_parent, quiet=quiet)
    else:
        parent_added = False

    extension_added = setup_python_path(str(extension_path), quiet=quiet)

    return parent_added, extension_added


def restore_python_path(original_path: list[str]) -> None:
    """
    Restore sys.path to a previous state.

    Useful for cleanup after extension loading.

    Args:
        original_path: The original sys.path to restore

    Examples:
        # Store original path
        original = sys.path[:]

        # Do some path manipulations
        setup_python_path("/some/path")

        # Restore when done
        restore_python_path(original)
    """
    sys.path[:] = original_path


def remove_from_python_path(path: str | Path) -> bool:
    """
    Remove a path from sys.path if present.

    Args:
        path: Path to remove from sys.path

    Returns:
        True if path was removed, False if not found

    Examples:
        # Clean up after extension loading
        remove_from_python_path(extension_dir)
    """
    path_str = str(path)

    if path_str in sys.path:
        sys.path.remove(path_str)
        return True

    return False


def get_python_path() -> list[str]:
    """
    Get a copy of the current sys.path.

    Useful for storing the current state before modifications.

    Returns:
        Copy of sys.path as a list

    Examples:
        # Store current state
        saved_path = get_python_path()

        # Make modifications
        setup_python_path("/new/path")

        # Later restore
        restore_python_path(saved_path)
    """
    return sys.path[:]


def is_in_python_path(path: str | Path) -> bool:
    """
    Check if a path is in sys.path.

    Args:
        path: Path to check

    Returns:
        True if path is in sys.path
    """
    return str(path) in sys.path


def setup_backend_paths(
    backend_dir: Optional[str | Path] = None,
    add_src: bool = True,
    add_vendor: bool = True,
) -> list[str]:
    """
    Set up common backend paths for imports.

    Adds frequently used backend directories to sys.path:
    - backend/src (for src.* imports)
    - backend/vendor (for ComfyUI and other vendored code)

    Args:
        backend_dir: Backend directory path (auto-detected if None)
        add_src: Whether to add src directory
        add_vendor: Whether to add vendor directory

    Returns:
        List of paths that were added (empty if already present)

    Examples:
        # At the start of main.py or similar entry points
        setup_backend_paths()
    """
    if backend_dir is None:
        # Auto-detect: this file is at backend/src/shared/utils/path_utils.py
        backend_dir = Path(__file__).resolve().parent.parent.parent.parent
    else:
        backend_dir = Path(backend_dir)

    added = []

    if add_src:
        src_dir = backend_dir / "src"
        if setup_python_path(src_dir):
            added.append(str(src_dir))

    if add_vendor:
        vendor_dir = backend_dir / "vendor"
        if setup_python_path(vendor_dir):
            added.append(str(vendor_dir))

    return added


def setup_backend_dir_from_file(
    file_path: str | Path,
    parent_levels: int = 6,
    quiet: bool = False,
) -> bool:
    """
    Add backend directory to sys.path based on current file location.

    This replaces the common pattern:
        backend_dir = Path(__file__).resolve().parent.parent.parent...
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

    Args:
        file_path: Usually __file__ from the calling module
        parent_levels: Number of parent directories to go up to reach backend
                      (default: 6 for deep_research graph nodes/helpers)
        quiet: If True, don't log the addition

    Returns:
        True if path was added, False if already present

    Examples:
        # In deep_research/graph/nodes/common.py
        setup_backend_dir_from_file(__file__)

        # In deep_research/graph/helpers/boss_helpers.py
        setup_backend_dir_from_file(__file__, parent_levels=4)

        # In deep_research/graph/helpers/deep_agent_helpers.py
        setup_backend_dir_from_file(__file__, parent_levels=5)
    """
    backend_dir = Path(file_path).resolve().parents[parent_levels - 1]
    return setup_python_path(backend_dir, quiet=quiet)


__all__ = [
    "setup_python_path",
    "setup_extension_path",
    "restore_python_path",
    "remove_from_python_path",
    "get_python_path",
    "is_in_python_path",
    "setup_backend_paths",
    "setup_backend_dir_from_file",
]
