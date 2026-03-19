"""CLI Plugin Discovery System

Scans extension directories recursively for CLI plugins (cli.py files) and loads them.
Features caching for performance optimization.
"""

import importlib
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple
import sys
import json

logger = logging.getLogger(__name__)

# Cache file location
_CACHE_DIR = Path.home() / ".cache" / "unified-llm-interface"
_CACHE_FILE = _CACHE_DIR / "cli_plugins_cache.json"


class CLIPlugin:
    """A CLI plugin from an extension."""

    def __init__(self, name: str, module_path: str, register_func: Callable, file_mtime: float):
        self.name = name
        self.module_path = module_path
        self.register_func = register_func
        self.file_mtime = file_mtime  # For cache invalidation

    def register(self, parser_manager):
        """Register this plugin's commands with the parser manager."""
        self.register_func(parser_manager)


def _get_cache_key(extension_dirs: List[str]) -> str:
    """Generate a cache key from the extension directories."""
    key_str = ":".join(sorted(extension_dirs))
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_cache(cache_key: str) -> Optional[Dict]:
    """Load cached plugin data if available."""
    if not _CACHE_FILE.exists():
        return None

    try:
        with open(_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is for the same directories
        if cache_data.get("cache_key") != cache_key:
            return None

        return cache_data.get("plugins", {})
    except (json.JSONDecodeError, IOError) as e:
        logger.debug(f"[CLI] Failed to load cache: {e}")
        return None


def _save_cache(cache_key: str, plugins: Dict[str, Dict]):
    """Save plugin data to cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "cache_key": cache_key,
            "plugins": plugins,
            "timestamp": time.time()
        }

        with open(_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"[CLI] Saved {len(plugins)} plugins to cache")
    except IOError as e:
        logger.warning(f"[CLI] Failed to save cache: {e}")


def _is_cache_valid(plugin_path: Path, cached_mtime: float) -> bool:
    """Check if cached plugin is still valid."""
    if not plugin_path.exists():
        return False

    current_mtime = plugin_path.stat().st_mtime
    return abs(current_mtime - cached_mtime) < 0.001  # Allow tiny float differences


def discover_cli_plugins(
    extension_dirs: List[str],
    use_cache: bool = False,  # Cache disabled - causes more problems than it solves
    force_refresh: bool = False
) -> Dict[str, CLIPlugin]:
    """
    Discover all CLI plugins in extension directories recursively.

    Args:
        extension_dirs: List of module paths to scan (e.g., ["src.extensions.features"])
        use_cache: Whether to use cached results (default: True)
        force_refresh: Force rediscovery even if cache is valid

    Returns:
        Dict mapping plugin names to CLIPlugin instances
    """
    cache_key = _get_cache_key(extension_dirs)
    plugins: Dict[str, CLIPlugin] = {}
    root_path = Path(__file__).parent.parent.parent  # backend/

    # Try to load from cache
    cached_plugins: Dict[str, Dict] = {}
    if use_cache and not force_refresh:
        cached_plugins = _load_cache(cache_key) or {}

    for ns in extension_dirs:
        # Convert module path to file path
        ns_path = root_path / ns.replace(".", "/")

        if not ns_path.exists():
            logger.debug(f"Extension directory not found: {ns_path}")
            continue

        # Recursively find all cli.py files, excluding certain directories
        for cli_file in ns_path.rglob("cli.py"):
            # Skip files in unwanted directories (.venv, node_modules, __pycache__, etc.)
            if any(part in cli_file.parts for part in [".venv", "node_modules", "__pycache__", ".git", "site-packages", "api"]):
                continue

            # Convert file path to module path
            full_path = cli_file.relative_to(root_path)
            module_path = ".".join(full_path.parent.parts + (full_path.stem,))

            # Extract extension name (last directory name)
            extension_name = full_path.parent.name

            # Check cache first
            file_mtime = cli_file.stat().st_mtime
            cache_entry = cached_plugins.get(extension_name)

            if cache_entry and _is_cache_valid(cli_file, cache_entry.get("mtime", 0)):
                # Use cached plugin info
                try:
                    mod = importlib.import_module(module_path)

                    if hasattr(mod, "register"):
                        plugin = CLIPlugin(
                            name=extension_name,
                            module_path=module_path,
                            register_func=mod.register,
                            file_mtime=file_mtime
                        )
                        plugins[extension_name] = plugin
                        logger.debug(f"[CLI] Loaded from cache: {extension_name}")
                    elif hasattr(mod, "register_commands"):
                        plugin = CLIPlugin(
                            name=extension_name,
                            module_path=module_path,
                            register_func=mod.register_commands,
                            file_mtime=file_mtime
                        )
                        plugins[extension_name] = plugin
                        logger.debug(f"[CLI] Loaded from cache: {extension_name} (register_commands)")
                    else:
                        logger.debug(f"[CLI] No register function in {module_path}")
                    continue
                except ImportError as e:
                    logger.debug(f"[CLI] Cache hit but import failed for {extension_name}: {e}")
                    # Fall through to rediscovery
                except Exception as e:
                    logger.debug(f"[CLI] Cache hit but load failed for {extension_name}: {e}")
                    # Fall through to rediscovery

            # Full discovery
            try:
                # Add parent directory to Python path if needed
                parent_dir = root_path
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))

                # Import cli.py directly WITHOUT triggering package __init__.py
                # This avoids loading heavy dependencies (langchain, etc.) during discovery
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_path, cli_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    # Set the module in sys.modules so relative imports work
                    sys.modules[module_path] = mod
                    spec.loader.exec_module(mod)
                else:
                    mod = importlib.import_module(module_path)

                # Check for required register function
                if hasattr(mod, "register"):
                    plugin = CLIPlugin(
                        name=extension_name,
                        module_path=module_path,
                        register_func=mod.register,
                        file_mtime=file_mtime
                    )
                    plugins[extension_name] = plugin
                    logger.debug(f"[CLI] Loaded plugin: {extension_name} from {module_path}")
                elif hasattr(mod, "register_commands"):
                    # Alternative naming convention
                    plugin = CLIPlugin(
                        name=extension_name,
                        module_path=module_path,
                        register_func=mod.register_commands,
                        file_mtime=file_mtime
                    )
                    plugins[extension_name] = plugin
                    logger.debug(f"[CLI] Loaded plugin: {extension_name} from {module_path} (register_commands)")
                else:
                    logger.debug(f"[CLI] No register function in {module_path}")

            except ImportError as e:
                # No cli.py or import error - this is expected for most extensions
                logger.debug(f"[CLI] No CLI module for {extension_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"[CLI] Error loading CLI plugin {extension_name}: {e}")
                continue

    # Update cache
    if use_cache:
        cache_data = {
            name: {
                "module_path": plugin.module_path,
                "mtime": plugin.file_mtime
            }
            for name, plugin in plugins.items()
        }
        _save_cache(cache_key, cache_data)

    return plugins


def invalidate_cache():
    """Invalidate the CLI plugin cache."""
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
            logger.info("[CLI] Plugin cache invalidated")
    except IOError as e:
        logger.warning(f"[CLI] Failed to invalidate cache: {e}")


def get_builtin_commands() -> Dict[str, Callable]:
    """
    Get built-in commands that don't come from extensions.

    Returns:
        Dict mapping command names to handler functions
    """
    from src.cli.commands import (
        handle_install,
        handle_server,
        handle_download,
        handle_help,
    )

    return {
        "install": handle_install,
        "server": handle_server,
        "download": handle_download,
        "help": handle_help,
    }
