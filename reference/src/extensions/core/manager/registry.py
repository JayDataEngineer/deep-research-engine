"""
Extension Registry

Stores and provides access to loaded extensions.
Simple in-memory storage with lookup methods.
Supports lazy loading via ExtensionProxy.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any
from ..base import BaseExtension, ExtensionStatus


class ExtensionProxy(BaseExtension):
    """
    Proxy for lazy-loaded extensions.

    Delays importing the actual extension module until first access.
    This allows fast startup time by not loading heavy dependencies (torch, etc.)
    until the extension is actually used.
    """

    def __init__(self, config: Dict[str, Any], ext_dir: Path, extensions_dir: Path):
        """
        Initialize the proxy.

        Args:
            config: Parsed config.yaml
            ext_dir: Extension directory path
            extensions_dir: Base extensions directory for module loading
        """
        self.config = config
        self.name = config['extension']['name']
        self.root_path = ext_dir
        self._extensions_dir = extensions_dir
        self._instance: Optional[BaseExtension] = None
        self._module_loaded = False
        self._initialized = False  # Track whether extension.initialize() was called
        self.status = ExtensionStatus.LOADED  # Proxy is "loaded" but defers actual loading

    def _ensure_loaded(self) -> BaseExtension:
        """
        Ensure the actual extension is loaded.

        Returns:
            The real extension instance
        """
        if self._instance is None:
            print(f"[ExtensionProxy] 🔄 Lazy loading {self.name}...")

            try:
                # Import the extension module
                ext_module_path = self.root_path / "extension.py"
                if not ext_module_path.exists():
                    raise ImportError(f"No extension.py found for {self.name}")

                # Build proper module spec
                module_name = f"src.extensions.{self.root_path.relative_to(self._extensions_dir).as_posix().replace('/', '.')}.extension"

                spec = importlib.util.spec_from_file_location(module_name, ext_module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    module.__package__ = module_name.rsplit('.', 1)[0]
                    sys.modules[module_name] = module

                    # Create parent packages if needed
                    parent_name = module.__package__
                    if parent_name and parent_name not in sys.modules:
                        parent_parts = parent_name.split('.')
                        for i in range(len(parent_parts)):
                            pkg_name = '.'.join(parent_parts[:i+1])
                            if pkg_name not in sys.modules:
                                pkg_module = type(sys)(pkg_name)
                                if i > 0:  # Not root 'src'
                                    levels_up = len(parent_parts) - i - 1
                                    pkg_path = self.root_path
                                    for _ in range(levels_up):
                                        pkg_path = pkg_path.parent
                                    if pkg_path.exists() and pkg_path.is_dir():
                                        pkg_module.__path__ = [str(pkg_path)]
                                sys.modules[pkg_name] = pkg_module

                    spec.loader.exec_module(module)

                    # Get the extension instance
                    if hasattr(module, 'register_extension'):
                        self._instance = module.register_extension()
                    elif hasattr(module, 'Extension'):
                        self._instance = module.Extension(self.config, self.root_path)
                    else:
                        raise AttributeError(f"No register_extension() or Extension class found in {module_name}")

                    self._module_loaded = True
                    print(f"[ExtensionProxy] ✅ Loaded {self.name}")

            except Exception as e:
                print(f"[ExtensionProxy] ❌ Failed to load {self.name}: {e}")
                # Return a failed extension
                from ..base import FailedExtension
                self._instance = FailedExtension(self.config, self.root_path, str(e))
                self.status = ExtensionStatus.ERROR

        return self._instance

    async def initialize(self, context: 'ExtensionContext') -> None:
        """Initialize the real extension when first accessed."""
        real_ext = self._ensure_loaded()
        if hasattr(real_ext, 'context'):
            real_ext.context = context
        if hasattr(real_ext, 'initialize') and not self._initialized:
            await real_ext.initialize(context)
            self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the real extension if loaded."""
        if self._instance is not None:
            await self._instance.shutdown()

    @property
    def category(self) -> str:
        """Get category from config (no loading needed)."""
        return self.config.get('extension', {}).get('category', 'unknown')

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities from config (no loading needed)."""
        return self.config.get('capabilities', {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI config from config (no loading needed)."""
        return self.config.get('ui', {})

    def register_routes(self, app) -> None:
        """Delegate route registration to the real extension."""
        real_ext = self._ensure_loaded()
        if hasattr(real_ext, 'register_routes'):
            real_ext.register_routes(app)

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """Delegate router info query to the real extension."""
        real_ext = self._ensure_loaded()
        if hasattr(real_ext, 'get_router_info'):
            return real_ext.get_router_info()
        return None

    # Delegate all other attributes to the real instance
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the real extension."""
        if name.startswith('_') or name in ('config', 'name', 'root_path', '_instance', '_module_loaded', 'status'):
            raise AttributeError(name)

        real_ext = self._ensure_loaded()
        return getattr(real_ext, name)


class ExtensionRegistry:
    """
    Storage and lookup for loaded extensions.
    """

    def __init__(self):
        self._extensions: Dict[str, BaseExtension] = {}

    def register(self, extension: BaseExtension) -> None:
        """Register an extension."""
        self._extensions[extension.name] = extension

    def get(self, name: str) -> Optional[BaseExtension]:
        """Get extension by exact name."""
        return self._extensions.get(name)

    def get_with_fallbacks(self, identifier: str) -> Optional[BaseExtension]:
        """
        Get extension with fallback lookups.

        Tries:
        1. Exact match
        2. chat_{mode} format
        3. Case-insensitive match
        """
        # Exact match
        ext = self.get(identifier)
        if ext:
            return ext

        # chat_ prefix for modes
        ext = self.get(f"chat_{identifier}")
        if ext:
            return ext

        # Case-insensitive
        identifier_lower = identifier.lower()
        for name, extension in self._extensions.items():
            if name.lower() == identifier_lower:
                return extension

        return None

    def get_all(self) -> Dict[str, BaseExtension]:
        """Get all extensions."""
        return self._extensions.copy()

    def get_by_status(self, status: ExtensionStatus) -> List[BaseExtension]:
        """Get all extensions with a specific status."""
        return [ext for ext in self._extensions.values() if ext.status == status]

    def get_by_category(self, category: str) -> List[BaseExtension]:
        """Get all extensions in a category."""
        return [ext for ext in self._extensions.values() if ext.category == category]

    def is_active(self, name: str) -> bool:
        """Check if extension is loaded and active."""
        ext = self.get(name)
        return ext and ext.status == ExtensionStatus.LOADED

    def remove(self, name: str) -> Optional[BaseExtension]:
        """Remove an extension from registry."""
        return self._extensions.pop(name, None)

    def clear(self) -> None:
        """Clear all extensions."""
        self._extensions.clear()

    @property
    def count(self) -> int:
        """Total number of registered extensions."""
        return len(self._extensions)

    @property
    def active_count(self) -> int:
        """Number of active (loaded) extensions."""
        return len(self.get_by_status(ExtensionStatus.LOADED))
