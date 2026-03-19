"""
Extension Manager

Main orchestrator for the extension system. Coordinates discovery, loading,
and provides a simple API for the rest of the application.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from ..base import BaseExtension, ExtensionStatus
from ..context import ExtensionContext
from ..lifecycle import ExtensionLifecycle
from .registry import ExtensionRegistry
from .discover import ExtensionDiscoverer
from .loader import ExtensionLoader
from .capabilities import CapabilityQuery
from .operations import ExtensionOperations
from .metrics import ExtensionMetrics
from .utils import unwrap_extension, is_extension_instance


class ExtensionManager:
    """
    Main extension manager.

    Provides a simple API while delegating to specialized modules.
    """

    def __init__(self, max_heavy: int = 2):
        """
        Initialize the extension manager.

        Args:
            max_heavy: Maximum number of heavy (service) extensions to keep active
        """
        # Core components
        self.registry = ExtensionRegistry()
        self.discoverer: Optional[ExtensionDiscoverer] = None
        self.loader = ExtensionLoader(self.registry)
        self.capabilities = CapabilityQuery(self.registry)
        self.metrics = ExtensionMetrics()
        self.operations = ExtensionOperations(
            self.registry,
            self.discoverer,  # Will be set in load_all
            self.loader,
            self.metrics,
        )
        self.lifecycle = ExtensionLifecycle(max_heavy)

        # Shared context
        self.context: Optional[ExtensionContext] = None

        # Extensions directory
        self._extensions_dir: Optional[Path] = None

    def set_context(self, context: ExtensionContext) -> None:
        """
        Set the shared context for all extensions.

        Args:
            context: The context with LLM factory, settings, etc.
        """
        self.context = context

    async def load_all(self, extensions_dir: Path, app=None) -> None:
        """
        Scan and load all extensions.

        Args:
            extensions_dir: Path to the extensions directory
            app: Optional FastAPI app instance for router registration
        """
        self._extensions_dir = extensions_dir
        print(f"[ExtensionManager] Scanning for extensions in {extensions_dir}")

        # Execute prestartup scripts before loading extensions
        self._execute_prestartup_scripts(extensions_dir)

        # Initialize discoverer with the extensions directory
        self.discoverer = ExtensionDiscoverer(extensions_dir)
        self.loader.set_extensions_dir(extensions_dir)
        self.operations.discoverer = self.discoverer

        # Discover all extensions
        discovered = await self.discoverer.scan()

        # Instantiate all extensions
        for config, ext_dir in discovered:
            ext_name = config['extension']['name']
            try:
                load_start = __import__('time').time()
                ext = await self.loader.instantiate(config, ext_dir)
                if ext:
                    self.registry.register(ext)
                    load_time = __import__('time').time() - load_start
                    self.metrics.record_load(ext_name, load_time)
                    print(f"[ExtensionManager] ✅ Instantiated: {ext_name}")
            except Exception as e:
                print(f"[ExtensionManager] ❌ Failed to instantiate {ext_name}: {e}")

        # Initialize all extensions
        print(f"[ExtensionManager] Initializing {self.registry.count} extensions...")
        await self.loader.initialize_all(self.registry.get_all(), self.context)

        # Register API routes from extensions (Micro-Kernel Pattern)
        if app is not None:
            print(f"[ExtensionManager] Registering API routes from extensions...")
            self._register_extension_routes(app)

    def _register_extension_routes(self, app) -> None:
        """
        Register API routes from all extensions.

        Extensions should implement register_routes(app) to register their routers.
        This is the micro-kernel pattern: extensions register their own routes.

        Args:
            app: The FastAPI application instance
        """
        from ..base import BaseExtension

        registered_count = 0
        for ext in self.registry.get_all().values():
            try:
                # Only call if extension overrides the base register_routes
                if self._has_custom_register_routes(ext, BaseExtension):
                    ext.register_routes(app)
                    registered_count += 1
                    self._log_route_registration(ext)
            except Exception as e:
                print(f"[ExtensionManager] ⚠️  Failed to register routes for {ext.name}: {e}")

        print(f"[ExtensionManager] Registered routes from {registered_count} extensions")

    def _has_custom_register_routes(self, ext, base_class: BaseExtension) -> bool:
        """Check if extension has a custom register_routes implementation."""
        if not hasattr(ext, 'register_routes'):
            return False

        # Get the method's actual function (unwrap bound method)
        method = ext.register_routes
        if hasattr(method, '__func__'):
            method_func = method.__func__
        else:
            method_func = method

        # Compare with base class implementation
        base_method = base_class.register_routes
        if hasattr(base_method, '__func__'):
            base_func = base_method.__func__
        else:
            base_func = base_method

        return method_func is not base_func

    def _log_route_registration(self, ext) -> None:
        """Log route registration for an extension."""
        router_info = ext.get_router_info()
        if router_info:
            prefix = router_info.get('prefix', 'unknown')
            print(f"[ExtensionManager] ✅ Registered routes for {ext.name} at {prefix}")
        else:
            print(f"[ExtensionManager] ✅ Registered routes for {ext.name}")

    def _execute_prestartup_scripts(self, extensions_dir: Path) -> None:
        """
        Execute pre-startup scripts in each extension directory.

        This follows the ComfyUI extension pattern where extensions can have
        install.py and prestartup_script.py files.

        Extensions with isolated .venv directories have their dependencies
        installed separately and scripts are only run during initial setup.
        """
        import importlib.util
        import warnings
        import sys

        if not extensions_dir.exists():
            return

        for category_dir in extensions_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'core':
                continue

            for ext_dir in category_dir.iterdir():
                if not ext_dir.is_dir() or ext_dir.name.startswith('_'):
                    continue

                # Check for install.py
                install_script = ext_dir / "install.py"
                if install_script.exists():
                    # Check if this extension has an isolated venv
                    ext_venv = ext_dir / ".venv"

                    if ext_venv.exists():
                        # Extension has isolated venv - skip install script
                        # The install script itself will handle dependencies in its venv
                        print(f"[ExtensionManager] Extension {ext_dir.name} has isolated venv, skipping install script")
                    else:
                        # Extension doesn't have isolated venv - check dependencies
                        warnings.warn(
                            f"Extension {ext_dir.name} does not have an isolated .venv directory. "
                            "Using shared dependencies is deprecated.",
                            DeprecationWarning,
                            stacklevel=2
                        )
                        req_file = ext_dir / "requirements.txt"
                        should_run_install = False

                        if req_file.exists():
                            # Check if dependencies are installed
                            with open(req_file) as f:
                                requirements = [
                                    line.strip()
                                    for line in f
                                    if line.strip() and not line.startswith("#")
                                ]

                            for req in requirements:
                                package_name = (
                                    req.split(">=")[0]
                                    .split("==")[0]
                                    .split("<=")[0]
                                    .split("~")[0]
                                    .strip()
                                )

                                spec = importlib.util.find_spec(package_name)
                                if spec is None:
                                    should_run_install = True
                                    break

                        if should_run_install:
                            print(f"[ExtensionManager] Executing install script for {ext_dir.name}")
                            self._execute_script(install_script)

                # Check for prestartup_script.py
                prestartup_script = ext_dir / "prestartup_script.py"
                if prestartup_script.exists():
                    print(f"[ExtensionManager] Executing prestartup script for {ext_dir.name}")
                    self._execute_script(prestartup_script)

    def _execute_script(self, script_path: Path) -> None:
        """Execute a Python script from a given path."""
        import importlib.util
        import sys
        import os

        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        if spec is None or spec.loader is None:
            print(f"[ExtensionManager] Failed to load script: {script_path}")
            return

        module = importlib.util.module_from_spec(spec)

        script_dir = str(script_path.parent)
        original_path = sys.path[:]

        try:
            # Add script's directory to sys.path temporarily
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            # Execute the module
            spec.loader.exec_module(module)

            # Execute main() or install_dependencies() if exists
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'install_dependencies'):
                module.install_dependencies()

        except Exception as e:
            print(f"[ExtensionManager] Failed to execute script {script_path}: {e}")
        finally:
            # Restore sys.path
            sys.path[:] = original_path

    # ==========================================================================
    # BACKWARD COMPATIBILITY
    # ==========================================================================

    @property
    def extensions(self) -> Dict[str, BaseExtension]:
        """
        Get all extensions (backward compatibility).

        Deprecated: Use registry.get_all() instead.
        """
        return self.registry.get_all()

    def get(self, name: str) -> Optional[BaseExtension]:
        """
        Get an extension by name.

        Args:
            name: The extension name

        Returns:
            The extension or None if not found
        """
        ext = self.registry.get(name)

        # Track usage for metrics
        if ext:
            self.metrics.record_usage(name)

        return ext

    def get_extension(self, identifier: str) -> Optional[BaseExtension]:
        """
        Get an extension by identifier with fallbacks.

        Tries:
        1. Exact match
        2. chat_{mode} format (for chat modes)
        3. Case-insensitive match

        Args:
            identifier: Extension ID or mode name

        Returns:
            The extension or None if not found
        """
        ext = self.registry.get_with_fallbacks(identifier)

        # Track usage
        if ext:
            self.metrics.record_usage(ext.name)

        return ext

    def get_all_router_info(self) -> List[Dict[str, Any]]:
        """
        Get router metadata from all extensions for frontend discovery.

        Returns:
            List of router info dicts with prefix, name, tags, description
        """
        return self.capabilities.get_all_router_info()

    def is_active(self, name: str) -> bool:
        """
        Check if an extension is active.

        Args:
            name: The extension name

        Returns:
            True if loaded and active
        """
        return self.registry.is_active(name)

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities for the frontend.

        Returns:
            Dict with modes, tools, tts, features, and extensions
        """
        return self.capabilities.get_capabilities()

    # ==========================================================================
    # OPERATIONS (delegated to ExtensionOperations)
    # ==========================================================================

    async def reload(self, name: str) -> bool:
        """Reload an extension."""
        return await self.operations.reload(name, self.context)

    async def health_check(self, name: str) -> Dict[str, Any]:
        """Check health of an extension."""
        return await self.operations.health_check(name)

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all extensions."""
        return await self.operations.health_check_all()

    async def install(self, extension_path: str, enabled: bool = True) -> bool:
        """Install a new extension."""
        return await self.operations.install(extension_path, self.context, enabled)

    async def uninstall(self, name: str) -> bool:
        """Uninstall an extension."""
        return await self.operations.uninstall(name)

    async def enable(self, name: str) -> bool:
        """Enable a disabled extension."""
        return await self.operations.enable(name, self.context)

    async def disable(self, name: str) -> bool:
        """Disable an extension."""
        return await self.operations.disable(name)

    async def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """Get telemetry metrics."""
        return await self.operations.get_metrics(name)

    # ==========================================================================
    # LIFECYCLE (for service extensions)
    # ==========================================================================

    async def activate_service(self, name: str) -> Optional[BaseExtension]:
        """
        Activate a service extension with LRU management.

        Args:
            name: Extension name

        Returns:
            The activated extension
        """
        from ..base import ServiceExtension

        ext = self.get_extension(name)
        if not ext or not is_extension_instance(ext, ServiceExtension):
            return None

        return await self.lifecycle.activate(name, ext)

    async def deactivate_service(self, name: str) -> None:
        """Deactivate a service extension."""
        await self.lifecycle.deactivate(name)

    async def shutdown_all(self) -> None:
        """Shutdown all extensions and services."""
        await self.lifecycle.deactivate_all()
        await self.loader.shutdown_all(self.registry.get_all())

    # ==========================================================================
    # TTS-SPECIFIC FUNCTIONALITY
    # ==========================================================================

    def get_all_voices(self) -> list:
        """
        Get all voices from all TTS extensions.

        Returns:
            List of VoiceInfo objects from all extensions that implement TTSExtensionInterface
        """
        all_voices = []
        from ..base import TTSExtensionInterface

        for ext in self.registry.get_all().values():
            if is_extension_instance(ext, TTSExtensionInterface):
                try:
                    voices = ext.get_voices()
                    if voices:
                        all_voices.extend(voices)
                except Exception as e:
                    print(f"[ExtensionManager] Error getting voices from {ext.name}: {e}")

        return all_voices

    def get_voices_for_extension(self, extension_name: str) -> list:
        """Get voices for a specific TTS extension."""
        from ..base import TTSExtensionInterface

        ext = self.registry.get(extension_name)
        if not ext or not is_extension_instance(ext, TTSExtensionInterface):
            return []

        try:
            return ext.get_voices()
        except Exception as e:
            print(f"[ExtensionManager] Error getting voices from {extension_name}: {e}")
            return []

    def get_extension_for_voice(self, voice_id: str) -> Optional['BaseExtension']:
        """Find which extension provides a specific voice."""
        from ..base import TTSExtensionInterface

        for ext in self.registry.get_all().values():
            if is_extension_instance(ext, TTSExtensionInterface):
                try:
                    voices = ext.get_voices()
                    for voice in voices:
                        if hasattr(voice, 'id') and voice.id == voice_id:
                            return ext
                        elif isinstance(voice, dict) and voice.get('id') == voice_id:
                            return ext
                except Exception:
                    continue

        return None

    def get_all_tts_configs(self) -> list:
        """Get configuration for all TTS extensions."""
        from ..base import TTSExtensionInterface

        configs = []
        for ext in self.registry.get_all().values():
            if is_extension_instance(ext, TTSExtensionInterface):
                try:
                    config = ext.get_tts_config()
                    if config:
                        configs.append(config)
                except Exception as e:
                    print(f"[ExtensionManager] Error getting config from {ext.name}: {e}")

        return configs

    def get_tts_config(self, name: str) -> Optional[dict]:
        """Get configuration for a specific TTS extension."""
        from ..base import TTSExtensionInterface

        ext = self.registry.get(name)
        if not ext or not is_extension_instance(ext, TTSExtensionInterface):
            return None

        try:
            return ext.get_tts_config()
        except Exception as e:
            print(f"[ExtensionManager] Error getting config from {name}: {e}")
            return None

    # ==========================================================================
    # COMFYUI-STYLE EXTENSION LOADING
    # ==========================================================================

    async def discover_single_extension(self, extension_name: str, extensions_dir: Path = None) -> bool:
        """
        Discover and load a single extension by name.

        This is useful for subprocess mode where each extension runs in its own
        isolated venv and we only want to load that specific extension.

        Args:
            extension_name: Name of the extension to discover
            extensions_dir: Path to extensions directory (uses stored dir if None)

        Returns:
            True if extension was discovered and registered
        """
        if extensions_dir is None:
            extensions_dir = self._extensions_dir

        if not extensions_dir:
            print("[ExtensionManager] No extensions directory set")
            return False

        # Find the extension folder
        target_folder = None
        for category_dir in extensions_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'core':
                continue
            for ext_dir in category_dir.iterdir():
                if ext_dir.is_dir() and ext_dir.name == extension_name:
                    target_folder = ext_dir
                    break
            if target_folder:
                break

        if not target_folder:
            print(f"[ExtensionManager] Extension folder for '{extension_name}' not found")
            return False

        print(f"[ExtensionManager] Discovering single extension: {extension_name} from {target_folder}")

        try:
            # Check for config.yaml first (core pattern)
            config_path = target_folder / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if config and 'extension' in config:
                    ext = await self.loader.instantiate(config, target_folder)
                    if ext:
                        self.registry.register(ext)
                        await ext.initialize(self.context)
                        print(f"[ExtensionManager] Successfully discovered extension: {extension_name}")
                        return True

            # Fall back to ComfyUI-style loading (extension.py with register_extension)
            extension_path = target_folder / "extension.py"
            if extension_path.exists():
                ext = self._load_comfy_style_extension(target_folder, extension_path)
                if ext:
                    # Create a minimal config for the extension
                    config = {'extension': {'name': extension_name, 'category': 'unknown'}}
                    ext.config = config
                    ext.name = extension_name
                    ext.root_path = target_folder

                    self.registry.register(ext)
                    await ext.initialize(self.context)
                    print(f"[ExtensionManager] Successfully discovered extension: {extension_name}")
                    return True

            print(f"[ExtensionManager] No valid extension found in {target_folder}")
            return False

        except Exception as e:
            print(f"[ExtensionManager] Error discovering extension {extension_name}: {e}")
            return False

    def _load_comfy_style_extension(self, folder_path: Path, extension_path: Path):
        """Load extension from extension.py following ComfyUI pattern."""
        import importlib.util
        import sys
        import types

        folder_name = folder_path.name
        module_name = f"{folder_name}.extension"
        original_path = sys.path[:]

        try:
            # Add parent directory to sys.path
            parent_dir = str(folder_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Create parent package module for relative imports
            parent_module_name = folder_name
            if parent_module_name not in sys.modules:
                parent_module = types.ModuleType(parent_module_name)
                parent_module.__path__ = [str(folder_path)]
                parent_module.__package__ = parent_module_name
                sys.modules[parent_module_name] = parent_module

            # Create module spec
            spec = importlib.util.spec_from_file_location(
                module_name,
                extension_path,
                submodule_search_locations=[str(folder_path)],
            )
            if not spec or not spec.loader:
                return None

            # Create and execute module
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(extension_path)
            module.__package__ = folder_name

            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get extension from register_extension function
            if hasattr(module, 'register_extension'):
                return module.register_extension()

            return None

        except Exception as e:
            print(f"[ExtensionManager] Error loading ComfyUI-style extension from {folder_name}: {e}")
            return None
        finally:
            # Restore sys.path
            sys.path[:] = original_path
            # Cleanup temp modules
            if module_name in sys.modules:
                del sys.modules[module_name]
            if folder_name in sys.modules:
                del sys.modules[folder_name]
