"""
Extension Loader

Handles extension instantiation, dependency checking, and initialization.
Supports lazy loading via ExtensionProxy for fast startup.
Supports parallel initialization with dependency resolution.
"""

import importlib.util
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from ..base import (
    BaseExtension,
    ExtensionType,
    ExtensionStatus,
    LogicExtension,
    JobExtension,
    ServiceExtension,
    FailedExtension,
)
from .registry import ExtensionRegistry, ExtensionProxy
from .utils import unwrap_extension, is_extension_instance


class DependencyResolver:
    """
    Resolves extension dependencies and determines loading order.
    """

    def __init__(self, registry: ExtensionRegistry):
        self.registry = registry

    def resolve_load_order(self, extensions: Dict[str, BaseExtension]) -> List[List[str]]:
        """
        Resolve dependency graph and return loading batches.

        Extensions in the same batch can be loaded in parallel.
        Extensions in later batches depend on extensions in earlier batches.

        Args:
            extensions: All extensions to load

        Returns:
            List of batches, where each batch is a list of extension names
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {name: set() for name in extensions}
        in_degree: Dict[str, int] = {name: 0 for name in extensions}

        for name, ext in extensions.items():
            if is_extension_instance(ext, FailedExtension):
                continue

            # Get dependencies from config
            deps = ext.config.get('relationships', {}).get('requires', [])
            if not deps:
                # Handle both dict format (dependencies.extensions) and list format (deprecated)
                deps_config = ext.config.get('dependencies', {})
                if isinstance(deps_config, dict):
                    deps = deps_config.get('extensions', [])
                elif isinstance(deps_config, list):
                    # Backward compatibility: dependencies as a list of extension names
                    deps = deps_config

            for dep in deps:
                dep_name = dep if isinstance(dep, str) else dep.get('name', '')

                # Only consider dependencies that are in our extension set
                if dep_name in extensions and dep_name != name:
                    graph[name].add(dep_name)
                    in_degree[name] = in_degree.get(name, 0) + 1

        # Kahn's algorithm for topological sorting with batching
        batches: List[List[str]] = []
        remaining = set(extensions.keys())

        while remaining:
            # Find all extensions with no unsatisfied dependencies
            ready = [
                name for name in remaining
                if in_degree.get(name, 0) == 0
            ]

            if not ready:
                # Circular dependency - break by taking remaining extensions by priority
                logger = __import__('logging').getLogger(__name__)
                logger.warning(
                    "[Loader] Circular dependency detected, "
                    "breaking by loading remaining extensions by priority"
                )
                ready = sorted(
                    remaining,
                    key=lambda x: getattr(extensions[x], 'priority', 50),
                    reverse=True
                )

            # Add as a batch (can load in parallel)
            batches.append(ready)

            # Remove from remaining and update in-degrees
            for name in ready:
                remaining.remove(name)
                for other_name in remaining:
                    if name in graph.get(other_name, set()):
                        in_degree[other_name] -= 1

        return batches

    def check_satisfied(self, ext: BaseExtension) -> bool:
        """
        Check if an extension's dependencies are satisfied.

        Args:
            ext: Extension to check

        Returns:
            True if all dependencies are satisfied
        """
        deps = ext.config.get('relationships', {}).get('requires', [])
        if not deps:
            # Handle both dict format (dependencies.extensions) and list format (deprecated)
            deps_config = ext.config.get('dependencies', {})
            if isinstance(deps_config, dict):
                deps = deps_config.get('extensions', [])
            elif isinstance(deps_config, list):
                # Backward compatibility: dependencies as a list of extension names
                deps = deps_config

        for dep in deps:
            dep_name = dep if isinstance(dep, str) else dep.get('name', '')
            dep_ext = self.registry.get(dep_name)

            if not dep_ext:
                if isinstance(dep, dict) and dep.get('critical', True):
                    return False
            elif dep_ext.status != ExtensionStatus.LOADED:
                if isinstance(dep, dict) and dep.get('critical', True):
                    return False

        return True


class ExtensionLoader:
    """
    Instantiates and initializes extensions with parallel loading support.
    """

    def __init__(self, registry: ExtensionRegistry):
        """
        Initialize the loader.

        Args:
            registry: Extension registry for dependency checks
        """
        self.registry = registry
        self._extensions_dir: Optional[Path] = None
        self._resolver = DependencyResolver(registry)

    def set_extensions_dir(self, extensions_dir: Path) -> None:
        """Set the extensions directory for module loading."""
        self._extensions_dir = extensions_dir

    async def instantiate(self, config: Dict[str, Any], ext_dir: Path) -> BaseExtension:
        """
        Create extension instance based on type.

        Args:
            config: Parsed config.yaml
            ext_dir: Extension directory path

        Returns:
            Extension instance (may be FailedExtension or ExtensionProxy)
        """
        ext_name = config['extension']['name']
        ext_config = config.get('extension', {})
        system_config = config.get('system', {})

        # Create a placeholder for subprocess-only extensions (e.g., TTS extensions)
        # These are registered for discovery but not loaded into memory
        # They can be activated later via /api/extensions/{name}/activate
        if ext_config.get('subprocess_only', False):
            print(f"[Loader] 📦 Creating subprocess-only placeholder: {ext_name}")
            return self._create_subprocess_placeholder(config, ext_dir)

        # Check dependencies first
        if not self._check_dependencies(config):
            print(f"[Loader] ❌ {ext_name} has missing dependencies")
            return FailedExtension(config, ext_dir, "Missing dependencies")

        # Check for conflicts
        conflict = self._check_conflicts(config)
        if conflict:
            print(f"[Loader] ❌ {ext_name} conflicts with {conflict}")
            return FailedExtension(config, ext_dir, f"Conflicts with {conflict}")

        # Check for lazy_load flag
        # Default to True for most extensions, False for core libraries
        lazy_load = system_config.get('lazy_load', True)

        if lazy_load:
            print(f"[Loader] 📦 Deferring load (lazy): {ext_name}")
            return ExtensionProxy(config, ext_dir, self._extensions_dir)

        # Eager load - load extension immediately
        print(f"[Loader] ⚡ Eager loading: {ext_name}")

        # Try to load extension.py
        ext = self._load_from_module(config, ext_dir)
        if ext:
            return ext

        # Fall back to category-based instantiation
        return self._create_generic_extension(config, ext_dir)

    def _load_from_module(self, config: Dict[str, Any], ext_dir: Path) -> Optional[BaseExtension]:
        """
        Try to load extension from extension.py file.

        Args:
            config: Extension config
            ext_dir: Extension directory

        Returns:
            Extension instance or None
        """
        ext_module_path = ext_dir / "extension.py"
        if not ext_module_path.exists():
            return None

        ext_name = config['extension']['name']

        try:
            # Build proper module spec with package context
            module_name = f"src.extensions.{ext_dir.relative_to(self._extensions_dir).as_posix().replace('/', '.')}.extension"

            spec = importlib.util.spec_from_file_location(module_name, ext_module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)

                # Set package for relative imports
                module.__package__ = module_name.rsplit('.', 1)[0]

                # Add to sys.modules
                sys.modules[module_name] = module

                parent_name = module.__package__
                if parent_name and parent_name not in sys.modules:
                    # Create parent as a proper package module (with __path__)
                    # Walk up the directory tree from ext_dir to create all parent packages
                    parent_parts = parent_name.split('.')
                    pkg_dir = ext_dir  # Start from the extension directory

                    # Walk DOWN from src, creating each package level
                    for i in range(len(parent_parts)):
                        pkg_name = '.'.join(parent_parts[:i+1])
                        if pkg_name not in sys.modules:
                            pkg_module = type(sys)(pkg_name)
                            # Set __path__ for packages (except for the root 'src')
                            if i > 0:  # Skip 'src' root package
                                # Walk up from ext_dir to find this package's directory
                                levels_up = len(parent_parts) - i - 1
                                pkg_path = ext_dir
                                for _ in range(levels_up):
                                    pkg_path = pkg_path.parent
                                if pkg_path.exists() and pkg_path.is_dir():
                                    pkg_module.__path__ = [str(pkg_path)]
                            sys.modules[pkg_name] = pkg_module

                spec.loader.exec_module(module)

                # Look for register function or Extension class
                if hasattr(module, 'register_extension'):
                    return module.register_extension()
                elif hasattr(module, 'Extension'):
                    return module.Extension(config, ext_dir)

        except Exception as e:
            print(f"[Loader] Failed to import {ext_module_path}: {e}")
            return FailedExtension(config, ext_dir, str(e))

        return None

    def _create_generic_extension(
        self,
        config: Dict[str, Any],
        ext_dir: Path,
    ) -> BaseExtension:
        """
        Create a generic extension based on category.

        Args:
            config: Extension config
            ext_dir: Extension directory

        Returns:
            Extension instance
        """
        category = config.get('extension', {}).get('category', 'unknown')
        ext_type = self._get_extension_type(category)

        # Get priority for load order
        priority = config.get('extension', {}).get('priority', 50)

        # Create stub class based on type
        if ext_type == ExtensionType.MODE:
            base_class = LogicExtension
        elif ext_type == ExtensionType.JOB:
            base_class = JobExtension
        else:
            base_class = ServiceExtension

        class GenericExtension(base_class):
            def __init__(self, cfg, path):
                super().__init__(cfg, path)
                self._priority = priority

            async def initialize(self, context):
                await super().initialize(context)

            async def shutdown(self):
                pass

            async def process(self, input_data, **kwargs):
                return {"error": "Not implemented"}

            async def execute(self, **kwargs):
                yield {"error": "Not implemented"}

            async def start_service(self):
                pass

            async def send_request(self, endpoint, data):
                return {"error": "Not implemented"}

            async def stop_service(self):
                pass

            @property
            def priority(self) -> int:
                return self._priority

        return GenericExtension(config, ext_dir)

    def _get_extension_type(self, category: str) -> ExtensionType:
        """Map category to extension type."""
        mode_categories = {'mode', 'text_modes'}
        job_categories = {'tool', 'text_processing', 'visualization', 'job'}
        service_categories = {'tts', 'stt', 'image_gen', 'service'}

        if category in mode_categories:
            return ExtensionType.MODE
        elif category in service_categories:
            return ExtensionType.SERVICE
        else:
            return ExtensionType.JOB

    def _check_dependencies(self, config: Dict[str, Any]) -> bool:
        """
        Validate extension dependencies.

        Args:
            config: Extension config

        Returns:
            True if all dependencies satisfied
        """
        requires = config.get('relationships', {}).get('requires', [])
        if not requires:
            # Handle both dict format (dependencies.extensions) and list format (deprecated)
            deps = config.get('dependencies', {})
            if isinstance(deps, dict):
                requires = deps.get('extensions', [])
            elif isinstance(deps, list):
                # Backward compatibility: dependencies as a list of extension names
                requires = deps

        for dep in requires:
            target_name = dep if isinstance(dep, str) else dep.get('name', '')

            # Check if dependency exists
            target_ext = self.registry.get(target_name)
            if not target_ext:
                if isinstance(dep, dict) and dep.get('critical', True):
                    return False
                continue

            # Check if dependency is active
            if target_ext.status != ExtensionStatus.LOADED:
                if isinstance(dep, dict) and dep.get('critical', True):
                    return False

        return True

    def _check_conflicts(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Check for extension conflicts.

        Args:
            config: Extension config

        Returns:
            Name of conflicting extension or None
        """
        conflicts = config.get('relationships', {}).get('conflicts', [])

        for conflict_name in conflicts:
            if self.registry.is_active(conflict_name):
                return conflict_name

        return None

    def _create_subprocess_placeholder(
        self,
        config: Dict[str, Any],
        ext_dir: Path,
    ) -> BaseExtension:
        """
        Create a placeholder extension for subprocess-only extensions.

        These extensions are registered for discovery but not loaded into memory.
        They can be activated later via /api/extensions/{name}/activate.

        Uses the unified SubprocessServiceExtension base class.

        Args:
            config: Extension config
            ext_dir: Extension directory

        Returns:
            A placeholder extension that can be activated on demand
        """
        # Import the unified base class
        from ..tts_base import SubprocessServiceExtension

        # Import the process manager class
        from src.extensions.libraries.tts_engine.extension_process_manager import ExtensionProcessManager

        ext_name = config['extension']['name']
        category = config.get('extension', {}).get('category', 'service')

        # Create a dynamic subclass with the proper process manager
        class DynamicSubprocessExtension(SubprocessServiceExtension):
            _process_manager_class = ExtensionProcessManager

            def get_capabilities(self) -> Dict[str, Any]:
                """Return basic capabilities for discovery."""
                return {
                    "name": ext_name,
                    "category": category,
                    "subprocess_only": True,
                    "supports_activation": True,
                }

            def get_ui_config(self) -> Dict[str, Any]:
                """Return UI config for the extension."""
                return config.get('ui', {})

        return DynamicSubprocessExtension(config, ext_dir)

    async def initialize_all(
        self,
        extensions: Dict[str, BaseExtension],
        context,
        parallel: bool = True,
    ) -> None:
        """
        Initialize all extensions with optional parallel loading.

        Args:
            extensions: Extensions to initialize
            context: Extension context
            parallel: Whether to use parallel loading (default: True)
        """
        if parallel:
            await self._initialize_parallel(extensions, context)
        else:
            await self._initialize_sequential(extensions, context)

    async def _initialize_sequential(
        self,
        extensions: Dict[str, BaseExtension],
        context
    ) -> None:
        """
        Initialize all extensions sequentially in priority order.

        Args:
            extensions: Extensions to initialize
            context: Extension context
        """
        # Sort by priority (higher first)
        sorted_exts = sorted(
            extensions.items(),
            key=lambda x: getattr(x[1], 'priority', 50),
            reverse=True
        )

        for name, ext in sorted_exts:
            if is_extension_instance(ext, FailedExtension):
                continue

            try:
                await ext.initialize(context)
                print(f"[Loader] ✅ Initialized: {name}")
            except Exception as e:
                import traceback
                print(f"[Loader] ❌ Failed to initialize {name}: {e}")
                print(f"[Loader] Full traceback:\n{traceback.format_exc()}")
                ext.status = ExtensionStatus.ERROR

    async def _initialize_parallel(
        self,
        extensions: Dict[str, BaseExtension],
        context
    ) -> None:
        """
        Initialize all extensions in parallel batches based on dependencies.

        Args:
            extensions: Extensions to initialize
            context: Extension context
        """
        # First register all extensions with the temporary registry for dependency resolution
        temp_registry = {}
        for name, ext in extensions.items():
            if not is_extension_instance(ext, FailedExtension):
                temp_registry[name] = ext

        # Resolve load order into batches
        batches = self._resolver.resolve_load_order(temp_registry)

        print(f"[Loader] Parallel initialization: {len(batches)} batches")

        # Initialize each batch in parallel
        for batch_num, batch in enumerate(batches, 1):
            print(f"[Loader] Batch {batch_num}/{len(batches)}: {', '.join(batch)}")

            # Create tasks for this batch
            tasks = []
            for name in batch:
                ext = extensions[name]
                if not is_extension_instance(ext, FailedExtension):
                    tasks.append((name, ext.initialize(context)))

            # Run all tasks in this batch in parallel
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            # Check results
            for (name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    import traceback
                    print(f"[Loader] ❌ Failed to initialize {name}: {result}")
                    if isinstance(result, RecursionError):
                        # Print last 50 frames of recursion for debugging
                        tb_lines = traceback.format_tb(result.__traceback__)
                        print(f"[Loader] Recursion traceback (last 50 frames):\n{''.join(tb_lines[-50:])}")
                    extensions[name].status = ExtensionStatus.ERROR
                else:
                    print(f"[Loader] ✅ Initialized: {name}")

    async def shutdown_all(self, extensions: Dict[str, BaseExtension]) -> None:
        """
        Shutdown all extensions.

        Args:
            extensions: Extensions to shutdown
        """
        # Shutdown in reverse priority order (lowest priority first)
        sorted_exts = sorted(
            extensions.items(),
            key=lambda x: getattr(x[1], 'priority', 50),
        )

        for name, ext in sorted_exts:
            if is_extension_instance(ext, FailedExtension):
                continue

            try:
                await ext.shutdown()
                print(f"[Loader] 🔒 Shut down: {name}")
            except Exception as e:
                print(f"[Loader] ⚠️  Error shutting down {name}: {e}")
