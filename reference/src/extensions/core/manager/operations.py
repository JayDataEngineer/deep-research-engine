"""
Extension Operations

Runtime operations: reload, health checks, install, uninstall, etc.
"""

import asyncio
import time
import shutil
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..base import BaseExtension, ExtensionStatus
from .discover import ExtensionDiscoverer, load_yaml_config, is_extension_enabled
from .loader import ExtensionLoader
from .metrics import ExtensionMetrics


def save_yaml_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save configuration to YAML file.

    Args:
        config_path: Path to save YAML file
        config: Configuration dictionary to save
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class ExtensionOperations:
    """
    Runtime operations for extensions.
    """

    def __init__(
        self,
        registry,
        discoverer: ExtensionDiscoverer,
        loader: ExtensionLoader,
        metrics: ExtensionMetrics,
    ):
        """
        Initialize operations.

        Args:
            registry: Extension registry
            discoverer: Extension discoverer
            loader: Extension loader
            metrics: Telemetry/metrics collector
        """
        self.registry = registry
        self.discoverer = discoverer
        self.loader = loader
        self.metrics = metrics

    async def reload(self, name: str, context) -> bool:
        """
        Reload an extension.

        Shuts down and reloads an extension, preserving state where possible.

        Args:
            name: Extension name
            context: Extension context

        Returns:
            True if successful
        """
        ext = self.registry.get(name)
        if not ext:
            print(f"[Operations] Extension not found: {name}")
            return False

        print(f"[Operations] Reloading: {name}")

        # Shutdown
        try:
            await ext.shutdown()
        except Exception as e:
            print(f"[Operations] Error during shutdown: {e}")

        # Remove from registry
        self.registry.remove(name)

        # Find and reload
        ext_dir = self.discoverer.get_extension_path(name)
        if not ext_dir:
            print(f"[Operations] Cannot find extension directory: {name}")
            return False

        config = self.discoverer._load_config(ext_dir / "config.yaml")
        if not config:
            print(f"[Operations] Cannot load config for: {name}")
            return False

        try:
            new_ext = await self.loader.instantiate(config, ext_dir)
            await new_ext.initialize(context)
            self.registry.register(new_ext)
            self.metrics.record_reload(name)
            print(f"[Operations] ✅ Reloaded: {name}")
            return True
        except Exception as e:
            print(f"[Operations] ❌ Failed to reload {name}: {e}")
            return False

    async def health_check(self, name: str) -> Dict[str, Any]:
        """
        Check health of an extension.

        Args:
            name: Extension name

        Returns:
            Health status dict
        """
        ext = self.registry.get_with_fallbacks(name)
        if not ext:
            return {
                "name": name,
                "status": "not_found",
                "healthy": False,
            }

        health = {
            "name": ext.name,
            "status": ext.status.value,
            "healthy": ext.status == ExtensionStatus.LOADED,
            "category": ext.category,
            "version": ext.config.get('extension', {}).get('version', 'unknown'),
        }

        # Call extension's health check if available
        if hasattr(ext, 'health_check'):
            try:
                ext_health = await ext.health_check()
                health.update(ext_health)
            except Exception as e:
                health["error"] = str(e)
                health["healthy"] = False

        # Add uptime if tracked
        if hasattr(ext, '_loaded_at'):
            health["uptime_seconds"] = time.time() - ext._loaded_at

        return health

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all extensions.

        Returns:
            Dict mapping extension names to health status
        """
        results = {}
        for name in self.registry.get_all().keys():
            results[name] = await self.health_check(name)
        return results

    async def install(
        self,
        extension_path: str,
        context,
        enabled: bool = True,
    ) -> bool:
        """
        Install a new extension from path.

        Args:
            extension_path: Path to extension directory or git URL
            context: Extension context
            enabled: Whether to enable after install

        Returns:
            True if successful
        """
        print(f"[Operations] Installing extension: {extension_path}")

        # Determine if it's a git URL or local path
        if extension_path.startswith(("http://", "https://", "git@")):
            return await self._install_from_git(extension_path, context, enabled)
        else:
            return await self._install_from_local(extension_path, context, enabled)

    async def _install_from_git(
        self,
        git_url: str,
        context,
        enabled: bool = True,
    ) -> bool:
        """Install extension from git repository."""
        import tempfile

        # Get extensions root directory
        extensions_root = self.discoverer.extensions_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repository
            try:
                print(f"[Operations] Cloning {git_url}...")
                subprocess.run(
                    ["git", "clone", git_url, temp_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[Operations] ❌ Failed to clone: {e.stderr}")
                return False

            # Find the actual extension directory (might be in a subdirectory)
            ext_dir = Path(temp_dir)
            config_path = ext_dir / "config.yaml"

            # If config.yaml not in root, look for it in subdirectories
            if not config_path.exists():
                for item in ext_dir.iterdir():
                    if item.is_dir() and (item / "config.yaml").exists():
                        ext_dir = item
                        config_path = item / "config.yaml"
                        break

            if not config_path.exists():
                print(f"[Operations] ❌ No config.yaml found in repository")
                return False

            # Load config to get extension name
            config = self.discoverer._load_config(config_path)
            if not config:
                print(f"[Operations] ❌ Failed to load config.yaml")
                return False

            ext_name = config.get("extension", {}).get("name")
            if not ext_name:
                print(f"[Operations] ❌ No extension name in config")
                return False

            # Determine target directory
            if config.get("extension", {}).get("category") == "service":
                target_dir = extensions_root / "services" / ext_name
            else:
                target_dir = extensions_root / "features" / ext_name

            # Copy files to target location
            if target_dir.exists():
                print(f"[Operations] ⚠️ Extension already exists: {ext_name}")
                return False

            print(f"[Operations] Installing to: {target_dir}")
            shutil.copytree(ext_dir, target_dir)

            # Set up venv if needed
            if config.get("dependencies", {}).get("use_venv"):
                await self._setup_venv(target_dir)

            # Update enabled flag in config
            if enabled is not None:
                await self._update_config_enabled(target_dir / "config.yaml", enabled)

            # Validate and register
            if enabled:
                try:
                    ext = await self.loader.instantiate(config, target_dir)
                    await ext.initialize(context)
                    self.registry.register(ext)
                    print(f"[Operations] ✅ Installed and enabled: {ext_name}")
                except Exception as e:
                    print(f"[Operations] ⚠️ Installed but failed to enable: {e}")
                    # Still return True since files were copied
            else:
                print(f"[Operations] ✅ Installed (disabled): {ext_name}")

            self.metrics.record_install(ext_name)
            return True

    async def _install_from_local(
        self,
        local_path: str,
        context,
        enabled: bool = True,
    ) -> bool:
        """Install extension from local directory."""
        source_dir = Path(local_path).resolve()

        if not source_dir.exists():
            print(f"[Operations] ❌ Path does not exist: {local_path}")
            return False

        config_path = source_dir / "config.yaml"
        if not config_path.exists():
            print(f"[Operations] ❌ No config.yaml found in: {local_path}")
            return False

        # Load config
        config = self.discoverer._load_config(config_path)
        if not config:
            print(f"[Operations] ❌ Failed to load config.yaml")
            return False

        ext_name = config.get("extension", {}).get("name")
        if not ext_name:
            print(f"[Operations] ❌ No extension name in config")
            return False

        extensions_root = self.discoverer.extensions_dir

        # Determine target directory based on current location
        # If it's already in extensions/, just register it
        try:
            relative_path = source_dir.relative_to(extensions_root)
            # Already in extensions directory
            target_dir = source_dir
        except ValueError:
            # Not in extensions directory, need to copy
            if config.get("extension", {}).get("category") == "service":
                target_dir = extensions_root / "services" / ext_name
            else:
                target_dir = extensions_root / "features" / ext_name

            if target_dir.exists():
                print(f"[Operations] ⚠️ Extension already exists: {ext_name}")
                return False

            print(f"[Operations] Copying to: {target_dir}")
            shutil.copytree(source_dir, target_dir)

        # Set up venv if needed
        if config.get("dependencies", {}).get("use_venv"):
            await self._setup_venv(target_dir)

        # Update enabled flag in config
        if enabled is not None:
            await self._update_config_enabled(target_dir / "config.yaml", enabled)

        # Validate and register
        if enabled:
            try:
                ext = await self.loader.instantiate(config, target_dir)
                await ext.initialize(context)
                self.registry.register(ext)
                print(f"[Operations] ✅ Installed and enabled: {ext_name}")
            except Exception as e:
                print(f"[Operations] ⚠️ Installed but failed to enable: {e}")
        else:
            print(f"[Operations] ✅ Installed (disabled): {ext_name}")

        self.metrics.record_install(ext_name)
        return True

    async def _setup_venv(self, extension_dir: Path) -> bool:
        """Set up virtual environment for extension."""
        venv_dir = extension_dir / ".venv"

        try:
            print(f"[Operations] Setting up venv for: {extension_dir.name}")

            # Create venv
            subprocess.run(
                ["python", "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True
            )

            # Install dependencies if requirements.txt exists
            requirements = extension_dir / "requirements.txt"
            if requirements.exists():
                pip_path = venv_dir / "bin" / "pip"
                if not pip_path.exists():
                    pip_path = venv_dir / "Scripts" / "pip.exe"  # Windows

                subprocess.run(
                    [str(pip_path), "install", "-r", str(requirements)],
                    check=True,
                    capture_output=True
                )
                print(f"[Operations] ✅ Installed dependencies from requirements.txt")

            return True
        except subprocess.CalledProcessError as e:
            print(f"[Operations] ⚠️ Venv setup failed: {e}")
            return False

    async def _update_config_enabled(self, config_path: Path, enabled: bool) -> bool:
        """Update the enabled flag in config.yaml."""
        try:
            # Use shared config utility for loading
            config = load_yaml_config(config_path, required=False)

            if config is None:
                config = {}

            if "extension" not in config:
                config["extension"] = {}

            config["extension"]["enabled"] = enabled

            # Use shared config utility for saving
            save_yaml_config(config_path, config)

            return True
        except Exception as e:
            print(f"[Operations] ⚠️ Failed to update config: {e}")
            return False

    async def uninstall(self, name: str) -> bool:
        """
        Uninstall an extension.

        Args:
            name: Extension name

        Returns:
            True if successful
        """
        ext = self.registry.get(name)
        if not ext:
            print(f"[Operations] Extension not found: {name}")
            return False

        print(f"[Operations] Uninstalling: {name}")

        # Get extension directory before shutdown
        ext_dir = self.discoverer.get_extension_path(name)
        if not ext_dir:
            print(f"[Operations] Cannot find extension directory: {name}")
            return False

        # Shutdown first
        try:
            await ext.shutdown()
        except Exception as e:
            print(f"[Operations] Error during shutdown: {e}")

        # Remove from registry
        self.registry.remove(name)

        # Remove files from disk
        try:
            if ext_dir.exists():
                shutil.rmtree(ext_dir)
                print(f"[Operations] ✅ Removed files: {ext_dir}")
            else:
                print(f"[Operations] ⚠️ Directory already removed: {ext_dir}")
        except Exception as e:
            print(f"[Operations] ⚠️ Failed to remove files: {e}")
            # Still return True since extension is unregistered

        self.metrics.record_uninstall(name)
        print(f"[Operations] ✅ Uninstalled: {name}")
        return True

    async def enable(self, name: str, context) -> bool:
        """
        Enable a disabled extension.

        Args:
            name: Extension name
            context: Extension context

        Returns:
            True if successful
        """
        ext_dir = self.discoverer.get_extension_path(name)
        if not ext_dir:
            print(f"[Operations] Extension not found: {name}")
            return False

        config_path = ext_dir / "config.yaml"

        # Update enabled flag in config
        await self._update_config_enabled(config_path, True)

        # Load and initialize
        config = self.discoverer._load_config(config_path)
        if not config:
            return False

        try:
            ext = await self.loader.instantiate(config, ext_dir)
            await ext.initialize(context)
            self.registry.register(ext)
            print(f"[Operations] ✅ Enabled: {name}")
            return True
        except Exception as e:
            print(f"[Operations] ❌ Failed to enable {name}: {e}")
            return False

    async def disable(self, name: str) -> bool:
        """
        Disable an extension without uninstalling.

        Args:
            name: Extension name

        Returns:
            True if successful
        """
        ext = self.registry.get(name)
        if not ext:
            print(f"[Operations] Extension not found: {name}")
            return False

        print(f"[Operations] Disabling: {name}")

        # Get extension directory before shutdown
        ext_dir = self.discoverer.get_extension_path(name)

        # Shutdown
        try:
            await ext.shutdown()
        except Exception as e:
            print(f"[Operations] Error during shutdown: {e}")

        # Remove from registry but keep files
        self.registry.remove(name)

        # Update config.yaml enabled flag
        if ext_dir:
            await self._update_config_enabled(ext_dir / "config.yaml", False)

        print(f"[Operations] ✅ Disabled: {name}")
        return True

    async def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """
        Get telemetry metrics.

        Args:
            name: Optional extension name. If None, returns all metrics.

        Returns:
            Metrics dict
        """
        if name:
            return self.metrics.get_extension_metrics(name)
        return self.metrics.get_all_metrics()
