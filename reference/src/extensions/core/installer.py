"""
Extension Installer

Handles declarative installation of extension dependencies:
- Git repositories
- Model downloads
- Virtual environments
"""

import asyncio
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import httpx


class ExtensionInstaller:
    """
    Installs extension dependencies from config.yaml.

    Supports:
    - Git repository cloning
    - Model file downloads with hash verification
    - Virtual environment creation
    """

    def __init__(self, extensions_dir: Path):
        """
        Initialize the installer.

        Args:
            extensions_dir: Path to the extensions directory
        """
        self.extensions_dir = extensions_dir

    async def install_extension(self, ext_name: str, config: Dict[str, Any]) -> None:
        """
        Install all dependencies for an extension.

        Args:
            ext_name: The extension name
            config: The extension config
        """
        ext_dir = self.extensions_dir / ext_name

        print(f"[Installer] Installing {ext_name}...")

        # 1. Clone git repositories
        await self._install_git_deps(ext_dir, config)

        # 2. Download models
        await self._install_model_deps(ext_dir, config)

        # 3. Create venv if needed
        use_host_env = config.get('python', {}).get('use_host_env', True)
        if not use_host_env:
            await self._create_venv(ext_dir, config)

        # 4. Run install hook if present
        install_hook = config.get('dependencies', {}).get('install_hook')
        if install_hook:
            await self._run_install_hook(ext_dir, install_hook)

        print(f"[Installer] ✅ Installed {ext_name}")

    async def _install_git_deps(self, ext_dir: Path, config: Dict[str, Any]) -> None:
        """Clone git repositories declared in config."""
        git_deps = config.get('dependencies', {}).get('git', [])

        for dep in git_deps:
            url = dep.get('url')
            dest = dep.get('dest')
            branch = dep.get('branch', 'main')

            if not url or not dest:
                continue

            dest_path = ext_dir / dest

            # Skip if already exists
            if dest_path.exists() and (dest_path / '.git').exists():
                print(f"[Installer] Git repo already exists: {dest}")
                continue

            print(f"[Installer] Cloning {url} -> {dest}")

            try:
                await self._git_clone(url, dest_path, branch)
            except Exception as e:
                print(f"[Installer] ⚠️ Failed to clone {url}: {e}")

    async def _git_clone(self, url: str, dest: Path, branch: str) -> None:
        """Clone a git repository."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'git', 'clone',
            '--depth', '1',
            '--branch', branch,
            url,
            str(dest)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Git clone failed: {stderr.decode()}")

    async def _install_model_deps(self, ext_dir: Path, config: Dict[str, Any]) -> None:
        """Download model files with hash verification."""
        model_deps = config.get('dependencies', {}).get('models', [])

        for dep in model_deps:
            url = dep.get('url')
            dest = dep.get('dest')
            expected_hash = dep.get('hash')

            if not url or not dest:
                continue

            dest_path = ext_dir / dest

            # Skip if already exists and hash matches
            if dest_path.exists():
                if expected_hash:
                    actual_hash = await self._hash_file(dest_path)
                    if actual_hash == expected_hash:
                        print(f"[Installer] Model already verified: {dest}")
                        continue
                else:
                    print(f"[Installer] Model already exists: {dest}")
                    continue

            print(f"[Installer] Downloading {url} -> {dest}")

            try:
                await self._download_with_progress(url, dest_path, expected_hash)
            except Exception as e:
                print(f"[Installer] ⚠️ Failed to download {url}: {e}")

    async def _download_with_progress(
        self,
        url: str,
        dest: Path,
        expected_hash: Optional[str] = None
    ) -> None:
        """Download file with progress indication."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=300.0) as response:
                response.raise_for_status()

                total = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(dest, 'wb') as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            pct = (downloaded / total) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                print(f"[Installer] Downloading: {pct:.1f}%")

        # Verify hash if provided
        if expected_hash:
            actual_hash = await self._hash_file(dest)
            if actual_hash != expected_hash:
                dest.unlink()
                raise RuntimeError(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")

    async def _hash_file(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()

        with open(path, 'rb') as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                sha256.update(data)

        return f"sha256:{sha256.hexdigest()}"

    async def _create_venv(self, ext_dir: Path, config: Dict[str, Any]) -> None:
        """Create a virtual environment for the extension."""
        venv_path = ext_dir / '.venv'

        if venv_path.exists():
            print(f"[Installer] Venv already exists: {venv_path}")
            return

        print(f"[Installer] Creating venv: {venv_path}")

        # Create venv
        subprocess.run(
            ['python', '-m', 'venv', str(venv_path)],
            check=True
        )

        # Get pip path
        if os.name == 'nt':
            pip_path = venv_path / 'Scripts' / 'pip'
        else:
            pip_path = venv_path / 'bin' / 'pip'

        # Install requirements
        requirements_file = ext_dir / config.get('python', {}).get('requirements', 'requirements.txt')

        if requirements_file.exists():
            print(f"[Installer] Installing requirements from {requirements_file}")
            subprocess.run(
                [str(pip_path), 'install', '-r', str(requirements_file)],
                check=True
            )

    async def _run_install_hook(self, ext_dir: Path, hook_script: str) -> None:
        """Run a custom install hook script."""
        hook_path = ext_dir / hook_script

        if not hook_path.exists():
            print(f"[Installer] Install hook not found: {hook_path}")
            return

        print(f"[Installer] Running install hook: {hook_script}")

        # Determine python to use
        venv_python = ext_dir / '.venv' / 'bin' / 'python'
        if not venv_python.exists():
            venv_python = ext_dir / '.venv' / 'Scripts' / 'python.exe'
        if not venv_python.exists():
            venv_python = 'python'

        process = await asyncio.create_subprocess_exec(
            str(venv_python),
            str(hook_path),
            cwd=ext_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[Installer] ⚠️ Install hook failed: {stderr.decode()}")

    async def install_all(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Install all extensions.

        Args:
            configs: Mapping of extension name to config
        """
        # Build dependency graph and install in order
        installed = set()

        def can_install(name: str) -> bool:
            """Check if all dependencies are installed."""
            requires = configs[name].get('relationships', {}).get('requires', [])
            for dep in requires:
                if dep['name'] not in installed:
                    return False
            return True

        # Multi-pass installation
        max_passes = len(configs) + 1
        for pass_num in range(max_passes):
            installed_this_pass = []

            for name, config in configs.items():
                if name in installed:
                    continue

                if can_install(name):
                    await self.install_extension(name, config)
                    installed.add(name)
                    installed_this_pass.append(name)

            if not installed_this_pass:
                # No progress made - remaining have unmet dependencies
                remaining = set(configs.keys()) - installed
                if remaining:
                    print(f"[Installer] ⚠️ Could not install (missing deps): {remaining}")
                break
