"""Centralized Virtual Environment Manager.

Provides unified venv detection and management for the backend and extensions.
Consolidates duplicate venv handling patterns from extension_installer.py
and extension_manager.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

# Default Python version - can be overridden by extensions
DEFAULT_PYTHON_VERSION = "3.10"


def get_backend_dir() -> Path:
    """Get the backend directory path from this script's location."""
    # This file is at: backend/src/core/venv_manager.py
    # Go up: src/core -> src -> backend (3 levels)
    return Path(__file__).resolve().parent.parent.parent


def get_extension_dir() -> Path | None:
    """
    Get the extension directory from the call stack.

    Inspects the call stack to find which extension directory is calling
    this function. Used by install.py scripts to auto-detect their location.

    Returns:
        Path to the extension directory, or None if not found
    """
    import inspect
    frame = inspect.currentframe()
    try:
        for _ in range(10):  # Check up to 10 frames
            frame = frame.f_back
            if frame is None:
                break
            frame_file = frame.f_code.co_filename
            if "install.py" in frame_file:
                # Found install.py - get its extension directory
                install_path = Path(frame_file).parent.resolve()
                # Check if it's in an extensions subdirectory
                if "extensions" in install_path.parts:
                    return install_path
    finally:
        del frame
    return None


def get_venv_python(
    backend_dir: Path | None = None,
    extension_name: str | None = None,
    extension_dir: Path | None = None,
) -> Path | None:
    """
    Get the Python executable path in a virtual environment.

    Tries multiple strategies to find the venv:
    1. Specific extension venv if extension_name provided
    2. Call stack inspection to find calling extension's venv
    3. Backend/.venv (legacy location)
    4. Project root .venv

    Args:
        backend_dir: Backend directory (defaults to auto-detection)
        extension_name: Specific extension name (for targeted lookup)
        extension_dir: Extension directory path (for direct lookup)

    Returns:
        Path to python executable, or None if not found
    """
    if backend_dir is None:
        backend_dir = get_backend_dir()

    # Strategy 1: If extension_name provided, look for that specific extension's venv
    if extension_name:
        ext_venv = backend_dir / "extensions" / extension_name / ".venv"
        if ext_venv.exists():
            python_exe = _get_python_from_venv(ext_venv)
            if python_exe:
                return python_exe

    # Strategy 2: If extension_dir provided, check its venv directly
    if extension_dir:
        ext_venv = extension_dir / ".venv"
        if ext_venv.exists():
            python_exe = _get_python_from_venv(ext_venv)
            if python_exe:
                return python_exe

    # Strategy 3: Check if we're being called from within an extension directory
    # by inspecting the call stack
    import inspect
    frame = inspect.currentframe()
    try:
        # Go up the stack to find the install.py file
        for _ in range(10):  # Check up to 10 frames
            frame = frame.f_back
            if frame is None:
                break
            frame_file = frame.f_code.co_filename
            if "install.py" in frame_file:
                # Found install.py - get its extension directory
                install_path = Path(frame_file).parent.resolve()
                # Check if it's in an extensions subdirectory
                if "extensions" in install_path.parts:
                    # Found the extension's venv
                    ext_venv = install_path / ".venv"
                    if ext_venv.exists():
                        python_exe = _get_python_from_venv(ext_venv)
                        if python_exe:
                            return python_exe
    finally:
        del frame

    # Strategy 4: Check backend/.venv (legacy location)
    venv_path = backend_dir / ".venv"
    if venv_path.exists():
        python_exe = _get_python_from_venv(venv_path)
        if python_exe:
            return python_exe

    # Strategy 5: Last resort - project root .venv
    project_root = backend_dir.parent
    venv_path = project_root / ".venv"
    if venv_path.exists():
        python_exe = _get_python_from_venv(venv_path)
        if python_exe:
            return python_exe

    return None


def _get_python_from_venv(venv_path: Path) -> Path | None:
    """Get the Python executable from a venv directory."""
    if os.name == "nt":  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        python_exe = venv_path / "bin" / "python"
    return python_exe if python_exe.exists() else None


def ensure_extension_venv(
    extension_dir: Path | None = None,
    python_version: str | None = None
) -> Path | None:
    """
    Ensure the extension has a virtual environment with the correct Python version.

    Creates a new venv if one doesn't exist, or recreates it if Python version
    doesn't match the required version.

    The intended Python version is stored in a .venv/python_version.json file
    to avoid oscillating between different Python versions when called without
    an explicit version parameter.

    Args:
        extension_dir: Extension directory (defaults to auto-detection via call stack)
        python_version: Required Python version (e.g., "3.10", "3.11", "3.12")

    Returns:
        Path to python executable, or None if failed
    """
    if extension_dir is None:
        extension_dir = get_extension_dir()

    if extension_dir is None:
        console.print("[red]Error: Could not detect extension directory[/red]")
        return None

    ext_venv = extension_dir / ".venv"
    version_marker = ext_venv / "python_version.json"

    # Determine the intended Python version
    # If venv exists and has a version marker, use that
    intended_version = python_version
    if intended_version is None and version_marker.exists():
        try:
            with open(version_marker) as f:
                data = json.load(f)
                intended_version = data.get("python_version")
        except (OSError, json.JSONDecodeError):
            pass

    # If still no version, use default
    if intended_version is None:
        intended_version = DEFAULT_PYTHON_VERSION

    # Check if venv exists
    if ext_venv.exists():
        # Check Python version in existing venv
        python_exe = _get_python_from_venv(ext_venv)
        if python_exe:
            # Check Python version
            try:
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Output is like "Python 3.12.3"
                current_version = result.stdout.strip().split()[-1]
                major_minor = ".".join(current_version.split(".")[:2])

                if major_minor == intended_version:
                    return python_exe
                else:
                    console.print(f"[yellow]Python version mismatch: {major_minor} != {intended_version}[/yellow]")
                    console.print(f"[yellow]Recreating venv with Python {intended_version}...[/yellow]")
                    # Recreate venv
                    import shutil
                    shutil.rmtree(ext_venv)
            except subprocess.CalledProcessError:
                console.print("[yellow]Failed to check Python version, recreating venv...[/yellow]")
                import shutil
                shutil.rmtree(ext_venv)

    # Create new venv with correct Python version
    console.print(f"[yellow]Creating venv with Python {intended_version}...[/yellow]")
    try:
        subprocess.run(
            ["uv", "venv", "--python", intended_version],
            cwd=str(extension_dir),
            check=True
        )
        console.print(f"[green]✓ Created venv with Python {intended_version}[/green]")

        # Store the intended Python version for future calls
        version_marker = ext_venv / "python_version.json"
        try:
            with open(version_marker, "w") as f:
                json.dump({"python_version": intended_version}, f)
        except OSError:
            pass  # Non-critical, continue anyway
    except FileNotFoundError:
        console.print("[red]Error: uv not found. Please install uv first.[/red]")
        return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error creating venv: {e}[/red]")
        return None

    # Return the python executable path
    return _get_python_from_venv(ext_venv)


def get_uv_command(backend_dir: Path | None = None, extension_name: str | None = None) -> list[str]:
    """
    Get the UV command with proper Python targeting.

    Args:
        backend_dir: Backend directory (defaults to auto-detection)
        extension_name: Specific extension name (optional)

    Returns:
        List of command parts starting with 'uv'
    """
    venv_python = get_venv_python(backend_dir, extension_name)

    if venv_python:
        return ["uv", "--python", str(venv_python)]
    else:
        # Fallback: use current Python
        return ["uv", "--python", sys.executable]


__all__ = [
    "get_backend_dir",
    "get_extension_dir",
    "get_venv_python",
    "ensure_extension_venv",
    "get_uv_command",
    "DEFAULT_PYTHON_VERSION",
]
