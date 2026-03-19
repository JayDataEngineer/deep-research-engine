"""
Cozy TTS Extension Installer Utilities

Provides reusable functions for extension installation scripts.
Inspired by ComfyUI's extension system but adapted for UV/Python environments.

Note: Venv management functions now imported from centralized VenvManager
to eliminate code duplication across the codebase.
"""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Import venv management from centralized module
from src.shared.utils.venv_manager import (
    get_backend_dir,
    get_extension_dir,
    get_venv_python,
    ensure_extension_venv,
    get_uv_command,
    DEFAULT_PYTHON_VERSION,
)

console = Console()


# Re-export for backward compatibility
__all__ = [
    "get_backend_dir",
    "get_extension_dir",
    "get_venv_python",
    "ensure_extension_venv",
    "get_uv_command",
    "DEFAULT_PYTHON_VERSION",
    "check_git_available",
    "check_git_lfs_available",
    "clone_or_update_repo",
    "install_requirements",
    "install_package",
    "install_local_package",
    "run_pip_command",
    "check_import",
    "check_version",
    "install_extension_template",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
]


# ============================================================================
# GIT UTILITIES
# ============================================================================

def check_git_available() -> bool:
    """Check if git is available."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_git_lfs_available() -> bool:
    """Check if git-lfs is available."""
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clone_or_update_repo(
    repo_url: str,
    target_dir: Path,
    use_lfs: bool = True,
    branch: str | None = None
) -> bool:
    """
    Clone a git repository or update if it exists.

    Args:
        repo_url: Git repository URL
        target_dir: Directory to clone to
        use_lfs: Whether to use git lfs pull
        branch: Branch to checkout (optional, defaults to repo default)

    Returns:
        True if successful
    """
    if not check_git_available():
        console.print("[red]Error: Git is not installed.[/red]")
        return False

    has_lfs = check_git_lfs_available() if use_lfs else False

    if target_dir.exists():
        console.print(f"[yellow]Repository already exists at {target_dir}[/yellow]")

        # Check if we're in detached HEAD state
        try:
            result = subprocess.run(
                ["git", "-C", str(target_dir), "symbolic-ref", "-q", "HEAD"],
                capture_output=True,
                text=True,
                check=False
            )
            is_detached = result.returncode != 0

            if is_detached:
                console.print("[yellow]Detached HEAD detected, fetching updates...[/yellow]")
                # Fetch all updates
                subprocess.run(
                    ["git", "-C", str(target_dir), "fetch", "--all", "--tags"],
                    capture_output=True,
                    check=True
                )

                # Checkout the latest tag or main branch
                if branch:
                    checkout_target = branch
                else:
                    # Try to get the latest tag
                    result = subprocess.run(
                        ["git", "-C", str(target_dir), "describe", "--tags", "--abbrev=0"],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0:
                        checkout_target = result.stdout.strip()
                        console.print(f"[yellow]Checking out latest tag: {checkout_target}[/yellow]")
                    else:
                        # Fall back to main/master
                        for branch_name in ["main", "master"]:
                            result = subprocess.run(
                                ["git", "-C", str(target_dir), "ls-remote", "--heads", "origin", branch_name],
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                checkout_target = branch_name
                                break
                        else:
                            console.print("[red]Error: Could not determine branch to checkout[/red]")
                            return False

                subprocess.run(
                    ["git", "-C", str(target_dir), "checkout", checkout_target],
                    capture_output=True,
                    check=True
                )
                console.print(f"[green]✓ Checked out {checkout_target}[/green]")
            else:
                # Normal branch - just pull
                console.print("[yellow]Updating...[/yellow]")
                subprocess.run(
                    ["git", "-C", str(target_dir), "pull"],
                    capture_output=True,
                    check=True
                )

            if has_lfs:
                subprocess.run(
                    ["git", "-C", str(target_dir), "lfs", "pull"],
                    capture_output=True,
                    check=False
                )
            console.print("[green]✓ Repository updated[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error updating repository: {e}[/red]")
            return False
    else:
        console.print(f"[yellow]Cloning {repo_url}...[/yellow]")
        try:
            # Create parent directory if needed
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            clone_cmd = ["git", "clone", repo_url, str(target_dir)]
            if branch:
                clone_cmd.extend(["--branch", branch])

            subprocess.run(
                clone_cmd,
                capture_output=True,
                check=True
            )

            if has_lfs:
                subprocess.run(
                    ["git", "-C", str(target_dir), "lfs", "pull"],
                    capture_output=True,
                    check=False
                )

            console.print("[green]✓ Repository cloned[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error cloning repository: {e}[/red]")
            return False


# ============================================================================
# INSTALLATION UTILITIES
# ============================================================================

def install_requirements(
    requirements_file: Path,
    backend_dir: Path | None = None
) -> bool:
    """
    Install packages from a requirements file.

    Args:
        requirements_file: Path to requirements.txt
        backend_dir: Backend directory (defaults to auto-detection)

    Returns:
        True if successful
    """
    if not requirements_file.exists():
        console.print(f"[yellow]Requirements file not found: {requirements_file}[/yellow]")
        return False

    # Check if requirements file contains --extra-index-url
    # If so, we need to use --index-strategy unsafe-best-match to avoid
    # dependency conflicts when packages exist in multiple indexes
    has_extra_index = False
    try:
        with open(requirements_file) as f:
            for line in f:
                if "--extra-index-url" in line:
                    has_extra_index = True
                    break
    except OSError:
        pass

    # If no backend_dir specified, check if we're in an extension
    if backend_dir is None:
        extension_dir = get_extension_dir()
        if extension_dir:
            # Use extension's venv
            venv_python = ensure_extension_venv(extension_dir)
            work_dir = extension_dir
        else:
            backend_dir = get_backend_dir()
            venv_python = get_venv_python(backend_dir)
            work_dir = backend_dir
    else:
        venv_python = get_venv_python(backend_dir)
        work_dir = backend_dir

    cmd = ["uv", "pip", "install", "-r", str(requirements_file)]
    if venv_python:
        cmd.extend(["--python", str(venv_python)])
    if has_extra_index:
        cmd.extend(["--index-strategy", "unsafe-best-match"])

    console.print(f"[yellow]Installing from {requirements_file.name}...[/yellow]")
    print(f"CozyTTS-Installer: EXECUTE => {cmd}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir)
        )

        if result.returncode == 0:
            console.print("[green]✓ Requirements installed[/green]")
            return True
        else:
            console.print(f"[red]Error: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def install_package(
    package_spec: str,
    backend_dir: Path | None = None,
    editable: bool = False
) -> bool:
    """
    Install a single package.

    Args:
        package_spec: Package specification (name, path, or URL)
        backend_dir: Backend directory (defaults to auto-detection)
        editable: Whether to install in editable mode (-e)

    Returns:
        True if successful
    """
    # If no backend_dir specified, check if we're in an extension
    if backend_dir is None:
        extension_dir = get_extension_dir()
        if extension_dir:
            # Use extension's venv
            venv_python = ensure_extension_venv(extension_dir)
            work_dir = extension_dir
        else:
            backend_dir = get_backend_dir()
            venv_python = get_venv_python(backend_dir)
            work_dir = backend_dir
    else:
        venv_python = get_venv_python(backend_dir)
        work_dir = backend_dir

    cmd = ["uv", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(package_spec)

    if venv_python:
        cmd.extend(["--python", str(venv_python)])

    console.print(f"[yellow]Installing {package_spec}...[/yellow]")
    print(f"CozyTTS-Installer: EXECUTE => {cmd}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir)
        )

        # Print output for debugging
        if result.stdout:
            print(f"stdout: {result.stdout[-500:]}")
        if result.stderr and "warning" in result.stderr.lower():
            # Only print stderr if it contains warnings (not errors)
            print(f"stderr (warnings): {result.stderr[-500:]}")

        if result.returncode == 0:
            console.print("[green]✓ Package installed[/green]")
            return True
        else:
            console.print(f"[red]Error (code {result.returncode}): {result.stderr[-500:]}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def install_local_package(
    package_dir: Path,
    backend_dir: Path | None = None
) -> bool:
    """
    Install a local package in editable mode.

    Looks for pyproject.toml or setup.py to determine if it's installable.

    Args:
        package_dir: Directory containing the package
        backend_dir: Backend directory (defaults to auto-detection)

    Returns:
        True if successful
    """
    if not package_dir.exists():
        console.print(f"[red]Package directory not found: {package_dir}[/red]")
        return False

    # Check if it's a Python package
    has_pyproject = (package_dir / "pyproject.toml").exists()
    has_setup = (package_dir / "setup.py").exists()

    if not (has_pyproject or has_setup):
        console.print(f"[yellow]No pyproject.toml or setup.py found in {package_dir}[/yellow]")
        return False

    return install_package(str(package_dir), backend_dir, editable=True)


def run_pip_command(
    args: list[str],
    backend_dir: Path | None = None
) -> bool:
    """
    Run a custom pip command via UV.

    Args:
        args: Arguments to pass to pip (e.g., ["install", "package==1.0.0"])
        backend_dir: Backend directory (defaults to auto-detection)

    Returns:
        True if successful
    """
    # If no backend_dir specified, check if we're in an extension
    if backend_dir is None:
        extension_dir = get_extension_dir()
        if extension_dir:
            # Use extension's venv
            venv_python = ensure_extension_venv(extension_dir)
            work_dir = extension_dir
        else:
            backend_dir = get_backend_dir()
            venv_python = get_venv_python(backend_dir)
            work_dir = backend_dir
    else:
        venv_python = get_venv_python(backend_dir)
        work_dir = backend_dir

    cmd = ["uv", "pip"] + args
    if venv_python:
        cmd.extend(["--python", str(venv_python)])

    print(f"CozyTTS-Installer: EXECUTE => {cmd}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir)
        )

        if result.returncode == 0:
            return True
        else:
            console.print(f"[red]Error: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def check_import(module_name: str) -> bool:
    """
    Check if a module can be imported.

    Args:
        module_name: Name of the module to check

    Returns:
        True if import succeeds
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_version(package_name: str) -> str | None:
    """
    Get the version of an installed package.

    Args:
        package_name: Name of the package

    Returns:
        Version string, or None if not installed
    """
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata

    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


# ============================================================================
# INSTALLATION TEMPLATE
# ============================================================================

def install_extension_template(
    extension_name: str,
    install_func: callable,
    dependencies: list[str] | None = None,
    python_version: str | None = None
) -> bool:
    """
    Template for extension installation scripts.

    Args:
        extension_name: Display name of the extension
        install_func: Function that performs the actual installation
        dependencies: Optional list of dependencies to check/install first
        python_version: Required Python version (e.g., "3.10", "3.11", "3.12")

    Returns:
        True if successful
    """
    console.print(Panel(f"[bold blue]Installing {extension_name}[/bold blue]"))

    # Ensure extension venv exists with correct Python version
    venv_python = ensure_extension_venv(python_version=python_version)

    if not venv_python:
        console.print("[red]Error: Failed to create extension virtual environment![/red]")
        return False

    console.print(f"[green]✓ Using Python: {venv_python}[/green]")

    # Install dependencies if provided
    if dependencies:
        for dep in dependencies:
            if not check_import(dep):
                console.print(f"[yellow]Installing dependency: {dep}[/yellow]")
                if not install_package(dep):
                    console.print(f"[red]Failed to install {dep}[/red]")
                    return False

    # Run the install function
    try:
        success = install_func()
        if success:
            console.print(Panel(f"[bold green]{extension_name} installed successfully![/bold green]"))
        else:
            console.print(Panel(f"[bold red]{extension_name} installation failed![/bold red]"))
        return success
    except Exception as e:
        console.print(f"[red]Installation error: {e}[/red]")
        console.print(Panel(f"[bold red]{extension_name} installation failed![/bold red]"))
        return False


# ============================================================================
# CONSOLE PRINTING HELPERS
# ============================================================================

def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓ {message}[/green]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗ {message}[/red]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠ {message}[/yellow]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ {message}[/blue]")
