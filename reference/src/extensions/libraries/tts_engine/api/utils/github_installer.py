"""
Shared utility for installing packages from GitHub repositories and common installation progress indicators.
"""

import os
import subprocess
import sys

from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


def install_from_github(repo_url: str, package_name: str = None, backend_dir: str = None) -> bool:
    """
    Install a package from a GitHub repository.

    Args:
        repo_url: URL of the GitHub repository to install from
        package_name: Optional package name for error reporting
        backend_dir: Optional backend directory to run the installation from (default: current directory)

    Returns:
        bool: True if installation was successful, False otherwise
    """
    console = Console()
    package_name = package_name or repo_url.split('/')[-1]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create a temporary directory for cloning
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, package_name)

            try:
                # Clone the repository
                task = progress.add_task(f"Cloning {package_name} from {repo_url}...", total=None)

                clone_result = subprocess.run(
                    ["git", "clone", repo_url, repo_dir],
                    capture_output=True,
                    text=True,
                )

                if clone_result.returncode != 0:
                    progress.update(task, description=f"Failed to clone {package_name} from {repo_url}", completed=True)
                    rprint(f"[red]✗ Failed to clone {package_name} repository: {clone_result.stderr}[/red]")
                    return False

                progress.update(task, description=f"Repository {package_name} cloned successfully", completed=True)

                # Install extra requirements if they exist
                extra_req_file = os.path.join(repo_dir, "extra-req.txt")
                if os.path.exists(extra_req_file):
                    task = progress.add_task(f"Installing extra requirements for {package_name}...", total=None)

                    # Use specific backend venv path instead of --system flag
                    backend_python = os.path.join(backend_dir, ".venv", "bin", "python") if backend_dir else None
                    req_cmd = ["uv", "pip", "install", "-r", "extra-req.txt", "--no-deps"]
                    if backend_python and os.path.exists(backend_python):
                        req_cmd.append(f"--python={backend_python}")
                    else:
                        # Fallback to using current environment
                        req_cmd.append("--python")
                        req_cmd.append(sys.executable)

                    # Print installation command in ComfyUI style
                    print(f"CozyTTS-Manager: EXECUTE => {req_cmd}")

                    extra_req_result = subprocess.run(
                        req_cmd,
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                    )

                    if extra_req_result.returncode != 0:
                        progress.update(task, description=f"Failed to install extra requirements for {package_name}", completed=True)
                        rprint(f"[yellow]⚠ Failed to install extra requirements: {extra_req_result.stderr}[/yellow]")
                    else:
                        progress.update(task, description=f"Extra requirements for {package_name} installed", completed=True)
                else:
                    rprint(f"[yellow]⚠ No extra requirements file found for {package_name}, skipping...[/yellow]")

                # Install requirements
                req_file = os.path.join(repo_dir, "requirements.txt")
                if os.path.exists(req_file):
                    task = progress.add_task(f"Installing requirements for {package_name}...", total=None)

                    # Use specific backend venv path instead of --system flag
                    backend_python = os.path.join(backend_dir, ".venv", "bin", "python") if backend_dir else None
                    req_cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
                    if backend_python and os.path.exists(backend_python):
                        req_cmd.append(f"--python={backend_python}")
                    else:
                        # Fallback to using current environment
                        req_cmd.append("--python")
                        req_cmd.append(sys.executable)

                    # Print installation command in ComfyUI style
                    print(f"CozyTTS-Manager: EXECUTE => {req_cmd}")

                    req_result = subprocess.run(
                        req_cmd,
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                    )

                    if req_result.returncode != 0:
                        progress.update(task, description=f"Failed to install requirements for {package_name}", completed=True)
                        rprint(f"[yellow]⚠ Failed to install requirements: {req_result.stderr}[/yellow]")
                        # Continue anyway as some dependencies might already be installed
                    else:
                        progress.update(task, description=f"Requirements for {package_name} installed", completed=True)
                else:
                    rprint(f"[yellow]⚠ No requirements.txt file found for {package_name}, skipping...[/yellow]")

                # Install the package in development mode
                task = progress.add_task(f"Installing {package_name} package...", total=None)

                # Determine the install command based on available files
                setup_files = ["setup.py", "pyproject.toml"]
                setup_exists = any(os.path.exists(os.path.join(repo_dir, f)) for f in setup_files)

                if setup_exists:
                    # Use --python flag to target the backend virtual environment
                    # Determine correct backend Python path
                    target_backend_dir = backend_dir if backend_dir else os.getcwd()
                    if not target_backend_dir.endswith("backend"):
                        # Go up one level from extension dir to backend dir
                        target_backend_dir = os.path.join(target_backend_dir, "..")

                    backend_python = os.path.join(target_backend_dir, ".venv", "bin", "python")
                    install_cmd = ["uv", "pip", "install", "-e", repo_dir]
                    if os.path.exists(backend_python):
                        install_cmd = ["uv", "pip", "install", "-e", repo_dir, f"--python={backend_python}"]
                    else:
                        # Fallback to using current Python environment
                        install_cmd = ["uv", "pip", "install", "-e", repo_dir, "--python", sys.executable]

                    # Print installation command in ComfyUI style
                    print(f"CozyTTS-Manager: EXECUTE => {install_cmd}")

                    # Run from the backend directory to ensure proper environment
                    install_result = subprocess.run(
                        install_cmd,
                        capture_output=True,
                        text=True,
                        cwd=target_backend_dir if os.path.exists(target_backend_dir) else os.getcwd(),
                    )

                    if install_result.returncode == 0:
                        progress.update(task, description=f"{package_name} installed successfully", completed=True)
                        rprint(f"[green]✓ {package_name} installed successfully[/green]")
                        return True
                    else:
                        progress.update(task, description=f"Failed to install {package_name} package", completed=True)
                        rprint(f"[red]✗ Failed to install {package_name}: {install_result.stderr}[/red]")
                        return False
                else:
                    # If no setup files, try to manually install the package
                    rprint(f"[yellow]⚠ No standard setup files found for {package_name}, attempting alternative installation...[/yellow]")

                    # Use --python flag to target the backend virtual environment
                    # Determine correct backend Python path
                    target_backend_dir = backend_dir if backend_dir else os.getcwd()
                    if not target_backend_dir.endswith("backend"):
                        # Go up one level from extension dir to backend dir
                        target_backend_dir = os.path.join(target_backend_dir, "..")

                    backend_python = os.path.join(target_backend_dir, ".venv", "bin", "python")
                    install_cmd = ["uv", "pip", "install", repo_dir]
                    if os.path.exists(backend_python):
                        install_cmd = ["uv", "pip", "install", repo_dir, f"--python={backend_python}"]
                    else:
                        # Fallback to using current Python environment
                        install_cmd = ["uv", "pip", "install", repo_dir, "--python", sys.executable]

                    # Print installation command in ComfyUI style
                    print(f"CozyTTS-Manager: EXECUTE => {install_cmd}")

                    # Run from the backend directory to ensure proper environment
                    install_result = subprocess.run(
                        install_cmd,
                        capture_output=True,
                        text=True,
                        cwd=target_backend_dir if os.path.exists(target_backend_dir) else os.getcwd(),
                    )

                    if install_result.returncode == 0:
                        progress.update(task, description=f"{package_name} installed successfully", completed=True)
                        rprint(f"[green]✓ {package_name} installed successfully[/green]")
                        return True
                    else:
                        progress.update(task, description=f"Failed to install {package_name} package", completed=True)
                        rprint(f"[red]✗ Failed to install {package_name}: {install_result.stderr}[/red]")
                        return False

            except Exception as e:
                rprint(f"[red]✗ Error during installation of {package_name}: {str(e)}[/red]")
                return False


def install_requirements_with_progress(req_file_path: str, backend_dir: str) -> bool:
    """
    Install requirements from a file with progress indicators.

    Args:
        req_file_path: Path to the requirements.txt file
        backend_dir: Backend directory to run the installation from

    Returns:
        bool: True if installation was successful, False otherwise
    """
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            task = progress.add_task(f"Installing dependencies from {os.path.basename(req_file_path)}...", total=None)

            # Use the backend virtual environment's Python
            backend_python = os.path.join(backend_dir, ".venv", "bin", "python")
            install_cmd = ["uv", "pip", "install", "-r", req_file_path]
            if os.path.exists(backend_python):
                install_cmd = ["uv", "pip", "install", "-r", req_file_path, f"--python={backend_python}"]
            else:
                # Fallback to using current Python environment
                install_cmd = ["uv", "pip", "install", "-r", req_file_path, "--python", sys.executable]

            # Print installation command in ComfyUI style
            print(f"CozyTTS-Manager: EXECUTE => {install_cmd}")

            install_result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                cwd=backend_dir,
            )

            if install_result.returncode == 0:
                progress.update(task, description="Dependencies installed successfully", completed=True)
                rprint("[green]✓ Dependencies installed successfully[/green]")
                return True
            else:
                progress.update(task, description="Failed to install dependencies", completed=True)
                rprint(f"[red]✗ Failed to install dependencies: {install_result.stderr}[/red]")
                return False

        except Exception as e:
            rprint(f"[red]✗ Error during dependency installation: {str(e)}[/red]")
            return False


def install_package_with_progress(package_name: str, backend_dir: str, install_cmd: list = None) -> bool:
    """
    Install a specific package with progress indicators.

    Args:
        package_name: Name of the package to install
        backend_dir: Backend directory to run the installation from
        install_cmd: Optional custom install command (default: ["uv", "pip", "install", package_name])

    Returns:
        bool: True if installation was successful, False otherwise
    """
    console = Console()

    if install_cmd is None:
        # Use the backend virtual environment's Python
        backend_python = os.path.join(backend_dir, ".venv", "bin", "python")
        if os.path.exists(backend_python):
            install_cmd = ["uv", "pip", "install", package_name, f"--python={backend_python}"]
        else:
            # Fallback to using current Python environment
            install_cmd = ["uv", "pip", "install", package_name, "--python", sys.executable]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        try:
            task = progress.add_task(f"Installing {package_name}...", total=None)

            # Print installation command in ComfyUI style
            print(f"CozyTTS-Manager: EXECUTE => {install_cmd}")

            install_result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                cwd=backend_dir,
            )

            if install_result.returncode == 0:
                progress.update(task, description=f"{package_name} installed successfully", completed=True)
                rprint(f"[green]✓ {package_name} installed successfully[/green]")
                return True
            else:
                progress.update(task, description=f"Failed to install {package_name}", completed=True)
                rprint(f"[red]✗ Failed to install {package_name}: {install_result.stderr}[/red]")
                return False

        except Exception as e:
            rprint(f"[red]✗ Error during installation of {package_name}: {str(e)}[/red]")
            return False
