"""Built-in CLI Commands

Commands for install, server, download, and help.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_backend_root():
    """Get the backend root directory."""
    script_path = Path(__file__).resolve()
    # src/cli/commands.py -> backend/
    return script_path.parent.parent.parent


def handle_install(args):
    """Handle install command - delegates to install script."""
    backend_root = get_backend_root()
    python_exe = sys.executable
    module = "scripts.install.install"
    full_cmd = [python_exe, "-m", module] + (args.extra or [])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_root)

    try:
        subprocess.run(full_cmd, check=True, cwd=backend_root, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


def handle_server(args):
    """Handle server command - delegates to server script."""
    backend_root = get_backend_root()
    python_exe = sys.executable
    module = "scripts.server.server"
    full_cmd = [python_exe, "-m", module] + (args.extra or [])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_root)

    try:
        subprocess.run(full_cmd, check=True, cwd=backend_root, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


def handle_download(args):
    """Handle download command - delegates to download script."""
    backend_root = get_backend_root()
    python_exe = sys.executable
    module = "scripts.download.download"
    full_cmd = [python_exe, "-m", module] + (args.extra or [])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_root)

    try:
        subprocess.run(full_cmd, check=True, cwd=backend_root, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


def handle_help(args):
    """Handle help command - show comprehensive help."""
    from src.cli.discovery import discover_cli_plugins

    backend_root = get_backend_root()
    extension_dirs = [
        "src.extensions.features",
        "src.extensions.libraries",
        "src.extensions.services",
    ]

    # Add parent dirs to path
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    print("\n" + "=" * 60)
    print("  Unified LLM Interface - Command Center")
    print("=" * 60)

    print("\nModular System Commands:")
    print("-" * 60)
    print("  install     - Installation system (ComfyUI, llama.cpp, etc.)")
    print("  server      - Server management (start/stop services)")
    print("  download    - Download system (models, extensions)")

    # Discover extension plugins
    plugins = discover_cli_plugins(extension_dirs)

    if plugins:
        print("\nExtension Commands:")
        print("-" * 60)
        for name in sorted(plugins.keys()):
            print(f"  {name:<20} - Extension commands")

    print("\nUsage:")
    print("  uv run backend/manage.py <command> [args...]")
    print()
    print("Examples:")
    print("  uv run backend/manage.py install --all")
    print("  uv run backend/manage.py server --start")
    print("  uv run backend/manage.py <extension> --help")
    print()
