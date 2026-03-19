"""Memory Monitor CLI Plugin.

Provides commands for GPU and system memory monitoring.
"""

import argparse
import asyncio
import time
from typing import Optional


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_memory_info(data: dict, show_gpu: bool = True, show_ram: bool = True):
    """Print memory information in a formatted way."""
    if show_ram and "ram" in data:
        ram = data["ram"]
        print(f"\n{'='*60}")
        print(f"RAM Memory:")
        print(f"{'='*60}")
        print(f"  Total:     {format_bytes(ram['total'])}")
        print(f"  Used:      {format_bytes(ram['used'])} ({ram['percent']:.1f}%)")
        print(f"  Free:      {format_bytes(ram['free'])}")
        bar_length = 40
        filled = int(bar_length * ram['percent'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"  [{bar}]")

    if show_gpu and "gpu" in data:
        gpu = data["gpu"]
        if gpu.get("count", 0) > 0:
            print(f"\n{'='*60}")
            print(f"GPU Memory ({gpu['count']} device{'s' if gpu['count'] > 1 else ''} found):")
            print(f"{'='*60}")

            # Aggregate stats
            mem = gpu["memory"]
            print(f"  Total:     {format_bytes(mem['total'])}")
            print(f"  Used:      {format_bytes(mem['used'])} ({mem['percent']:.1f}%)")
            print(f"  Free:      {format_bytes(mem['free'])}")
            print(f"  Util:      {gpu['utilization']:.1f}%")
            bar_length = 40
            filled = int(bar_length * mem['percent'] / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  [{bar}]")

            # Per-GPU details
            for gpu_id, info in gpu.get("info", {}).items():
                print(f"\n  {gpu_id.upper()}: {info.get('name', 'Unknown')}")
                if 'total_memory' in info:
                    total = info['total_memory']
                    used = info.get('used_memory', info.get('allocated_memory', 0))
                    percent = (used / total * 100) if total > 0 else 0
                    print(f"    Memory: {format_bytes(used)} / {format_bytes(total)} ({percent:.1f}%)")
                if 'utilization' in info:
                    print(f"    Util:   {info['utilization']}%")
        else:
            print(f"\n{'='*60}")
            print(f"GPU: No NVIDIA GPU detected or pynvml/torch not available")
            print(f"{'='*60}")


def cmd_monitor(args):
    """Display current memory usage."""
    from src.extensions.features.assistant.memory_monitor.extension import MemoryMonitorExtension

    try:
        import yaml
        from pathlib import Path

        root_path = Path(__file__).parent
        config_path = root_path / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        extension = MemoryMonitorExtension(config, root_path)

        async def get_and_display():
            data = await extension.get_memory()
            print_memory_info(data, show_gpu=args.gpu, show_ram=args.ram)

        asyncio.run(get_and_display())

    except Exception as e:
        print(f"Error getting memory info: {e}")
        import traceback
        traceback.print_exc()


def cmd_watch(args):
    """Continuously monitor memory usage."""
    from src.extensions.features.assistant.memory_monitor.extension import MemoryMonitorExtension

    try:
        import yaml
        from pathlib import Path

        root_path = Path(__file__).parent
        config_path = root_path / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        extension = MemoryMonitorExtension(config, root_path)

        print(f"Monitoring memory every {args.interval}s (Press Ctrl+C to stop)...")

        async def monitor_loop():
            try:
                while True:
                    # Clear screen (works on most terminals)
                    print("\033[2J\033[H", end="")

                    data = await extension.get_memory()
                    print_memory_info(data, show_gpu=args.gpu, show_ram=args.ram)

                    print(f"\nUpdating every {args.interval}s... (Ctrl+C to exit)")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")

        asyncio.run(monitor_loop())

    except Exception as e:
        print(f"Error in monitor loop: {e}")
        import traceback
        traceback.print_exc()


def cmd_gpu(args):
    """Display GPU information only."""
    from src.extensions.features.assistant.memory_monitor.extension import MemoryMonitorExtension

    try:
        import yaml
        from pathlib import Path

        root_path = Path(__file__).parent
        config_path = root_path / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        extension = MemoryMonitorExtension(config, root_path)

        async def get_and_display():
            gpu_info = await extension.get_gpu_info()
            print_memory_info({"gpu": gpu_info}, show_gpu=True, show_ram=False)

        asyncio.run(get_and_display())

    except Exception as e:
        print(f"Error getting GPU info: {e}")
        import traceback
        traceback.print_exc()


def cmd_ram(args):
    """Display RAM information only."""
    from src.extensions.features.assistant.memory_monitor.extension import MemoryMonitorExtension

    try:
        import yaml
        from pathlib import Path

        root_path = Path(__file__).parent
        config_path = root_path / "config.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        extension = MemoryMonitorExtension(config, root_path)

        async def get_and_display():
            ram_info = await extension.get_ram_info()
            print_memory_info({"ram": ram_info}, show_gpu=False, show_ram=True)

        asyncio.run(get_and_display())

    except Exception as e:
        print(f"Error getting RAM info: {e}")
        import traceback
        traceback.print_exc()


def register(parser: argparse.ArgumentParser):
    """Register memory monitor CLI commands.

    Args:
        parser: The CLI argument parser (typically a subparser)
    """
    # Create subparsers for memory commands
    subparsers = parser.add_subparsers(dest="memory_command", help="Memory monitor commands")

    # Main monitor command (show all)
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Display current GPU and RAM memory usage",
        description="Show current memory usage for GPU and RAM"
    )
    monitor_parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Show GPU memory (default: True)"
    )
    monitor_parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Hide GPU memory"
    )
    monitor_parser.add_argument(
        "--ram",
        action="store_true",
        default=True,
        help="Show RAM memory (default: True)"
    )
    monitor_parser.add_argument(
        "--no-ram",
        action="store_false",
        dest="ram",
        help="Hide RAM memory"
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # Watch command (continuous monitoring)
    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously monitor memory usage",
        description="Continuously monitor and display memory usage"
    )
    watch_parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)"
    )
    watch_parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Show GPU memory (default: True)"
    )
    watch_parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Hide GPU memory"
    )
    watch_parser.add_argument(
        "--ram",
        action="store_true",
        default=True,
        help="Show RAM memory (default: True)"
    )
    watch_parser.add_argument(
        "--no-ram",
        action="store_false",
        dest="ram",
        help="Hide RAM memory"
    )
    watch_parser.set_defaults(func=cmd_watch)

    # GPU-only command
    gpu_parser = subparsers.add_parser(
        "gpu",
        help="Display GPU information only",
        description="Show GPU memory and utilization"
    )
    gpu_parser.set_defaults(func=cmd_gpu)

    # RAM-only command
    ram_parser = subparsers.add_parser(
        "ram",
        help="Display RAM information only",
        description="Show RAM memory usage"
    )
    ram_parser.set_defaults(func=cmd_ram)
