"""Logging utilities with rich console output + loguru."""

import logging
import sys

from loguru import logger as loguru_logger
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

# Global console for rich output
console = Console()

# Configure loguru with rich formatting
loguru_logger.remove()  # Remove default handler
loguru_logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def log_phase_start(phase_name: str, details: dict | None = None) -> None:
    """Log a phase start notification with rich formatting.

    Args:
        phase_name: Name of the phase (boss, workers, critic, synthesis)
        details: Optional additional details to display
    """
    phase_icons = {
        "boss": "🎯",
        "workers": "👷",
        "critic": "🔍",
        "synthesis": "📝",
    }

    icon = phase_icons.get(phase_name, "⚙️")
    title = f"{icon} {phase_name.upper()} PHASE"

    content = Text()
    content.append(f"Starting {phase_name} phase", style="bold cyan")

    if details:
        content.append("\n\n", style="")
        for key, value in details.items():
            content.append(f"{key}: ", style="bold")
            content.append(f"{value}\n", style="")

    panel = Panel(
        content,
        title=title,
        border_style="bright_blue",
        padding=(1, 2),
    )

    console.print()
    console.print(panel)
    console.print()


def log_phase_complete(phase_name: str, duration: float | None = None) -> None:
    """Log a phase completion notification.

    Args:
        phase_name: Name of the completed phase
        duration: Optional duration in seconds
    """
    duration_str = f" ({duration:.2f}s)" if duration else ""

    console.print(
        f"[green]✓[/green] [bold green]{phase_name}[/bold green] complete{duration_str}",
    )


def log_phase_error(phase_name: str, error: str) -> None:
    """Log a phase error notification.

    Args:
        phase_name: Name of the failed phase
        error: Error message
    """
    console.print()
    console.print(
        Panel(
            f"[red]{error}[/red]",
            title=f"[red bold]❌ {phase_name.upper()} FAILED[/red bold]",
            border_style="red",
            padding=(1, 2),
        )
    )
    console.print()


def log_worker_spawn(worker_id: str, task: str, worker_num: int | None = None) -> None:
    """Log a worker spawn notification.

    Args:
        worker_id: Worker identifier
        task: Task description
        worker_num: Optional worker number for display
    """
    prefix = f"[dim]Worker {worker_num}:[/dim] " if worker_num else ""
    console.print(
        f"  [cyan]🔨[/cyan] {prefix}[bold]{worker_id}[/bold]: {task[:80]}{'...' if len(task) > 80 else ''}",
    )


def log_worker_result(worker_id: str, success: bool, duration: float | None = None) -> None:
    """Log a worker completion notification.

    Args:
        worker_id: Worker identifier
        success: Whether the worker succeeded
        duration: Optional duration in seconds
    """
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    duration_str = f" ({duration:.2f}s)" if duration else ""
    console.print(f"    {status} [dim]{worker_id}[/dim]{duration_str}")


def setup_logging(
    level: str = "info",
    format: str | None = None,
    use_rich: bool = True,
) -> None:
    """Setup logging configuration.

    Args:
        level: Log level (debug, info, warning, error)
        format: Optional custom format
        use_rich: If True, use RichHandler for formatted output
    """
    # Configure loguru
    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper(),
        colorize=True,
    )

    if use_rich:
        # Also configure standard library logging to use RichHandler
        log_level = getattr(logging, level.upper(), logging.INFO)

        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=(level == "debug"),
            markup=True,
        )
        rich_handler.setLevel(log_level)
        logging.basicConfig(
            level=log_level,
            handlers=[rich_handler],
            force=True,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a standard library logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_loguru_logger(name: str | None = None):
    """Get a loguru logger instance.

    Args:
        name: Optional name for contextual logging

    Returns:
        Loguru logger instance
    """
    if name:
        return loguru_logger.bind(name=name)
    return loguru_logger
