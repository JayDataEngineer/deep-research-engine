"""Simple logging with Loguru + Rich.

Tells us when we change stages of the process,
and gives critic review on fail.
"""

import sys
from typing import Optional, Dict, Any, List

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint


# =============================================================================
# CONSOLE SETUP
# =============================================================================

console = Console()


# =============================================================================
# LOGURU CONFIGURATION
# =============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup Loguru with Rich sink.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with Rich formatting
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Optional file handler
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days",
        )

    logger.info(f"Logging initialized at {level} level")


def get_logger(name: str):
    """Get a logger instance with context binding.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger with context binding
    """
    return logger.bind(name=name)


# =============================================================================
# STAGE LOGGING
# =============================================================================

def log_stage(
    stage: str,
    details: Optional[Dict[str, Any]] = None,
    style: str = "blue",
):
    """Log stage transition with Rich panel.

    Args:
        stage: Stage name (e.g., "INGEST", "BUILD", "CRITIC")
        details: Optional details dict
        style: Border style (blue, green, yellow, red)
    """
    subtitle = ""
    if details:
        parts = [f"{k}={v}" for k, v in details.items()]
        subtitle = " | ".join(parts)

    console.print(Panel(
        f"[bold]{stage}[/bold]",
        subtitle=subtitle,
        border_style=style,
        expand=False,
    ))

    logger.info(f"STAGE: {stage}", **(details or {}))


# =============================================================================
# CRITIC REVIEW LOGGING
# =============================================================================

def log_critic_review(
    score: float,
    passed: bool,
    issues: List[str],
    suggestions: Optional[List[str]] = None,
):
    """Log critic review with Rich formatting.

    Args:
        score: Score (0.0 - 1.0)
        passed: Whether passed
        issues: List of issues found
        suggestions: Optional list of suggestions
    """
    if passed:
        # Success panel
        console.print(Panel(
            f"[green bold]PASSED[/green bold]\n\n"
            f"Score: {score:.2f}",
            title="[bold]Critic Review[/bold]",
            border_style="green",
            expand=False,
        ))
        logger.info(f"CRITIC PASSED: score={score:.2f}")

    else:
        # Failure panel with details
        issues_text = "\n".join(f"  • {issue}" for issue in issues) if issues else "  (none)"

        suggestions_text = ""
        if suggestions:
            suggestions_text = "\n\n[yellow]Suggestions:[/yellow]\n" + "\n".join(f"  → {s}" for s in suggestions)

        console.print(Panel(
            f"[red bold]FAILED[/red bold]\n\n"
            f"Score: {score:.2f}\n\n"
            f"[red]Issues:[/red]\n{issues_text}"
            f"{suggestions_text}",
            title="[bold]Critic Review[/bold]",
            border_style="red",
            expand=False,
        ))

        logger.warning(
            f"CRITIC FAILED: score={score:.2f}, issues={len(issues)}"
        )


# =============================================================================
# ERROR LOGGING
# =============================================================================

def log_error(
    error: str,
    phase: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
):
    """Log error with Rich formatting.

    Args:
        error: Error message
        phase: Phase where error occurred
        details: Optional details dict
        exception: Optional exception object
    """
    # Error panel
    phase_text = f" in [bold]{phase}[/bold]" if phase else ""
    details_text = ""
    if details:
        details_text = "\n\n" + "\n".join(f"  {k}: {v}" for k, v in details.items())

    console.print(Panel(
        f"[red bold]ERROR{phase_text}[/red bold]\n\n"
        f"{error}"
        f"{details_text}",
        border_style="red",
        expand=False,
    ))

    # Log to file with full traceback
    if exception:
        logger.exception(f"ERROR: {error}", phase=phase, **(details or {}))
    else:
        logger.error(f"ERROR: {error}", phase=phase, **(details or {}))


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(
    session_id: str,
    score: float,
    passed: bool,
    iterations: int,
    context_count: int = 0,
):
    """Print final summary table.

    Args:
        session_id: Session ID
        score: Final score
        passed: Whether passed
        iterations: Number of iterations
        context_count: Number of context items stored
    """
    table = Table(title="Pipeline Summary", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Session ID", session_id)
    table.add_row("Status", "[green]PASSED[/green]" if passed else "[red]FAILED[/red]")
    table.add_row("Score", f"{score:.2f}")
    table.add_row("Iterations", str(iterations))
    table.add_row("Context Items", str(context_count))

    console.print(table)
    logger.info(
        f"Pipeline complete: session={session_id}, passed={passed}, score={score:.2f}"
    )
