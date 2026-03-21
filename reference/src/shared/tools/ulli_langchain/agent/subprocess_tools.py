"""Subprocess wrapper tools for calling isolated CLI services.

These tools act as clients that spawn subprocess calls to isolated services
running in their own environments. This preserves CLI capability while
allowing the AI to trigger these services.
"""

import asyncio
import json
from .logging_config import get_logger
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from langchain_core.tools import tool

logger = get_logger(__name__)

# =============================================================================
# DEEP RESEARCH SUBPROCESS WRAPPER
# =============================================================================


@tool
async def deep_research_subprocess(
    query: str,
    use_case: str = "rag",
    overdrive: bool = False,
) -> str:
    """Spawn deep research as an isolated subprocess.

    This tool runs the Deep Research CLI script in a subprocess, allowing
    the AI to trigger deep research without loading it into the main process.

    Use this tool for:
    - Comprehensive research on complex topics
    - Multi-source information gathering
    - OSINT investigations
    - Trading analysis (with use_case="trading")

    Args:
        query: Research query or topic
        use_case: Research mode - rag, podcast, trading, osint, news, content, web_research
        overdrive: Enable max quality/max context mode (slower but better results)

    Returns:
        Research results summary with output file location
    """
    # Path to the deep_research CLI script
    script_dir = Path(__file__).parent.parent.parent / "extensions" / "features" / "deep_research"
    script_path = script_dir / "cli.py"

    if not script_path.exists():
        return f"Error: Deep research CLI not found at {script_path}"

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "src.extensions.features.deep_research.cli",
        "--topic", query,
        "--use-case", use_case,
    ]

    if overdrive:
        cmd.append("--overdrive")

    logger.info(f"[DEEP_RESEARCH] Spawning subprocess: {' '.join(cmd)}")

    try:
        # Run subprocess and capture output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_dir.parent.parent.parent.parent)
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            logger.error(f"[DEEP_RESEARCH] Subprocess failed: {error_msg}")
            return f"Deep research failed: {error_msg[:500]}"

        # Parse output
        output = stdout.decode()
        return f"Deep research completed successfully.\n\n{output[-1000:]}"  # Last 1000 chars

    except Exception as e:
        logger.error(f"[DEEP_RESEARCH] Error: {e}")
        return f"Error running deep research: {str(e)}"


# =============================================================================
# INDEX TTS SUBPROCESS WRAPPER
# =============================================================================


@tool
async def index_tts_subprocess(
    text: str,
    voice_sample: str,
    speaker: str = "narrator",
) -> str:
    """Spawn IndexTTS generation as an isolated subprocess.

    This tool runs the IndexTTS CLI script in a subprocess for TTS generation
    with emotion-aware voice cloning.

    Args:
        text: Text to synthesize
        voice_sample: Path to voice sample for cloning
        speaker: Speaker identifier (e.g., narrator, left, right)

    Returns:
        Path to generated audio file
    """
    # This would point to the IndexTTS CLI script when created
    # For now, return a placeholder
    return f"IndexTTS subprocess not yet implemented. Text: {text[:50]}..., Voice: {voice_sample}"


# =============================================================================
# GENERIC SUBPROCESS RUNNER
# =============================================================================


@tool
async def run_cli_script(
    script_path: str,
    args: List[str],
    working_dir: Optional[str] = None,
) -> str:
    """Run any CLI script as a subprocess.

    Generic tool for running standalone CLI scripts. Use this for
    custom scripts that aren't covered by specific wrappers.

    Args:
        script_path: Path to Python script or module (e.g., "path/to/script.py" or "module.submodule")
        args: Command line arguments to pass to the script
        working_dir: Working directory for the subprocess (default: backend root)

    Returns:
        Script output
    """
    backend_root = Path(__file__).parent.parent.parent.parent
    work_dir = Path(working_dir) if working_dir else backend_root

    # Determine if it's a module or script
    if script_path.endswith(".py"):
        cmd = [sys.executable, script_path] + args
    else:
        cmd = [sys.executable, "-m", script_path] + args

    logger.info(f"[SUBPROCESS] Running: {' '.join(cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir)
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            return f"Script failed with code {process.returncode}: {error_msg}"

        return stdout.decode() or "Script completed successfully"

    except Exception as e:
        return f"Error running script: {str(e)}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "deep_research_subprocess",
    "index_tts_subprocess",
    "run_cli_script",
]
