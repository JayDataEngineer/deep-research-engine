"""Read Artifact Tool - Lazy loading for Manifest Flow architecture.

This tool provides on-demand access to full worker findings stored as artifacts.
It implements the "Manifest Flow" pattern where:
1. Workers save full findings to disk (artifacts/)
2. Workers return structured CitationManifest (claims + artifact references)
3. Synthesis gets manifest (tiny, no clipping)
4. Synthesis calls read_artifact when it needs full details

This ELIMINATES the need for "clipping" (lossy truncation).

Usage:
    from .tools.ulli_langchain.web.read_artifact import read_artifact_tool

    tool = read_artifact_tool()
    content = await tool.ainvoke({"artifact_path": "artifacts/worker_findings_Philosopher_0_20240203_120000.md"})
"""

from .logging_config import get_logger
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class ReadArtifactInput(BaseModel):
    """Input for reading artifact content."""
    artifact_path: str = Field(
        ...,
        description="Path to artifact file (e.g., 'artifacts/worker_findings_Philosopher_0_20240203_120000.md')"
    )
    max_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum characters to retrieve (default: 10000)"
    )
    section: Optional[str] = Field(
        default=None,
        description="Optional section filter (e.g., 'Line 50-100', 'Section II')"
    )


class ReadArtifactTool(BaseTool):
    """Tool for lazy-loading worker artifact content (Manifest Flow).

    This tool implements the Manifest Flow architecture where:
    - Synthesis gets CitationManifest (structured claims + file paths)
    - If synthesis needs more detail, it calls this tool to retrieve full content
    - NO CLIPPING NEEDED - data is retrieved on-demand

    Example flow:
        Worker returns: "Claim X -> artifact: worker_1.md, Line 50"
        Synthesis sees claim, needs more detail
        Synthesis calls: read_artifact("artifacts/worker_1.md")
        Tool returns: Full content from worker_1.md
    """

    name: str = "read_artifact"
    description: str = """Retrieve full content from worker artifacts.

Use this tool when you need to:
- Verify a specific claim or get more details from a worker's findings
- Read the full research content behind a structured claim
- Access the complete context that was summarized in a CitationManifest
- Get exact quotes or detailed explanations from worker artifacts

The artifact path is provided in CitationClaim.source_artifact or in worker output.

Args:
    artifact_path: Path to artifact file (e.g., 'artifacts/worker_findings_Philosopher_0_20240203_120000.md')
    max_length: Maximum characters to retrieve (default: 10000, max: 100000)
    section: Optional section filter (e.g., 'Line 50-100', 'Section II')

Returns:
    Full artifact content with worker metadata and research findings.
"""
    args_schema: type[BaseModel] = ReadArtifactInput

    # Base directory for artifacts (set at runtime)
    _run_dir: Optional[Path] = None

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def __init__(self, run_dir: Optional[Path] = None, **kwargs):
        """Initialize the tool with artifact directory.

        Args:
            run_dir: Base directory containing artifacts/ folder
            **kwargs: Additional kwargs for BaseTool
        """
        super().__init__(**kwargs)
        self._run_dir = Path(run_dir) if run_dir else None
        logger.info("[READ_ARTIFACT] Tool initialized")

    def _run(self, **kwargs):
        """Synchronous run - not supported."""
        raise NotImplementedError("Use async for read_artifact")

    async def _arun(
        self,
        artifact_path: str,
        max_length: int = 10000,
        section: Optional[str] = None,
    ) -> str:
        """Async run to retrieve artifact content from disk.

        Args:
            artifact_path: Path to artifact file (relative or absolute)
            max_length: Maximum characters to retrieve
            section: Optional section filter

        Returns:
            Formatted artifact content
        """
        if not self._run_dir:
            return "Error: Tool not initialized with run_dir"

        # Resolve artifact path (handle relative paths)
        artifact_file = Path(artifact_path)
        if not artifact_file.is_absolute():
            artifact_file = self._run_dir / artifact_path

        if not artifact_file.exists():
            # Try artifacts/ subdirectory
            artifact_file = self._run_dir / "artifacts" / Path(artifact_path).name

        if not artifact_file.exists():
            return f"Error: Artifact not found at {artifact_path}"

        try:
            # Read artifact content
            content = artifact_file.read_text(encoding='utf-8')

            # Apply section filter if specified
            if section:
                content = self._filter_by_section(content, section)

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + f"\n\n[... Artifact truncated at {max_length} chars (total: {len(content)} chars)]"

            # Add metadata header
            header = [
                f"**Artifact: {artifact_file.name}**",
                f"Path: {artifact_file}",
                f"Size: {len(content)} chars",
                ""
            ]

            return "\n".join(header) + "\n\n" + content

        except Exception as e:
            logger.error(f"[READ_ARTIFACT] Error reading artifact: {e}")
            return f"Error reading artifact: {type(e).__name__}: {str(e)}"

    def _filter_by_section(self, content: str, section: str) -> str:
        """Filter content by section/line range.

        Args:
            content: Full artifact content
            section: Section filter (e.g., "Line 50-100", "Section II")

        Returns:
            Filtered content
        """
        # Simple line range extraction
        if "Line" in section and "-" in section:
            try:
                # Extract line numbers (e.g., "Line 50-100" -> 50, 100)
                parts = section.replace("Line", "").strip().split("-")
                start_line = int(parts[0].strip())
                end_line = int(parts[1].strip()) if len(parts) > 1 else start_line + 50

                lines = content.split("\n")
                # Adjust for 0-indexing
                extracted = lines[start_line-1:min(end_line, len(lines))]
                return "\n".join(extracted)
            except (ValueError, IndexError):
                logger.warning(f"[READ_ARTIFACT] Invalid line range: {section}")

        # Section keyword search
        if section.startswith("Section"):
            lines = content.split("\n")
            result = []
            capturing = False
            section_header = section.upper()

            for line in lines:
                if section_header in line.upper():
                    capturing = True
                if capturing:
                    result.append(line)
                    # Stop at next major section
                    if line.strip().startswith("#") and section_header not in line.upper():
                        break

            return "\n".join(result) if result else content

        # No filter matched, return original
        return content

    def set_run_dir(self, run_dir: Path):
        """Update the run directory (for LangGraph integration).

        Args:
            run_dir: Base directory containing artifacts/ folder
        """
        self._run_dir = Path(run_dir)
        logger.debug(f"[READ_ARTIFACT] Run dir updated: {run_dir}")


# Singleton instance (run_dir will be set at runtime)
_tool_instance: Optional[ReadArtifactTool] = None


def read_artifact_tool(run_dir: Optional[Path] = None) -> ReadArtifactTool:
    """Get or create the shared read_artifact tool instance.

    Args:
        run_dir: Optional base directory containing artifacts/ folder

    Returns:
        The shared ReadArtifactTool instance
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = ReadArtifactTool(run_dir=run_dir)
        logger.info("[READ_ARTIFACT] Created shared instance")
    elif run_dir is not None:
        _tool_instance.set_run_dir(run_dir)
    return _tool_instance
