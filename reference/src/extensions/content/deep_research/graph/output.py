"""Output formatters and validators for different output modes.

DEEP_RESEARCH IS DUMB ABOUT FORMATTING:
- This module routes to content_formatter extension for complex formats
- Built-in formatters only for simple cases (mdx, text, json, etc.)
- Single source of truth: content_formatter

Supported output modes:
- Simple (built-in): mdx, text, md, json, script, code_repo, llm_docs
- Complex (via content_formatter): podcast, twitter, tweets, etc.
"""

import json
import logging
import sys
import os
import re
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from pydantic import BaseModel, Field
from ..logging_utils import get_logger

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# =============================================================================
# CONTENT_FORMATTER BRIDGE - DEEP_RESEARCH IS DUMB
# =============================================================================

# Formats handled by content_formatter extension
CONTENT_FORMATTER_FORMATS = {
    "podcast",
    "twitter",
    "tweets",
}


async def _format_via_content_formatter(
    response: str,
    format_name: str,
    session_id: Optional[str] = None,
    character: Optional[str] = None,
    **kwargs
) -> str:
    """Bridge to content_formatter extension via extension manager.

    DEEP_RESEARCH IS DUMB: We just pass data and get back formatted output.
    content_formatter owns all the logic for formatting.

    Uses extension manager for proper dependency injection instead of direct imports.

    Args:
        response: Research output to format
        format_name: Format type (podcast, tweets, etc.)
        session_id: Session ID for Context Engine access
        character: Optional character name for character-based formats
        **kwargs: Additional format-specific parameters

    Returns:
        Formatted output as string

    Raises:
        RuntimeError: If content_formatter extension is not available
    """
    try:
        from src.api.core.extensions import get_extension_manager

        manager = get_extension_manager()
        formatter_ext = manager.get_extension("content_formatter")

        if not formatter_ext:
            raise RuntimeError(
                f"content_formatter extension not found. "
                f"Format '{format_name}' requires content_formatter to be available."
            )

        if not getattr(formatter_ext, 'enabled', False):
            raise RuntimeError(
                f"content_formatter extension is not enabled. "
                f"Format '{format_name}' requires content_formatter to be enabled."
            )

        # Get formats_dir from content_formatter extension
        backend_root = Path(__file__).parent.parent.parent.parent.parent
        formats_dir = backend_root / "src" / "extensions" / "content" / "content_formatter" / "formats"

        # Get base config (minimal - let format config do the work)
        base_config = {"base": {}}

        # Use extension's public API via the instance
        from src.extensions.content.content_formatter.base import format_content

        result = await format_content(
            format_name=format_name,
            content=response,
            base_config=base_config,
            formats_dir=formats_dir,
            session_id=session_id,
            character=character,
            **kwargs
        )

        # Convert Pydantic model to string representation
        return _format_result_to_string(result, format_name)

    except RuntimeError:
        # Re-raise RuntimeError as-is (crash-only philosophy)
        raise
    except Exception as e:
        logger.error(f"[OUTPUT] content_formatter bridge failed for '{format_name}': {e}")
        raise RuntimeError(
            f"content_formatter failed to format '{format_name}': {e}"
        ) from e


def _format_result_to_string(result: Any, format_name: str) -> str:
    """Convert content_formatter Pydantic result to string.

    Args:
        result: Pydantic model from content_formatter
        format_name: Format type for custom formatting

    Returns:
        String representation of the formatted result
    """
    # Convert Pydantic model to dict, then to formatted string
    if hasattr(result, 'model_dump'):
        data = result.model_dump()
    elif hasattr(result, 'dict'):
        data = result.dict()
    else:
        return str(result)

    # Format-specific rendering
    if format_name in ("podcast", "PodcastScript"):
        return _render_podcast_result(data)
    elif format_name in ("twitter", "tweets", "TweetThread"):
        return _render_tweets_result(data)
    else:
        # Default JSON rendering
        return json.dumps(data, indent=2)


def _render_podcast_result(data: Dict[str, Any]) -> str:
    """Render podcast result to readable script format."""
    lines = []
    lines.append(f"# {data.get('title', 'Podcast Script')}")
    lines.append("")
    if data.get('summary'):
        lines.append(f"**Summary:** {data['summary']}")
        lines.append("")

    segments = data.get('segments', [])
    for i, segment in enumerate(segments, 1):
        lines.append(f"## Segment {i}: {segment.get('segment_type', 'Untitled')}")
        lines.append("")

        for line in segment.get('dialogue', []):
            speaker = line.get('speaker', 'Unknown')
            text = line.get('text', '')
            emotion = line.get('emotion', '')
            lines.append(f"**{speaker}** ({emotion}): {text}")

        lines.append("")

    if data.get('notes'):
        lines.append("---")
        lines.append(f"**Notes:** {data['notes']}")

    return "\n".join(lines)


def _render_tweets_result(data: Dict[str, Any]) -> str:
    """Render tweets result to readable thread format."""
    lines = []
    lines.append("# Tweet Thread")
    lines.append("")

    tweets = data.get('tweets', [])
    for i, tweet in enumerate(tweets, 1):
        content = tweet.get('content', '')
        hashtags = tweet.get('hashtags', [])
        lines.append(f"**Tweet {i}**")
        lines.append(content)
        if hashtags:
            lines.append(f"Hashtags: {' '.join(hashtags)}")
        lines.append("")

    lines.append("---")
    lines.append(f"**Hook Strength:** {data.get('hook_strength', 'N/A')}")
    lines.append(f"**Call to Action:** {data.get('call_to_action', 'N/A')}")
    lines.append(f"**Total Characters:** {data.get('total_char_count', 0)}")

    return "\n".join(lines)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT VALIDATION
# =============================================================================

class MDXOutput(BaseModel):
    """MDX file output with frontmatter."""

    title: str = Field(description="Document title")
    slug: Optional[str] = Field(default=None, description="URL slug")
    content: str = Field(description="Main content in MDX format")
    frontmatter: Dict[str, Any] = Field(default_factory=dict, description="YAML frontmatter")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TextOutput(BaseModel):
    """Plain text file output."""

    content: str = Field(description="Plain text content")
    encoding: str = Field(default="utf-8", description="File encoding")


class MarkdownOutput(BaseModel):
    """Markdown file output."""

    title: Optional[str] = Field(default=None, description="Document title")
    content: str = Field(description="Markdown content")
    toc: bool = Field(default=True, description="Include table of contents")


class JSONOutput(BaseModel):
    """JSON file output."""

    data: Dict[str, Any] = Field(description="JSON data")
    pretty: bool = Field(default=True, description="Pretty print JSON")


class ScriptOutput(BaseModel):
    """Executable script output."""

    language: str = Field(description="Script language (python, bash, etc.)")
    shebang: Optional[str] = Field(default=None, description="Shebang line")
    content: str = Field(description="Script content")
    executable: bool = Field(default=False, description="Make executable")


class CodeRepoFile(BaseModel):
    """Single file in a code repository."""

    path: str = Field(description="File path relative to repo root")
    content: str = Field(description="File content")
    language: Optional[str] = Field(default=None, description="Programming language")


class CodeRepoOutput(BaseModel):
    """Code repository structure output."""

    files: List[CodeRepoFile] = Field(description="List of files")
    readme: Optional[str] = Field(default=None, description="README.md content")
    structure: str = Field(default="", description="Directory structure representation")
    output_dir: Optional[str] = Field(default=None, description="Output directory path")


class LLMDocsFile(BaseModel):
    """Single file in LLM-optimized documentation."""

    path: str = Field(description="File path relative to docs root")
    content: str = Field(description="File content (simplified for LLM input)")
    tokens_estimate: Optional[int] = Field(default=None, description="Estimated token count")
    summary: Optional[str] = Field(default=None, description="Brief summary of this file")


class LLMDocsOutput(BaseModel):
    """LLM-optimized documentation output."""

    files: List[LLMDocsFile] = Field(description="List of documentation files")
    readme: str = Field(description="Main README content")
    structure: str = Field(default="", description="Directory structure")
    output_dir: Optional[str] = Field(default=None, description="Output directory path")
    source_info: Optional[Dict[str, Any]] = Field(default=None, description="Source information")


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

class OutputFormatter(ABC):
    """Base class for output formatters."""

    @abstractmethod
    def format(self, response: str, config: Dict[str, Any]) -> str:
        """Format response according to mode.

        Args:
            response: The LLM response text
            config: Output configuration

        Returns:
            Formatted output as string
        """
        pass


class MDXFormatter(OutputFormatter):
    """Format output as MDX file with frontmatter."""

    def format(self, response: str, config: Dict[str, Any]) -> str:
        # Validate against MDXOutput model if configured
        if config.get("validate", False):
            output = MDXOutput(
                title=config.get("title", "Untitled"),
                content=response,
                frontmatter=config.get("frontmatter", {}),
                metadata=config.get("metadata", {})
            )
            return self._render(output)
        else:
            # Simple rendering without validation
            return self._simple_render(response, config)

    def _render(self, output: MDXOutput) -> str:
        """Render validated MDX output."""
        lines = ["---"]

        # Add frontmatter
        if output.frontmatter:
            for key, value in output.frontmatter.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"{key}: {value}")
                elif isinstance(value, list):
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                elif isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")

        lines.append("---")
        lines.append("")

        # Add title if present
        if output.title:
            lines.append(f"# {output.title}")
            lines.append("")

        # Add content
        lines.append(output.content)

        return "\n".join(lines)

    def _simple_render(self, response: str, config: Dict[str, Any]) -> str:
        """Simple MDX render without validation."""
        lines = ["---"]

        # Add frontmatter from config
        frontmatter = config.get("frontmatter", {})
        if frontmatter:
            for key, value in frontmatter.items():
                if isinstance(value, str):
                    lines.append(f'{key}: "{value}"')
                else:
                    lines.append(f"{key}: {value}")

        lines.append("---")
        lines.append("")

        # Add title
        if config.get("title"):
            lines.append(f"# {config['title']}")
            lines.append("")

        lines.append(response)
        return "\n".join(lines)


class TextFormatter(OutputFormatter):
    """Format output as plain text file."""

    def format(self, response: str, config: Dict[str, Any]) -> str:
        if config.get("validate", False):
            output = TextOutput(content=response)
            return output.content
        return response


class MarkdownFormatter(OutputFormatter):
    """Format output as Markdown file."""

    def format(self, response: str, config: Dict[str, Any]) -> str:
        if config.get("validate", False):
            output = MarkdownOutput(
                title=config.get("title"),
                content=response,
                toc=config.get("toc", True)
            )
            return self._render(output)
        return response

    def _render(self, output: MarkdownOutput) -> str:
        lines = []
        if output.title:
            lines.append(f"# {output.title}")
            lines.append("")
        lines.append(output.content)
        return "\n".join(lines)


class JSONFormatter(OutputFormatter):
    """Format output as JSON file.

    CRASH-ONLY: Response MUST be valid JSON. No fallback wrapper.
    LLMs should use with_structured_output() to return structured data.
    """

    def format(self, response: str, config: Dict[str, Any]) -> str:
        # CRASH-ONLY: Response must be valid JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSONFormatter requires response to be valid JSON. "
                f"The LLM should use with_structured_output() to return structured data. "
                f"JSON parsing fallback has been EXTERMINATED. Error: {e}"
            ) from e

        if config.get("validate", False):
            output = JSONOutput(data=data, pretty=config.get("pretty", True))
            return json.dumps(output.data, indent=2 if output.pretty else None)

        return json.dumps(data, indent=2 if config.get("pretty", True) else None)


class ScriptFormatter(OutputFormatter):
    """Format output as executable script."""

    def format(self, response: str, config: Dict[str, Any]) -> str:
        language = config.get("language", "bash")

        shebang = config.get("shebang")
        if not shebang:
            # Default shebangs
            shebangs = {
                "python": "#!/usr/bin/env python3",
                "bash": "#!/bin/bash",
                "sh": "#!/bin/sh",
                "zsh": "#!/bin/zsh"
            }
            shebang = shebangs.get(language, f"#!/usr/bin/env {language}")

        lines = [shebang, "", response]

        if config.get("validate", False):
            output = ScriptOutput(
                language=language,
                shebang=shebang,
                content=response,
                executable=config.get("executable", False)
            )

        return "\n".join(lines)


class CodeRepoFormatter(OutputFormatter):
    """Format output as code repository structure and writes actual files.

    CRASH-ONLY: Response MUST be valid JSON with file structure.
    LLMs should use with_structured_output() to return structured data.
    """

    def format(self, response: str, config: Dict[str, Any]) -> str:
        # CRASH-ONLY: Response must be valid JSON
        try:
            files_data = json.loads(response)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"CodeRepoFormatter requires response to be valid JSON. "
                f"The LLM should use with_structured_output() to return structured data. "
                f"JSON parsing fallback has been EXTERMINATED. Error: {e}"
            ) from e

        if isinstance(files_data, dict) and "files" in files_data:
            files = files_data["files"]
            output_dir = files_data.get("output_dir")
        elif isinstance(files_data, list):
            files = files_data
            output_dir = None
        else:
            raise RuntimeError(
                f"CodeRepoFormatter requires JSON with 'files' key or a list. "
                f"Got: {type(files_data)}"
            )

        if config.get("validate", False):
            output = CodeRepoOutput(
                files=[CodeRepoFile(**f) for f in files],
                readme=config.get("readme"),
                structure=config.get("structure", ""),
                output_dir=output_dir
            )
            return self._render(output)

        return self._render(CodeRepoOutput(
            files=[CodeRepoFile(**f) for f in files],
            output_dir=output_dir
        ))

    def _render(self, output: CodeRepoOutput) -> str:
        """Render code repository and optionally write files to disk."""
        lines = []
        lines.append("Code Repository Structure")
        lines.append("=" * 50)
        lines.append("")

        # Show structure
        if output.structure:
            lines.append(output.structure)
            lines.append("")

        # List files
        lines.append(f"Files ({len(output.files)}):")
        lines.append("")
        for file in output.files:
            lines.append(f"  {file.path}")
            if file.language:
                lines.append(f"    Language: {file.language}")

        lines.append("")
        lines.append("=" * 50)
        lines.append("File Contents:")
        lines.append("")

        for file in output.files:
            lines.append(f"### {file.path}")
            lines.append("```" + (file.language or ""))
            lines.append(file.content)
            lines.append("```")
            lines.append("")

        # Write files to disk if output_dir specified
        if output.output_dir:
            self._write_files_to_disk(output)

        return "\n".join(lines)

    def _write_files_to_disk(self, output: CodeRepoOutput) -> None:
        """Write actual files to disk and create a zip file."""
        # Create temp directory for the repo
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "code_repo"
            repo_path.mkdir()

            # Write each file
            for file in output.files:
                file_path = repo_path / file.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file.content)

            # Create zip file
            output_path = Path(output.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = output_path / f"code_repo_{timestamp}.zip"

            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in repo_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(repo_path)
                        zipf.write(file_path, arcname)

            logger.info(f"[CODE_REPO] Written {len(output.files)} files to: {zip_filename}")


class LLMDocsFormatter(OutputFormatter):
    """Format output as LLM-optimized documentation.

    CRASH-ONLY: Response MUST be valid JSON with docs structure.
    LLMs should use with_structured_output() to return structured data.
    """

    def format(self, response: str, config: Dict[str, Any]) -> str:
        # CRASH-ONLY: Response must be valid JSON
        try:
            docs_data = json.loads(response)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLMDocsFormatter requires response to be valid JSON. "
                f"The LLM should use with_structured_output() to return structured data. "
                f"JSON parsing fallback has been EXTERMINATED. Error: {e}"
            ) from e

        if isinstance(docs_data, dict) and "files" in docs_data:
            files = docs_data["files"]
            output_dir = docs_data.get("output_dir")
            source_info = docs_data.get("source_info", {})
            readme = docs_data.get("readme", "# LLM-Optimized Documentation\n\n")
        elif isinstance(docs_data, list):
            files = docs_data
            output_dir = None
            source_info = {}
            readme = "# LLM-Optimized Documentation\n\n"
        else:
            raise RuntimeError(
                f"LLMDocsFormatter requires JSON with 'files' key or a list. "
                f"Got: {type(docs_data)}"
            )

        if config.get("validate", False):
            output = LLMDocsOutput(
                files=[LLMDocsFile(**f) for f in files],
                readme=readme,
                structure=config.get("structure", ""),
                output_dir=output_dir,
                source_info=source_info
            )
            return self._render(output)

        return self._render(LLMDocsOutput(
            files=[LLMDocsFile(**f) for f in files],
            readme=readme,
            output_dir=output_dir,
            source_info=source_info
        ))

    def _render(self, output: LLMDocsOutput) -> str:
        """Render LLM-optimized documentation and write files to disk."""
        lines = []
        lines.append("# LLM-Optimized Documentation")
        lines.append("=" * 50)
        lines.append("")

        # Source info
        if output.source_info:
            lines.append("## Source Information")
            lines.append("")
            for key, value in output.source_info.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Structure
        if output.structure:
            lines.append("## Structure")
            lines.append("")
            lines.append("```")
            lines.append(output.structure)
            lines.append("```")
            lines.append("")

        # File listing with token estimates
        lines.append("## Files")
        lines.append("")
        total_tokens = 0
        for file in output.files:
            tokens = file.tokens_estimate or len(file.content) // 4
            total_tokens += tokens
            lines.append(f"### {file.path}")
            lines.append(f"Tokens: ~{tokens:,}")
            if file.summary:
                lines.append(f"Summary: {file.summary}")
            lines.append("")

        lines.append(f"**Total Tokens: ~{total_tokens:,}**")
        lines.append("")
        lines.append("=" * 50)
        lines.append("## Contents")
        lines.append("")

        # README first
        lines.append("### README.md")
        lines.append("")
        lines.append(output.readme)
        lines.append("")
        lines.append("---")
        lines.append("")

        # File contents
        for file in output.files:
            lines.append(f"### {file.path}")
            lines.append("")
            lines.append(file.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Write files to disk if output_dir specified
        if output.output_dir:
            self._write_docs_to_disk(output)

        return "\n".join(lines)

    def _write_docs_to_disk(self, output: LLMDocsOutput) -> None:
        """Write documentation files to disk."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_path = Path(tmpdir) / "llm_docs"
            docs_path.mkdir()

            # Write README
            readme_path = docs_path / "README.md"
            readme_path.write_text(output.readme)

            # Write each file
            for file in output.files:
                file_path = docs_path / file.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file.content)

            # Create zip file
            output_path = Path(output.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = output_path / f"llm_docs_{timestamp}.zip"

            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in docs_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(docs_path)
                        zipf.write(file_path, arcname)

            logger.info(f"[LLM_DOCS] Written {len(output.files) + 1} files to: {zip_filename}")


# =============================================================================
# OUTPUT FACTORY - Dynamic Extension Discovery
# =============================================================================

# Built-in formatters for basic output modes (non-extension based)
BUILTIN_FORMATTERS = {
    "mdx": MDXFormatter,
    "text": TextFormatter,
    "md": MarkdownFormatter,
    "json": JSONFormatter,
    "script": ScriptFormatter,
    "code_repo": CodeRepoFormatter,
    "llm_docs": LLMDocsFormatter,
}


def _get_research_formatter(mode: str):
    """Get formatter from research extensions dynamically.

    This discovers and loads formatters from research extensions (podcast, twitter, etc.)
    following the Open/Closed Principle - new output modes can be added via extensions
    without modifying this file.

    Args:
        mode: Output mode (podcast, twitter, etc.)

    Returns:
        Formatter class or None if not found
    """
    from src.constants import RESEARCH_EXTENSIONS_DIR

    # Only handle research-specific modes
    if mode in BUILTIN_FORMATTERS:
        return None

    # Try to load from research extension
    research_dir = RESEARCH_EXTENSIONS_DIR
    if not research_dir.exists():
        return None

    # Find the extension directory for this mode
    ext_dir = research_dir / mode
    if not ext_dir.exists():
        return None

    # Import and get the output formatting from the extension
    try:
        output_formatting_module = ext_dir / "output_formatting"
        if not output_formatting_module.exists():
            return None

        # Import the output module
        import importlib.util
        import sys

        output_path = output_formatting_module / "output.py"
        if not output_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(
            f"{mode}_output",
            output_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{mode}_output"] = module
            spec.loader.exec_module(module)

            # Get the formatter class (expected pattern: {Mode}Formatter)
            formatter_class_name = f"{mode.capitalize()}Formatter"
            if hasattr(module, formatter_class_name):
                logger.info(f"[OUTPUT] Loaded formatter for '{mode}' from research extension")
                return getattr(module, formatter_class_name)
    except Exception as e:
        logger.warning(f"[OUTPUT] Failed to load formatter for '{mode}': {e}")

    return None


async def format_output(
    response: str,
    mode: str,
    config: Dict[str, Any],
    session_id: Optional[str] = None
) -> str:
    """Format response according to output mode.

    DEEP_RESEARCH IS DUMB:
    - Routes to content_formatter for complex formats (podcast, tweets, etc.)
    - Uses built-in formatters for simple formats (mdx, text, json, etc.)
    - Single source of truth: content_formatter extension

    Args:
        response: The LLM response text
        mode: Output mode (mdx, text, md, json, script, code_repo, llm_docs, podcast, twitter, etc.)
        config: Output configuration for the mode
        session_id: Optional session ID for Context Engine access (passed to content_formatter)

    Returns:
        Formatted output as string
    """
    # Route to content_formatter for complex formats
    if mode in CONTENT_FORMATTER_FORMATS:
        logger.info(f"[OUTPUT] Routing '{mode}' to content_formatter extension")
        character = config.get("character")  # Extract character from config if present
        return await _format_via_content_formatter(
            response=response,
            format_name=mode,
            session_id=session_id,
            character=character,
        )

    # Fall back to built-in formatters for simple cases
    formatter_class = BUILTIN_FORMATTERS.get(mode)

    if not formatter_class:
        raise ValueError(
            f"Unknown output mode: {mode}. "
            f"Content formatter formats: {CONTENT_FORMATTER_FORMATS}. "
            f"Built-in formats: {list(BUILTIN_FORMATTERS.keys())}"
        )

    formatter = formatter_class()
    return formatter.format(response, config)
