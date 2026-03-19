"""Context Management Extension

Conversation context and token management.
Provides CLI tools for managing conversation context, tokens, and segments.

CLI Usage:
    uv run backend/manage.py context info
    uv run backend/manage.py context analyze <text>
    uv run backend/manage.py context trim --max-tokens 2000
    uv run backend/manage.py context segments list
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


@dataclass
class ContextSegment:
    """A segment of conversation context."""

    id: str
    type: str  # "system", "user", "assistant", "lorebook", "author_note", "character_card"
    content: str
    tokens: int
    enabled: bool = True
    position: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextInfo:
    """Information about the conversation context."""

    total_tokens: int
    max_tokens: int
    segments: List[ContextSegment]
    completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "completion_tokens": self.completion_tokens,
            "available_tokens": self.max_tokens - self.total_tokens - self.completion_tokens,
            "segments": [s.to_dict() for s in self.segments],
        }


class ContextExtension(LogicExtension):
    """Context extension - CLI wrapper for context management.

    This extension provides CLI access to context management tools
    for tracking tokens, managing segments, and analyzing context.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._segments: List[ContextSegment] = []
        self._max_tokens = 4096
        self._completion_tokens = 512

    async def initialize(self, context):
        """Initialize the context extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "context"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "context"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self._load_config()

        logger.info("Context extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            info                    - Show context information
            analyze <text>          - Analyze text for token count
            trim --max-tokens N     - Trim context to max tokens
            segments list           - List all context segments
            segments add            - Add a context segment
            segments remove <id>    - Remove a context segment
            segments enable <id>    - Enable a segment
            segments disable <id>   - Disable a segment
            export <file>           - Export context to file
            import <file>           - Import context from file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("context", "ctx", "conversation", "conv") or args:
            subcommand = args[0] if args else "info"
            subargs = args[1:] if len(args) > 1 else []

            # Check if it's a segments command
            if subcommand == "segments" and subargs:
                segment_cmd = subargs[0]
                segment_args = subargs[1:]
                async for result in self._handle_segments_command(segment_cmd, segment_args):
                    yield result
            else:
                async for result in self._handle_cli_command(subcommand, subargs):
                    yield result
        else:
            # API mode
            operation = input_data.get("operation")
            async for result in self._handle_api_operation(operation, input_data):
                yield result

    async def _handle_cli_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle CLI subcommands."""
        if subcommand == "info":
            yield await self._cmd_info()
        elif subcommand == "analyze":
            if not args:
                yield {"error": "Please provide text to analyze"}
                return
            yield await self._cmd_analyze(" ".join(args))
        elif subcommand in ("trim", "optimize"):
            yield await self._cmd_trim(args)
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_export(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_segments_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle segments subcommands."""
        if subcommand == "list":
            yield await self._cmd_segments_list()
        elif subcommand in ("add", "create"):
            if not args:
                yield {"error": "Please provide segment type"}
                return
            yield await self._cmd_segments_add(args[0], args[1:])
        elif subcommand in ("remove", "delete", "rm"):
            if not args:
                yield {"error": "Please provide a segment ID"}
                return
            yield await self._cmd_segments_remove(args[0])
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a segment ID"}
                return
            yield await self._cmd_segments_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a segment ID"}
                return
            yield await self._cmd_segments_disable(args[0])
        elif subcommand == "clear":
            yield await self._cmd_segments_clear()
        else:
            yield {"error": f"Unknown segments subcommand: {subcommand}"}

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "get_info":
            info = self._get_context_info()
            yield {"info": info.to_dict()}
        elif operation == "analyze_text":
            text = input_data.get("text", "")
            tokens = self._estimate_tokens(text)
            yield {"text": text, "tokens": tokens, "chars": len(text)}
        elif operation == "add_segment":
            segment_type = input_data.get("type")
            content = input_data.get("content")
            if not segment_type or not content:
                yield {"error": "type and content required"}
                return
            segment = self._add_segment(segment_type, content, input_data.get("metadata", {}))
            yield {"segment": segment.to_dict()}
        elif operation == "remove_segment":
            segment_id = input_data.get("id")
            if not segment_id:
                yield {"error": "id required"}
                return
            success = self._remove_segment(segment_id)
            yield {"success": success}
        elif operation == "set_max_tokens":
            max_tokens = input_data.get("max_tokens")
            if max_tokens:
                self._max_tokens = max_tokens
            yield {"max_tokens": self._max_tokens}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_config(self):
        """Load context configuration."""
        config_file = self._storage_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    self._max_tokens = config.get("max_tokens", 4096)
                    self._completion_tokens = config.get("completion_tokens", 512)
            except Exception as e:
                logger.error(f"Error loading config: {e}")

    def _save_config(self):
        """Save context configuration."""
        config_file = self._storage_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump({
                "max_tokens": self._max_tokens,
                "completion_tokens": self._completion_tokens,
            }, f, indent=2)

    # ========================================================================
    # Token Estimation
    # ========================================================================

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Rough estimation: ~4 chars per token for English text.
        For better accuracy, a proper tokenizer would be used.
        """
        if not text:
            return 0

        # Count words (rough approximation)
        words = len(text.split())

        # Count characters
        chars = len(text)

        # Use average of both estimates
        # Tokens ≈ words * 1.3 or chars / 4
        word_tokens = int(words * 1.3)
        char_tokens = chars // 4

        return max(word_tokens, char_tokens)

    # ========================================================================
    # Context Management
    # ========================================================================

    def _get_context_info(self) -> ContextInfo:
        """Get current context information."""
        total_tokens = sum(s.tokens for s in self._segments if s.enabled)
        return ContextInfo(
            total_tokens=total_tokens,
            max_tokens=self._max_tokens,
            segments=[s for s in self._segments],
            completion_tokens=self._completion_tokens,
        )

    def _add_segment(
        self,
        segment_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextSegment:
        """Add a context segment."""
        import uuid
        segment = ContextSegment(
            id=str(uuid.uuid4())[:8],
            type=segment_type,
            content=content,
            tokens=self._estimate_tokens(content),
            enabled=True,
            position=len(self._segments),
            metadata=metadata or {},
        )
        self._segments.append(segment)
        return segment

    def _remove_segment(self, segment_id: str) -> bool:
        """Remove a context segment."""
        for i, segment in enumerate(self._segments):
            if segment.id == segment_id:
                self._segments.pop(i)
                # Reorder positions
                for j, seg in enumerate(self._segments):
                    seg.position = j
                return True
        return False

    def _set_segment_enabled(self, segment_id: str, enabled: bool) -> bool:
        """Set enabled state of a segment."""
        for segment in self._segments:
            if segment.id == segment_id:
                segment.enabled = enabled
                return True
        return False

    def _trim_context(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Trim context to fit within max tokens."""
        max_toks = max_tokens or self._max_tokens
        removed = []

        # Calculate available tokens for context
        available = max_toks - self._completion_tokens

        # Sort segments by position (keep earlier segments)
        current_tokens = 0
        for segment in self._segments:
            if segment.enabled:
                current_tokens += segment.tokens

        # Disable segments until we fit
        while current_tokens > available and self._segments:
            # Find lowest priority segment to disable
            # For now, disable from the end (most recent)
            for segment in reversed(self._segments):
                if segment.enabled:
                    segment.enabled = False
                    current_tokens -= segment.tokens
                    removed.append(segment)
                    break

        return {
            "removed_count": len(removed),
            "remaining_tokens": current_tokens,
            "max_tokens": max_toks,
        }

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_info(self) -> Dict[str, Any]:
        """Show context information."""
        try:
            info = self._get_context_info()

            lines = ["\nContext Information:"]
            lines.append(f"Max Tokens: {info.max_tokens}")
            lines.append(f"Completion Tokens: {info.completion_tokens}")
            lines.append(f"Used Tokens: {info.total_tokens}")
            lines.append(f"Available Tokens: {info.max_tokens - info.total_tokens - info.completion_tokens}")
            lines.append(f"\nSegments: {len(info.segments)}")

            # Group by type
            by_type: Dict[str, List[ContextSegment]] = {}
            for segment in info.segments:
                if segment.type not in by_type:
                    by_type[segment.type] = []
                by_type[segment.type].append(segment)

            for seg_type, segments in by_type.items():
                total = sum(s.tokens for s in segments if s.enabled)
                enabled = sum(1 for s in segments if s.enabled)
                lines.append(f"  {seg_type}: {enabled}/{len(segments)} enabled, {total} tokens")

            return {
                "output": "\n".join(lines),
                "data": {"info": info.to_dict()}
            }
        except Exception as e:
            logger.error(f"Error getting context info: {e}")
            return {"error": str(e)}

    async def _cmd_analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for token count."""
        try:
            tokens = self._estimate_tokens(text)
            chars = len(text)
            words = len(text.split())

            lines = [f"\nText Analysis:"]
            lines.append(f"Characters: {chars}")
            lines.append(f"Words: {words}")
            lines.append(f"Estimated Tokens: {tokens}")
            lines.append(f"Tokens per 1000 chars: {int((tokens / chars) * 1000) if chars > 0 else 0}")

            return {
                "output": "\n".join(lines),
                "data": {"text": text, "tokens": tokens, "chars": chars, "words": words}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_trim(self, args: List[str]) -> Dict[str, Any]:
        """Trim context to max tokens."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Trim context")
            parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
            parser.add_argument("--completion", type=int, default=self._completion_tokens,
                               help="Completion tokens")

            parsed = parser.parse_args(args)

            if parsed.max_tokens:
                self._max_tokens = parsed.max_tokens
            if parsed.completion:
                self._completion_tokens = parsed.completion

            result = self._trim_context()

            lines = [f"\nContext Trimmed:"]
            lines.append(f"Segments removed: {result['removed_count']}")
            lines.append(f"Remaining tokens: {result['remaining_tokens']}")
            lines.append(f"Max tokens: {result['max_tokens']}")

            return {
                "output": "\n".join(lines),
                "data": result
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_segments_list(self) -> Dict[str, Any]:
        """List all context segments."""
        try:
            if not self._segments:
                return {"output": "No context segments found.", "data": {"segments": []}}

            lines = ["\nContext Segments:"]
            for segment in self._segments:
                status = "✓" if segment.enabled else "✗"
                content_preview = segment.content[:40] + "..." if len(segment.content) > 40 else segment.content
                lines.append(f"  {status} [{segment.id}] {segment.type} - {segment.tokens} tokens")
                lines.append(f"      {content_preview}")

            return {
                "output": "\n".join(lines),
                "data": {"segments": [s.to_dict() for s in self._segments]}
            }
        except Exception as e:
            logger.error(f"Error listing segments: {e}")
            return {"error": str(e)}

    async def _cmd_segments_add(self, segment_type: str, args: List[str]) -> Dict[str, Any]:
        """Add a context segment."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Add segment")
            parser.add_argument("--content", required=True, help="Segment content")
            parser.add_argument("--metadata", help="Metadata as JSON string")

            parsed = parser.parse_args(args)

            metadata = {}
            if parsed.metadata:
                metadata = json.loads(parsed.metadata)

            segment = self._add_segment(segment_type, parsed.content, metadata)

            return {
                "output": f"Segment '{segment.id}' added ({segment.tokens} tokens)",
                "data": {"segment": segment.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_segments_remove(self, segment_id: str) -> Dict[str, Any]:
        """Remove a context segment."""
        try:
            success = self._remove_segment(segment_id)
            if success:
                return {"output": f"Segment '{segment_id}' removed"}
            return {"error": f"Segment '{segment_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_segments_enable(self, segment_id: str) -> Dict[str, Any]:
        """Enable a context segment."""
        try:
            success = self._set_segment_enabled(segment_id, True)
            if success:
                return {"output": f"Segment '{segment_id}' enabled"}
            return {"error": f"Segment '{segment_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_segments_disable(self, segment_id: str) -> Dict[str, Any]:
        """Disable a context segment."""
        try:
            success = self._set_segment_enabled(segment_id, False)
            if success:
                return {"output": f"Segment '{segment_id}' disabled"}
            return {"error": f"Segment '{segment_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_segments_clear(self) -> Dict[str, Any]:
        """Clear all context segments."""
        try:
            count = len(self._segments)
            self._segments.clear()
            return {"output": f"Cleared {count} segments"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export context to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            data = {
                "max_tokens": self._max_tokens,
                "completion_tokens": self._completion_tokens,
                "segments": [s.to_dict() for s in self._segments],
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                "output": f"Context exported to {path}",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import context from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._max_tokens = data.get("max_tokens", 4096)
            self._completion_tokens = data.get("completion_tokens", 512)

            self._segments.clear()
            for segment_data in data.get("segments", []):
                segment = ContextSegment(**segment_data)
                self._segments.append(segment)

            self._save_config()

            return {
                "output": f"Context imported from {path} ({len(self._segments)} segments)",
                "data": {"imported": len(self._segments)}
            }
        except Exception as e:
            logger.error(f"Error importing context: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Context Management CLI

Commands:
    info                    Show context information
    analyze <text>          Analyze text for token count
    trim --max-tokens N     Trim context to max tokens
    segments list           List all context segments
    segments add <type>     Add a context segment
    segments remove <id>    Remove a context segment
    segments enable <id>    Enable a segment
    segments disable <id>   Disable a segment
    segments clear          Clear all segments
    export <file>           Export context to file
    import <file>           Import context from file

Trim Options:
    --max-tokens <int>      Maximum tokens
    --completion <int>      Completion tokens (default: 512)

Segment Types:
    system, user, assistant, lorebook, author_note, character_card

Add Segment Options:
    --content <text>        Segment content (required)
    --metadata <json>       Metadata as JSON string

Examples:
    uv run backend/manage.py context info
    uv run backend/manage.py context analyze "Hello world"
    uv run backend/manage.py context trim --max-tokens 2000
    uv run backend/manage.py context segments add system --content "You are a helpful assistant"
    uv run backend/manage.py context export backup.json
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/context", tags=["context"])

        @router.get("/info")
        async def get_info():
            """Get context information."""
            return {"info": self._get_context_info().to_dict()}

        @router.post("/analyze")
        async def analyze_text(request: dict):
            """Analyze text for tokens."""
            text = request.get("text", "")
            return {"text": text, "tokens": self._estimate_tokens(text)}

        @router.post("/segments")
        async def add_segment(request: dict):
            """Add a context segment."""
            segment_type = request.get("type")
            content = request.get("content")
            if not segment_type or not content:
                return {"error": "type and content required"}, 400
            segment = self._add_segment(segment_type, content, request.get("metadata", {}))
            return {"segment": segment.to_dict()}

        @router.delete("/segments/{segment_id}")
        async def remove_segment(segment_id: str):
            """Remove a context segment."""
            success = self._remove_segment(segment_id)
            if success:
                return {"success": True}
            return {"error": "Segment not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ContextExtension)
