"""
Inline Commands

Commands embedded directly in prompts:
- /hide - Hide text from output
- /show - Show text in output
- /comment - Add comment not sent to LLM
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class InlineCommandType(Enum):
    """Types of inline commands."""

    HIDE = "hide"  # Hide text from output
    SHOW = "show"  # Show text in output
    COMMENT = "comment"  # Comment not sent to LLM


@dataclass
class InlineCommand:
    """An inline command found in text."""

    command_type: InlineCommandType
    content: str
    start_pos: int
    end_pos: int


class InlineCommandProcessor:
    """
    Process inline commands in prompts.

    These commands allow users to control what gets sent to the LLM
    and what gets displayed in the output.
    """

    def __init__(self):
        """Initialize inline command processor."""
        pass

    def process(
        self,
        text: str,
    ) -> Tuple[str, List[InlineCommand], Dict[str, Any]]:
        """
        Process inline commands in text.

        Args:
            text: Input text possibly containing inline commands

        Returns:
            Tuple of:
            - Cleaned text (with commands removed or processed)
            - List of found commands
            - Metadata dict with processing info
        """
        commands = []
        metadata = {
            "had_hide": False,
            "had_show": False,
            "had_comment": False,
        }

        # Find all inline commands
        # Pattern: /command content ... /end_command (or end of line)

        # Process /hide ... /show (or end)
        hide_pattern = r"/hide\s+(.*?)(?=/show|$)"
        for match in re.finditer(hide_pattern, text, re.DOTALL):
            commands.append(InlineCommand(
                command_type=InlineCommandType.HIDE,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end(),
            ))
            metadata["had_hide"] = True

        # Process /show ... /hide (or end)
        show_pattern = r"/show\s+(.*?)(?=/hide|$)"
        for match in re.finditer(show_pattern, text, re.DOTALL):
            commands.append(InlineCommand(
                command_type=InlineCommandType.SHOW,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end(),
            ))
            metadata["had_show"] = True

        # Process /comment ... (until end or newline)
        comment_pattern = r"/comment\s+(.*?)(?=\n|$)"
        for match in re.finditer(comment_pattern, text, re.DOTALL):
            commands.append(InlineCommand(
                command_type=InlineCommandType.COMMENT,
                content=match.group(1).strip(),
                start_pos=match.start(),
                end_pos=match.end(),
            ))
            metadata["had_comment"] = True

        # Clean the text
        clean_text = text

        # Remove /hide.../show blocks (they're hidden from LLM)
        clean_text = re.sub(r"/hide\s+.*?(?=/show|$)", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"/show\s+.*?(?=/hide|$)", "", clean_text, flags=re.DOTALL)

        # Remove /comment blocks
        clean_text = re.sub(r"/comment\s+.*?(?=\n|$)", "", clean_text, flags=re.DOTALL)

        return clean_text, commands, metadata

    def get_llm_content(self, text: str) -> str:
        """
        Get content that should be sent to LLM.

        Removes /hide blocks and /comment content.
        Keeps /show blocks.
        """
        clean_text = text

        # Remove /hide.../show blocks (hidden from LLM)
        clean_text = re.sub(r"/hide\s+.*?(?=/show|$)", "", clean_text, flags=re.DOTALL)

        # Remove /comment blocks
        clean_text = re.sub(r"/comment\s+.*?(?=\n|$)", "", clean_text, flags=re.DOTALL)

        return clean_text.strip()

    def get_display_content(self, text: str) -> str:
        """
        Get content that should be displayed.

        Removes /hide blocks and /comment content.
        Keeps /show blocks.
        """
        # Same logic as LLM content for now
        return self.get_llm_content(text)

    def get_hidden_content(self, text: str) -> List[str]:
        """
        Get content that was hidden (/hide blocks).

        This might be useful for debugging or admin views.
        """
        hidden = []

        pattern = r"/hide\s+(.*?)(?=/show|$)"
        for match in re.finditer(pattern, text, re.DOTALL):
            hidden.append(match.group(1).strip())

        return hidden

    def get_comments(self, text: str) -> List[str]:
        """Get all /comment content."""
        comments = []

        pattern = r"/comment\s+(.*?)(?=\n|$)"
        for match in re.finditer(pattern, text, re.DOTALL):
            comments.append(match.group(1).strip())

        return comments


def create_inline_command_processor() -> InlineCommandProcessor:
    """Factory function to create InlineCommandProcessor."""
    return InlineCommandProcessor()
