"""
SillyTavern Special Commands Detection

Detects and processes special SillyTavern-style commands:
- OOC (out of character)
- Story mode
- Inline commands (/hide, /show, /comment)
- Bias control (/bias, /bias-list)
- Text manipulation (/reverse, /continue)
"""

import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class SpecialCommandResult:
    """Result of special command detection."""

    command_type: str  # "ooc", "story_mode", "inline_hide", "inline_show", "inline_comment", "bias", "bias_list", "reverse", "continue"
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def detect_special_commands(text: str) -> Optional[SpecialCommandResult]:
    """
    Detect SillyTavern-style special commands in text.

    Args:
        text: Input text to check

    Returns:
        SpecialCommandResult if command detected, None otherwise
    """
    # Check for slash commands first (priority)
    slash_result = _detect_slash_commands(text)
    if slash_result:
        return slash_result

    # Check for OOC markers
    ooc_result = _detect_ooc(text)
    if ooc_result:
        return ooc_result

    # Check for story mode
    story_result = _detect_story_mode(text)
    if story_result:
        return story_result

    # Check for inline commands
    inline_result = _detect_inline_commands(text)
    if inline_result:
        return inline_result

    return None


def _detect_ooc(text: str) -> Optional[SpecialCommandResult]:
    """Detect out-of-character content."""
    # OOC markers: ((text))
    pattern = r"\(\((.*?)\)\)"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Combine all OOC content
        ooc_content = " ".join(matches).strip()
        return SpecialCommandResult(
            command_type="ooc",
            content=ooc_content,
            metadata={"markers_found": len(matches)},
        )

    return None


def _detect_story_mode(text: str) -> Optional[SpecialCommandResult]:
    """Detect story mode (action prefixed with >)."""
    if text.strip().startswith(">"):
        # Remove > and extract content
        content = text.strip()[1:].strip()
        return SpecialCommandResult(
            command_type="story_mode",
            content=content,
            metadata={"full_text": text},
        )

    return None


def _detect_inline_commands(text: str) -> Optional[SpecialCommandResult]:
    """Detect inline commands (/hide, /show, /comment)."""

    # /hide - hide text from output
    if "/hide" in text:
        # Extract content after /hide
        pattern = r"/hide\s+(.*?)(?:/show|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return SpecialCommandResult(
                command_type="inline_hide",
                content=match.group(1).strip(),
                metadata={"full_text": text},
            )

    # /show - show text in output
    if "/show" in text:
        pattern = r"/show\s+(.*?)(?:/hide|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return SpecialCommandResult(
                command_type="inline_show",
                content=match.group(1).strip(),
                metadata={"full_text": text},
            )

    # /comment - add comment not sent to LLM
    if "/comment" in text:
        pattern = r"/comment\s+(.*?)(?:/$|\n|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return SpecialCommandResult(
                command_type="inline_comment",
                content=match.group(1).strip(),
                metadata={"full_text": text},
            )

    return None


def _detect_slash_commands(text: str) -> Optional[SpecialCommandResult]:
    """Detect slash commands (/bias, /bias-list, /reverse, /continue)."""

    # /bias <value> - Set bias level for AI responses
    # /bias - Show current bias level
    bias_match = re.match(r"^/bias(?:\s+(.+))?$", text.strip())
    if bias_match:
        bias_value = bias_match.group(1) if bias_match.group(1) else None
        return SpecialCommandResult(
            command_type="bias",
            content=bias_value or "",
            metadata={
                "action": "get" if bias_value is None else "set",
                "value": bias_value
            },
        )

    # /bias-list - Show all bias settings
    if re.match(r"^/bias-list(?:\s|$)", text.strip()):
        return SpecialCommandResult(
            command_type="bias_list",
            content="",
            metadata={"action": "list"},
        )

    # /reverse - Reverse the last message or input
    if re.match(r"^/reverse(?:\s|$)", text.strip()):
        # Extract any additional text after /reverse
        reverse_match = re.match(r"^/reverse(?:\s+(.*))?$", text.strip())
        reverse_content = reverse_match.group(1) if reverse_match.group(1) else ""
        return SpecialCommandResult(
            command_type="reverse",
            content=reverse_content,
            metadata={"action": "reverse"},
        )

    # /continue - Continue the last incomplete generation
    if re.match(r"^/continue(?:\s|$)", text.strip()):
        # Extract any additional context after /continue
        continue_match = re.match(r"^/continue(?:\s+(.*))?$", text.strip())
        continue_content = continue_match.group(1) if continue_match.group(1) else ""
        return SpecialCommandResult(
            command_type="continue",
            content=continue_content,
            metadata={"action": "continue"},
        )

    # /note <text> - Set author's note
    note_match = re.match(r"^/note(?:\s+(.*))?$", text.strip())
    if note_match:
        note_content = note_match.group(1) if note_match.group(1) else ""
        return SpecialCommandResult(
            command_type="note",
            content=note_content,
            metadata={"action": "set_note"},
        )

    # /popup <text> - Show as popup/notification
    popup_match = re.match(r"^/popup(?:\s+(.*))?$", text.strip())
    if popup_match:
        popup_content = popup_match.group(1) if popup_match.group(1) else ""
        return SpecialCommandResult(
            command_type="popup",
            content=popup_content,
            metadata={"action": "show_popup"},
        )

    # /dry <count> - Run N times without changing context (dry run)
    dry_match = re.match(r"^/dry(?:\s+(\d+))?$", text.strip())
    if dry_match:
        dry_count = int(dry_match.group(1)) if dry_match.group(1) else 1
        return SpecialCommandResult(
            command_type="dry",
            content=str(dry_count),
            metadata={"action": "dry_run", "count": dry_count},
        )

    # /query <text> - Query system without sending to LLM
    query_match = re.match(r"^/query(?:\s+(.*))?$", text.strip())
    if query_match:
        query_content = query_match.group(1) if query_match.group(1) else ""
        return SpecialCommandResult(
            command_type="query",
            content=query_content,
            metadata={"action": "query_system"},
        )

    # /regen - Regenerate last response
    if re.match(r"^/regen(?:\s|$)", text.strip()):
        return SpecialCommandResult(
            command_type="regen",
            content="",
            metadata={"action": "regenerate"},
        )

    # /edit <text> - Edit last message
    edit_match = re.match(r"^/edit(?:\s+(.*))?$", text.strip())
    if edit_match:
        edit_content = edit_match.group(1) if edit_match.group(1) else ""
        return SpecialCommandResult(
            command_type="edit",
            content=edit_content,
            metadata={"action": "edit_last"},
        )

    # /last - Show last message
    if re.match(r"^/last(?:\s|$)", text.strip()):
        return SpecialCommandResult(
            command_type="last",
            content="",
            metadata={"action": "show_last"},
        )

    # /clear - Clear conversation context
    if re.match(r"^/clear(?:\s|$)", text.strip()):
        return SpecialCommandResult(
            command_type="clear",
            content="",
            metadata={"action": "clear_context"},
        )

    return None


def strip_special_commands(text: str) -> str:
    """
    Strip special commands from text, returning clean version.

    Args:
        text: Input text possibly containing special commands

    Returns:
        Text with special commands removed
    """
    clean_text = text

    # Remove slash commands (must be on their own line)
    clean_text = re.sub(r"^/bias(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/bias-list\s*$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/reverse(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/continue(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/note(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/popup(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/dry(?:\s+\d+)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/query(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/regen\s*$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/edit(?:\s+.*)?$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/last\s*$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^/clear\s*$", "", clean_text, flags=re.MULTILINE)

    # Remove OOC content
    clean_text = re.sub(r"\(\((.*?)\)\)", "", clean_text, flags=re.DOTALL)

    # Remove story mode marker
    clean_text = re.sub(r"^>\s*", "", clean_text.strip(), flags=re.MULTILINE)

    # Remove inline commands
    clean_text = re.sub(r"/hide\s+.*?(?=/show|/hide|$)", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"/show\s+.*?(?=/hide|/show|$)", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"/comment\s+.*?(?=/\n|$)", "", clean_text, flags=re.DOTALL)

    return clean_text.strip()


def process_ooc_content(ooc_text: str, context: Dict[str, Any]) -> str:
    """
    Process OOC content and return appropriate response.

    Args:
        ooc_text: The OOC content
        context: Current conversation context

    Returns:
        Response to the OOC comment
    """
    # Common OOC requests
    ooc_lower = ooc_text.lower()

    # System commands
    if "restart" in ooc_lower or "reset" in ooc_lower:
        return "[System: Would you like to restart the conversation? This will clear current context.]"

    if "save" in ooc_lower:
        return "[System: Conversation auto-saved. Manual save coming soon.]"

    if "help" in ooc_lower:
        return """[System: Available commands:
        - ((restart)) - Restart conversation
        - ((save)) - Save conversation
        - ((clear)) - Clear context

        Slash Commands:
        - /note <text> - Set author's note
        - /popup <text> - Show as popup/notification
        - /bias <value> - Set AI bias level (0.0-2.0, default 1.0)
        - /bias-list - Show all bias settings
        - /reverse [text] - Reverse last message or specified text
        - /continue [text] - Continue last generation with optional context
        - /dry <count> - Run N times without changing context (dry run)
        - /query <text> - Query system without sending to LLM
        - /regen - Regenerate last response
        - /edit <text> - Edit last message
        - /last - Show last message
        - /clear - Clear conversation context

        Story Mode:
        - >action - Story mode (action prefixed with >)

        Inline Commands:
        - /hide <text> - Hide text from output
        - /show <text> - Show text in output
        - /comment <text> - Add comment not sent to LLM]"""

    # Meta comments
    if "good response" in ooc_lower or "nice" in ooc_lower:
        return "[OOC acknowledged: Thanks for the feedback!]"

    if "bad response" in ooc_lower or "wrong" in ooc_lower:
        return "[OOC acknowledged: I'll try to do better next time.]"

    # Default acknowledgment
    return f"[OOC: I received your comment: '{ooc_text}'. Continuing in character...]"
