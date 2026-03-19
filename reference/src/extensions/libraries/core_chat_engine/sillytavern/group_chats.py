"""
Group Chats Feature

Support for multiple characters in a single conversation.
Manages turn order, character-to-character interaction, and group dialogue formatting.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random


class TurnOrder(Enum):
    """Turn order strategies."""

    ROUND_ROBIN = "round_robin"  # Each character speaks in order
    RANDOM = "random"  # Random character speaks
    INTENT_BASED = "intent_based"  # Character most relevant to input
    USER_CHOICE = "user_choice"  # User specifies who speaks


@dataclass
class GroupCharacter:
    """A character in a group chat."""

    name: str
    personality_config: Any = None  # PersonalityConfig
    is_active: bool = True
    speak_probability: float = 1.0  # How likely to speak when called

    # State
    last_spoke_turn: int = 0
    turn_count: int = 0


@dataclass
class GroupTurn:
    """A single turn in a group conversation."""

    turn_number: int
    speaker: str  # Character name
    content: str
    target: Optional[str] = None  # Who they're addressing
    was_user: bool = False


class GroupChat:
    """
    Manages multi-character conversations.

    Features:
    - Multiple characters in single conversation
    - Turn order management
    - Character-to-character interaction
    - Group dialogue formatting
    """

    def __init__(
        self,
        characters: List[GroupCharacter] = None,
        turn_order: TurnOrder = TurnOrder.ROUND_ROBIN,
    ):
        """
        Initialize group chat.

        Args:
            characters: List of characters in the group
            turn_order: Strategy for determining who speaks next
        """
        self.characters: Dict[str, GroupCharacter] = {}

        if characters:
            for char in characters:
                self.characters[char.name] = char

        self.turn_order = turn_order
        self.turn_history: List[GroupTurn] = []
        self.current_turn = 0
        self.last_speaker: Optional[str] = None

    def add_character(
        self,
        name: str,
        personality_config: Any = None,
        speak_probability: float = 1.0,
    ) -> None:
        """Add a character to the group."""
        self.characters[name] = GroupCharacter(
            name=name,
            personality_config=personality_config,
            is_active=True,
            speak_probability=speak_probability,
        )

    def remove_character(self, name: str) -> None:
        """Remove a character from the group."""
        if name in self.characters:
            del self.characters[name]

    def set_character_active(self, name: str, active: bool) -> None:
        """Set whether a character is active."""
        if name in self.characters:
            self.characters[name].is_active = active

    def get_next_speaker(
        self,
        user_input: str = None,
        context: Any = None,
    ) -> Optional[str]:
        """
        Determine who should speak next.

        Args:
            user_input: Current user input
            context: Conversation context

        Returns:
            Character name who should speak, or None if no active characters
        """
        active_chars = [c for c in self.characters.values() if c.is_active]

        if not active_chars:
            return None

        if self.turn_order == TurnOrder.ROUND_ROBIN:
            return self._round_robin_speaker(active_chars)
        elif self.turn_order == TurnOrder.RANDOM:
            return self._random_speaker(active_chars)
        elif self.turn_order == TurnOrder.INTENT_BASED:
            return self._intent_based_speaker(active_chars, user_input, context)
        else:
            # Default to round robin
            return self._round_robin_speaker(active_chars)

    def _round_robin_speaker(self, active_chars: List[GroupCharacter]) -> str:
        """Get next speaker in round robin order."""
        if not self.last_speaker:
            return active_chars[0].name

        # Find index of last speaker
        last_index = next(
            i for i, c in enumerate(active_chars) if c.name == self.last_speaker
        )

        # Next speaker
        next_index = (last_index + 1) % len(active_chars)
        return active_chars[next_index].name

    def _random_speaker(self, active_chars: List[GroupCharacter]) -> str:
        """Get random speaker."""
        weights = [c.speak_probability for c in active_chars]
        return random.choices(active_chars, weights=weights)[0].name

    def _intent_based_speaker(
        self,
        active_chars: List[GroupCharacter],
        user_input: str,
        context: Any,
    ) -> str:
        """Get speaker based on intent/relevance."""
        # Simple implementation: return random
        # Could be enhanced with NLP to match input to character expertise
        return self._random_speaker(active_chars)

    def format_dialogue(
        self,
        speaker: str,
        content: str,
        target: Optional[str] = None,
    ) -> str:
        """
        Format character dialogue for display.

        Args:
            speaker: Character name
            content: What they're saying
            target: Who they're addressing

        Returns:
            Formatted dialogue string
        """
        if target:
            return f"{speaker} → {target}: {content}"
        else:
            return f"{speaker}: {content}"

    def get_active_characters(self) -> List[str]:
        """Get list of active character names."""
        return [c.name for c in self.characters.values() if c.is_active]

    def get_character_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a character."""
        if name not in self.characters:
            return {}

        char = self.characters[name]

        return {
            "name": char.name,
            "is_active": char.is_active,
            "turn_count": char.turn_count,
            "last_spoke_turn": char.last_spoke_turn,
            "speak_probability": char.speak_probability,
        }

    def record_turn(
        self,
        speaker: str,
        content: str,
        target: Optional[str] = None,
        was_user: bool = False,
    ) -> GroupTurn:
        """Record a turn in the conversation."""
        self.current_turn += 1

        turn = GroupTurn(
            turn_number=self.current_turn,
            speaker=speaker,
            content=content,
            target=target,
            was_user=was_user,
        )

        self.turn_history.append(turn)

        # Update character stats
        if speaker in self.characters and not was_user:
            self.characters[speaker].turn_count += 1
            self.characters[speaker].last_spoke_turn = self.current_turn

        self.last_speaker = speaker

        return turn


def create_group_chat(
    characters: List[GroupCharacter] = None,
    turn_order: TurnOrder = TurnOrder.ROUND_ROBIN,
) -> GroupChat:
    """Factory function to create GroupChat."""
    return GroupChat(
        characters=characters,
        turn_order=turn_order,
    )
