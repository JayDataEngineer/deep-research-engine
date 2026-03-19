"""
Conversation Variety Module

Ensures conversations are different and personalized by introducing:
- Dynamic temperature adjustment
- Response pattern variation
- Persona micro-shifts based on context
- Randomness injection for freshness
"""

import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ResponsePattern(Enum):
    """Different response patterns for variety."""

    CONVERSATIONAL = "conversational"  # Normal back-and-forth
    DETAILED = "detailed"  # Longer, more explanatory
    CONCISE = "concise"  # Short and to the point
    PLAYFUL = "playful"  # Light-hearted, fun
    THOUGHTFUL = "thoughtful"  # Reflective, deeper
    ENTHUSIASTIC = "enthusiastic"  # High energy
    CALM = "calm"  # Relaxed, measured


@dataclass
class VarietyConfig:
    """Configuration for conversation variety."""

    # Enable variety features
    enabled: bool = True

    # Temperature variance
    base_temperature: float = 0.7
    temperature_variance: float = 0.2  # +/- variance

    # Response pattern randomization
    enable_pattern_rotation: bool = True
    pattern_weights: Dict[ResponsePattern, float] = None

    # Persona micro-shifts
    enable_persona_shifts: bool = True
    shift_probability: float = 0.15  # 15% chance to shift slightly

    # Response length variation
    enable_length_variation: bool = True
    short_response_probability: float = 0.2  # 20% chance of shorter response
    long_response_probability: float = 0.15  # 15% chance of longer response

    def __post_init__(self):
        if self.pattern_weights is None:
            # Default balanced weights
            self.pattern_weights = {
                ResponsePattern.CONVERSATIONAL: 0.35,
                ResponsePattern.DETAILED: 0.15,
                ResponsePattern.CONCISE: 0.15,
                ResponsePattern.PLAYFUL: 0.10,
                ResponsePattern.THOUGHTFUL: 0.10,
                ResponsePattern.ENTHUSIASTIC: 0.10,
                ResponsePattern.CALM: 0.05,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "base_temperature": self.base_temperature,
            "temperature_variance": self.temperature_variance,
            "enable_pattern_rotation": self.enable_pattern_rotation,
            "pattern_weights": {p.value: w for p, w in self.pattern_weights.items()},
            "enable_persona_shifts": self.enable_persona_shifts,
            "shift_probability": self.shift_probability,
            "enable_length_variation": self.enable_length_variation,
            "short_response_probability": self.short_response_probability,
            "long_response_probability": self.long_response_probability,
        }


class ConversationVarietyManager:
    """
    Manages conversation variety to keep interactions fresh.

    Features:
    - Dynamic temperature adjustment based on context
    - Response pattern rotation
    - Persona micro-shifts for subtle personality changes
    - Length variation for more natural flow
    """

    def __init__(self, config: Optional[VarietyConfig] = None):
        """
        Initialize the variety manager.

        Args:
            config: Variety configuration
        """
        self.config = config or VarietyConfig()
        self._current_pattern: Optional[ResponsePattern] = None
        self._consecutive_same_pattern = 0
        self._max_consecutive_same = 3

        # Track recent temperatures for smoothing
        self._recent_temperatures: List[float] = []
        self._temp_history_size = 5

    def get_temperature(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get dynamic temperature for this turn.

        Adds variance to base temperature while maintaining consistency.

        Args:
            context: Optional context about the conversation

        Returns:
            Temperature value for this turn
        """
        if not self.config.enabled:
            return self.config.base_temperature

        # Calculate variance
        variance = random.uniform(
            -self.config.temperature_variance,
            self.config.temperature_variance
        )

        # Apply some smoothing based on recent temperatures
        if self._recent_temperatures:
            avg_temp = sum(self._recent_temperatures) / len(self._recent_temperatures)
            # Blend new temp with average for smoothness
            variance = (variance * 0.7) + ((avg_temp - self.config.base_temperature) * 0.3)

        temperature = max(0.1, min(2.0, self.config.base_temperature + variance))

        # Update history
        self._recent_temperatures.append(temperature)
        if len(self._recent_temperatures) > self._temp_history_size:
            self._recent_temperatures.pop(0)

        return temperature

    def get_response_pattern(self, force_change: bool = False) -> ResponsePattern:
        """
        Get the response pattern for this turn.

        Rotates through patterns based on weights and prevents
        too much repetition.

        Args:
            force_change: Force a different pattern than current

        Returns:
            Selected response pattern
        """
        if not self.config.enabled or not self.config.enable_pattern_rotation:
            return ResponsePattern.CONVERSATIONAL

        # Check if we should change pattern
        should_change = (
            force_change or
            self._current_pattern is None or
            self._consecutive_same_pattern >= self._max_consecutive_same
        )

        if should_change:
            # Weighted random selection
            patterns = list(self.config.pattern_weights.keys())
            weights = list(self.config.pattern_weights.values())

            # Exclude current pattern if we've used it too much
            if self._current_pattern and self._consecutive_same_pattern >= 2:
                current_idx = patterns.index(self._current_pattern)
                weights[current_idx] *= 0.1  # Reduce weight significantly

            new_pattern = random.choices(patterns, weights=weights)[0]

            if new_pattern != self._current_pattern:
                self._current_pattern = new_pattern
                self._consecutive_same_pattern = 1
            else:
                self._consecutive_same_pattern += 1
        else:
            self._consecutive_same_pattern += 1

        return self._current_pattern

    def get_pattern_instruction(self, pattern: ResponsePattern) -> str:
        """
        Get instruction for the current pattern.

        This can be injected into the prompt to guide the response.

        Args:
            pattern: The response pattern

        Returns:
            Instruction string for the pattern
        """
        instructions = {
            ResponsePattern.CONVERSATIONAL: "",
            ResponsePattern.DETAILED: (
                "Provide a thorough, detailed response that fully explores the topic. "
                "Include relevant context and examples."
            ),
            ResponsePattern.CONCISE: (
                "Keep your response brief and to the point. "
                "Focus on the key points without unnecessary elaboration."
            ),
            ResponsePattern.PLAYFUL: (
                "Feel free to be playful and light-hearted in your response. "
                "Use a friendly, fun tone."
            ),
            ResponsePattern.THOUGHTFUL: (
                "Take a thoughtful, reflective approach to your response. "
                "Consider deeper implications and nuances."
            ),
            ResponsePattern.ENTHUSIASTIC: (
                "Bring high energy and enthusiasm to your response. "
                "Show excitement and engagement."
            ),
            ResponsePattern.CALM: (
                "Maintain a calm, measured tone in your response. "
                "Be relaxed and composed."
            ),
        }
        return instructions.get(pattern, "")

    def get_max_tokens_adjustment(self) -> Optional[int]:
        """
        Get adjustment to max_tokens based on variety settings.

        Returns:
            Adjustment factor or None
        """
        if not self.config.enabled or not self.config.enable_length_variation:
            return None

        roll = random.random()

        # Short response
        if roll < self.config.short_response_probability:
            return -150  # Reduce token count

        # Long response
        if roll < (self.config.short_response_probability + self.config.long_response_probability):
            return 200  # Increase token count

        return None

    def should_shift_persona(self) -> bool:
        """
        Determine if we should apply a persona micro-shift.

        Returns:
            True if persona should shift slightly
        """
        if not self.config.enabled or not self.config.enable_persona_shifts:
            return False

        return random.random() < self.config.shift_probability

    def get_persona_shift_instruction(self) -> str:
        """
        Get instruction for persona micro-shift.

        Returns:
            Instruction string for the shift
        """
        shifts = [
            "Be slightly more formal than usual.",
            "Be a bit more casual and relaxed.",
            "Show more empathy in your response.",
            "Be more direct and straightforward.",
            "Add a touch of humor to your response.",
            "Be more inquisitive and ask follow-up questions.",
            "Focus on being encouraging and supportive.",
            "Take a more analytical approach.",
        ]

        return random.choice(shifts)

    def get_variety_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get complete variety context for this turn.

        Args:
            context: Optional conversation context

        Returns:
            Dictionary with temperature, pattern, and instructions
        """
        pattern = self.get_response_pattern()
        temperature = self.get_temperature(context)

        result = {
            "temperature": temperature,
            "pattern": pattern.value,
            "pattern_instruction": self.get_pattern_instruction(pattern),
            "max_tokens_adjustment": self.get_max_tokens_adjustment(),
        }

        # Add persona shift if applicable
        if self.should_shift_persona():
            result["persona_shift"] = self.get_persona_shift_instruction()

        return result

    def reset_pattern(self):
        """Reset the current pattern (e.g., after a topic change)."""
        self._current_pattern = None
        self._consecutive_same_pattern = 0


def create_variety_manager(
    enabled: bool = True,
    temperature_variance: float = 0.2,
    enable_pattern_rotation: bool = True,
) -> ConversationVarietyManager:
    """
    Factory function to create a variety manager.

    Args:
        enabled: Enable variety features
        temperature_variance: Amount of temperature variance
        enable_pattern_rotation: Enable response pattern rotation

    Returns:
        Initialized variety manager
    """
    config = VarietyConfig(
        enabled=enabled,
        temperature_variance=temperature_variance,
        enable_pattern_rotation=enable_pattern_rotation,
    )
    return ConversationVarietyManager(config)
