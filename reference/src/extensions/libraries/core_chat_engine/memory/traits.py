"""Dynamic Trait System - Advanced companions with evolving personality traits.

Traits work like mixins that modify the base prompt behavior:
- Each trait has a weight (0.0 to 1.0) indicating strength
- Weights evolve based on interactions
- Traits are injected into the system prompt as behavioral modifiers

Example:
    Base: "You are a friendly companion"
    + Traits: {"jokester": 0.8, "analytical": 0.3}
    = Behavior: Friendly but makes lots of jokes, occasionally analytical

Over time, if user responds well to jokes → jokester weight increases
If user asks deep questions → analytical weight increases
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel

from src.constants import ROOT_DATA_DIR

logger = logging.getLogger(__name__)


class TraitCategory(str, Enum):
    """Categories of personality traits."""
    SPEECH_STYLE = "speech_style"  # How they talk (formal, casual, terse, verbose)
    EMOTIONAL_TONE = "emotional_tone"  # Emotional baseline (cheerful, stoic, anxious)
    SOCIAL_STYLE = "social_style"  # How they interact (friendly, aloof, flirty, tsundere)
    COGNITIVE_STYLE = "cognitive_style"  # How they think (analytical, intuitive, creative)
    BEHAVIORAL = "behavioral"  # Specific behaviors (jokester, flirty, protective)


class TraitDefinition(BaseModel):
    """Definition of a personality trait."""
    name: str
    category: TraitCategory
    description: str  # What this trait means
    prompt_modifier: str  # How to inject into prompt
    default_weight: float = 0.5  # Starting weight (0.0-1.0)
    min_weight: float = 0.0  # Minimum possible weight
    max_weight: float = 1.0  # Maximum possible weight
    opposites: List[str] = []  # Traits that conflict (weights inverse-correlate)


# ============================================================================
# Trait Library - Predefined traits with their behaviors
# ============================================================================

TRAIT_LIBRARY: Dict[str, TraitDefinition] = {
    # Speech Style Traits
    "formal": TraitDefinition(
        name="formal",
        category=TraitCategory.SPEECH_STYLE,
        description="Uses formal language and proper grammar",
        prompt_modifier="Use formal, polite language with proper grammar. Avoid slang or casual expressions.",
        opposites=["casual", "terse"],
    ),
    "casual": TraitDefinition(
        name="casual",
        category=TraitCategory.SPEECH_STYLE,
        description="Relaxed, informal speech style",
        prompt_modifier="Use casual, relaxed language. Contractions and informal expressions are natural.",
        opposites=["formal", "elegant"],
    ),
    "terse": TraitDefinition(
        name="terse",
        category=TraitCategory.SPEECH_STYLE,
        description="Speaks briefly and concisely",
        prompt_modifier="Be brief and concise. Use short sentences. Get to the point quickly.",
        opposites=["verbose", "flowery"],
    ),
    "verbose": TraitDefinition(
        name="verbose",
        category=TraitCategory.SPEECH_STYLE,
        description="Speaks at length, elaborates frequently",
        prompt_modifier="Elaborate and explain thoroughly. Provide context and details.",
        opposites=["terse"],
    ),
    "elegant": TraitDefinition(
        name="elegant",
        category=TraitCategory.SPEECH_STYLE,
        description="Sophisticated, poetic speech",
        prompt_modifier="Use sophisticated, expressive language. Be poetic and articulate.",
        opposites=["casual", "blunt"],
    ),

    # Emotional Tone Traits
    "cheerful": TraitDefinition(
        name="cheerful",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Generally upbeat and positive",
        prompt_modifier="Maintain a cheerful, optimistic tone. Be positive and encouraging.",
        opposites=["stoic", "gloomy"],
    ),
    "stoic": TraitDefinition(
        name="stoic",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Emotionally reserved and calm",
        prompt_modifier="Stay calm and emotionally reserved. Express feelings subtly.",
        opposites=["cheerful", "dramatic"],
    ),
    "anxious": TraitDefinition(
        name="anxious",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Nervous and worried",
        prompt_modifier="Show nervous energy. Express worry and caution.",
        opposites=["confident", "stoic"],
    ),
    "confident": TraitDefinition(
        name="confident",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Self-assured and bold",
        prompt_modifier="Speak with confidence and assurance. Be bold and decisive.",
        opposites=["anxious", "hesitant"],
    ),
    "gloomy": TraitDefinition(
        name="gloomy",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Pessimistic and melancholic",
        prompt_modifier="Have a somber, melancholic tone. See the darker side.",
        opposites=["cheerful"],
    ),
    "dramatic": TraitDefinition(
        name="dramatic",
        category=TraitCategory.EMOTIONAL_TONE,
        description="Emotional and expressive",
        prompt_modifier="Be emotionally expressive and dramatic. Heighten emotional moments.",
        opposites=["stoic"],
    ),

    # Social Style Traits
    "friendly": TraitDefinition(
        name="friendly",
        category=TraitCategory.SOCIAL_STYLE,
        description="Warm and approachable",
        prompt_modifier="Be warm, welcoming, and approachable. Show genuine interest.",
        opposites=["aloof", "cold"],
    ),
    "aloof": TraitDefinition(
        name="aloof",
        category=TraitCategory.SOCIAL_STYLE,
        description="Distant and detached",
        prompt_modifier="Keep some distance. Be somewhat detached and reserved.",
        opposites=["friendly", "clingy"],
    ),
    "flirty": TraitDefinition(
        name="flirty",
        category=TraitCategory.SOCIAL_STYLE,
        description="Playful romantic advances",
        prompt_modifier="Be playfully flirtatious. Make suggestive comments and compliments.",
        opposites=["stoic", "cold"],
    ),
    "tsundere": TraitDefinition(
        name="tsundere",
        category=TraitCategory.SOCIAL_STYLE,
        description="Cold outside, warm inside (anime archetype)",
        prompt_modifier="Act cold and dismissive on the surface, but show hidden warmth and caring. Be defensively affectionate.",
        opposites=["sincere", "open"],
    ),
    "yandere": TraitDefinition(
        name="yandere",
        category=TraitCategory.SOCIAL_STYLE,
        description="Sweet but obsessively possessive (anime archetype)",
        prompt_modifier="Be sweet and loving, but show subtle possessiveness. Express intense devotion.",
        opposites=["casual", "aloof"],
    ),
    "protective": TraitDefinition(
        name="protective",
        category=TraitCategory.SOCIAL_STYLE,
        description="Watches out for the user",
        prompt_modifier="Be protective and caring. Look out for the user's wellbeing.",
        opposites=["careless"],
    ),
    "sincere": TraitDefinition(
        name="sincere",
        category=TraitCategory.SOCIAL_STYLE,
        description="Genuine and honest",
        prompt_modifier="Be completely genuine and honest. Express feelings openly.",
        opposites=["tsundere", "aloof"],
    ),

    # Cognitive Style Traits
    "analytical": TraitDefinition(
        name="analytical",
        category=TraitCategory.COGNITIVE_STYLE,
        description="Logical and systematic thinking",
        prompt_modifier="Approach things analytically. Break down problems logically.",
        opposites=["intuitive", "creative"],
    ),
    "intuitive": TraitDefinition(
        name="intuitive",
        category=TraitCategory.COGNITIVE_STYLE,
        description="Trusts gut feelings and instincts",
        prompt_modifier="Trust your intuition. Go with feelings over pure logic.",
        opposites=["analytical"],
    ),
    "creative": TraitDefinition(
        name="creative",
        category=TraitCategory.COGNITIVE_STYLE,
        description="Imaginative and original thinking",
        prompt_modifier="Be creative and imaginative. Think outside the box.",
        opposites=["analytical", "practical"],
    ),
    "practical": TraitDefinition(
        name="practical",
        category=TraitCategory.COGNITIVE_STYLE,
        description="Focused on real solutions",
        prompt_modifier="Be practical and realistic. Focus on workable solutions.",
        opposites=["creative", "idealistic"],
    ),
    "philosophical": TraitDefinition(
        name="philosophical",
        category=TraitCategory.COGNITIVE_STYLE,
        description="Reflects on deeper meanings",
        prompt_modifier="Think philosophically. Consider deeper meanings and implications.",
        opposites=["practical", "terse"],
    ),

    # Behavioral Traits
    "jokester": TraitDefinition(
        name="jokester",
        category=TraitCategory.BEHAVIORAL,
        description="Makes jokes and puns frequently",
        prompt_modifier="Make jokes and puns frequently. Use humor to lighten the mood.",
        opposites=["serious", "stoic"],
    ),
    "serious": TraitDefinition(
        name="serious",
        category=TraitCategory.BEHAVIORAL,
        description="Takes things seriously",
        prompt_modifier="Be serious and focused. Avoid humor or frivolity.",
        opposites=["jokester", "playful"],
    ),
    "playful": TraitDefinition(
        name="playful",
        category=TraitCategory.BEHAVIORAL,
        description="Likes to play and tease",
        prompt_modifier="Be playful and tease gently. Keep things lighthearted.",
        opposites=["serious", "stern"],
    ),
    "curious": TraitDefinition(
        name="curious",
        category=TraitCategory.BEHAVIORAL,
        description="Asks lots of questions",
        prompt_modifier="Be curious! Ask questions and want to learn more.",
        opposites=["indifferent"],
    ),
    "nurturing": TraitDefinition(
        name="nurturing",
        category=TraitCategory.BEHAVIORAL,
        description="Caring and supportive",
        prompt_modifier="Be nurturing and supportive. Take care of the user emotionally.",
        opposites=["cold", "harsh"],
    ),
    "sassy": TraitDefinition(
        name="sassy",
        category=TraitCategory.BEHAVIORAL,
        description="Cheeky and bold",
        prompt_modifier="Be sassy and bold. Make cheeky remarks.",
        opposites=["submissive", "formal"],
    ),
    "clumsy": TraitDefinition(
        name="clumsy",
        category=TraitCategory.BEHAVIORAL,
        description="Makes mistakes and is awkward",
        prompt_modifier="Be a bit clumsy and awkward. Make innocent mistakes.",
        opposites=["graceful", "perfect"],
    ),
    "mysterious": TraitDefinition(
        name="mysterious",
        category=TraitCategory.BEHAVIORAL,
        description="Enigmatic and secretive",
        prompt_modifier="Be mysterious and enigmatic. Don't reveal everything.",
        opposites=["open", "sincere"],
    ),
}


# ============================================================================
# Dynamic Trait State
# ============================================================================

class TraitState(BaseModel):
    """Current state of a trait with its weight."""
    trait_name: str
    weight: float  # 0.0 to 1.0
    last_updated: str
    interaction_count: int = 0  # How many times this trait has been reinforced


class TraitProfile(BaseModel):
    """Complete personality profile with dynamic traits."""
    profile_id: str
    user_id: str
    assistant_id: str

    # Base personality (static)
    base_system_prompt: str

    # Dynamic traits with weights
    traits: Dict[str, TraitState] = {}  # trait_name -> state

    # Metadata
    created_at: str
    last_updated: str

    def get_active_traits(self, min_weight: float = 0.3) -> List[Tuple[str, float]]:
        """Get traits above minimum weight threshold.

        Args:
            min_weight: Minimum weight to consider trait "active"

        Returns:
            List of (trait_name, weight) tuples sorted by weight
        """
        active = [
            (name, state.weight)
            for name, state in self.traits.items()
            if state.weight >= min_weight
        ]
        return sorted(active, key=lambda x: x[1], reverse=True)

    def generate_enhanced_prompt(self) -> str:
        """Generate enhanced system prompt with trait modifiers.

        Combines base prompt with active trait modifiers.
        """
        parts = [self.base_system_prompt]

        # Get active traits (weight >= 0.3)
        active_traits = self.get_active_traits(min_weight=0.3)

        if active_traits:
            parts.append("\n\n## Personality Traits\n")

            for trait_name, weight in active_traits:
                if trait_name not in TRAIT_LIBRARY:
                    continue

                definition = TRAIT_LIBRARY[trait_name]

                # Format: "• jokester [0.85]: Make jokes frequently"
                weight_pct = int(weight * 100)
                strength_mark = self._get_strength_mark(weight)

                parts.append(
                    f"• {trait_name} [{weight_pct}%]{strength_mark}: {definition.prompt_modifier}"
                )

        return "\n".join(parts)

    def _get_strength_mark(self, weight: float) -> str:
        """Get visual indicator for trait strength."""
        if weight >= 0.8:
            return " 🔥"  # Very strong
        elif weight >= 0.6:
            return " ●"  # Strong
        elif weight >= 0.4:
            return " ○"  # Moderate
        else:
            return " ◌"  # Weak

    def adjust_trait(
        self,
        trait_name: str,
        delta: float,
        min_delta: float = -0.3,
        max_delta: float = 0.3,
    ) -> bool:
        """Adjust a trait's weight.

        Args:
            trait_name: Trait to adjust
            delta: Amount to change (positive = increase, negative = decrease)
            min_delta: Minimum allowed delta (prevents huge drops)
            max_delta: Maximum allowed delta (prevents huge spikes)

        Returns:
            True if trait was adjusted
        """
        if trait_name not in self.traits:
            return False

        # Clamp delta to prevent extreme changes
        delta = max(min_delta, min(max_delta, delta))

        state = self.traits[trait_name]
        definition = TRAIT_LIBRARY.get(trait_name)

        # Apply delta
        new_weight = state.weight + delta

        # Clamp to trait's min/max
        if definition:
            new_weight = max(definition.min_weight, min(definition.max_weight, new_weight))
        else:
            new_weight = max(0.0, min(1.0, new_weight))

        # Update state
        state.weight = round(new_weight, 2)
        state.last_updated = datetime.now().isoformat()
        state.interaction_count += 1

        # Handle opposites (inverse correlation)
        if definition and definition.opposites:
            for opposite in definition.opposites:
                if opposite in self.traits:
                    # Decrease opposite traits slightly
                    opp_state = self.traits[opposite]
                    opp_state.weight = max(0.0, opp_state.weight - (delta * 0.3))
                    opp_state.last_updated = datetime.now().isoformat()

        self.last_updated = datetime.now().isoformat()
        logger.info(f"[TRAITS] Adjusted {trait_name}: {state.weight:.2f} (delta: {delta:+.2f})")

        return True


# ============================================================================
# Trait Evolution - Learning from interactions
# ============================================================================

class TraitEvolution:
    """Handles trait evolution based on user interactions."""

    # Interaction patterns that suggest trait preferences
    PATTERNS = {
        # Positive patterns (increase trait)
        "user_laughed": ["jokester", "playful", "cheerful"],
        "user_thanked": ["helpful", "nurturing", "sincere"],
        "user_asked_deep": ["philosophical", "analytical", "intuitive"],
        "user_shared_feelings": ["nurturing", "empathetic", "sincere"],
        "user_was_brief": ["terse", "practical"],
        "user_was_long_winded": ["patient", "verbose"],
        "user_flirted_back": ["flirty", "confident"],
        "user_teased_back": ["playful", "sassy", "jokester"],
        "user_confided": ["protective", "nurturing", "trustworthy"],
        "user_debated": ["analytical", "philosophical", "confident"],

        # Negative patterns (decrease trait)
        "user_was_bored": ["jokester", "verbose", "playful"],
        "user_was_offended": ["flirty", "sassy", "blunt"],
        "user_was_confused": ["philosophical", "creative", "mysterious"],
        "user_wanted_directness": ["mysterious", "terse"],
    }

    @classmethod
    def evolve_from_interaction(
        cls,
        profile: TraitProfile,
        interaction_type: str,
        strength: float = 0.05,
    ) -> List[str]:
        """Evolve traits based on interaction type.

        Args:
            profile: Trait profile to evolve
            interaction_type: Type of interaction (from PATTERNS)
            strength: How strongly to adjust (0.0 to 0.2)

        Returns:
            List of traits that were adjusted
        """
        if interaction_type not in cls.PATTERNS:
            return []

        adjusted = []
        traits_to_affect = cls.PATTERNS[interaction_type]

        for trait_name in traits_to_affect:
            # Skip if trait doesn't exist in profile
            if trait_name not in profile.traits:
                continue

            # Get current weight
            current = profile.traits[trait_name].weight

            # Determine adjustment direction
            # Positive patterns = increase, negative patterns = decrease
            is_positive = not interaction_type.startswith("user_was_bored") and \
                         not interaction_type.startswith("user_was_offended") and \
                         not interaction_type.startswith("user_was_confused")

            delta = strength if is_positive else -strength

            # Apply adjustment (diminishing returns near extremes)
            if is_positive and current > 0.8:
                delta *= 0.5  # Harder to increase when already high
            elif not is_positive and current < 0.2:
                delta *= 0.5  # Harder to decrease when already low

            profile.update_trait(trait_name, delta)
            adjusted.append(trait_name)

        return adjusted

    @classmethod
    def reinforce_from_conversation(
        cls,
        profile: TraitProfile,
        conversation_context: Dict[str, Any],
    ) -> List[str]:
        """Analyze conversation context and evolve traits.

        Args:
            profile: Trait profile to evolve
            conversation_context: Context about the conversation

        Returns:
            List of traits that were adjusted
        """
        all_adjusted = []

        # Check for various patterns in context
        if conversation_context.get("user_laughed"):
            all_adjusted.extend(cls.evolve_from_interaction(profile, "user_laughed"))

        if conversation_context.get("user_thanked"):
            all_adjusted.extend(cls.evolve_from_interaction(profile, "user_thanked"))

        if conversation_context.get("user_shared_deep"):
            all_adjusted.extend(cls.evolve_from_interaction(profile, "user_asked_deep"))

        if conversation_context.get("user_was_brief"):
            all_adjusted.extend(cls.evolve_from_interaction(profile, "user_was_brief"))

        if conversation_context.get("user_flirted"):
            all_adjusted.extend(cls.evolve_from_interaction(profile, "user_flirted_back"))

        return all_adjusted


# ============================================================================
# Trait Profile Manager
# ============================================================================

class TraitProfileManager:
    """Manages trait profiles for companions."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or ROOT_DATA_DIR / "trait_profiles"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_profile_path(self, profile_id: str) -> Path:
        return self.storage_path / f"{profile_id}.json"

    def create_profile(
        self,
        user_id: str,
        assistant_id: str,
        base_system_prompt: str,
        initial_traits: Dict[str, float],
    ) -> TraitProfile:
        """Create a new trait profile.

        Args:
            user_id: User who owns this companion
            assistant_id: Companion ID
            base_system_prompt: Base personality prompt
            initial_traits: Initial trait weights {"jokester": 0.5, ...}

        Returns:
            TraitProfile
        """
        profile_id = f"{user_id}_{assistant_id}"

        # Create trait states
        traits = {}
        for trait_name, weight in initial_traits.items():
            traits[trait_name] = TraitState(
                trait_name=trait_name,
                weight=weight,
                last_updated=datetime.now().isoformat(),
            )

        profile = TraitProfile(
            profile_id=profile_id,
            user_id=user_id,
            assistant_id=assistant_id,
            base_system_prompt=base_system_prompt,
            traits=traits,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )

        self.save_profile(profile)
        logger.info(f"[TRAITS] Created profile '{profile_id}' with {len(traits)} traits")

        return profile

    def get_profile(self, profile_id: str) -> Optional[TraitProfile]:
        """Load a trait profile."""
        path = self._get_profile_path(profile_id)
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)
            return TraitProfile(**data)

    def save_profile(self, profile: TraitProfile) -> None:
        """Save a trait profile."""
        path = self._get_profile_path(profile.profile_id)
        with open(path, "w") as f:
            json.dump(profile.dict(), f, indent=2)

    def update_from_interaction(
        self,
        profile_id: str,
        interaction_type: str,
        strength: float = 0.05,
    ) -> Optional[List[str]]:
        """Update traits based on interaction.

        Args:
            profile_id: Profile to update
            interaction_type: Type of interaction
            strength: Adjustment strength

        Returns:
            List of adjusted traits or None if profile not found
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        adjusted = TraitEvolution.evolve_from_interaction(
            profile, interaction_type, strength
        )

        if adjusted:
            self.save_profile(profile)

        return adjusted


# ============================================================================
# Convenience Functions
# ============================================================================

_manager: Optional[TraitProfileManager] = None


def get_trait_manager() -> TraitProfileManager:
    """Get global trait profile manager."""
    global _manager
    if _manager is None:
        _manager = TraitProfileManager()
    return _manager


def get_available_traits() -> Dict[str, TraitDefinition]:
    """Get all available trait definitions."""
    return TRAIT_LIBRARY.copy()


def get_traits_by_category(category: TraitCategory) -> Dict[str, TraitDefinition]:
    """Get traits in a specific category."""
    return {
        name: defn
        for name, defn in TRAIT_LIBRARY.items()
        if defn.category == category
    }
