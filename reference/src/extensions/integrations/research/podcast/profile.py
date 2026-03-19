"""Podcast Profile - Character-driven political podcast generation.

This profile implements character-specific customizations on top of the generic
preset graph provided by BaseResearchProfile, using Jinja2 for prompt templating.
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from jinja2 import Environment, FileSystemLoader

from ....content.deep_research.profiles.base import BaseResearchProfile


logger = logging.getLogger(__name__)

# Setup Jinja environment for this profile only
THIS_DIR = Path(__file__).parent
jinja_env = Environment(loader=FileSystemLoader(THIS_DIR / "templates"))


class PodcastProfile(BaseResearchProfile):
    """Profile for generating podcast scripts with contrasting characters.

    Character System:
    - Personalities are loaded from YAML config (podcast_personalities.yaml)
    - Each personality has: name, role, description, emotion_bias, speech_patterns, ideological_framework
    - Worker behavior (research_directive, sources, etc.) is embedded in each personality

    Pipeline:
    Configurable via YAML specialist_configs and worker_graph.
    Default: news_aggregator → character_a_research → character_b_research → character_a → character_b → scriptwriter

    Note: The preset graph logic is inherited from BaseResearchProfile.get_preset_boss_decision()
    which works generically for any use case with specialist_configs + worker_graph in YAML.
    """

    @property
    def name(self) -> str:
        return "podcast"

    @property
    def description(self) -> str:
        return "Character-driven political podcast generation with contrasting personalities"

    # ==========================================================================
    # CHARACTER CARD LOADING
    # ==========================================================================

    def _load_personalities(self) -> Dict[str, Any]:
        """Load character personalities from local text_injections.yaml.

        Returns:
            Dict mapping personality_id to personality data
        """
        local_path = THIS_DIR / "text_injections.yaml"
        if local_path.exists():
            with open(local_path) as f:
                config = yaml.safe_load(f)
            personalities = config.get("personalities", {})
            logger.info(f"[PODCAST] Loaded text_injections from local extension: {local_path}")
            return personalities

        logger.error("[PODCAST] text_injections.yaml not found in extension directory")
        return {}

    def _get_personality(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific personality by ID.

        Args:
            personality_id: Personality ID (e.g., 'PERSONALITY_A', 'PERSONALITY_B')

        Returns:
            Personality dict with name, role, description, etc.
        """
        personalities = self._load_personalities()
        return personalities.get(personality_id)

    def _get_worker_behavior(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get worker behavior instructions for a specific personality.

        Worker behavior is now embedded WITHIN the personality itself.

        Args:
            personality_id: Personality ID (e.g., 'PERSONALITY_A', 'PERSONALITY_B')

        Returns:
            Worker behavior dict with research_sources, community_platforms, research_directive
        """
        # Get the personality - worker behavior is now inside it
        personality = self._get_personality(personality_id)
        if not personality:
            logger.warning(f"[PODCAST] No personality found for {personality_id}")
            return None

        # Extract worker behavior fields from personality
        behavior = {
            "research_sources": personality.get("research_sources", []),
            "community_platforms": personality.get("community_platforms", []),
            "research_directive": personality.get("research_directive", ""),
            "prefill_prompt": personality.get("prefill_prompt", ""),
        }

        # Only return if we have actual content
        if behavior["research_directive"] or behavior["research_sources"]:
            logger.info(f"[PODCAST] Loaded worker_behavior for {personality_id} from personality entry")
            return behavior

        logger.warning(f"[PODCAST] No worker_behavior found for {personality_id}")
        return None

    def _extract_voice_traits(self, personality: Dict[str, Any]) -> list[str]:
        """Extract voice traits from personality description.

        Args:
            personality: Personality dict

        Returns:
            List of voice trait strings
        """
        description = personality.get("description", "")
        traits = []

        # Common voice traits to look for
        if "bubbly" in description.lower():
            traits.append("Bubbly and energetic")
        if "moody" in description.lower() or "melancholy" in description.lower():
            traits.append("Moody and introspective")
        if "sarcastic" in description.lower():
            traits.append("Sarcastic and cynical")
        if "calculating" in description.lower():
            traits.append("Strategic and calculating")
        if "passionate" in description.lower():
            traits.append("Intensely passionate")
        if "philosophical" in description.lower():
            traits.append("Philosophical and analytical")

        return traits

    # ==========================================================================
    # PROMPT ENHANCEMENT - Character Injection with Jinja
    # ==========================================================================

    def enhance_system_prompt(
        self,
        base_prompt: str,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> str:
        """Enhance worker prompts with character cards using Jinja templates.

        For character workers (character_a, character_b), this renders
        a Jinja template with the full personality definition.

        Args:
            base_prompt: The generic system prompt
            worker_type: Type of worker (e.g., 'character_a', 'scriptwriter')
            worker_id: Unique worker ID
            state: Current global state

        Returns:
            Enhanced prompt with character card if applicable
        """
        # Get character card mappings from spawner_config
        character_card_mappings = self.spawner_config.get("character_card_mappings", {})

        if worker_type not in character_card_mappings:
            return base_prompt

        # Load the personality
        personality_id = character_card_mappings[worker_type]
        personality = self._get_personality(personality_id)

        if not personality:
            logger.warning(f"[PODCAST] Personality '{personality_id}' not found for worker {worker_type}")
            return base_prompt

        # Render Jinja template for character persona
        try:
            template = jinja_env.get_template("character_persona.j2")
            persona_block = template.render(
                character_role=personality.get("role"),
                character_name=personality.get("name"),
                character_description=personality.get("description"),
                voice_traits=self._extract_voice_traits(personality),
                speech_patterns=personality.get("speech_patterns", []),
                ideological_framework=personality.get("ideological_framework", "")
            )

            logger.info(f"[PODCAST] Enhanced prompt for {worker_type} with personality {personality_id} (Jinja)")
            return f"{base_prompt}\n\n{persona_block}"

        except Exception as e:
            logger.error(f"[PODCAST] Failed to render character_persona.j2: {e}")
            # Fallback to simple format
            return base_prompt

    # ==========================================================================
    # CRITIC FEEDBACK - Character Consistency with Jinja
    # ==========================================================================

    def process_critic_feedback(self, feedback: str, state: Dict[str, Any]) -> str:
        """Add character consistency checking to critic feedback using Jinja templates.

        This renders a Jinja template with character cards for the critic to check.

        Args:
            feedback: Raw critic feedback
            state: Current global state

        Returns:
            Enhanced feedback with character consistency requirements
        """
        # Get character mappings from config
        character_mappings = self.spawner_config.get("character_card_mappings", {})

        if not character_mappings:
            return feedback

        # Load personalities
        personalities = self._load_personalities()

        if not personalities:
            return feedback

        # Build character list for template
        characters = []
        for worker_type, personality_id in character_mappings.items():
            if personality_id in personalities:
                personality = personalities[personality_id]
                characters.append({
                    "name": personality.get("name", personality_id),
                    "role": personality.get("role", ""),
                    "description": personality.get("description", ""),
                    "key_phrases": personality.get("key_phrases", []),
                    "ideological_stance": personality.get("ideological_stance", "")
                })

        if not characters:
            return feedback

        # Render Jinja template for critic consistency
        try:
            template = jinja_env.get_template("critic_consistency.j2")
            consistency_block = template.render(characters=characters)

            logger.info(f"[PODCAST] Added character consistency for critic: {len(characters)} characters (Jinja)")
            return feedback + "\n\n" + consistency_block

        except Exception as e:
            logger.error(f"[PODCAST] Failed to render critic_consistency.j2: {e}")
            # Fallback to simple format
            return feedback
