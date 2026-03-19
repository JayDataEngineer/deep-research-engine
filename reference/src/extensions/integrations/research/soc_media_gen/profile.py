"""Social Media Content Generator Profile - Personality-driven content generation.

This profile implements social media content generation using personality cards
loaded from YAML, similar to the podcast profile but optimized for social platforms.
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from ....content.deep_research.profiles.base import BaseResearchProfile


logger = logging.getLogger(__name__)

# Setup Jinja environment for this profile only
THIS_DIR = Path(__file__).parent


class SocMediaGenProfile(BaseResearchProfile):
    """Profile for generating social media content with personality-driven approach.

    Personality System:
    - Social media personalities are loaded from YAML config (text_injections.yaml)
    - Each personality has: name, role, description, prefill_prompt, research_directive, content_style
    - Worker behavior (research approach, style guidance) is embedded in each personality

    Pipeline:
    Configurable via YAML character_card_mappings in spawner_config.
    Default: Uses 4 personalities (HOOK_SPECIALIST, STORYTELLER, LISTICLE_EXPERT, HOT_TAKE_ARTIST)
    """

    @property
    def name(self) -> str:
        return "soc_media_gen"

    @property
    def description(self) -> str:
        return "Personality-driven social media content generation with viral optimization"

    # ==========================================================================
    # SOCIAL MEDIA PERSONALITY LOADING
    # ==========================================================================

    def _load_personalities(self) -> Dict[str, Any]:
        """Load social media personalities from local text_injections.yaml.

        Returns:
            Dict mapping personality_id to personality data
        """
        local_path = THIS_DIR / "text_injections.yaml"
        if local_path.exists():
            with open(local_path) as f:
                config = yaml.safe_load(f)
            personalities = config.get("personalities", {})
            logger.info(f"[SOC_MEDIA_GEN] Loaded text_injections from local extension: {local_path}")
            return personalities

        logger.error("[SOC_MEDIA_GEN] text_injections.yaml not found in extension directory")
        return {}

    def _get_personality(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific personality by ID.

        Args:
            personality_id: Personality ID (e.g., 'HOOK_SPECIALIST', 'STORYTELLER')

        Returns:
            Personality dict with name, role, description, prefill_prompt, etc.
        """
        personalities = self._load_personalities()
        return personalities.get(personality_id)

    def _get_worker_behavior(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get worker behavior instructions for a specific personality.

        Worker behavior is embedded WITHIN the personality itself.

        Args:
            personality_id: Personality ID

        Returns:
            Worker behavior dict with prefill_prompt, research_directive, thread_style
        """
        personality = self._get_personality(personality_id)
        if not personality:
            logger.warning(f"[SOC_MEDIA_GEN] No personality found for {personality_id}")
            return None

        # Extract worker behavior fields from personality
        behavior = {
            "prefill_prompt": personality.get("prefill_prompt", ""),
            "research_directive": personality.get("research_directive", ""),
            "thread_style": personality.get("thread_style", ""),
        }

        # Only return if we have actual content
        if behavior["prefill_prompt"] or behavior["research_directive"]:
            logger.info(f"[SOC_MEDIA_GEN] Loaded worker_behavior for {personality_id}")
            return behavior

        logger.warning(f"[SOC_MEDIA_GEN] No worker_behavior found for {personality_id}")
        return None

    # ==========================================================================
    # PROMPT ENHANCEMENT - Twitter Personality Injection
    # ==========================================================================

    def enhance_system_prompt(
        self,
        base_prompt: str,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> str:
        """Enhance worker prompts with Twitter personality cards.

        For mapped workers, injects the full personality definition.

        Args:
            base_prompt: The generic system prompt
            worker_type: Type of worker (e.g., 'researcher_0', 'researcher_1')
            worker_id: Unique worker ID
            state: Current global state

        Returns:
            Enhanced prompt with personality card if applicable
        """
        # Get character card mappings from spawner_config
        character_card_mappings = self.spawner_config.get("character_card_mappings", {})

        if worker_type not in character_card_mappings:
            return base_prompt

        # Load the personality
        personality_id = character_card_mappings[worker_type]
        personality = self._get_personality(personality_id)

        if not personality:
            logger.warning(f"[SOC_MEDIA_GEN] Personality '{personality_id}' not found for worker {worker_type}")
            return base_prompt

        # Build personality injection block
        name = personality.get("name", personality_id)
        role = personality.get("role", "")
        description = personality.get("description", "")
        thread_style = personality.get("thread_style", "")

        persona_block = f"""
## YOUR TWITTER PERSONALITY: {name}

**Role:** {role}

**Your Style:**
{thread_style}

**Who You Are:**
{description}

---
"""

        logger.info(f"[SOC_MEDIA_GEN] Enhanced prompt for {worker_type} with personality {personality_id}")
        return base_prompt + persona_block

    # ==========================================================================
    # CRITIC FEEDBACK - Twitter Quality Standards
    # ==========================================================================

    def process_critic_feedback(self, feedback: str, state: Dict[str, Any]) -> str:
        """Add Twitter-specific quality checks to critic feedback.

        This adds Twitter thread quality requirements to the critic's evaluation.

        Args:
            feedback: Raw critic feedback
            state: Current global state

        Returns:
            Enhanced feedback with Twitter quality requirements
        """
        twitter_requirements = """

## TWITTER THREAD QUALITY REQUIREMENTS:

1. HOOK STRENGTH (30%):
   - Stops the scroll immediately?
   - Creates curiosity gap?
   - Bold or surprising?
   - Under 280 characters?

2. VALUE DELIVERY (25%):
   - Each tweet independently valuable?
   - Actionable insights?
   - Worth saving/sharing?
   - No fluff?

3. THREAD FLOW (20%):
   - Narrative builds properly?
   - Best content in middle tweets?
   - Satisfying conclusion?
   - Makes sense top-to-bottom?

4. VIRAL FACTORS (15%):
   - Shareable insights?
   - Retweet-worthy?
   - Comment bait?
   - Strong CTA?

5. FORMAT (10%):
   - Numbered correctly (1/N, 2/N)?
   - Thread connectors (🧵)?
   - Under 280 chars per tweet?
   - Clean spacing?
"""

        logger.info("[SOC_MEDIA_GEN] Added thread quality requirements to critic feedback")
        return feedback + twitter_requirements

    # ==========================================================================
    # SPECIALIST PROMPTS - Twitter Worker Types
    # ==========================================================================

    def get_worker_specialist_prompt(
        self,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """Get specialist prompt for Twitter workers.

        This allows personality-based specialist prompts.

        Args:
            worker_type: Type of worker
            worker_id: Worker ID
            state: Current state

        Returns:
            Specialist prompt or None
        """
        character_card_mappings = self.spawner_config.get("character_card_mappings", {})

        if worker_type not in character_card_mappings:
            return None

        personality_id = character_card_mappings[worker_type]
        personality = self._get_personality(personality_id)

        if not personality:
            return None

        # Return the research directive as the specialist prompt
        return personality.get("research_directive", None)
