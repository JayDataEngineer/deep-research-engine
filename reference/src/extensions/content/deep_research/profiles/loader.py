"""Profile Loader System - Engine vs Soul separation.

This module implements the "Holy Grail" of agentic software architecture:
**Separating the Engine (Logic) from the Fuel (Content/Persona).**

**Layer A: The Engine (Git Tracked)**
- Python code, LangGraph definitions, Information Funnel logic
- Responsibility: "I know HOW to research, HOW to scrape, HOW to format"

**Layer B: The Soul (Not Git Tracked / User Data)**
- YAML profiles with brand voice, personality, research preferences
- Responsibility: "I know WHO I am, WHAT I care about, WHERE to look"

This enables:
1. Dynamic Boss strategy based on profile research_preferences
2. Context-aware Critic evaluation based on profile voice/banned_words
3. User customizations without touching code
4. DRY configuration inheritance

**CENTRAL CHARACTER REPOSITORY:**
- Uses CHARACTER_TEXT_INJECTIONS_FILE from constants.py as central source of truth
- Path is hot-swappable via constants for easy migration
"""

from pathlib import Path
import os
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.constants import CHARACTER_TEXT_INJECTIONS_FILE

sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger

from typing import Dict, Any, Optional, List, Union
import yaml
from jinja2 import Template, Environment, FileSystemLoader

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


class ProfileLoader:
    """Load and merge configuration profiles with user override support.

    Priority (highest to lowest):
    1. User profiles (data/user_profiles/) - User customizations
    2. System profiles (src/extensions/.../profiles/) - Default profiles

    This enables the "Engine vs Soul" separation:
    - Engine: Python code that reads profile config
    - Soul: YAML profiles with brand voice, personality, preferences
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        user_data_dir: Optional[Path] = None
    ):
        """Initialize the profile loader.

        Args:
            base_dir: Base directory for SYSTEM profiles (defaults to this package)
            user_data_dir: User data directory for USER profiles (overrides system)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent

        self.base_dir = Path(base_dir)
        self.brands_dir = self.base_dir / "brands"
        self.personalities_dir = self.base_dir / "personalities"
        self.templates_dir = self.base_dir / "prompt_templates"

        # User data directory (Layer B: The Soul)
        if user_data_dir is None:
            # Default: data/user_profiles/ relative to backend
            backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            user_data_dir = backend_dir / "data" / "user_profiles"

        self.user_dir = Path(user_data_dir)
        self.user_brands_dir = self.user_dir / "brands"
        self.user_personalities_dir = self.user_dir / "characters"
        self.user_tools_dir = self.user_dir / "tools"

        # Create user directories if they don't exist
        self.user_brands_dir.mkdir(parents=True, exist_ok=True)
        self.user_personalities_dir.mkdir(parents=True, exist_ok=True)
        self.user_tools_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def load_brand_profile(self, brand_name: str) -> Dict[str, Any]:
        """Load a brand profile by name.

        Priority:
        1. User profiles (data/user_profiles/brands/)
        2. System profiles (src/extensions/.../profiles/brands/)

        Args:
            brand_name: Name of brand (e.g., "iron_forge_fitness")

        Returns:
            Brand profile dict

        Raises:
            FileNotFoundError: If brand profile not found in either location
        """
        # 1. Try user profiles first (user customizations override system)
        user_path = self.user_brands_dir / f"{brand_name}.yaml"
        if user_path.exists():
            logger.info(f"[PROFILE_LOADER] Loading user brand profile: {brand_name}")
            with open(user_path) as f:
                return yaml.safe_load(f)

        # 2. Fall back to system profiles (defaults)
        system_path = self.brands_dir / f"{brand_name}.yaml"
        if system_path.exists():
            logger.info(f"[PROFILE_LOADER] Loading system brand profile: {brand_name}")
            with open(system_path) as f:
                return yaml.safe_load(f)

        # 3. Not found anywhere
        raise FileNotFoundError(
            f"Brand profile '{brand_name}' not found in user or system directories.\n"
            f"  User: {self.user_brands_dir}\n"
            f"  System: {self.brands_dir}"
        )

    def load_personality(self, personality_id: str) -> Dict[str, Any]:
        """Load a personality definition.

        Priority:
        1. User profiles (data/user_profiles/characters/)
        2. System profiles (profiles/personalities/)
        3. Central repository (CHARACTER_TEXT_INJECTIONS_FILE from constants)

        Args:
            personality_id: Personality ID (e.g., "EMMA", "OLIVIA", "custom_waifu")

        Returns:
            Personality dict

        Raises:
            FileNotFoundError: If personality not found
        """
        # 1. Try user profiles first (user customizations override system)
        user_path = self.user_personalities_dir / f"{personality_id}.yaml"
        if user_path.exists():
            logger.info(f"[PROFILE_LOADER] Loading user personality: {personality_id}")
            with open(user_path) as f:
                return yaml.safe_load(f)

        # 2. Fall back to system personalities
        system_path = self.personalities_dir / f"{personality_id}.yaml"
        if system_path.exists():
            logger.info(f"[PROFILE_LOADER] Loading system personality: {personality_id}")
            with open(system_path) as f:
                return yaml.safe_load(f)

        # 3. Fall back to central character repository (text_injections.yaml)
        if CHARACTER_TEXT_INJECTIONS_FILE.exists():
            logger.info(f"[PROFILE_LOADER] Loading from central repository: {personality_id}")
            with open(CHARACTER_TEXT_INJECTIONS_FILE) as f:
                data = yaml.safe_load(f)

            if data and "personalities" in data and personality_id in data["personalities"]:
                return data["personalities"][personality_id]

        # 4. Not found anywhere
        raise FileNotFoundError(
            f"\n"
            "╔═══════════════════════════════════════════════════════════════════╗\n"
            "║           PERSONALITY ERROR: Profile Not Found                        ║\n"
            "╚═══════════════════════════════════════════════════════════════════╝\n"
            "\n"
            f"Personality '{personality_id}' not found.\n"
            f"  User: {self.user_personalities_dir}\n"
            f"  System: {self.personalities_dir}\n"
            f"  Central: {CHARACTER_TEXT_INJECTIONS_FILE}\n"
            "\n"
        )

    def load_prompt_template(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Load and render a Jinja2 prompt template.

        Args:
            template_name: Template name (without .j2 extension)
            context: Template variables

        Returns:
            Rendered template string

        Raises:
            FileNotFoundError: If template not found
        """
        try:
            template = self.jinja_env.get_template(f"{template_name}.j2")
            return template.render(**context)
        except Exception as e:
            raise FileNotFoundError(f"Template not found or render failed: {template_name}.j2: {e}")

    def merge_configs(
        self,
        *configs: Dict[str, Any],
        override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge multiple config dicts, with later configs taking precedence.

        Args:
            *configs: Config dicts to merge (in order of precedence)
            override: Optional final override dict

        Returns:
            Merged config dict
        """
        result = {}

        for config in configs:
            result = self._deep_merge(result, config)

        if override:
            result = self._deep_merge(result, override)

        return result

    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dicts.

        Args:
            base: Base dict
            override: Override dict (takes precedence)

        Returns:
            Merged dict
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override with new value
                result[key] = value

        return result

    def load_variant_with_brand(
        self,
        variant_config: Dict[str, Any],
        brand_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a variant config with brand profile inheritance.

        Args:
            variant_config: Variant-specific YAML config
            brand_name: Optional brand profile to inherit from

        Returns:
            Merged config with brand profile applied
        """
        # If no brand specified, return variant as-is
        if not brand_name:
            return variant_config

        # Load brand profile
        try:
            brand_profile = self.load_brand_profile(brand_name)
        except FileNotFoundError:
            logger.warning(f"Brand profile '{brand_name}' not found, using variant config only")
            return variant_config

        # Check if variant has _extends key
        extends = variant_config.get("_extends", [])
        if isinstance(extends, str):
            extends = [extends]

        # Merge configs in order: brand → variant
        configs_to_merge = []
        if brand_profile.get("brand_context"):
            # Wrap brand_context to match variant structure
            configs_to_merge.append({"brand_context": brand_profile["brand_context"]})

        configs_to_merge.append(variant_config)

        merged = self.merge_configs(*configs_to_merge)

        # Store brand metadata
        merged["_brand"] = brand_name
        merged["_brand_profile"] = brand_profile.get("_metadata", {})

        return merged

    def build_worker_prompt_from_template(
        self,
        worker_role: str,
        brand_context: Dict[str, Any],
        specific_instructions: Optional[str] = None,
        query: Optional[str] = None,
        previous_findings: Optional[str] = None,
        template_name: str = "worker_base"
    ) -> str:
        """Build a worker prompt from template.

        Args:
            worker_role: Worker role (e.g., "MOTIVATION HAMMER")
            brand_context: Brand context dict
            specific_instructions: Worker-specific instructions
            query: Research query/topic
            previous_findings: Previous worker outputs
            template_name: Template to use (default: worker_base)

        Returns:
            Rendered prompt string
        """
        context = {
            "worker_role": worker_role,
            "brand_context": brand_context,
            "specific_instructions": specific_instructions,
            "query": query,
            "previous_findings": previous_findings,
        }

        return self.load_prompt_template(template_name, context)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_profile_loader: Optional[ProfileLoader] = None


def get_profile_loader() -> ProfileLoader:
    """Get the singleton profile loader instance.

    Returns:
        ProfileLoader instance
    """
    global _profile_loader
    if _profile_loader is None:
        _profile_loader = ProfileLoader()
    return _profile_loader


def load_variant_config(
    variant_config: Dict[str, Any],
    brand_name: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to load variant config with brand inheritance.

    Args:
        variant_config: Variant-specific YAML config
        brand_name: Optional brand profile to inherit from

    Returns:
        Merged config with brand profile applied
    """
    loader = get_profile_loader()
    return loader.load_variant_with_brand(variant_config, brand_name)


def build_worker_prompt(
    worker_role: str,
    brand_context: Dict[str, Any],
    specific_instructions: Optional[str] = None,
    query: Optional[str] = None,
    previous_findings: Optional[str] = None,
    template_name: str = "worker_base"
) -> str:
    """Convenience function to build worker prompt from template.

    Args:
        worker_role: Worker role (e.g., "MOTIVATION HAMMER")
        brand_context: Brand context dict
        specific_instructions: Worker-specific instructions
        query: Research query/topic
        previous_findings: Previous worker outputs
        template_name: Template to use (default: worker_base)

    Returns:
        Rendered prompt string
    """
    loader = get_profile_loader()
    return loader.build_worker_prompt_from_template(
        worker_role=worker_role,
        brand_context=brand_context,
        specific_instructions=specific_instructions,
        query=query,
        previous_findings=previous_findings,
        template_name=template_name
    )


# =============================================================================
# BOSS & CRITIC PROFILE INTEGRATION
# =============================================================================

def get_active_profile(yaml_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the active profile from config for Boss/Critic injection.

    This extracts the brand_context from config for use by Boss and Critic nodes.

    Args:
        yaml_config: Full YAML configuration dict

    Returns:
        Brand profile dict or None
    """
    brand_context = yaml_config.get("brand_context", {})

    # Also check for _brand metadata
    brand_name = yaml_config.get("_brand")
    if brand_name and not brand_context:
        try:
            loader = get_profile_loader()
            profile = loader.load_brand_profile(brand_name)
            return profile.get("brand_context", {})
        except FileNotFoundError:
            pass

    return brand_context or None


def get_boss_strategy_from_profile(
    yaml_config: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    """Extract Boss strategy settings from active profile.

    The Boss uses this to:
    - Determine research bias (e.g., "Favor evolutionary biology")
    - Set quality thresholds (tier_a_threshold)
    - Choose worker specializations

    Args:
        yaml_config: Full YAML configuration dict
        query: User query to research

    Returns:
        Strategy dict with keys like research_bias, quality_threshold, etc.
    """
    profile = get_active_profile(yaml_config)
    if not profile:
        return {}

    # Extract strategy-relevant settings from profile
    strategy = {
        "research_bias": profile.get("research_preferences", {}).get("bias"),
        "tier_a_threshold": profile.get("research_preferences", {}).get("tier_a_threshold"),
        "focus_themes": profile.get("themes", [])[:3],  # Top 3 themes for this query
        "avoid_topics": profile.get("avoid", []),
        "voice_style": profile.get("tone", {}).get("style"),
    }

    # Remove None values
    return {k: v for k, v in strategy.items() if v is not None}


def get_critic_criteria_from_profile(
    yaml_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract Critic evaluation criteria from active profile.

    The Critic uses this to:
    - Check for banned words/phrases
    - Validate brand voice alignment
    - Enforce style guidelines

    Args:
        yaml_config: Full YAML configuration dict

    Returns:
        Criteria dict with validation rules

    Example:
        {
            "banned_words": ["try", "gentle", "maybe"],
            "required_tone": "aggressive, stoic",
            "voice_style": "direct_address",
            "quality_threshold": 0.8
        }
    """
    profile = get_active_profile(yaml_config)
    if not profile:
        return {}

    # Extract evaluation criteria from profile
    criteria = {
        "banned_words": profile.get("voice", {}).get("banned_words", []),
        "banned_phrases": profile.get("voice", {}).get("banned_phrases", []),
        "required_tone": profile.get("tone", {}).get("style"),
        "voice_style": profile.get("tone", {}).get("voice"),
        "energy_level": profile.get("tone", {}).get("energy"),
        "keywords": profile.get("tone", {}).get("keywords", []),
        "avoid": profile.get("avoid", []),
    }

    # Remove None values
    return {k: v for k, v in criteria.items() if v is not None}


def validate_content_against_profile(
    content: str,
    yaml_config: Dict[str, Any]
) -> tuple[bool, List[str]]:
    """Validate content against profile criteria.

    This is what the Critic uses to check:
    - No banned words
    - Brand voice matches
    - Required keywords present

    Args:
        content: Generated content to validate
        yaml_config: Full YAML configuration dict

    Returns:
        (is_valid, issues) tuple
    """
    criteria = get_critic_criteria_from_profile(yaml_config)
    if not criteria:
        return True, []

    issues = []
    content_lower = content.lower()

    # Check for banned words
    banned = criteria.get("banned_words", [])
    for word in banned:
        if word.lower() in content_lower:
            issues.append(f"Contains banned word: '{word}'")

    # Check for banned phrases
    banned_phrases = criteria.get("banned_phrases", [])
    for phrase in banned_phrases:
        if phrase.lower() in content_lower:
            issues.append(f"Contains banned phrase: '{phrase}'")

    # Check for required keywords (if specified)
    keywords = criteria.get("keywords", [])
    if keywords:
        # At least some keywords should be present
        found = sum(1 for kw in keywords if kw.lower() in content_lower)
        if found == 0:
            issues.append(f"None of the required keywords present: {keywords}")

    is_valid = len(issues) == 0
    return is_valid, issues
