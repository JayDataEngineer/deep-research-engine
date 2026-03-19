"""Research Profile Interface - Strategy Pattern for Deep Research.

This module defines the abstract contract for all run profiles (e.g., podcast, news, trading).
Each profile can customize behavior without modifying the core orchestrator.

Architecture:
- Deep Research Orchestrator = Context
- ResearchProfile implementations = Strategies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

from langgraph.graph import StateGraph


class ResearchProfile(ABC):
    """Abstract base class for deep research run profiles.

    A profile defines how a specific use case (podcast, news, trading) customizes
    the generic deep research engine. Profiles use the Strategy Pattern to avoid
    hardcoded if/else chains in the orchestrator.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Profile identifier (e.g., 'podcast', 'news', 'trading')."""
        pass

    def __init__(self, config: Dict[str, Any]):
        """Initialize the profile with its configuration.

        Args:
            config: The full YAML config dict for this profile (merged from
                    base + use_case + variant). Contains all profile-specific settings.
        """
        self.config = config
        self.yaml_config = config  # Alias for consistency

    @property
    def spawner_config(self) -> Dict[str, Any]:
        """Get spawner_config from config."""
        return self.config.get("spawner_config", {})

    # ==========================================================================
    # HOOK 1: Topology Customization
    # ==========================================================================

    def get_custom_topology(self, builder: StateGraph) -> Optional[StateGraph]:
        """Optionally modify the graph structure before compilation.

        This allows profiles to add custom nodes, edges, or entire subgraphs.
        For example, a podcast profile might add a TTS generation node.

        Args:
            builder: The StateGraph builder with base nodes already added

        Returns:
            Modified builder, or None to use default topology
        """
        return None

    # ==========================================================================
    # HOOK 2: Prompt Enhancement
    # ==========================================================================

    def enhance_system_prompt(
        self,
        base_prompt: str,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> str:
        """Enhance a worker's system prompt with profile-specific content.

        This is the primary hook for injecting domain-specific instructions.
        For example, a podcast profile might inject character cards here.

        Args:
            base_prompt: The generic system prompt from the YAML config
            worker_type: Type of worker (e.g., 'scriptwriter', 'character_a', 'character_b')
            worker_id: Unique worker ID (e.g., 'worker_0', 'news_aggregator')
            state: Current global state (for accessing research_brief, etc.)

        Returns:
            Enhanced prompt with profile-specific content injected
        """
        return base_prompt

    # ==========================================================================
    # HOOK 3: Critic Feedback Processing
    # ==========================================================================

    def process_critic_feedback(self, feedback: str, state: Dict[str, Any]) -> str:
        """Optionally process or augment critic feedback for profile-specific needs.

        Different profiles may care about different quality aspects.
        For example, a podcast profile might add character consistency checking.

        Args:
            feedback: Raw feedback from the critic node
            state: Current global state

        Returns:
            Processed feedback (may be augmented with profile-specific checks)
        """
        return feedback

    # ==========================================================================
    # HOOK 4: Worker Specialization
    # ==========================================================================

    def get_worker_specialist_prompt(
        self,
        worker_type: str,
        worker_id: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """Get a custom specialist prompt for a specific worker type.

        This is called when the YAML config defines a specialist_configs section.
        The profile can customize how specialist prompts are constructed.

        Args:
            worker_type: Type from specialist_configs (e.g., 'scriptwriter')
            worker_id: Unique ID for this worker instance
            state: Current global state

        Returns:
            Custom specialist prompt, or None to use YAML config default
        """
        return None

    # ==========================================================================
    # UTILITY: Config Access Helpers
    # ==========================================================================

    @property
    def spawner_config(self) -> Dict[str, Any]:
        """Quick access to spawner_config section."""
        return self.config.get("spawner_config", {})

    @property
    def output_config(self) -> Dict[str, Any]:
        """Quick access to output config section."""
        return self.config.get("output", {})

    @property
    def prompts_config(self) -> Dict[str, Any]:
        """Quick access to prompts config section."""
        return self.config.get("prompts", {})


class BaseResearchProfile(ResearchProfile):
    """Default implementation with no customizations.

    This is the fallback profile when no specific profile is found.
    It implements all hooks as no-ops, preserving the default behavior.
    """

    @property
    def name(self) -> str:
        return "base"


def load_profile_config(
    use_case: str,
    variant: str = "default",
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Load merged config for a specific profile.

    This is a convenience wrapper for loading configs before profile instantiation.

    Note: config_dir parameter is ignored in the new extension system.
    Configs are loaded from fixed paths defined by constants (DEEP_RESEARCH_CONFIG_DIR,
    RESEARCH_EXTENSIONS_DIR).

    Args:
        use_case: Profile name (e.g., 'podcast', 'soc_media_gen')
        variant: Variant name (default: 'default')
        config_dir: Ignored (kept for backwards compatibility)

    Returns:
        Merged config dict (base + use_case + variant)
    """
    from .. import load_research_config

    # config_dir is ignored - new extension system uses fixed paths from constants
    return load_research_config(
        use_case=use_case,
        variant=variant
    )
